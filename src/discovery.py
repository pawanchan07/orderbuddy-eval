"""
The intent-discovery method, as a single reusable implementation.

This module is deliberately the *only* place the method is defined. Step 2
(discover the OrderBuddy taxonomy) and step 3 (validate the method against
Banking77 / CLINC150) both import from here, so the benchmark numbers describe
the same pipeline that produced the taxonomy — not a re-implementation of it.

Pipeline:
    text -> all-MiniLM-L6-v2 embedding -> L2 normalise
         -> UMAP dimensionality reduction (cosine)
         -> HDBSCAN (euclidean, excess-of-mass)
         -> cluster labels, -1 = noise

UMAP sits between the encoder and HDBSCAN because density-based clustering
degrades badly in 384 dimensions (distance concentration). This is the
standard BERTopic-style arrangement and is applied identically to the
synthetic corpus and to both public benchmarks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "interim"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Method hyper-parameters. Held constant across every dataset the method is
# run on — tuning these per-dataset would invalidate the validation in step 3.
# Selected by src/sweep_params.py on the synthetic corpus only; the public
# benchmarks in step 3 are then scored with these values unchanged.
UMAP_PARAMS = dict(
    n_components=10,
    n_neighbors=80,
    min_dist=0.0,
    metric="cosine",
    random_state=20260824,
)
HDBSCAN_PARAMS = dict(
    min_samples=15,
    metric="euclidean",
    cluster_selection_method="eom",
)

# `min_cluster_size` is the method's one scale-dependent knob and cannot be a
# fixed constant: a value tuned for a 16-intent corpus (150) exceeds the mean
# class size of Banking77 (~170) and CLINC150 (~150), so it would erase whole
# classes at benchmark scale and make step 3 meaningless.
#
# Instead it is derived from corpus size and one stated prior — roughly how
# many intents the taxonomy is expected to contain. GRANULARITY_FACTOR was
# fixed at the value that reproduces the swept optimum on the synthetic corpus
# (10,000 / 16 * 0.25 = 156 ~= the swept best of 150) and is then applied
# unchanged to both benchmarks.
GRANULARITY_FACTOR = 0.25
MIN_CLUSTER_FLOOR = 15


def min_cluster_size_for(n_rows: int, k_expected: int) -> int:
    """Derive min_cluster_size from corpus size and the expected intent count."""
    return max(MIN_CLUSTER_FLOOR,
               round(GRANULARITY_FACTOR * n_rows / max(k_expected, 1)))


@dataclass
class DiscoveryResult:
    labels: np.ndarray                 # cluster id per row, -1 = noise
    embeddings: np.ndarray             # L2-normalised sentence embeddings
    reduced: np.ndarray                # UMAP projection
    n_clusters: int
    noise_fraction: float
    params: dict = field(default_factory=dict)


_model_cache: dict[str, object] = {}


def _get_model():
    """Load the encoder once per process."""
    if MODEL_NAME not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[MODEL_NAME] = SentenceTransformer(MODEL_NAME)
    return _model_cache[MODEL_NAME]


def embed(texts: list[str], cache_key: str | None = None,
          batch_size: int = 256) -> np.ndarray:
    """Encode texts with all-MiniLM-L6-v2 and L2-normalise.

    Embeddings are cached to data/interim (gitignored) keyed by a hash of the
    input, so reruns are fast but a changed corpus always re-encodes.
    """
    digest = hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()[:16]
    cache_path = None
    if cache_key:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"emb_{cache_key}_{digest}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if cache_path is not None:
        np.save(cache_path, vecs)
    return vecs


def discover(texts: list[str], k_expected: int, cache_key: str | None = None,
             hdbscan_params: dict | None = None) -> DiscoveryResult:
    """Run the full discovery pipeline over a list of texts.

    `k_expected` is the granularity prior described above — the approximate
    number of intents the taxonomy is expected to contain. It sets
    min_cluster_size and nothing else; no label information reaches the
    clustering itself.
    """
    import umap
    import hdbscan as hdbscan_lib

    vecs = embed(texts, cache_key=cache_key)

    reducer = umap.UMAP(**UMAP_PARAMS)
    reduced = reducer.fit_transform(vecs)

    params = {
        **HDBSCAN_PARAMS,
        "min_cluster_size": min_cluster_size_for(len(texts), k_expected),
        **(hdbscan_params or {}),
    }
    clusterer = hdbscan_lib.HDBSCAN(**params)
    labels = clusterer.fit_predict(reduced)

    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_fraction = float((labels == -1).mean())

    return DiscoveryResult(
        labels=labels,
        embeddings=vecs,
        reduced=np.asarray(reduced),
        n_clusters=n_clusters,
        noise_fraction=noise_fraction,
        params={"umap": UMAP_PARAMS, "hdbscan": params, "encoder": MODEL_NAME},
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_against_gold(labels: np.ndarray, gold: list[str],
                       recovery_f1_threshold: float = 0.50) -> dict:
    """Score discovered clusters against ground-truth intent labels.

    Two families of number are reported, because they answer different
    questions and are easy to conflate:

    `intents_recovered` — how many of the gold intents the method found *at
        all*. Clusters are matched one-to-one to gold intents by Hungarian
        assignment on overlap count; a gold intent counts as recovered when
        its assigned cluster reaches `recovery_f1_threshold` F1 against it.
        This is the "75/77"-style headline number.

    `accuracy` / `ari` / `nmi` — how well the *row-level* assignment matches
        gold. Accuracy is computed over the Hungarian mapping, with noise
        (-1) rows counted as errors rather than silently dropped, since a row
        the method refuses to cluster is a row it did not classify.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    gold_arr = np.asarray(gold)
    gold_names = sorted(set(gold_arr.tolist()))
    gold_index = {g: i for i, g in enumerate(gold_names)}

    cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)
    cluster_index = {c: i for i, c in enumerate(cluster_ids)}

    n_gold, n_clust = len(gold_names), len(cluster_ids)

    # contingency[cluster, gold]
    contingency = np.zeros((n_clust, n_gold), dtype=np.int64)
    for lab, g in zip(labels, gold_arr):
        if lab != -1:
            contingency[cluster_index[lab], gold_index[g]] += 1

    recovered: list[str] = []
    per_intent: dict[str, dict] = {}
    mapping: dict[int, str] = {}

    if n_clust:
        # Hungarian assignment maximising total overlap (pad to square).
        size = max(n_clust, n_gold)
        cost = np.zeros((size, size), dtype=np.int64)
        cost[:n_clust, :n_gold] = -contingency
        rows, cols = linear_sum_assignment(cost)

        cluster_totals = contingency.sum(axis=1)
        gold_totals = np.array(
            [int((gold_arr == g).sum()) for g in gold_names], dtype=np.int64
        )

        for r, c in zip(rows, cols):
            if r >= n_clust or c >= n_gold:
                continue
            overlap = int(contingency[r, c])
            if overlap == 0:
                continue
            gold_name = gold_names[c]
            precision = overlap / cluster_totals[r] if cluster_totals[r] else 0.0
            recall = overlap / gold_totals[c] if gold_totals[c] else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) else 0.0)

            mapping[cluster_ids[r]] = gold_name
            per_intent[gold_name] = {
                "cluster_id": int(cluster_ids[r]),
                "overlap": overlap,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "recovered": bool(f1 >= recovery_f1_threshold),
            }
            if f1 >= recovery_f1_threshold:
                recovered.append(gold_name)

    # Row-level accuracy under the Hungarian mapping; noise counts as wrong.
    correct = sum(
        1 for lab, g in zip(labels, gold_arr)
        if lab != -1 and mapping.get(lab) == g
    )
    accuracy = correct / len(gold_arr) if len(gold_arr) else 0.0

    # Clustered-only accuracy, for transparency about what noise costs.
    clustered_mask = labels != -1
    acc_clustered = (
        correct / int(clustered_mask.sum()) if clustered_mask.sum() else 0.0
    )

    missed = [g for g in gold_names if g not in recovered]

    return {
        "n_gold_intents": n_gold,
        "n_clusters_found": n_clust,
        "intents_recovered": len(recovered),
        "intent_recovery_str": f"{len(recovered)}/{n_gold}",
        "recovery_f1_threshold": recovery_f1_threshold,
        "accuracy": round(float(accuracy), 4),
        "accuracy_clustered_only": round(float(acc_clustered), 4),
        "noise_fraction": round(float((labels == -1).mean()), 4),
        "ari": round(float(adjusted_rand_score(gold_arr, labels)), 4),
        "nmi": round(float(normalized_mutual_info_score(gold_arr, labels)), 4),
        "missed_intents": missed,
        "per_intent": per_intent,
        "cluster_to_intent": {str(k): v for k, v in mapping.items()},
    }


def score_many_to_one(labels: np.ndarray, gold: list[str]) -> dict:
    """Score under the lenient many-to-one criterion.

    Every cluster is mapped to its own plurality gold label. Several clusters
    may map to the same intent, so a fragmented-but-pure clustering scores
    well here where the strict one-to-one criterion penalises it.

    This is the conventional "clustering accuracy / many-to-one accuracy"
    used throughout the clustering literature. It measures cluster *purity*:
    given that a message landed in some cluster, does that cluster's dominant
    label match the message's true label?

    It does not measure whether a usable taxonomy was produced — that is what
    the strict criterion measures — because nothing here penalises splitting
    one intent across ten clusters.

    Noise (-1) rows count as incorrect, as in the strict scorer: a row the
    method declined to cluster is a row it did not classify.
    """
    gold_arr = np.asarray(gold)
    gold_names = sorted(set(gold_arr.tolist()))
    gold_index = {g: i for i, g in enumerate(gold_names)}
    cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)
    n_gold = len(gold_names)

    if not cluster_ids:
        return {
            "criterion": "many_to_one_plurality",
            "n_gold_intents": n_gold,
            "n_clusters_found": 0,
            "intents_recovered": 0,
            "intent_recovery_str": f"0/{n_gold}",
            "accuracy": 0.0,
            "accuracy_clustered_only": 0.0,
            "noise_fraction": round(float((labels == -1).mean()), 4),
            "mean_clusters_per_recovered_intent": 0.0,
            "missed_intents": gold_names,
        }

    cluster_index = {c: i for i, c in enumerate(cluster_ids)}
    contingency = np.zeros((len(cluster_ids), n_gold), dtype=np.int64)
    for lab, g in zip(labels, gold_arr):
        if lab != -1:
            contingency[cluster_index[lab], gold_index[g]] += 1

    # Each cluster takes its plurality label.
    mapping: dict[int, str] = {}
    fragments: dict[str, int] = {}
    for cid in cluster_ids:
        row = contingency[cluster_index[cid]]
        if row.sum() == 0:
            continue
        name = gold_names[int(row.argmax())]
        mapping[cid] = name
        fragments[name] = fragments.get(name, 0) + 1

    recovered = sorted(fragments)
    correct = sum(
        1 for lab, g in zip(labels, gold_arr)
        if lab != -1 and mapping.get(int(lab)) == g
    )
    n = len(gold_arr)
    clustered = int((labels != -1).sum())

    return {
        "criterion": "many_to_one_plurality",
        "n_gold_intents": n_gold,
        "n_clusters_found": len(cluster_ids),
        "intents_recovered": len(recovered),
        "intent_recovery_str": f"{len(recovered)}/{n_gold}",
        "accuracy": round(correct / n, 4) if n else 0.0,
        "accuracy_clustered_only": round(correct / clustered, 4) if clustered else 0.0,
        "noise_fraction": round(float((labels == -1).mean()), 4),
        "mean_clusters_per_recovered_intent": (
            round(len(cluster_ids) / len(recovered), 2) if recovered else 0.0
        ),
        "max_clusters_for_one_intent": max(fragments.values()) if fragments else 0,
        "missed_intents": [g for g in gold_names if g not in fragments],
    }


def score_dual(labels: np.ndarray, gold: list[str],
               recovery_f1_threshold: float = 0.50) -> dict:
    """Score under both criteria and report them side by side.

    Neither number is the whole truth. The strict criterion answers "did this
    produce a usable taxonomy"; the lenient one answers "are the clusters
    pure". The gap between them is itself the finding: it is a direct measure
    of how fragmented the clustering is.
    """
    strict = score_against_gold(labels, gold, recovery_f1_threshold)
    lenient = score_many_to_one(labels, gold)
    return {
        "strict": strict,
        "lenient": lenient,
        "criterion_gap": {
            "recovery_delta": lenient["intents_recovered"] - strict["intents_recovered"],
            "accuracy_delta": round(lenient["accuracy"] - strict["accuracy"], 4),
            "interpretation": (
                "The gap is fragmentation: clusters that are pure enough to "
                "take an intent's plurality but too small or too numerous to "
                "be that intent's single one-to-one match."
            ),
        },
    }


def recovery_variants(labels: np.ndarray, gold: list[str]) -> dict:
    """Count recovered intents under four different published conventions.

    "How many intents did the method find?" has no single agreed definition,
    and the conventions differ by a wide margin on the same clustering. This
    reports all four side by side so a headline figure can never be quoted
    without the rule that produced it.

    strict_1to1_f1_50   - one-to-one Hungarian match, F1 >= 0.50 (this repo's
                          headline metric; the most conservative)
    strict_1to1_any     - one-to-one Hungarian match, any non-zero overlap
    many_to_one_plurality
                        - a gold intent is recovered if ANY cluster's plurality
                          label is that intent. Several fragments of one intent
                          all count, so this rewards over-clustering.
    many_to_one_pure_50 - as above, but the cluster must also be >= 50% pure.
    """
    from scipy.optimize import linear_sum_assignment

    gold_arr = np.asarray(gold)
    gold_names = sorted(set(gold_arr.tolist()))
    gold_index = {g: i for i, g in enumerate(gold_names)}
    cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)

    n_gold, n_clust = len(gold_names), len(cluster_ids)
    if n_clust == 0:
        zero = f"0/{n_gold}"
        return {k: zero for k in
                ("strict_1to1_f1_50", "strict_1to1_any",
                 "many_to_one_plurality", "many_to_one_pure_50")}

    cluster_index = {c: i for i, c in enumerate(cluster_ids)}
    contingency = np.zeros((n_clust, n_gold), dtype=np.int64)
    for lab, g in zip(labels, gold_arr):
        if lab != -1:
            contingency[cluster_index[lab], gold_index[g]] += 1

    cluster_totals = contingency.sum(axis=1)
    gold_totals = np.array([int((gold_arr == g).sum()) for g in gold_names])

    # Many-to-one: plurality label of each cluster.
    plurality = set()
    plurality_pure = set()
    for r in range(n_clust):
        if cluster_totals[r] == 0:
            continue
        c = int(contingency[r].argmax())
        plurality.add(gold_names[c])
        if contingency[r, c] / cluster_totals[r] >= 0.50:
            plurality_pure.add(gold_names[c])

    # One-to-one Hungarian.
    size = max(n_clust, n_gold)
    cost = np.zeros((size, size), dtype=np.int64)
    cost[:n_clust, :n_gold] = -contingency
    rows, cols = linear_sum_assignment(cost)

    strict_f1, strict_any = 0, 0
    for r, c in zip(rows, cols):
        if r >= n_clust or c >= n_gold:
            continue
        overlap = int(contingency[r, c])
        if overlap == 0:
            continue
        strict_any += 1
        p = overlap / cluster_totals[r] if cluster_totals[r] else 0.0
        rec = overlap / gold_totals[c] if gold_totals[c] else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        if f1 >= 0.50:
            strict_f1 += 1

    return {
        "strict_1to1_f1_50": f"{strict_f1}/{n_gold}",
        "strict_1to1_any": f"{strict_any}/{n_gold}",
        "many_to_one_plurality": f"{len(plurality)}/{n_gold}",
        "many_to_one_pure_50": f"{len(plurality_pure)}/{n_gold}",
    }


# --------------------------------------------------------------------------
# Taxonomy labelling
# --------------------------------------------------------------------------

def cluster_keywords(texts: list[str], labels: np.ndarray,
                     top_n: int = 8) -> dict[int, list[str]]:
    """Class-based TF-IDF: the terms that distinguish each cluster.

    Documents in a cluster are concatenated into one pseudo-document, so the
    scoring answers "what is distinctive about this cluster" rather than
    "what is frequent in this document".
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)
    docs = [
        " ".join(t for t, lab in zip(texts, labels) if lab == cid)
        for cid in cluster_ids
    ]
    if not docs:
        return {}

    vec = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", min_df=1, sublinear_tf=True
    )
    matrix = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())

    out: dict[int, list[str]] = {}
    for i, cid in enumerate(cluster_ids):
        row = matrix[i].toarray().ravel()
        top = row.argsort()[::-1][:top_n]
        out[int(cid)] = [str(vocab[j]) for j in top if row[j] > 0]
    return out


def representative_texts(texts: list[str], labels: np.ndarray,
                         embeddings: np.ndarray, n: int = 5) -> dict[int, list[str]]:
    """The n messages closest to each cluster centroid."""
    cluster_ids = sorted(c for c in set(labels.tolist()) if c != -1)
    out: dict[int, list[str]] = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        centroid = embeddings[idx].mean(axis=0)
        centroid /= (np.linalg.norm(centroid) or 1.0)
        sims = embeddings[idx] @ centroid
        best = idx[sims.argsort()[::-1][:n]]
        out[int(cid)] = [texts[i] for i in best]
    return out
