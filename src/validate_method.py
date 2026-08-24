"""
Step 3 — Method validation against public benchmarks.

Runs the *same* discovery pipeline used to build the OrderBuddy taxonomy
(imported from discovery.py, not re-implemented) against two standard intent
classification benchmarks, and reports what it actually recovers.

  Banking77  - 77 fine-grained banking intents (PolyAI)
  CLINC150   - 150 in-scope intents + 1 out-of-scope class (Larson et al.)

Both are used as *unlabelled* corpora: the pipeline sees only the text. Labels
are read afterwards, purely to score.

One favourable assumption is made and is recorded in the output: the method's
granularity prior (`k_expected`, which sets min_cluster_size) is given the
benchmark's true class count. A real discovery run does not know this number.
The `k_expected_sensitivity` block quantifies what that assumption is worth by
re-scoring at +/-25% and +/-50% of the true count.

Usage:
    py src/validate_method.py
Writes:
    results/method_validation.json
"""

from __future__ import annotations

import json
from datetime import date, timezone, datetime
from pathlib import Path

import pandas as pd

from discovery import (
    GRANULARITY_FACTOR,
    HDBSCAN_PARAMS,
    MODEL_NAME,
    UMAP_PARAMS,
    discover,
    min_cluster_size_for,
    recovery_variants,
    score_against_gold,
    score_many_to_one,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "method_validation.json"

# Figures the CV cites from the original run, carried here so the report can
# compare against them explicitly rather than leaving the reader to remember.
CLAIMED = {"banking77": "75/77", "clinc150": "143/151"}


def load_banking77() -> tuple[list[str], list[str]]:
    frames = [
        pd.read_csv(ROOT / "data" / "raw" / f"banking77_{split}.csv")
        for split in ("train", "test")
    ]
    df = pd.concat(frames, ignore_index=True)
    return df["text"].astype(str).tolist(), df["category"].astype(str).tolist()


def load_clinc150() -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    ds = load_dataset("clinc_oos", "plus", cache_dir=str(ROOT / "data" / "hf_cache"))
    names = ds["train"].features["intent"].names
    texts, gold = [], []
    for split in ("train", "validation", "test"):
        texts.extend(ds[split]["text"])
        gold.extend(names[i] for i in ds[split]["intent"])
    return texts, gold


def evaluate(name: str, texts: list[str], gold: list[str]) -> dict:
    k_true = len(set(gold))
    print(f"\n{'=' * 70}\n{name}: {len(texts):,} rows, {k_true} gold intents\n{'=' * 70}")

    result = discover(texts, k_expected=k_true, cache_key=name)
    scores = score_against_gold(result.labels, gold)
    lenient = score_many_to_one(result.labels, gold)

    print(f"  clusters found   : {result.n_clusters}")
    print(f"  ARI / NMI        : {scores['ari']:.3f} / {scores['nmi']:.3f}")
    print(f"  noise            : {scores['noise_fraction']:.1%}")
    print(f"  {'':<18}{'STRICT (1-to-1)':>18}{'LENIENT (m-to-1)':>18}")
    print(f"  {'intents recovered':<18}{scores['intent_recovery_str']:>18}"
          f"{lenient['intent_recovery_str']:>18}")
    print(f"  {'accuracy':<18}{scores['accuracy']:>17.1%}"
          f"{lenient['accuracy']:>17.1%}")
    print(f"  fragmentation    : {lenient['mean_clusters_per_recovered_intent']} "
          f"clusters per recovered intent "
          f"(max {lenient['max_clusters_for_one_intent']} for one intent)")

    variants = recovery_variants(result.labels, gold)
    print("  recovery under other published conventions:")
    for k, v in variants.items():
        print(f"      {k:<24} {v}")

    # How much of the result rests on being handed the true class count?
    sensitivity = []
    for mult in (0.5, 0.75, 1.25, 1.5):
        k_alt = max(2, round(k_true * mult))
        alt = discover(texts, k_expected=k_alt, cache_key=name)
        alt_scores = score_against_gold(alt.labels, gold)
        sensitivity.append({
            "k_expected": k_alt,
            "multiplier": mult,
            "min_cluster_size": min_cluster_size_for(len(texts), k_alt),
            "n_clusters": alt_scores["n_clusters_found"],
            "intent_recovery": alt_scores["intent_recovery_str"],
            "accuracy": alt_scores["accuracy"],
            "ari": alt_scores["ari"],
        })
        print(f"  k_expected={k_alt:<4} -> recovered "
              f"{alt_scores['intent_recovery_str']:<8} acc {alt_scores['accuracy']:.3f}")

    return {
        "dataset": name,
        "n_rows": len(texts),
        "n_gold_intents": k_true,
        "min_cluster_size_used": min_cluster_size_for(len(texts), k_true),
        "n_clusters_found": result.n_clusters,
        "strict": {
            "criterion": "one_to_one_hungarian_f1_50",
            "intent_recovery": scores["intent_recovery_str"],
            "intents_recovered": scores["intents_recovered"],
            "accuracy": scores["accuracy"],
            "accuracy_clustered_only": scores["accuracy_clustered_only"],
        },
        "lenient": lenient,
        "criterion_gap": {
            "recovery_delta": lenient["intents_recovered"] - scores["intents_recovered"],
            "accuracy_delta": round(lenient["accuracy"] - scores["accuracy"], 4),
        },
        "intent_recovery": scores["intent_recovery_str"],
        "intents_recovered": scores["intents_recovered"],
        "accuracy": scores["accuracy"],
        "accuracy_clustered_only": scores["accuracy_clustered_only"],
        "ari": scores["ari"],
        "nmi": scores["nmi"],
        "noise_fraction": scores["noise_fraction"],
        "recovery_f1_threshold": scores["recovery_f1_threshold"],
        "missed_intents": scores["missed_intents"],
        "claimed_in_cv": CLAIMED.get(name),
        "recovery_by_metric_definition": variants,
        "k_expected_sensitivity": sensitivity,
    }


def main() -> None:
    results = {}

    b_texts, b_gold = load_banking77()
    results["banking77"] = evaluate("banking77", b_texts, b_gold)

    c_texts, c_gold = load_clinc150()
    results["clinc150"] = evaluate("clinc150", c_texts, c_gold)

    # CLINC150's out-of-scope class is a deliberate grab-bag of unrelated
    # queries. It has no coherent centroid, so scoring it as if it were a
    # discoverable intent understates the method. Reported both ways.
    c_in_scope = [(t, g) for t, g in zip(c_texts, c_gold) if g != "oos"]
    results["clinc150"]["oos_note"] = (
        "The 'oos' class is an intentionally incoherent catch-all; it is "
        "counted in the denominator above to match the CV's 143/151 framing. "
        f"In-scope rows: {len(c_in_scope)} of {len(c_texts)}."
    )
    results["clinc150"]["oos_recovered"] = (
        "oos" not in results["clinc150"]["missed_intents"]
    )

    payload = {
        "run_date": date.today().isoformat(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {
            "encoder": MODEL_NAME,
            "umap": UMAP_PARAMS,
            "hdbscan_fixed": HDBSCAN_PARAMS,
            "granularity_factor": GRANULARITY_FACTOR,
            "min_cluster_size_rule":
                "max(15, round(GRANULARITY_FACTOR * n_rows / k_expected))",
            "tuned_on": "synthetic OrderBuddy corpus only (src/sweep_params.py); "
                        "applied unchanged to both benchmarks",
        },
        "headline_criterion": "strict",
        "headline_rationale": (
            "The deliverable of this stage is a usable taxonomy, and only the "
            "strict criterion tests for one. Many-to-one lets several "
            "fragments of a single intent each count as a recovery, which "
            "inflates the score without yielding a taxonomy anyone could "
            "build a routing workflow on."
        ),
        "criterion_definitions": {
            "strict": (
                "Clusters are matched to gold intents by Hungarian "
                "assignment maximising total overlap, so each cluster claims at "
                "most one intent and each intent at most one cluster. A gold "
                "intent counts as recovered when its assigned cluster reaches "
                "F1 >= 0.50 against it (F1 over that cluster's precision and "
                "recall for that intent). Row accuracy uses the same mapping. "
                "HDBSCAN noise (-1) counts as incorrect."
            ),
            "lenient": (
                "Many-to-one plurality, the conventional clustering-accuracy "
                "criterion. Every cluster is mapped to its own plurality gold "
                "label; several clusters may map to the same intent. A gold "
                "intent counts as recovered if it is the plurality label of at "
                "least one cluster. Row accuracy asks whether a row's cluster "
                "has that row's label as its plurality. HDBSCAN noise (-1) "
                "counts as incorrect, as in strict."
            ),
        },
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\nSUMMARY vs CV claims\n{'=' * 70}")
    for key, r in results.items():
        print(f"  {key:<12} this run: {r['intent_recovery']:<8} "
              f"CV claims: {r['claimed_in_cv']:<8} acc {r['accuracy']:.1%}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
