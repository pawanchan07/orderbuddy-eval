"""
Step 2 — Intent discovery over the OrderBuddy synthetic corpus.

Embeds all 10,000 messages, clusters them, derives a taxonomy from the
clusters, and scores the result against the generator's ground-truth intents.

The ground-truth column is used **only** for scoring after the fact. Nothing
upstream of the score sees it, so the discovery is genuinely unsupervised.

Usage:
    py src/run_discovery.py
Writes:
    results/intent_discovery.json
    data/interim/clustered_messages.csv
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from discovery import (
    HDBSCAN_PARAMS,
    MODEL_NAME,
    UMAP_PARAMS,
    cluster_keywords,
    discover,
    representative_texts,
    score_against_gold,
)

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "raw" / "support_messages.csv"
OUT_JSON = ROOT / "results" / "intent_discovery.json"
OUT_CLUSTERED = ROOT / "data" / "interim" / "clustered_messages.csv"


def main() -> None:
    df = pd.read_csv(IN_CSV)
    texts = df["text"].astype(str).tolist()
    gold = df["intent"].astype(str).tolist()

    print(f"Embedding {len(texts):,} messages with {MODEL_NAME} ...")
    result = discover(texts, k_expected=16, cache_key="orderbuddy")

    print(f"\nClusters found: {result.n_clusters}")
    print(f"Noise: {result.noise_fraction:.1%}")

    scores = score_against_gold(result.labels, gold)
    keywords = cluster_keywords(texts, result.labels)
    reps = representative_texts(texts, result.labels, result.embeddings)

    # Derive the taxonomy: each cluster becomes a candidate intent, named by
    # the gold intent it maps onto where one exists, otherwise left explicitly
    # unnamed for human review rather than silently invented.
    cluster_to_intent = {int(k): v for k, v in scores["cluster_to_intent"].items()}
    taxonomy = []
    for cid in sorted(keywords):
        size = int((result.labels == cid).sum())
        mapped = cluster_to_intent.get(cid)
        taxonomy.append({
            "cluster_id": cid,
            "size": size,
            "share": round(size / len(texts), 4),
            "derived_name": mapped or f"UNMAPPED_cluster_{cid}",
            "top_terms": keywords[cid],
            "representative_messages": reps.get(cid, []),
        })

    df_out = df.copy()
    df_out["cluster"] = result.labels
    df_out["derived_intent"] = [
        cluster_to_intent.get(int(c), "noise" if c == -1 else f"UNMAPPED_cluster_{c}")
        for c in result.labels
    ]
    OUT_CLUSTERED.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CLUSTERED, index=False, encoding="utf-8")

    payload = {
        "run_date": date.today().isoformat(),
        "dataset": {
            "path": "data/raw/support_messages.csv",
            "n_rows": len(df),
            "n_gold_intents": df["intent"].nunique(),
        },
        "method": {
            "encoder": MODEL_NAME,
            "umap": UMAP_PARAMS,
            "hdbscan": result.params["hdbscan"],
            "granularity_prior_k_expected": 16,
        },
        "clustering": {
            "n_clusters": result.n_clusters,
            "noise_fraction": round(result.noise_fraction, 4),
        },
        "scores": scores,
        "taxonomy": taxonomy,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nIntent recovery : {scores['intent_recovery_str']}")
    print(f"Accuracy        : {scores['accuracy']:.1%} "
          f"(clustered rows only: {scores['accuracy_clustered_only']:.1%})")
    print(f"ARI / NMI       : {scores['ari']:.3f} / {scores['nmi']:.3f}")
    if scores["missed_intents"]:
        print(f"Missed intents  : {', '.join(scores['missed_intents'])}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
