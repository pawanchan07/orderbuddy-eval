"""
Parameter sweep for the discovery method (development tool, run once).

Selects a single hyper-parameter set on the OrderBuddy synthetic corpus. That
set is then frozen into discovery.py and applied unchanged to Banking77 and
CLINC150 in step 3.

Tuning on synthetic and reporting on the benchmarks (rather than tuning on the
benchmarks) is the point: it keeps step 3 an honest out-of-sample test of the
method instead of a report of how well it can be fitted to a known answer.

Usage:
    py src/sweep_params.py
Writes:
    results/param_sweep.json
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from discovery import embed, score_against_gold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "param_sweep.json"


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "support_messages.csv")
    texts = df["text"].astype(str).tolist()
    gold = df["intent"].astype(str).tolist()

    vecs = embed(texts, cache_key="orderbuddy")
    print(f"Embeddings: {vecs.shape}")

    import umap
    import hdbscan as hdbscan_lib

    umap_grid = list(itertools.product([15, 30, 50, 80], [5, 10]))
    hdb_grid = list(itertools.product([50, 100, 150, 250], [5, 15]))

    rows = []
    for n_neighbors, n_components in umap_grid:
        print(f"\nUMAP n_neighbors={n_neighbors} n_components={n_components}")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=20260824,
        )
        reduced = reducer.fit_transform(vecs)

        for min_cluster_size, min_samples in hdb_grid:
            labels = hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            ).fit_predict(reduced)

            s = score_against_gold(labels, gold)
            rows.append({
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": s["n_clusters_found"],
                "recovered": s["intents_recovered"],
                "accuracy": s["accuracy"],
                "ari": s["ari"],
                "nmi": s["nmi"],
                "noise": s["noise_fraction"],
            })
            print(f"  mcs={min_cluster_size:<4} ms={min_samples:<3} "
                  f"-> {s['n_clusters_found']:>3} clusters, "
                  f"recovered {s['intents_recovered']:>2}/16, "
                  f"acc {s['accuracy']:.3f}, ari {s['ari']:.3f}, "
                  f"noise {s['noise_fraction']:.1%}")

    res = pd.DataFrame(rows).sort_values(
        ["recovered", "ari", "accuracy"], ascending=False
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== TOP 12 ===")
    print(res.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
