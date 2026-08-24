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

    print(f"  clusters found   : {result.n_clusters}")
    print(f"  intents recovered: {scores['intent_recovery_str']}")
    print(f"  accuracy         : {scores['accuracy']:.1%}")
    print(f"  ARI / NMI        : {scores['ari']:.3f} / {scores['nmi']:.3f}")
    print(f"  noise            : {scores['noise_fraction']:.1%}")

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
        "recovery_metric_definition": (
            "Clusters are matched one-to-one to gold intents by Hungarian "
            "assignment on overlap count. A gold intent counts as recovered "
            "when its assigned cluster reaches F1 >= 0.50 against it. "
            "Row-level accuracy uses the same mapping and counts HDBSCAN "
            "noise (-1) as incorrect."
        ),
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
