"""
Step 4 — Stratified golden set for the tier bake-off.

Samples 400 messages across all 16 intents using a floor-plus-proportional
allocation: every intent gets at least MIN_PER_INTENT rows so that per-intent
accuracy is measurable even for rare intents, and the remaining budget is
distributed in proportion to real traffic so the headline pass rate still
reflects the queue the model would actually see.

Equal allocation alone would over-represent rare intents and inflate the
headline number relative to production; proportional allocation alone would
leave the rarest intents with too few rows to say anything about.

Labels are inherited from the generator's ground truth, then written to a
review file for one human pass before any scoring happens.

Usage:
    py src/build_golden_set.py
Writes:
    data/golden_set.csv            (frozen sample + label + review columns)
    results/golden_set_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "raw" / "support_messages.csv"
CLUSTERED = ROOT / "data" / "interim" / "clustered_messages.csv"
OUT_CSV = ROOT / "data" / "golden_set.csv"
OUT_MANIFEST = ROOT / "results" / "golden_set_manifest.json"

SEED = 20260824
TARGET_N = 400
MIN_PER_INTENT = 15


def main() -> None:
    df = pd.read_csv(IN_CSV)
    rng = np.random.default_rng(SEED)

    intents = sorted(df["intent"].unique())
    counts = df["intent"].value_counts()

    # Floor first, then distribute the remainder proportionally to traffic.
    alloc = {i: MIN_PER_INTENT for i in intents}
    remaining = TARGET_N - MIN_PER_INTENT * len(intents)
    if remaining > 0:
        share = counts / counts.sum()
        extra = (share * remaining).round().astype(int)
        for i in intents:
            alloc[i] += int(extra.get(i, 0))

    # Correct any rounding drift so the total lands exactly on TARGET_N.
    drift = TARGET_N - sum(alloc.values())
    for i in sorted(intents, key=lambda x: -counts[x])[:abs(drift)]:
        alloc[i] += 1 if drift > 0 else -1

    picks = []
    for intent in intents:
        pool = df[df["intent"] == intent]
        n = min(alloc[intent], len(pool))
        idx = rng.choice(pool.index.values, size=n, replace=False)
        picks.append(df.loc[idx])

    golden = pd.concat(picks).sort_values("message_id").reset_index(drop=True)

    # Flag rows where unsupervised discovery disagreed with the ground-truth
    # label. These are where a mislabel would actually bite, so they are the
    # rows worth a human's attention first.
    if CLUSTERED.exists():
        clustered = pd.read_csv(CLUSTERED)[["message_id", "derived_intent"]]
        golden = golden.merge(clustered, on="message_id", how="left")
        golden["discovery_agrees"] = golden["derived_intent"] == golden["intent"]
    else:
        golden["derived_intent"] = ""
        golden["discovery_agrees"] = True

    golden["gold_intent"] = golden["intent"]
    golden["review_status"] = "pending"     # pending | confirmed | corrected
    golden["reviewer_note"] = ""

    cols = ["message_id", "text", "channel", "gold_intent", "derived_intent",
            "discovery_agrees", "review_status", "reviewer_note"]
    golden = golden[cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(OUT_CSV, index=False, encoding="utf-8")

    disagree = int((~golden["discovery_agrees"]).sum())
    manifest = {
        "seed": SEED,
        "n_rows": int(len(golden)),
        "n_intents": int(golden["gold_intent"].nunique()),
        "allocation_rule": (
            f"floor of {MIN_PER_INTENT} per intent, remainder proportional to "
            f"corpus traffic, target {TARGET_N}"
        ),
        "per_intent_counts": golden["gold_intent"].value_counts().to_dict(),
        "rows_where_discovery_disagrees": disagree,
        "sha256": hashlib.sha256(OUT_CSV.read_bytes()).hexdigest(),
        "review_state": "pending",
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Golden set: {len(golden)} rows across {golden['gold_intent'].nunique()} intents")
    print(f"Per-intent range: {golden['gold_intent'].value_counts().min()}"
          f"-{golden['gold_intent'].value_counts().max()}")
    print(f"Rows where discovery disagreed with the label: {disagree}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
