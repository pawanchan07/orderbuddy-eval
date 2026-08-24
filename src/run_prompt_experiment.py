"""
A2 prompt experiment — does naming the rule conflict close the safety gap?

The v1 run showed every safety-gate failure was A2 (echoing an order reference
into the output), and that it hit the cheaper tiers ~10x harder than the
premium tier. The hypothesis was that this is not a capability gap but an
unstated-precedence gap: v1 gives "quote verbatim" and "never repeat
identifiers" as independent rules and leaves the model to notice they collide.

v2 adds one sentence naming the conflict and saying which rule wins. Nothing
else about the prompt, the model set, or the scoring changes.

Scope — two row groups, both re-run with v2:

  treatment (80 rows)  every golden row whose message contains an identifier.
                       This is the complete set of rows where A2 can fire; the
                       v1 run had zero A2 failures outside it, so nothing is
                       missed by not re-running the rest.
  control   (100 rows) a seeded sample of the remaining 320. A2 cannot fire
                       here, so this group exists only to detect whether the
                       added sentence degrades accuracy elsewhere — the
                       obvious way a prompt fix could quietly cost more than
                       it saves.

Usage:
    py src/run_prompt_experiment.py [--poll-seconds S]
Writes:
    results/prompt_experiment.json
    results/raw_calls_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from classify_prompt import build_system_prompt, load_prices
from gates import (
    EMAIL_PATTERN,
    ORDER_REF_PATTERNS,
    PHONE_PATTERN,
    POSTCODE_PATTERN,
)
from run_bakeoff_batch import MODELS, build_requests, submit_and_wait, to_record

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "prompt_experiment.json"
OUT_CALLS = ROOT / "results" / "raw_calls_v2.jsonl"
V1_CALLS = ROOT / "results" / "raw_calls.jsonl"

CONTROL_N = 100
SEED = 20260825


def has_identifier(text: str) -> bool:
    return (any(re.search(p, text, re.IGNORECASE) for p in ORDER_REF_PATTERNS)
            or any(re.search(p, text)
                   for p in (EMAIL_PATTERN, PHONE_PATTERN, POSTCODE_PATTERN)))


def summarise(records: list[dict]) -> dict:
    """Gate + outcome summary for one group of records."""
    from run_bakeoff_batch import score_records
    s = score_records(records)
    a2 = sum(1 for r in records
             if any("A2" in x for x in r.get("safety_reasons", [])))
    return {
        "n": len(records),
        "accuracy": s["accuracy"],
        "wrong_rate": s["wrong_rate"],
        "abstain_rate": s["abstain_rate"],
        "safety_pass_rate": s["safety_pass_rate"],
        "groundedness_pass_rate": s["groundedness_pass_rate"],
        "clean_correct_rate": s["clean_correct_rate"],
        "a2_failures": a2,
        "gate_failure_counts": s["gate_failure_counts"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=int, default=20)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (expected in .env)")

    import anthropic
    client = anthropic.Anthropic()

    golden = pd.read_csv(ROOT / "data" / "golden_set.csv")
    golden["a2_relevant"] = golden["text"].map(has_identifier)

    treatment = golden[golden["a2_relevant"]].copy()
    rest = golden[~golden["a2_relevant"]]
    rng = np.random.default_rng(SEED)
    control_idx = rng.choice(rest.index.values, size=min(CONTROL_N, len(rest)),
                             replace=False)
    control = golden.loc[control_idx].copy()

    subset = pd.concat([treatment, control]).sort_values("message_id")
    treatment_ids = set(treatment["message_id"])
    control_ids = set(control["message_id"])

    print(f"treatment (identifier present): {len(treatment)} rows")
    print(f"control   (no identifier)     : {len(control)} rows")
    print(f"total per tier                : {len(subset)} rows")

    system_v2 = build_system_prompt("v2")
    prices = load_prices()
    started = datetime.now(timezone.utc)

    v2_records: dict[str, list[dict]] = {}
    batch_ids: dict[str, str] = {}

    for model, cfg in MODELS.items():
        print(f"\n=== {cfg['tier']}: {model} (v2 prompt, {len(subset)} rows) ===")
        requests = build_requests(subset, model, cfg, system_v2)
        batch_id, results = submit_and_wait(client, requests, model,
                                            args.poll_seconds)
        batch_ids[model] = batch_id
        records = [
            to_record(row, model, cfg, results.get(str(row["message_id"])))
            for _, row in subset.iterrows()
        ]
        v2_records[model] = records
        errs = sum(1 for r in records if r["error"])
        if errs:
            print(f"  !! {errs} requests did not succeed")

    # Attach gate verdicts before writing the log.
    from run_bakeoff_batch import score_records
    for records in v2_records.values():
        score_records(records)
    with OUT_CALLS.open("w", encoding="utf-8") as fh:
        for records in v2_records.values():
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Load v1 results and restrict them to the same rows, so before/after is
    # compared on identical inputs rather than on differently-sized sets.
    v1_all: dict[str, list[dict]] = {}
    with V1_CALLS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                v1_all.setdefault(rec["model"], []).append(rec)

    comparison = {}
    total_cost = 0.0
    for model in MODELS:
        meta = prices["models"][model]
        v1 = v1_all[model]
        v2 = v2_records[model]

        def split(recs, ids):
            return [r for r in recs if r["message_id"] in ids]

        in_tok = sum(r["input_tokens"] for r in v2)
        out_tok = sum(r["output_tokens"] for r in v2)
        cost = ((in_tok / 1e6) * meta["batch_input"]
                + (out_tok / 1e6) * meta["batch_output"])
        total_cost += cost

        comparison[model] = {
            "tier": MODELS[model]["tier"],
            "treatment": {
                "v1": summarise(split(v1, treatment_ids)),
                "v2": summarise(split(v2, treatment_ids)),
            },
            "control": {
                "v1": summarise(split(v1, control_ids)),
                "v2": summarise(split(v2, control_ids)),
            },
            "experiment_cost_usd": round(cost, 4),
            "batch_id": batch_ids[model],
        }

    payload = {
        "run_date": date.today().isoformat(),
        "run_started_utc": started.isoformat(timespec="seconds"),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "PRICES_AS_OF": prices["PRICES_AS_OF"],
        "api_path": "batch",
        "hypothesis": (
            "The v1 A2 gap between tiers is an unstated-precedence problem, "
            "not a capability gap: v1 gives 'quote verbatim' and 'never repeat "
            "identifiers' as independent rules without saying which wins."
        ),
        "change": (
            "v2 appends one sentence to rule 4 naming the conflict and stating "
            "that rule 4 outranks rule 2. No other change."
        ),
        "groups": {
            "treatment": f"{len(treatment)} rows whose message contains an "
                         "identifier (complete set where A2 can fire)",
            "control": f"{len(control)} sampled rows without an identifier "
                       "(regression check only)",
        },
        "total_experiment_cost_usd": round(total_cost, 4),
        "results": comparison,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'':<28}{'A2 fails':>10}{'safety':>9}{'clean&corr':>12}{'acc':>8}")
    for model, c in comparison.items():
        for grp in ("treatment", "control"):
            for ver in ("v1", "v2"):
                d = c[grp][ver]
                print(f"{c['tier'] + '/' + grp + '/' + ver:<28}"
                      f"{d['a2_failures']:>10}{d['safety_pass_rate']:>9.1%}"
                      f"{d['clean_correct_rate']:>12.1%}{d['accuracy']:>8.1%}")
    print(f"\nExperiment cost: ${total_cost:.3f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
