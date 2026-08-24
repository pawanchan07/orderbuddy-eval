"""
Step 5 — Tier bake-off across three Claude tiers, via the Batch API.

One batch per tier. Batch pricing is 50% of standard on both input and output,
so cost is computed from `batch_input` / `batch_output` in config/prices.json.

Fairness notes, all deliberate and all recorded in the output:
  * Every tier receives a byte-identical system prompt and user message.
  * Structured outputs are used for all three, so a tier is never penalised
    for JSON formatting rather than classification.
  * effort="low" is set on Sonnet 5 and Opus 5. Haiku 4.5 does not accept the
    parameter. Low effort is the realistic production setting for single-label
    classification; the residual asymmetry is that Haiku has no adaptive
    thinking to modulate at all, and that is stated rather than corrected for.

Abstentions are tracked as a third outcome throughout — never folded into
"wrong". A model that declines an ambiguous message is doing something
materially different from one that guesses incorrectly.

Usage:
    py src/run_bakeoff_batch.py [--limit N] [--models a,b] [--poll-seconds S]
Writes:
    results/tier_comparison.json
    results/raw_calls.jsonl
    results/batch_ids.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from classify_prompt import (
    RESPONSE_SCHEMA,
    build_system_prompt,
    build_user_message,
    load_prices,
)
from gates import check_groundedness, check_safety

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "tier_comparison.json"
OUT_CALLS = ROOT / "results" / "raw_calls.jsonl"
OUT_BATCH_IDS = ROOT / "results" / "batch_ids.json"

MODELS = {
    "claude-haiku-4-5": {"tier": "budget", "effort": None},
    "claude-sonnet-5": {"tier": "mid", "effort": "low"},
    "claude-opus-5": {"tier": "premium", "effort": "low"},
}

MAX_TOKENS = 512


def build_requests(golden: pd.DataFrame, model: str, cfg: dict, system: str) -> list[dict]:
    output_config: dict = {
        "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}
    }
    if cfg["effort"]:
        output_config["effort"] = cfg["effort"]

    return [
        {
            "custom_id": str(row["message_id"]),
            "params": {
                "model": model,
                "max_tokens": MAX_TOKENS,
                "system": system,
                "messages": [
                    {"role": "user", "content": build_user_message(row["text"])}
                ],
                "output_config": output_config,
            },
        }
        for _, row in golden.iterrows()
    ]


def submit_and_wait(client, requests: list[dict], label: str,
                    poll_seconds: int) -> tuple[str, dict]:
    """Submit one batch and poll until it ends. Returns (batch_id, results_by_id)."""
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch {batch.id} submitted ({len(requests)} requests)")

    t0 = time.time()
    while True:
        current = client.messages.batches.retrieve(batch.id)
        counts = current.request_counts
        if current.processing_status == "ended":
            print(f"  ended after {time.time() - t0:.0f}s — "
                  f"succeeded={counts.succeeded} errored={counts.errored} "
                  f"canceled={counts.canceled} expired={counts.expired}")
            break
        print(f"  [{time.time() - t0:>5.0f}s] {current.processing_status}: "
              f"processing={counts.processing} succeeded={counts.succeeded} "
              f"errored={counts.errored}")
        time.sleep(poll_seconds)

    results = {}
    for entry in client.messages.batches.results(batch.id):
        results[entry.custom_id] = entry
    return batch.id, results


def to_record(row, model: str, cfg: dict, entry) -> dict:
    """Normalise one batch result entry into the same record shape as the sync path."""
    base = {
        "message_id": str(row["message_id"]),
        "model": model,
        "tier": cfg["tier"],
        "text": row["text"],
        "gold_intent": row["gold_intent"],
        "predicted_intent": "",
        "evidence": "",
        "confidence": "",
        "raw_response": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "stop_reason": None,
        "error": None,
    }
    if entry is None:
        base["error"] = "no result returned for custom_id"
        return base

    result_type = entry.result.type
    if result_type != "succeeded":
        detail = getattr(entry.result, "error", None)
        base["error"] = f"{result_type}: {detail}"
        return base

    msg = entry.result.message
    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}

    base.update({
        "predicted_intent": parsed.get("intent", ""),
        "evidence": parsed.get("evidence", ""),
        "confidence": parsed.get("confidence", ""),
        "raw_response": raw,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "stop_reason": msg.stop_reason,
    })
    return base


def score_records(records: list[dict]) -> dict:
    """Apply docs/rubric.md. Abstentions are a separate outcome from wrong answers."""
    n = len(records)
    correct = abstain = wrong = unparseable = 0
    safe = grounded = both = clean = errors = 0
    per_intent = defaultdict(lambda: {"n": 0, "correct": 0, "abstain": 0, "wrong": 0})
    failures = defaultdict(int)

    for r in records:
        if r["error"]:
            errors += 1
        pred, goldv = r["predicted_intent"], r["gold_intent"]

        s_ok, s_why = check_safety(r["raw_response"], r["evidence"], r["text"])
        g_ok, g_why = check_groundedness(r["evidence"], r["text"], pred or "")
        for why in s_why + g_why:
            failures[why.split(":")[0]] += 1
        r["safety_pass"], r["safety_reasons"] = s_ok, s_why
        r["groundedness_pass"], r["groundedness_reasons"] = g_ok, g_why

        # Three-way outcome, never collapsed.
        if not pred:
            unparseable += 1
            outcome = "wrong"
        elif pred == "abstain":
            abstain += 1
            outcome = "abstain"
        elif pred == goldv:
            correct += 1
            outcome = "correct"
        else:
            wrong += 1
            outcome = "wrong"
        r["outcome"] = outcome

        if s_ok:
            safe += 1
        if g_ok:
            grounded += 1
        if s_ok and g_ok:
            both += 1
            if outcome == "correct":
                clean += 1

        pi = per_intent[goldv]
        pi["n"] += 1
        pi["correct"] += int(outcome == "correct")
        pi["abstain"] += int(outcome == "abstain")
        pi["wrong"] += int(outcome == "wrong")

    attempted = n - abstain
    return {
        "n_rows": n,
        "errors": errors,
        "unparseable": unparseable,
        "n_correct": correct,
        "n_wrong": wrong,
        "n_abstain": abstain,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "wrong_rate": round(wrong / n, 4) if n else 0.0,
        "abstain_rate": round(abstain / n, 4) if n else 0.0,
        "accuracy_excl_abstain": round(correct / attempted, 4) if attempted else 0.0,
        "safety_pass_rate": round(safe / n, 4) if n else 0.0,
        "groundedness_pass_rate": round(grounded / n, 4) if n else 0.0,
        "both_gates_pass_rate": round(both / n, 4) if n else 0.0,
        "clean_correct_rate": round(clean / n, 4) if n else 0.0,
        "gate_failure_counts": dict(sorted(failures.items())),
        "per_intent": {
            k: {
                "n": v["n"],
                "accuracy": round(v["correct"] / v["n"], 4),
                "abstain_rate": round(v["abstain"] / v["n"], 4),
                "wrong_rate": round(v["wrong"] / v["n"], 4),
            }
            for k, v in sorted(per_intent.items()) if v["n"]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--poll-seconds", type=int, default=20)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (expected in .env)")

    import anthropic
    client = anthropic.Anthropic()

    golden = pd.read_csv(ROOT / "data" / "golden_set.csv")
    pending = int((golden["review_status"] == "pending").sum())
    if pending:
        raise SystemExit(
            f"{pending} golden rows are still review_status=pending. "
            "The rubric requires a completed label review before scoring."
        )
    if args.limit:
        golden = golden.head(args.limit)

    models = MODELS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = {k: v for k, v in MODELS.items() if k in wanted}

    system = build_system_prompt()
    prices = load_prices()
    started = datetime.now(timezone.utc)

    all_records: dict[str, list[dict]] = {}
    batch_ids: dict[str, str] = {}

    for model, cfg in models.items():
        print(f"\n=== {cfg['tier']}: {model} ({len(golden)} rows) ===")
        requests = build_requests(golden, model, cfg, system)
        batch_id, results = submit_and_wait(
            client, requests, model, args.poll_seconds
        )
        batch_ids[model] = batch_id

        records = [
            to_record(row, model, cfg, results.get(str(row["message_id"])))
            for _, row in golden.iterrows()
        ]
        all_records[model] = records
        errs = sum(1 for r in records if r["error"])
        if errs:
            print(f"  !! {errs} requests did not succeed")

    OUT_BATCH_IDS.write_text(json.dumps(batch_ids, indent=2), encoding="utf-8")

    # Score first: score_records attaches gate verdicts and the three-way
    # outcome onto each record, and the raw log is only useful for re-deriving
    # results if it carries them.
    comparison = {}
    scored: dict[str, dict] = {}
    for model, records in all_records.items():
        scored[model] = score_records(records)

    with OUT_CALLS.open("w", encoding="utf-8") as fh:
        for records in all_records.values():
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    for model, records in all_records.items():
        scores = scored[model]
        in_tok = sum(r["input_tokens"] for r in records)
        out_tok = sum(r["output_tokens"] for r in records)
        meta = prices["models"][model]
        n = len(records)

        # Batch path -> batch rates.
        cost = ((in_tok / 1e6) * meta["batch_input"]
                + (out_tok / 1e6) * meta["batch_output"])
        cost_sync = (in_tok / 1e6) * meta["input"] + (out_tok / 1e6) * meta["output"]

        comparison[model] = {
            "model_version_string": model,
            "tier": MODELS[model]["tier"],
            "display_name": meta["display_name"],
            "effort": MODELS[model]["effort"],
            "api_path": "batch",
            "batch_id": batch_ids.get(model),
            "scores": scores,
            "tokens": {
                "total_input": in_tok,
                "total_output": out_tok,
                "total": in_tok + out_tok,
                "mean_input_per_call": round(in_tok / n, 1) if n else 0,
                "mean_output_per_call": round(out_tok / n, 1) if n else 0,
            },
            "cost": {
                "run_cost_usd": round(cost, 4),
                "cost_per_1000_classifications_usd": round(cost / n * 1000, 4) if n else 0,
                "cost_per_1000_if_sync_usd": round(cost_sync / n * 1000, 4) if n else 0,
                "batch_input_rate_per_mtok": meta["batch_input"],
                "batch_output_rate_per_mtok": meta["batch_output"],
                "standard_input_rate_per_mtok": meta["input"],
                "standard_output_rate_per_mtok": meta["output"],
            },
        }

    payload = {
        "run_date": date.today().isoformat(),
        "run_started_utc": started.isoformat(timespec="seconds"),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "PRICES_AS_OF": prices["PRICES_AS_OF"],
        "price_source": prices["source"],
        "api_path": "batch (50% of standard rates on input and output)",
        "golden_set": {
            "path": "data/golden_set.csv",
            "n_rows": len(golden),
            "n_intents": int(golden["gold_intent"].nunique()),
            "review_state": "confirmed",
        },
        "rubric": "docs/rubric.md",
        "gates": "deterministic code checks (src/gates.py), not an LLM judge",
        "outcome_model": (
            "Three-way per row: correct / wrong / abstain. Abstentions are "
            "never counted as correct and never folded into wrong."
        ),
        "prompt_note": (
            "byte-identical system prompt and user message for all tiers; "
            "structured outputs (json_schema) on all tiers; effort='low' on "
            "Sonnet 5 and Opus 5, unsupported on Haiku 4.5"
        ),
        "results": comparison,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'tier':<9}{'model':<19}{'corr':>7}{'wrong':>7}{'abst':>7}"
          f"{'safe':>7}{'grnd':>7}{'clean':>7}{'$/1k':>8}")
    print("-" * 78)
    for model, c in comparison.items():
        s = c["scores"]
        print(f"{c['tier']:<9}{model:<19}{s['accuracy']:>7.1%}{s['wrong_rate']:>7.1%}"
              f"{s['abstain_rate']:>7.1%}{s['safety_pass_rate']:>7.1%}"
              f"{s['groundedness_pass_rate']:>7.1%}{s['clean_correct_rate']:>7.1%}"
              f"{c['cost']['cost_per_1000_classifications_usd']:>8.3f}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
