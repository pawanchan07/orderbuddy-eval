"""
Step 5 — Tier bake-off across three Claude tiers.

Classifies the reviewed golden set with each tier, logs real token usage from
every API response, and applies the two binary gates from docs/rubric.md.

Fairness notes, all deliberate and all recorded in the output:
  * Every tier receives a byte-identical system prompt and user message.
  * Structured outputs are used for all three, so a tier is never penalised
    for JSON formatting rather than classification.
  * effort="low" is set on Sonnet 5 and Opus 5. Haiku 4.5 does not accept the
    parameter. Low effort is the realistic production setting for single-label
    classification and keeps the tiers comparable on spend; the asymmetry is
    that Haiku has no adaptive thinking to modulate at all.

Usage:
    py src/run_bakeoff.py [--limit N] [--models a,b]
Writes:
    results/tier_comparison.json
    results/raw_calls.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MODELS = {
    "claude-haiku-4-5": {"tier": "budget", "effort": None},
    "claude-sonnet-5": {"tier": "mid", "effort": "low"},
    "claude-opus-5": {"tier": "premium", "effort": "low"},
}

MAX_TOKENS = 512
MAX_WORKERS = 8
MAX_RETRIES = 4

_write_lock = threading.Lock()


def classify_one(client, model: str, cfg: dict, system: str, row) -> dict:
    """One classification call, with retries. Returns a fully-logged record."""
    kwargs = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": build_user_message(row["text"])}],
        output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
    )
    if cfg["effort"]:
        kwargs["output_config"]["effort"] = cfg["effort"]

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.messages.create(**kwargs)
            latency = time.time() - t0

            raw = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}

            return {
                "message_id": row["message_id"],
                "model": model,
                "tier": cfg["tier"],
                "text": row["text"],
                "gold_intent": row["gold_intent"],
                "predicted_intent": parsed.get("intent", ""),
                "evidence": parsed.get("evidence", ""),
                "confidence": parsed.get("confidence", ""),
                "raw_response": raw,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "stop_reason": resp.stop_reason,
                "latency_s": round(latency, 3),
                "attempts": attempt + 1,
                "error": None,
            }
        except Exception as exc:                      # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return {
        "message_id": row["message_id"], "model": model, "tier": cfg["tier"],
        "text": row["text"], "gold_intent": row["gold_intent"],
        "predicted_intent": "", "evidence": "", "confidence": "",
        "raw_response": "", "input_tokens": 0, "output_tokens": 0,
        "stop_reason": None, "latency_s": 0.0, "attempts": MAX_RETRIES,
        "error": last_err,
    }


def score_records(records: list[dict]) -> dict:
    """Apply the rubric to one tier's records."""
    n = len(records)
    correct = abstain = safe = grounded = both = clean = errors = 0
    per_intent = defaultdict(lambda: {"n": 0, "correct": 0})
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

        is_correct = bool(pred) and pred == goldv
        if pred == "abstain":
            abstain += 1
        if is_correct:
            correct += 1
        if s_ok:
            safe += 1
        if g_ok:
            grounded += 1
        if s_ok and g_ok:
            both += 1
            if is_correct:
                clean += 1

        per_intent[goldv]["n"] += 1
        per_intent[goldv]["correct"] += int(is_correct)

    non_abstain = n - abstain
    return {
        "n_rows": n,
        "errors": errors,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "accuracy_excl_abstain": round(correct / non_abstain, 4) if non_abstain else 0.0,
        "abstain_rate": round(abstain / n, 4) if n else 0.0,
        "safety_pass_rate": round(safe / n, 4) if n else 0.0,
        "groundedness_pass_rate": round(grounded / n, 4) if n else 0.0,
        "both_gates_pass_rate": round(both / n, 4) if n else 0.0,
        "clean_correct_rate": round(clean / n, 4) if n else 0.0,
        "gate_failure_counts": dict(sorted(failures.items())),
        "per_intent_accuracy": {
            k: round(v["correct"] / v["n"], 4)
            for k, v in sorted(per_intent.items()) if v["n"]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only classify the first N rows (smoke testing)")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated subset of model ids")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (expected in .env)")

    import anthropic
    client = anthropic.Anthropic()

    golden = pd.read_csv(ROOT / "data" / "golden_set.csv")
    pending = (golden["review_status"] == "pending").sum()
    if pending and not args.limit:
        print(f"WARNING: {pending} golden rows still marked review_status=pending")

    if args.limit:
        golden = golden.head(args.limit)

    models = MODELS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = {k: v for k, v in MODELS.items() if k in wanted}

    system = build_system_prompt()
    prices = load_prices()

    all_records: dict[str, list[dict]] = {}
    OUT_CALLS.parent.mkdir(parents=True, exist_ok=True)
    call_log = OUT_CALLS.open("a", encoding="utf-8")

    started = datetime.now(timezone.utc)

    for model, cfg in models.items():
        print(f"\n=== {cfg['tier']}: {model} ({len(golden)} rows) ===")
        records: list[dict] = []
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [
                pool.submit(classify_one, client, model, cfg, system, row)
                for _, row in golden.iterrows()
            ]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                records.append(rec)
                with _write_lock:
                    call_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 50 == 0 or i == len(futures):
                    print(f"  {i}/{len(futures)} "
                          f"({time.time() - t0:.0f}s elapsed)")

        all_records[model] = records
        errs = sum(1 for r in records if r["error"])
        if errs:
            print(f"  !! {errs} calls failed after {MAX_RETRIES} retries")

    call_log.close()

    # Assemble the comparison, computing cost from *logged* tokens only.
    comparison = {}
    for model, records in all_records.items():
        scores = score_records(records)
        in_tok = sum(r["input_tokens"] for r in records)
        out_tok = sum(r["output_tokens"] for r in records)
        meta = prices["models"][model]
        cost = (in_tok / 1e6) * meta["input"] + (out_tok / 1e6) * meta["output"]
        n = len(records)

        comparison[model] = {
            "model_version_string": model,
            "tier": MODELS[model]["tier"],
            "display_name": meta["display_name"],
            "effort": MODELS[model]["effort"],
            "scores": scores,
            "tokens": {
                "total_input": in_tok,
                "total_output": out_tok,
                "mean_input_per_call": round(in_tok / n, 1) if n else 0,
                "mean_output_per_call": round(out_tok / n, 1) if n else 0,
            },
            "latency": {
                "mean_s": round(sum(r["latency_s"] for r in records) / n, 3) if n else 0,
                "p95_s": round(
                    sorted(r["latency_s"] for r in records)[int(n * 0.95) - 1], 3
                ) if n else 0,
            },
            "cost": {
                "run_cost_usd": round(cost, 4),
                "cost_per_1000_classifications_usd": round(cost / n * 1000, 4) if n else 0,
                "input_rate_per_mtok": meta["input"],
                "output_rate_per_mtok": meta["output"],
            },
        }

    payload = {
        "run_date": date.today().isoformat(),
        "run_started_utc": started.isoformat(timespec="seconds"),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "PRICES_AS_OF": prices["PRICES_AS_OF"],
        "price_source": prices["source"],
        "golden_set": {
            "path": "data/golden_set.csv",
            "n_rows": len(golden),
            "n_intents": int(golden["gold_intent"].nunique()),
        },
        "rubric": "docs/rubric.md",
        "gates": "deterministic code checks (src/gates.py), not an LLM judge",
        "prompt_note": "byte-identical system prompt and user message for all tiers; "
                       "structured outputs (json_schema) on all tiers",
        "results": comparison,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'tier':<9}{'model':<20}{'acc':>7}{'safe':>7}{'grnd':>7}"
          f"{'clean':>7}{'$/1k':>9}")
    print("-" * 66)
    for model, c in comparison.items():
        s = c["scores"]
        print(f"{c['tier']:<9}{model:<20}{s['accuracy']:>7.1%}"
              f"{s['safety_pass_rate']:>7.1%}{s['groundedness_pass_rate']:>7.1%}"
              f"{s['clean_correct_rate']:>7.1%}"
              f"{c['cost']['cost_per_1000_classifications_usd']:>9.2f}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
