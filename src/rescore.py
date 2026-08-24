"""
Re-derive the bake-off results from the logged calls, without re-spending.

`results/raw_calls.jsonl` preserves every model response from the run. This
script replays the rubric over that log and rewrites both the log (with gate
verdicts attached) and `results/tier_comparison.json`.

That matters for two reasons:
  * The gates can be corrected or extended after a run and the scoring
    re-derived for free, so a rubric bug never costs another API bill.
  * The pinned model snapshots will eventually be deprecated. The log is the
    durable artifact; the scoring on top of it is reproducible from here.

Usage:
    py src/rescore.py
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from classify_prompt import load_prices
from run_bakeoff_batch import MODELS, score_records

ROOT = Path(__file__).resolve().parents[1]
CALLS = ROOT / "results" / "raw_calls.jsonl"
OUT_JSON = ROOT / "results" / "tier_comparison.json"
BATCH_IDS = ROOT / "results" / "batch_ids.json"


def main() -> None:
    if not CALLS.exists():
        raise SystemExit(f"{CALLS} not found — run the bake-off first")

    records: dict[str, list[dict]] = {}
    with CALLS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.setdefault(rec["model"], []).append(rec)

    prices = load_prices()
    batch_ids = (json.loads(BATCH_IDS.read_text(encoding="utf-8"))
                 if BATCH_IDS.exists() else {})

    # Preserve the original run's provenance if a previous comparison exists.
    prior = (json.loads(OUT_JSON.read_text(encoding="utf-8"))
             if OUT_JSON.exists() else {})

    comparison = {}
    scored = {model: score_records(recs) for model, recs in records.items()}

    with CALLS.open("w", encoding="utf-8") as fh:
        for recs in records.values():
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    for model, recs in records.items():
        scores = scored[model]
        in_tok = sum(r["input_tokens"] for r in recs)
        out_tok = sum(r["output_tokens"] for r in recs)
        meta = prices["models"][model]
        n = len(recs)

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
        **{k: v for k, v in prior.items() if k != "results"},
        "run_date": prior.get("run_date", date.today().isoformat()),
        "rescored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rescored_from": "results/raw_calls.jsonl (no API calls made)",
        "PRICES_AS_OF": prices["PRICES_AS_OF"],
        "price_source": prices["source"],
        "results": comparison,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'tier':<9}{'model':<19}{'corr':>7}{'wrong':>7}{'abst':>7}"
          f"{'safe':>7}{'grnd':>7}{'clean':>7}{'$/1k':>8}")
    print("-" * 78)
    for model, c in sorted(comparison.items(),
                           key=lambda kv: ["budget", "mid", "premium"].index(kv[1]["tier"])):
        s = c["scores"]
        print(f"{c['tier']:<9}{model:<19}{s['accuracy']:>7.1%}{s['wrong_rate']:>7.1%}"
              f"{s['abstain_rate']:>7.1%}{s['safety_pass_rate']:>7.1%}"
              f"{s['groundedness_pass_rate']:>7.1%}{s['clean_correct_rate']:>7.1%}"
              f"{c['cost']['cost_per_1000_classifications_usd']:>8.3f}")
    print(f"\nRewrote {OUT_JSON} and {CALLS}")


if __name__ == "__main__":
    main()
