"""
Budget guard — measure the bake-off's cost before spending anything.

Input tokens are *measured*, not guessed: every golden-set prompt is sent to
the free `/v1/messages/count_tokens` endpoint, which bills nothing. Output
tokens are projected from a stated per-call assumption, since they cannot be
known before the run; the assumption is printed so it can be argued with.

Usage:
    py src/estimate_cost.py
Writes:
    results/cost_estimate.json
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from classify_prompt import build_system_prompt, build_user_message, load_prices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cost_estimate.json"

# Output-token assumptions per tier. Haiku 4.5 has no adaptive thinking, so it
# emits little beyond the JSON object. Sonnet 5 and Opus 5 run adaptive
# thinking by default and their thinking tokens bill as output, so they are
# budgeted far higher even at effort="low".
OUTPUT_TOKENS_ASSUMED = {
    "claude-haiku-4-5": 90,
    "claude-sonnet-5": 400,
    "claude-opus-5": 500,
}
SAFETY_MULTIPLIER = 2.0   # headroom for retries, smoke tests and re-runs


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (expected in .env)")

    import anthropic
    client = anthropic.Anthropic()

    golden = pd.read_csv(ROOT / "data" / "golden_set.csv")
    prices = load_prices()
    system = build_system_prompt()

    # Measure input tokens on a sample, then scale. Token counting is free but
    # still a network round trip per call, so a 60-row sample is plenty to fix
    # the mean for messages this uniform.
    sample = golden.sample(60, random_state=20260824)
    model_for_counting = "claude-sonnet-5"
    counts = []
    print(f"Counting input tokens on {len(sample)} sample prompts "
          f"(free endpoint, model={model_for_counting}) ...")
    for _, row in sample.iterrows():
        res = client.messages.count_tokens(
            model=model_for_counting,
            system=system,
            messages=[{"role": "user", "content": build_user_message(row["text"])}],
        )
        counts.append(res.input_tokens)

    mean_input = sum(counts) / len(counts)
    n_rows = len(golden)
    print(f"  mean input tokens/call: {mean_input:.1f} "
          f"(min {min(counts)}, max {max(counts)})")

    per_model = {}
    total = 0.0
    for model_id, meta in prices["models"].items():
        out_per_call = OUTPUT_TOKENS_ASSUMED[model_id]
        in_tokens = mean_input * n_rows
        out_tokens = out_per_call * n_rows
        cost = (in_tokens / 1e6) * meta["input"] + (out_tokens / 1e6) * meta["output"]
        batch_cost = ((in_tokens / 1e6) * meta["batch_input"]
                      + (out_tokens / 1e6) * meta["batch_output"])
        per_model[model_id] = {
            "tier": meta["tier"],
            "display_name": meta["display_name"],
            "rows": n_rows,
            "mean_input_tokens_per_call": round(mean_input, 1),
            "assumed_output_tokens_per_call": out_per_call,
            "projected_input_tokens": int(in_tokens),
            "projected_output_tokens": int(out_tokens),
            "projected_cost_usd": round(cost, 4),
            "projected_cost_usd_if_batched": round(batch_cost, 4),
        }
        total += cost

    payload = {
        "estimate_date": date.today().isoformat(),
        "PRICES_AS_OF": prices["PRICES_AS_OF"],
        "golden_set_rows": n_rows,
        "input_tokens_measured_via": "messages.count_tokens (free endpoint)",
        "output_token_assumptions": OUTPUT_TOKENS_ASSUMED,
        "output_assumption_note": (
            "Sonnet 5 and Opus 5 run adaptive thinking by default and thinking "
            "tokens bill as output; both are budgeted at 400-500 output tokens "
            "per call even at effort='low'. Haiku 4.5 has no adaptive thinking."
        ),
        "per_model": per_model,
        "projected_total_usd": round(total, 4),
        "safety_multiplier": SAFETY_MULTIPLIER,
        "budget_ceiling_usd": round(total * SAFETY_MULTIPLIER, 2),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'tier':<10} {'model':<20} {'in tok':>9} {'out tok':>9} {'cost':>9}")
    print("-" * 62)
    for mid, m in per_model.items():
        print(f"{m['tier']:<10} {mid:<20} {m['projected_input_tokens']:>9,} "
              f"{m['projected_output_tokens']:>9,} ${m['projected_cost_usd']:>8.2f}")
    print("-" * 62)
    print(f"{'TOTAL':<31} {'':>9} {'':>9} ${total:>8.2f}")
    print(f"With {SAFETY_MULTIPLIER}x headroom for retries: "
          f"${total * SAFETY_MULTIPLIER:.2f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
