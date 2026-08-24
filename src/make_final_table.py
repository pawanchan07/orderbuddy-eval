"""
Final summary table for the website widget.

Composes the headline per-tier figures under the **final (v2) prompt**.

The v2 run covered 180 of the 400 golden rows: all 80 rows where the safety
gate's A2 rule can fire, plus a 100-row control sample. The remaining 220 rows
carry their v1 result forward. That composition is sound for the safety
figure and is stated wherever the number appears:

  * A2 cannot fire on a row whose message contains no identifier — there is
    nothing to echo — and the v1 run confirmed this empirically, with zero A2
    failures outside the treatment group across all three tiers.
  * The control group exists to test the other direction: whether the added
    sentence degrades accuracy on rows it was not aimed at. Its measured v1
    vs v2 delta is reported alongside, so the assumption is evidenced rather
    than asserted.

Both the composed full-set figure and the directly-measured treatment-group
figure are emitted. Where they disagree in what they support, the
directly-measured one is the safer citation.

Usage:
    py src/make_final_table.py
Writes:
    results/final_summary.md
    results/final_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "results" / "final_summary.md"
OUT_JSON = ROOT / "results" / "final_summary.json"
TIER_ORDER = ["budget", "mid", "premium"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load_jsonl(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out.setdefault(rec["model"], []).append(rec)
    return out


def main() -> None:
    tier = json.loads((ROOT / "results" / "tier_comparison.json").read_text(encoding="utf-8"))
    exp = json.loads((ROOT / "results" / "prompt_experiment.json").read_text(encoding="utf-8"))
    prices = json.loads((ROOT / "config" / "prices.json").read_text(encoding="utf-8"))

    v1 = load_jsonl(ROOT / "results" / "raw_calls.jsonl")
    v2 = load_jsonl(ROOT / "results" / "raw_calls_v2.jsonl")

    from run_bakeoff_batch import score_records

    final: dict[str, dict] = {}
    for model, meta in prices["models"].items():
        v2_by_id = {r["message_id"]: r for r in v2[model]}
        # Compose: v2 where it was run, v1 elsewhere.
        composed = [v2_by_id.get(r["message_id"], r) for r in v1[model]]
        scores = score_records(composed)

        in_tok = sum(r["input_tokens"] for r in composed)
        out_tok = sum(r["output_tokens"] for r in composed)
        n = len(composed)
        cost = ((in_tok / 1e6) * meta["batch_input"]
                + (out_tok / 1e6) * meta["batch_output"])
        cost_sync = (in_tok / 1e6) * meta["input"] + (out_tok / 1e6) * meta["output"]

        e = exp["results"][model]
        final[model] = {
            "tier": meta["tier"],
            "display_name": meta["display_name"],
            "model_version_string": model,
            "n_rows": n,
            "n_rows_measured_under_v2": len(v2[model]),
            "accuracy": scores["accuracy"],
            "wrong_rate": scores["wrong_rate"],
            "abstain_rate": scores["abstain_rate"],
            "safety_pass_rate": scores["safety_pass_rate"],
            "groundedness_pass_rate": scores["groundedness_pass_rate"],
            "clean_correct_rate": scores["clean_correct_rate"],
            "clean_correct_rate_v1": tier["results"][model]["scores"]["clean_correct_rate"],
            "tokens": {"input": in_tok, "output": out_tok},
            "cost_per_1000_usd": round(cost / n * 1000, 4),
            "cost_per_1000_usd_sync": round(cost_sync / n * 1000, 4),
            "treatment_group": {
                "n": e["treatment"]["v1"]["n"],
                "a2_failures_v1": e["treatment"]["v1"]["a2_failures"],
                "a2_failures_v2": e["treatment"]["v2"]["a2_failures"],
                "safety_v1": e["treatment"]["v1"]["safety_pass_rate"],
                "safety_v2": e["treatment"]["v2"]["safety_pass_rate"],
                "clean_correct_v1": e["treatment"]["v1"]["clean_correct_rate"],
                "clean_correct_v2": e["treatment"]["v2"]["clean_correct_rate"],
            },
            "control_group": {
                "n": e["control"]["v1"]["n"],
                "accuracy_v1": e["control"]["v1"]["accuracy"],
                "accuracy_v2": e["control"]["v2"]["accuracy"],
                "accuracy_delta": round(
                    e["control"]["v2"]["accuracy"] - e["control"]["v1"]["accuracy"], 4
                ),
            },
        }

    ordered = sorted(final.items(), key=lambda kv: TIER_ORDER.index(kv[1]["tier"]))
    run_date = exp["run_date"]

    lines: list[str] = []
    add = lines.append

    add("# OrderBuddy tier comparison — final")
    add("")
    add(f"Run date **{run_date}** · prices **PRICES_AS_OF "
        f"{prices['PRICES_AS_OF']}** · Batch API · prompt **v2** · "
        f"golden set **400 messages, 16 intents**")
    add("")

    add("| Tier | Model version | Accuracy | Abstained | Clean & correct | Cost / 1,000 |")
    add("|---|---|---:|---:|---:|---:|")
    for model, f in ordered:
        add(f"| {f['tier']} | `{f['model_version_string']}` | "
            f"{pct(f['accuracy'])} | {pct(f['abstain_rate'])} | "
            f"{pct(f['clean_correct_rate'])} | ${f['cost_per_1000_usd']:.3f} |")
    add("")
    add(f"*Accuracy* counts exact intent matches against the reviewed gold "
        f"label. *Abstained* is reported separately and is never counted as "
        f"correct nor folded into wrong; correct + wrong + abstained = 100%. "
        f"*Clean & correct* is correct **and** passing both the safety and "
        f"groundedness gates. Cost is computed from logged token usage at "
        f"published Batch API rates as of {prices['PRICES_AS_OF']}.")
    add("")

    add("## Effect of the v2 prompt fix")
    add("")
    add("v2 adds one sentence to the prompt stating that the "
        "\"never repeat identifiers\" rule outranks the \"quote verbatim\" "
        "rule when they conflict. Measured on the 80 rows whose message "
        "contains an identifier — the complete set where the A2 safety rule "
        "can fire:")
    add("")
    add("| Tier | Model | A2 failures v1 | A2 failures v2 | Safety v1 → v2 | Clean & correct v1 → v2 (treatment) |")
    add("|---|---|---:|---:|---:|---:|")
    for model, f in ordered:
        t = f["treatment_group"]
        add(f"| {f['tier']} | `{model}` | {t['a2_failures_v1']}/{t['n']} | "
            f"{t['a2_failures_v2']}/{t['n']} | "
            f"{pct(t['safety_v1'])} → {pct(t['safety_v2'])} | "
            f"{pct(t['clean_correct_v1'])} → {pct(t['clean_correct_v2'])} |")
    add("")
    add("| Tier | Clean & correct, full set v1 | Clean & correct, full set v2 |")
    add("|---|---:|---:|")
    for model, f in ordered:
        add(f"| {f['tier']} | {pct(f['clean_correct_rate_v1'])} | "
            f"{pct(f['clean_correct_rate'])} |")
    add("")

    add("### Control group — did the fix cost anything elsewhere?")
    add("")
    add("100 sampled rows with no identifier, where A2 cannot fire. This "
        "group exists only to detect accuracy regression from the added "
        "sentence:")
    add("")
    add("| Tier | Accuracy v1 | Accuracy v2 | Delta |")
    add("|---|---:|---:|---:|")
    for model, f in ordered:
        c = f["control_group"]
        add(f"| {f['tier']} | {pct(c['accuracy_v1'])} | {pct(c['accuracy_v2'])} | "
            f"{c['accuracy_delta'] * 100:+.1f} pts |")
    add("")

    add("### How the full-set v2 figure is composed")
    add("")
    add(f"The v2 run covered 180 of 400 rows: all 80 treatment rows plus the "
        f"100-row control. The remaining 220 rows carry their v1 result "
        f"forward. This is sound for the safety figure because A2 cannot fire "
        f"on a message containing no identifier, and the v1 run confirmed it "
        f"empirically — zero A2 failures outside the treatment group across "
        f"all three tiers. The control group tests the other direction and its "
        f"delta is reported above. Where a directly-measured number is "
        f"preferred to a composed one, cite the treatment-group table.")
    add("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "run_date": run_date,
        "PRICES_AS_OF": prices["PRICES_AS_OF"],
        "prompt_version": "v2",
        "api_path": "batch",
        "golden_set_rows": 400,
        "composition_note": (
            "v2 measured on 180/400 rows (80 treatment + 100 control); "
            "remaining 220 carry v1 results forward. A2 cannot fire on rows "
            "without an identifier, verified empirically in the v1 run."
        ),
        "tiers": {m: f for m, f in ordered},
    }, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
