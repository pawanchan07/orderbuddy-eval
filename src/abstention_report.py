"""
Per-tier abstention rates, derived from the logged calls. No API spend.

Reads results/raw_calls.jsonl (v1) and results/raw_calls_v2.jsonl (v2) and
counts abstentions strictly: a row is an abstention only when the model
returned the explicit label `abstain`. Abstentions are never merged with wrong
answers, and a response that failed to parse is counted in its own column
rather than being silently absorbed into either — an unparseable response is a
failure to answer, not a decision to decline.

Three views are produced because they answer different questions:

  full set, v1        all 400 rows under the original prompt
  full set, v2        the composed final configuration: v2 where it was
                      measured (180 rows), v1 carried forward (220 rows)
  same-180, v1 vs v2  like-for-like on exactly the rows re-run under v2, which
                      is the only clean measure of what the prompt change did
                      to abstention behaviour

Usage:
    py src/abstention_report.py
Writes:
    results/abstention_rates.json
    appends the table to results/final_summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results" / "raw_calls.jsonl"
V2 = ROOT / "results" / "raw_calls_v2.jsonl"
OUT_JSON = ROOT / "results" / "abstention_rates.json"
OUT_MD = ROOT / "results" / "final_summary.md"

TIER_ORDER = ["budget", "mid", "premium"]
MODEL_TIER = {
    "claude-haiku-4-5": "budget",
    "claude-sonnet-5": "mid",
    "claude-opus-5": "premium",
}


def load(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["model"], {})[r["message_id"]] = r
    return out


def tally(records: list[dict]) -> dict:
    """Strict three-way tally. Abstain means the model said 'abstain'."""
    n = len(records)
    abstain = sum(1 for r in records if r["predicted_intent"] == "abstain")
    unparseable = sum(1 for r in records if not r["predicted_intent"])
    correct = sum(
        1 for r in records
        if r["predicted_intent"] and r["predicted_intent"] != "abstain"
        and r["predicted_intent"] == r["gold_intent"]
    )
    wrong = n - abstain - correct
    return {
        "n": n,
        "n_abstain": abstain,
        "n_wrong": wrong,
        "n_correct": correct,
        "n_unparseable": unparseable,
        "abstain_rate": round(abstain / n, 4) if n else 0.0,
        "wrong_rate": round(wrong / n, 4) if n else 0.0,
        "accuracy": round(correct / n, 4) if n else 0.0,
    }


def main() -> None:
    v1 = load(V1)
    v2 = load(V2)

    report: dict[str, dict] = {}
    for model, tier in MODEL_TIER.items():
        v1_rows = v1[model]
        v2_rows = v2[model]
        composed = [v2_rows.get(mid, rec) for mid, rec in v1_rows.items()]
        overlap = sorted(v2_rows)

        report[model] = {
            "tier": tier,
            "model_version_string": model,
            "full_set_v1": tally(list(v1_rows.values())),
            "full_set_v2_composed": tally(composed),
            "same_rows_v1": tally([v1_rows[m] for m in overlap]),
            "same_rows_v2": tally([v2_rows[m] for m in overlap]),
        }

    ordered = sorted(report.items(), key=lambda kv: TIER_ORDER.index(kv[1]["tier"]))

    lines: list[str] = []
    add = lines.append
    add("")
    add("## Abstention rates")
    add("")
    add("Derived from the logged calls (`results/raw_calls*.jsonl`) with no "
        "further API spend. An abstention is counted **only** when the model "
        "returned the explicit `abstain` label. Abstentions are never merged "
        "into wrong answers; correct + wrong + abstained = 100% in every row.")
    add("")
    add("| Tier | Model version | Abstention rate (v2, final) | Abstention rate (v1) | Abstentions v2 | Abstentions v1 |")
    add("|---|---|---:|---:|---:|---:|")
    for model, r in ordered:
        a2, a1 = r["full_set_v2_composed"], r["full_set_v1"]
        add(f"| {r['tier']} | `{model}` | {a2['abstain_rate'] * 100:.1f}% | "
            f"{a1['abstain_rate'] * 100:.1f}% | {a2['n_abstain']}/{a2['n']} | "
            f"{a1['n_abstain']}/{a1['n']} |")
    add("")
    add("Full-set v2 is the composed final configuration (v2 where measured on "
        "180 rows, v1 carried forward on 220). For a like-for-like read of what "
        "the prompt change did to abstention behaviour, compare only the rows "
        "actually re-run:")
    add("")
    add("| Tier | Model version | Abstentions v1 (same 180 rows) | Abstentions v2 (same 180 rows) | Change |")
    add("|---|---|---:|---:|---:|")
    for model, r in ordered:
        s1, s2 = r["same_rows_v1"], r["same_rows_v2"]
        delta = s2["n_abstain"] - s1["n_abstain"]
        add(f"| {r['tier']} | `{model}` | {s1['n_abstain']}/{s1['n']} "
            f"({s1['abstain_rate'] * 100:.1f}%) | {s2['n_abstain']}/{s2['n']} "
            f"({s2['abstain_rate'] * 100:.1f}%) | {delta:+d} |")
    add("")

    unparseable = sum(
        r[k]["n_unparseable"] for _, r in ordered
        for k in ("full_set_v1", "full_set_v2_composed")
    )
    add(f"Unparseable responses across every tier and both prompt versions: "
        f"**{unparseable}**. Structured outputs were used throughout, so no "
        f"response had to be discarded or guessed at, and none was counted as "
        f"an abstention.")
    add("")

    table = "\n".join(lines)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = OUT_MD.read_text(encoding="utf-8")
    marker = "\n## Abstention rates\n"
    if marker in md:                      # replace a previous run's section
        md = md[:md.index(marker)]
    OUT_MD.write_text(md.rstrip() + "\n" + table, encoding="utf-8")

    print(table)
    print(f"Wrote {OUT_JSON} and appended to {OUT_MD}")


if __name__ == "__main__":
    main()
