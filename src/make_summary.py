"""
Step 7 — Paste-ready summary table for the website build.

Reads the result JSONs and emits one markdown table plus the supporting
figures. Nothing is recomputed here: every number is read from
results/tier_comparison.json and results/method_validation.json so the
website can never drift from the run that produced it.

Abstentions are carried as their own column everywhere. Correct / wrong /
abstain are three separate outcomes and always sum to 100%.

Usage:
    py src/make_summary.py
Writes:
    results/summary_table.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The tables contain characters (>=, en dashes) outside the Windows console's
# default cp1252 codepage. The file itself is always written as UTF-8; this
# only stops the echo to stdout from raising.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "summary_table.md"

TIER_ORDER = ["budget", "mid", "premium"]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    tier = json.loads((ROOT / "results" / "tier_comparison.json").read_text(encoding="utf-8"))
    val = json.loads((ROOT / "results" / "method_validation.json").read_text(encoding="utf-8"))
    disc = json.loads((ROOT / "results" / "intent_discovery.json").read_text(encoding="utf-8"))
    gold = json.loads((ROOT / "results" / "golden_set_manifest.json").read_text(encoding="utf-8"))

    results = tier["results"]
    ordered = sorted(results.items(), key=lambda kv: TIER_ORDER.index(kv[1]["tier"]))

    lines: list[str] = []
    add = lines.append

    add("# OrderBuddy evaluation — summary")
    add("")
    add(f"Run date **{tier['run_date']}** · prices **PRICES_AS_OF "
        f"{tier['PRICES_AS_OF']}** · API path **{tier['api_path']}**")
    add("")

    # ---- Main table -------------------------------------------------------
    add("## Tier bake-off")
    add("")
    add(f"Golden set: {tier['golden_set']['n_rows']} messages across "
        f"{tier['golden_set']['n_intents']} intents, labels human-reviewed "
        f"({gold.get('corrections_made', 0)} corrections). Gates are "
        "deterministic code checks, not an LLM judge.")
    add("")
    add("| Tier | Model | Correct | Wrong | Abstained | Safety gate | Groundedness gate | Clean & correct | Input tok | Output tok | Cost / 1k |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model, c in ordered:
        s, t, cost = c["scores"], c["tokens"], c["cost"]
        add(
            f"| {c['tier']} | `{model}` | {pct(s['accuracy'])} | "
            f"{pct(s['wrong_rate'])} | {pct(s['abstain_rate'])} | "
            f"{pct(s['safety_pass_rate'])} | {pct(s['groundedness_pass_rate'])} | "
            f"{pct(s['clean_correct_rate'])} | {t['total_input']:,} | "
            f"{t['total_output']:,} | "
            f"${cost['cost_per_1000_classifications_usd']:.3f} |"
        )
    add("")
    add("**Correct + Wrong + Abstained = 100%.** An abstention is never counted "
        "as correct and never folded into wrong — a model declining an "
        "ambiguous message is behaving differently from one guessing wrong. "
        "*Clean & correct* is correct **and** passing both gates; a correct "
        "label that trips a gate is not a usable output.")
    add("")

    # ---- Cost detail ------------------------------------------------------
    add("## Cost per 1,000 classifications")
    add("")
    add(f"Computed from logged token totals at published prices as of "
        f"**{tier['PRICES_AS_OF']}** ([source]({tier['price_source']})).")
    add("")
    add("| Tier | Model | Batch rate in/out $/MTok | Cost / 1k (batch) | Cost / 1k (sync) |")
    add("|---|---|---|---:|---:|")
    for model, c in ordered:
        cost = c["cost"]
        add(
            f"| {c['tier']} | `{model}` | "
            f"${cost['batch_input_rate_per_mtok']} / ${cost['batch_output_rate_per_mtok']} | "
            f"${cost['cost_per_1000_classifications_usd']:.3f} | "
            f"${cost['cost_per_1000_if_sync_usd']:.3f} |"
        )
    add("")

    # ---- Method validation ------------------------------------------------
    add("## Method validation (public benchmarks)")
    add("")
    add("The intent-discovery method scored against two standard benchmarks, "
        "using the same code path that built the OrderBuddy taxonomy.")
    add("")
    add("Both criteria are reported side by side. They answer different "
        "questions and neither is the whole truth.")
    add("")
    add("| Benchmark | Rows | Gold intents | Clusters | Recovered (strict) | Recovered (lenient) | Accuracy (strict) | Accuracy (lenient) | ARI | NMI |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("banking77", "clinc150"):
        r = val["results"][key]
        st, le = r["strict"], r["lenient"]
        add(
            f"| {key} | {r['n_rows']:,} | {r['n_gold_intents']} | "
            f"{r['n_clusters_found']} | **{st['intent_recovery']}** | "
            f"{le['intent_recovery_str']} | **{pct(st['accuracy'])}** | "
            f"{pct(le['accuracy'])} | {r['ari']} | {r['nmi']} |"
        )
    add("")
    add(f"**Headline criterion: strict** (bold). {val['headline_rationale']}")
    add("")

    add("### The two criteria, precisely")
    add("")
    add(f"- **Strict (one-to-one).** {val['criterion_definitions']['strict']}")
    add("")
    add(f"- **Lenient (many-to-one plurality).** "
        f"{val['criterion_definitions']['lenient']}")
    add("")
    add("The gap between them measures fragmentation — clusters pure enough to "
        "take an intent's plurality, but too numerous to be that intent's "
        "single one-to-one match:")
    add("")
    add("| Benchmark | Recovery gap | Accuracy gap | Clusters per recovered intent | Worst-split intent |")
    add("|---|---:|---:|---:|---:|")
    for key in ("banking77", "clinc150"):
        r = val["results"][key]
        g, le = r["criterion_gap"], r["lenient"]
        add(
            f"| {key} | +{g['recovery_delta']} intents | "
            f"+{g['accuracy_delta'] * 100:.1f} pts | "
            f"{le['mean_clusters_per_recovered_intent']} | "
            f"{le['max_clusters_for_one_intent']} clusters |"
        )
    add("")

    add("### Two further conventions, for context")
    add("")
    add("| Benchmark | Strict 1-to-1, F1≥0.50 | Strict 1-to-1, any overlap | Many-to-one plurality | Many-to-one, ≥50% pure | Cited on CV |")
    add("|---|---:|---:|---:|---:|---:|")
    for key in ("banking77", "clinc150"):
        r = val["results"][key]
        v = r["recovery_by_metric_definition"]
        add(
            f"| {key} | **{v['strict_1to1_f1_50']}** | {v['strict_1to1_any']} | "
            f"{v['many_to_one_plurality']} | {v['many_to_one_pure_50']} | "
            f"{r['claimed_in_cv']} |"
        )
    add("")

    # ---- Discovery --------------------------------------------------------
    d = disc["scores"]
    dl = disc.get("scores_lenient", {})
    add("## Intent discovery (OrderBuddy corpus)")
    add("")
    add("Same two criteria as the benchmarks above; strict is the headline.")
    add("")
    add("| Metric | Strict (1-to-1) | Lenient (m-to-1) |")
    add("|---|---:|---:|")
    add(f"| Corpus | {disc['dataset']['n_rows']:,} messages | |")
    add(f"| Ground-truth intents | {disc['dataset']['n_gold_intents']} | |")
    add(f"| Clusters found | {disc['clustering']['n_clusters']} | |")
    add(f"| Intents recovered | **{d['intent_recovery_str']}** | "
        f"{dl.get('intent_recovery_str', 'n/a')} |")
    add(f"| Row accuracy | **{pct(d['accuracy'])}** | "
        f"{pct(dl['accuracy']) if dl else 'n/a'} |")
    add(f"| ARI / NMI | {d['ari']} / {d['nmi']} | |")
    add(f"| Noise (unclustered) | {pct(d['noise_fraction'])} | |")
    if dl:
        add(f"| Clusters per recovered intent | | "
            f"{dl['mean_clusters_per_recovered_intent']} "
            f"(max {dl['max_clusters_for_one_intent']}) |")
    add("")
    if d["missed_intents"]:
        add(f"Intents not recovered: {', '.join(f'`{m}`' for m in d['missed_intents'])}.")
        add("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n\nWrote {OUT}")


if __name__ == "__main__":
    main()
