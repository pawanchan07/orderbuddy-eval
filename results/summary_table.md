# OrderBuddy evaluation — summary

Run date **2026-08-25** · prices **PRICES_AS_OF 2026-08-24** · API path **batch (50% of standard rates on input and output)**

## Tier bake-off

Golden set: 400 messages across 16 intents, labels human-reviewed (0 corrections). Gates are deterministic code checks, not an LLM judge.

| Tier | Model | Correct | Wrong | Abstained | Safety gate | Groundedness gate | Clean & correct | Input tok | Output tok | Cost / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| budget | `claude-haiku-4-5` | 98.5% | 0.2% | 1.2% | 88.0% | 99.8% | 86.2% | 396,817 | 13,560 | $0.581 |
| mid | `claude-sonnet-5` | 98.8% | 0.8% | 0.5% | 87.8% | 100.0% | 86.5% | 512,913 | 17,853 | $1.505 |
| premium | `claude-opus-5` | 98.8% | 0.0% | 1.2% | 98.8% | 100.0% | 97.5% | 512,913 | 18,828 | $3.794 |

**Correct + Wrong + Abstained = 100%.** An abstention is never counted as correct and never folded into wrong — a model declining an ambiguous message is behaving differently from one guessing wrong. *Clean & correct* is correct **and** passing both gates; a correct label that trips a gate is not a usable output.

## Cost per 1,000 classifications

Computed from logged token totals at published prices as of **2026-08-24** ([source](https://platform.claude.com/docs/en/about-claude/pricing)).

| Tier | Model | Batch rate in/out $/MTok | Cost / 1k (batch) | Cost / 1k (sync) |
|---|---|---|---:|---:|
| budget | `claude-haiku-4-5` | $0.5 / $2.5 | $0.581 | $1.161 |
| mid | `claude-sonnet-5` | $1.0 / $5.0 | $1.505 | $3.011 |
| premium | `claude-opus-5` | $2.5 / $12.5 | $3.794 | $7.588 |

## Method validation (public benchmarks)

The intent-discovery method scored against two standard benchmarks, using the same code path that built the OrderBuddy taxonomy.

Both criteria are reported side by side. They answer different questions and neither is the whole truth.

| Benchmark | Rows | Gold intents | Clusters | Recovered (strict) | Recovered (lenient) | Accuracy (strict) | Accuracy (lenient) | ARI | NMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 13,083 | 77 | 91 | **55/77** | 71/77 | **58.7%** | 65.9% | 0.3442 | 0.7738 |
| clinc150 | 23,850 | 151 | 152 | **117/151** | 126/151 | **64.1%** | 69.5% | 0.4231 | 0.8502 |

**Headline criterion: strict** (bold). The deliverable of this stage is a usable taxonomy, and only the strict criterion tests for one. Many-to-one lets several fragments of a single intent each count as a recovery, which inflates the score without yielding a taxonomy anyone could build a routing workflow on.

### The two criteria, precisely

- **Strict (one-to-one).** Clusters are matched to gold intents by Hungarian assignment maximising total overlap, so each cluster claims at most one intent and each intent at most one cluster. A gold intent counts as recovered when its assigned cluster reaches F1 >= 0.50 against it (F1 over that cluster's precision and recall for that intent). Row accuracy uses the same mapping. HDBSCAN noise (-1) counts as incorrect.

- **Lenient (many-to-one plurality).** Many-to-one plurality, the conventional clustering-accuracy criterion. Every cluster is mapped to its own plurality gold label; several clusters may map to the same intent. A gold intent counts as recovered if it is the plurality label of at least one cluster. Row accuracy asks whether a row's cluster has that row's label as its plurality. HDBSCAN noise (-1) counts as incorrect, as in strict.

The gap between them measures fragmentation — clusters pure enough to take an intent's plurality, but too numerous to be that intent's single one-to-one match:

| Benchmark | Recovery gap | Accuracy gap | Clusters per recovered intent | Worst-split intent |
|---|---:|---:|---:|---:|
| banking77 | +16 intents | +7.2 pts | 1.28 | 3 clusters |
| clinc150 | +9 intents | +5.4 pts | 1.21 | 2 clusters |

### Two further conventions, for context

| Benchmark | Strict 1-to-1, F1≥0.50 | Strict 1-to-1, any overlap | Many-to-one plurality | Many-to-one, ≥50% pure | Cited on CV |
|---|---:|---:|---:|---:|---:|
| banking77 | **55/77** | 72/77 | 71/77 | 65/77 | 75/77 |
| clinc150 | **117/151** | 128/151 | 126/151 | 112/151 | 143/151 |

## Intent discovery (OrderBuddy corpus)

Same two criteria as the benchmarks above; strict is the headline.

| Metric | Strict (1-to-1) | Lenient (m-to-1) |
|---|---:|---:|
| Corpus | 10,000 messages | |
| Ground-truth intents | 16 | |
| Clusters found | 23 | |
| Intents recovered | **14/16** | 15/16 |
| Row accuracy | **60.8%** | 70.0% |
| ARI / NMI | 0.4816 / 0.6614 | |
| Noise (unclustered) | 14.8% | |
| Clusters per recovered intent | | 1.53 (max 3) |

Intents not recovered: `exchange_item`, `product_availability`.
