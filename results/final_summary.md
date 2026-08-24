# OrderBuddy tier comparison — final

Run date **2026-08-25** · prices **PRICES_AS_OF 2026-08-24** · Batch API · prompt **v2** · golden set **400 messages, 16 intents**

| Tier | Model version | Accuracy | Abstained | Clean & correct | Cost / 1,000 |
|---|---|---:|---:|---:|---:|
| budget | `claude-haiku-4-5` | 98.2% | 1.2% | 94.0% | $0.590 |
| mid | `claude-sonnet-5` | 98.5% | 0.5% | 97.5% | $1.534 |
| premium | `claude-opus-5` | 98.5% | 1.5% | 98.5% | $3.868 |

*Accuracy* counts exact intent matches against the reviewed gold label. *Abstained* is reported separately and is never counted as correct nor folded into wrong; correct + wrong + abstained = 100%. *Clean & correct* is correct **and** passing both the safety and groundedness gates. Cost is computed from logged token usage at published Batch API rates as of 2026-08-24.

## Effect of the v2 prompt fix

v2 adds one sentence to the prompt stating that the "never repeat identifiers" rule outranks the "quote verbatim" rule when they conflict. Measured on the 80 rows whose message contains an identifier — the complete set where the A2 safety rule can fire:

| Tier | Model | A2 failures v1 | A2 failures v2 | Safety v1 → v2 | Clean & correct v1 → v2 (treatment) |
|---|---|---:|---:|---:|---:|
| budget | `claude-haiku-4-5` | 48/80 | 5/80 | 40.0% → 93.8% | 40.0% → 80.0% |
| mid | `claude-sonnet-5` | 49/80 | 4/80 | 38.8% → 95.0% | 38.8% → 95.0% |
| premium | `claude-opus-5` | 5/80 | 0/80 | 93.8% → 100.0% | 93.8% → 100.0% |

| Tier | Clean & correct, full set v1 | Clean & correct, full set v2 |
|---|---:|---:|
| budget | 86.2% | 94.0% |
| mid | 86.5% | 97.5% |
| premium | 97.5% | 98.5% |

### Control group — did the fix cost anything elsewhere?

100 sampled rows with no identifier, where A2 cannot fire. This group exists only to detect accuracy regression from the added sentence:

| Tier | Accuracy v1 | Accuracy v2 | Delta |
|---|---:|---:|---:|
| budget | 99.0% | 98.0% | -1.0 pts |
| mid | 100.0% | 99.0% | -1.0 pts |
| premium | 100.0% | 99.0% | -1.0 pts |

### How the full-set v2 figure is composed

The v2 run covered 180 of 400 rows: all 80 treatment rows plus the 100-row control. The remaining 220 rows carry their v1 result forward. This is sound for the safety figure because A2 cannot fire on a message containing no identifier, and the v1 run confirmed it empirically — zero A2 failures outside the treatment group across all three tiers. The control group tests the other direction and its delta is reported above. Where a directly-measured number is preferred to a composed one, cite the treatment-group table.

## Abstention rates

Derived from the logged calls (`results/raw_calls*.jsonl`) with no further API spend. An abstention is counted **only** when the model returned the explicit `abstain` label. Abstentions are never merged into wrong answers; correct + wrong + abstained = 100% in every row.

| Tier | Model version | Abstention rate (v2, final) | Abstention rate (v1) | Abstentions v2 | Abstentions v1 |
|---|---|---:|---:|---:|---:|
| budget | `claude-haiku-4-5` | 1.2% | 1.2% | 5/400 | 5/400 |
| mid | `claude-sonnet-5` | 0.5% | 0.5% | 2/400 | 2/400 |
| premium | `claude-opus-5` | 1.5% | 1.2% | 6/400 | 5/400 |

Full-set v2 is the composed final configuration (v2 where measured on 180 rows, v1 carried forward on 220). For a like-for-like read of what the prompt change did to abstention behaviour, compare only the rows actually re-run:

| Tier | Model version | Abstentions v1 (same 180 rows) | Abstentions v2 (same 180 rows) | Change |
|---|---|---:|---:|---:|
| budget | `claude-haiku-4-5` | 1/180 (0.6%) | 1/180 (0.6%) | +0 |
| mid | `claude-sonnet-5` | 0/180 (0.0%) | 0/180 (0.0%) | +0 |
| premium | `claude-opus-5` | 0/180 (0.0%) | 1/180 (0.6%) | +1 |

Unparseable responses across every tier and both prompt versions: **0**. Structured outputs were used throughout, so no response had to be discarded or guessed at, and none was counted as an abstention.
