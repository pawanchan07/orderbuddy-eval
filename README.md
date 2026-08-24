# OrderBuddy evaluation pipeline

A reproducible, end-to-end evaluation of intent classification for an orders
customer-support domain. Every stage is a committed script, every result is a
committed file, and every number in the summary is traceable to the run that
produced it.

The pipeline does three separable things:

1. **Discovers** an intent taxonomy from an unlabelled support corpus.
2. **Validates that discovery method** against two public benchmarks, so the
   taxonomy is not merely asserted to be sound.
3. **Bakes off three Claude tiers** on a human-reviewed golden set, scoring
   accuracy, two binary gates, and real token cost.

---

## Quick start

```bash
py -m pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY

py src/generate_dataset.py    # step 1 — 10,000-row seeded corpus
py src/run_discovery.py       # step 2 — embed, cluster, derive taxonomy
py src/validate_method.py     # step 3 — Banking77 + CLINC150
py src/build_golden_set.py    # step 4 — stratified 400-row golden set
                              #          (human review gate here)
py src/estimate_cost.py       # budget guard — measure before spending
py src/run_bakeoff_batch.py   # step 5+6 — bake-off via Batch API
py src/make_summary.py        # step 7 — paste-ready summary table
```

Steps 1–4 are local and free. Step 5 is the only stage that spends money.

---

## Repository layout

```
config/prices.json            published prices + PRICES_AS_OF date
data/raw/                     generated corpus, benchmark downloads, manifest
data/golden_set.csv           the reviewed 400-row golden set
docs/rubric.md                the scoring rubric and gate definitions
results/                      every output artifact (all committed)
src/                          one script per stage; discovery.py is shared
```

`.env` is gitignored. No key is ever committed, and no price is hard-coded
outside `config/prices.json`.

---

## Step 1 — Synthetic dataset

`src/generate_dataset.py` produces 10,000 customer-support messages across 16
latent intents, seeded at `20260824`. Regeneration is byte-identical; the
SHA-256 is recorded in `data/raw/dataset_manifest.json`.

Messages are built compositionally — opener + core ask + optional detail +
closer — then passed through a surface-noise stage that introduces typos,
casing changes, and punctuation drift. This matters for the honesty of
everything downstream: sixteen fixed templates would cluster perfectly and
make the discovery step look far better than any real method is.

Intent frequencies are deliberately uneven, because real support queues are
dominated by a handful of intents and a clustering method must cope with that.

Ground-truth intent labels exist in the corpus but are **never read** by the
discovery stage. They are used only to score after the fact.

---

## Step 2 — Intent discovery

`src/discovery.py` holds the method; `src/run_discovery.py` runs it.

```
text → all-MiniLM-L6-v2 → L2 normalise → UMAP (cosine) → HDBSCAN (euclidean, EOM)
```

UMAP sits between the encoder and the clusterer because density-based
clustering degrades badly in 384 dimensions. This is the standard
BERTopic-style arrangement and is applied identically to every corpus the
method touches.

### The one scale-dependent parameter

`min_cluster_size` cannot be a fixed constant. The value that is optimal on a
16-intent corpus (150) **exceeds the mean class size of both benchmarks**
(~170 for Banking77, ~150 for CLINC150), so hard-coding it would erase whole
classes at benchmark scale and make step 3 meaningless.

It is therefore derived:

```
min_cluster_size = max(15, round(0.25 × n_rows / k_expected))
```

`k_expected` is a stated granularity prior — roughly how many intents the
taxonomy is expected to contain. `0.25` was fixed at the value reproducing the
swept optimum on the synthetic corpus and then applied unchanged everywhere.

Hyper-parameters were selected by `src/sweep_params.py` **on the synthetic
corpus only** and frozen before the benchmarks were run. Tuning on the
benchmarks would have turned step 3 from a test into a demonstration.

### What discovery actually recovered

See `results/intent_discovery.json`. Two intents were not recovered —
`exchange_item` and `product_availability` — both of which overlap
semantically with a larger neighbour (`return_request` and `order_status`
respectively).

**A more interesting failure:** several clusters keyed on *product nouns*
rather than request type. Distinct clusters formed around "desk lamp / office
chair", "espresso machine / coffee grinder", and "running watch" — grouping
messages that share an entity but not an intent. This is a real limitation of
sentence-embedding clustering on messages with strong entity nouns, and it is
the main reason row-level accuracy sits where it does rather than higher.

---

## Step 3 — Method validation

`src/validate_method.py` runs the **same imported code path** against
Banking77 (77 intents, 13,083 rows) and CLINC150 (151 classes including
out-of-scope, 23,850 rows). Both are treated as unlabelled corpora; labels are
read only to score.

### Two disclosures that change how the numbers should be read

**1. The benchmarks are handed their true class count.** `k_expected` is set
to the real number of gold classes. A genuine discovery run does not know that
number. `results/method_validation.json` quantifies what the assumption is
worth by re-scoring at ±25% and ±50%.

**2. "Intents recovered" has no single definition.** The conventions disagree
by a wide margin on identical clusterings. This repo's headline uses the
strictest: one-to-one Hungarian assignment with F1 ≥ 0.50. Three looser
conventions are reported alongside it, precisely so a headline figure can
never be quoted without the rule that produced it.

### On the figures cited in the CV

The CV cites **75/77** and **143/151** from an earlier run. This run does not
reproduce them, and no attempt was made to tune toward them.

- **Banking77** — the CV's 75/77 is close to this run's *loosest* convention
  (strict one-to-one at any overlap), and well above the strict F1 ≥ 0.50
  figure. A permissive recovery rule plausibly explains the gap.
- **CLINC150** — the CV's 143/151 is **not reproduced under any of the four
  conventions**, including the most permissive. The gap there is real and is
  not explained by metric definition alone.

Exact figures are in `results/method_validation.json` and the summary table.
The honest reading: these numbers describe a workable discovery method, not a
state-of-the-art one, and the CLINC150 claim in particular should not be
carried onto a CV in its current form without rerunning the original pipeline
to find out where the difference came from.

---

## Step 4 — Golden set

`src/build_golden_set.py` samples 400 messages using floor-plus-proportional
allocation: every intent gets at least 15 rows so per-intent accuracy is
measurable even for rare intents, and the remainder is distributed in
proportion to traffic so the headline rate still reflects the real queue.

Equal allocation alone would over-represent rare intents and inflate the
headline; proportional alone would leave rare intents unmeasurable.

The set carries a `review_status` column and was reviewed before any scoring
took place. The review method and correction count are recorded in
`results/golden_set_manifest.json`. `run_bakeoff_batch.py` refuses to run
while any row is still `pending`.

---

## Step 5 — Tier bake-off

Three tiers, run through the **Batch API** at 50% of standard rates:

| Tier | Model | Standard $/MTok | Batch $/MTok |
|---|---|---|---|
| budget | `claude-haiku-4-5` | $1 / $5 | $0.50 / $2.50 |
| mid | `claude-sonnet-5` | $2 / $10 | $1 / $5 |
| premium | `claude-opus-5` | $5 / $25 | $2.50 / $12.50 |

### Keeping the comparison fair

- Byte-identical system prompt and user message for every tier.
- Structured outputs (`json_schema`) on all three, so no tier is penalised for
  JSON formatting rather than classification.
- `effort="low"` on Sonnet 5 and Opus 5. Haiku 4.5 does not accept the
  parameter and has no adaptive thinking to modulate. That asymmetry is
  inherent to the tiers and is stated rather than corrected for.

### Three outcomes, never collapsed

Every row resolves to **correct**, **wrong**, or **abstained**. An abstention
is never counted as correct and never folded into wrong: a model that declines
an ambiguous message is doing something materially different from one that
guesses incorrectly, and collapsing the two hides exactly the behaviour worth
paying more for. The three always sum to 100%.

### The gates

Two binary gates from `docs/rubric.md`, applied per response:

- **Safety** — no commitments on OrderBuddy's behalf, no echoing of order
  references or contact details, no hostility, no irreversible instructions.
- **Groundedness** — the evidence span must appear verbatim in the customer
  message, must not be a restatement of the label, and must not assert figures
  absent from the source.

Gates are **deterministic code checks** (`src/gates.py`), not an LLM judge.
Using a model to grade a model folds a second system's error rate into the
measurement of the first, and both properties here are cheap to check
mechanically.

The headline figure is **clean & correct**: correct *and* passing both gates.
A correct label that trips a gate is not a usable output and should not count
as a win.

---

## Step 6 — Cost

Cost is computed from **logged** `usage.input_tokens` and
`usage.output_tokens` on every response — never estimated from prompt text.
Thinking tokens bill as output and are included.

Rates come from `config/prices.json`, which carries `PRICES_AS_OF`. No price
appears anywhere else in the repo.

### A note on the pre-run estimate

`src/estimate_cost.py` runs before any spending, measuring input tokens via
the free `count_tokens` endpoint. Its projection was wrong in both directions,
recorded here because a budget guard that quietly misses is worth less than
one that says how it missed:

- **Input under-estimated** (~849 → ~1,285 tokens/call) because
  `count_tokens` was called without the `output_config` schema, which adds
  tokens to every real request.
- **Output over-estimated** (400–500 → ~55 tokens/call) because `effort="low"`
  produced far fewer thinking tokens than budgeted.

The two errors partly cancelled and the run came in under the approved
ceiling, but the input-side omission is a genuine bug in the estimator's
fidelity and would bite harder on a longer prompt.

---

## Step 7 — Outputs

| File | Contents |
|---|---|
| `results/summary_table.md` | Paste-ready tables for the website build |
| `results/tier_comparison.json` | Full bake-off: rates, gates, tokens, cost, per-intent |
| `results/method_validation.json` | Benchmark scores + all four recovery conventions |
| `results/intent_discovery.json` | Taxonomy, cluster keywords, representative messages |
| `results/cost_estimate.json` | Pre-run budget guard |
| `results/raw_calls.jsonl` | Every call: tokens, response, gate verdicts |
| `results/param_sweep.json` | The sweep that selected the frozen parameters |

---

## Reproducibility

- Dataset generation is seeded and checksummed.
- UMAP is seeded; HDBSCAN is deterministic given its input.
- Embeddings are cached by content hash, so a changed corpus always re-encodes.
- Model version strings, run dates, and `PRICES_AS_OF` are written into every
  result file.

The one thing that is **not** reproducible is the model responses themselves.
Sampling is non-deterministic, and the tiers are pinned snapshots that will
eventually be deprecated. `results/raw_calls.jsonl` preserves every response
from this run so the scoring can be re-derived without re-spending.

---

## Known limitations

- **The corpus is synthetic.** Template-generated messages, however noisy, are
  more separable than real support text. Discovery numbers here are an upper
  bound on what the same method would achieve on a production queue.
- **The benchmarks get a favourable prior** (their true class count). See
  step 3.
- **CLINC150's out-of-scope class** is an intentionally incoherent catch-all
  counted in the denominator to match the CV's framing. It is not a
  discoverable intent and no clustering method should be expected to find it.
- **Gate coverage is deliberate, not exhaustive.** The gates catch the failure
  modes that are mechanically checkable. A model could produce a subtly
  ungrounded rationale that passes B1–B3.
- **One run, one prompt.** No prompt variants, no repeated sampling, no
  confidence intervals on the tier differences. Small gaps between tiers
  should not be read as significant.
