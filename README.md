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

## Headline findings

**1. Model tier barely mattered for accuracy; prompt clarity mattered enormously.**
All three tiers classify at 98.2–98.5% and sit within 0.3 points of each
other — statistically indistinguishable on 400 rows. What separated them was a
safety rule, and the gap turned out to be my prompt's fault, not the models'.

**2. One ambiguous instruction created what looked like a capability gap.**
The prompt asked for a verbatim quote *and* forbade repeating order
references, without saying which wins when they collide. The budget tier
violated the safety rule 48 times in 80 chances; the premium tier, 5. Adding
one sentence naming the precedence cut budget-tier violations from 48 to 5 —
premium-tier behaviour at a sixth of the cost. The original gap was measuring
prompt ambiguity, not model capability.

**3. Fixing one gate exposed cheating on another.** Told not to repeat the
identifier, the budget tier began *rewriting the quote* to remove it —
emitting `"wrong address on my order"` where the message said `"wrong address
on 32435772"`. A verbatim span that was not verbatim. It traded 43 safety
failures for 11 groundedness failures. Scoring safety alone would have
declared it fixed; the second gate caught it. That is the strongest argument
in this repo for cheap deterministic gates that constrain each other.

**4. The public-benchmark numbers came in well below the figures this work
previously claimed**, and no attempt was made to tune toward them. See
[On the figures cited in the CV](#on-the-figures-cited-in-the-cv).

### Final numbers

| Tier | Model | Accuracy | Abstained | Clean & correct | Cost / 1,000 |
|---|---|---:|---:|---:|---:|
| budget | `claude-haiku-4-5` | 98.2% | 1.2% | 94.0% | $0.590 |
| mid | `claude-sonnet-5` | 98.5% | 0.5% | 97.5% | $1.534 |
| premium | `claude-opus-5` | 98.5% | 1.5% | **98.5%** | $3.868 |

400-message golden set · Batch API · prompt v2 · run 2026-08-25 ·
PRICES_AS_OF 2026-08-24. Abstentions are counted separately and never folded
into wrong answers. Full detail in [Results](#results) and
[The A2 prompt experiment](#the-a2-prompt-experiment); limitations that
qualify all of the above are in [Known limitations](#known-limitations).

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
by a wide margin on identical clusterings, so every score is reported under
**two** precisely-defined criteria, side by side.

#### The strict criterion (one-to-one)

Clusters are matched to gold intents by **Hungarian assignment** maximising
total overlap, so each cluster claims at most one intent and each intent is
claimed by at most one cluster. A gold intent counts as **recovered** when its
assigned cluster reaches **F1 ≥ 0.50** against it, where precision is the
share of that cluster's rows carrying the intent and recall is the share of
the intent's rows landing in that cluster. Row accuracy uses the same mapping.
HDBSCAN noise (`-1`) counts as incorrect.

#### The lenient criterion (many-to-one plurality)

Every cluster is mapped to **its own plurality gold label**. Several clusters
may map to the same intent. A gold intent counts as **recovered** if it is the
plurality label of at least one cluster. Row accuracy asks whether a row's
cluster carries that row's label as its plurality. Noise counts as incorrect,
as in strict.

This is the conventional "clustering accuracy / many-to-one accuracy" of the
clustering literature. It measures **cluster purity**: given that a message
landed somewhere, does that cluster's dominant label match?

#### Which is the honest headline

**The strict criterion is the honest headline, and it is what this repo
reports in bold.**

The deliverable of this stage is a *usable taxonomy* — a label set someone can
build routing on. Only the strict criterion tests for one, because it demands
that each intent have a single cluster that both covers it and is mostly it.
The lenient criterion has no term that penalises splitting one intent across
ten clusters; each fragment can independently claim a recovery. A method could
score near-perfectly on lenient recovery while producing a taxonomy far too
fragmented to use.

The lenient number is not padding, though. It isolates a genuine property —
purity — and the **gap between the two is itself the diagnostic**: it measures
exactly how fragmented the clustering is. On these runs the gap is driven by
intents split across 2–3 clusters apiece, which is why lenient recovery runs
9–16 intents ahead of strict while row accuracy moves only 5–7 points.

Two further conventions (strict at any non-zero overlap; many-to-one requiring
≥50% cluster purity) are reported in `results/method_validation.json` for
context, so a headline figure can never be quoted without the rule that
produced it.

### On the figures cited in the CV

The CV cites **75/77** and **143/151** from an earlier run. This run does not
reproduce them, and no attempt was made to tune toward them.

| Benchmark | Strict (headline) | Lenient | Loosest tested | Cited on CV |
|---|---:|---:|---:|---:|
| Banking77 | **55/77** | 71/77 | 72/77 | 75/77 |
| CLINC150 | **117/151** | 126/151 | 128/151 | 143/151 |

- **Banking77** — the CV's 75/77 sits just above this run's *loosest*
  convention (72/77). A permissive recovery rule plausibly explains most of
  that gap.
- **CLINC150** — the CV's 143/151 is **not reproduced under any of the four
  conventions**, including the most permissive (128/151). A 15-intent gap
  remains that metric definition does not explain.

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

## Results

Full numbers in `results/summary_table.md`. The headline:

**Final (v2 prompt)** — the table for the website widget, also in
`results/final_summary.md`:

| Tier | Model version | Accuracy | Abstained | Clean & correct | Cost / 1,000 |
|---|---|---:|---:|---:|---:|
| budget | `claude-haiku-4-5` | 98.2% | 1.2% | 94.0% | $0.590 |
| mid | `claude-sonnet-5` | 98.5% | 0.5% | 97.5% | $1.534 |
| premium | `claude-opus-5` | 98.5% | 1.5% | **98.5%** | $3.868 |

Run date 2026-08-25 · PRICES_AS_OF 2026-08-24 · Batch API · 400-row golden set.

**Initial (v1 prompt)**, kept for the before/after:

| Tier | Model | Correct | Wrong | Abstained | Safety | Groundedness | Clean & correct | Cost / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| budget | `claude-haiku-4-5` | 98.5% | 0.2% | 1.2% | 88.0% | 99.8% | 86.2% | $0.581 |
| mid | `claude-sonnet-5` | 98.8% | 0.8% | 0.5% | 87.8% | 100.0% | 86.5% | $1.505 |
| premium | `claude-opus-5` | 98.8% | 0.0% | 1.2% | 98.8% | 100.0% | 97.5% | $3.794 |

**Accuracy does not separate these tiers.** All three land within 0.3 points
of each other, and on a 400-row set that difference is noise. If accuracy were
the only criterion, the budget tier would win outright at a sixth the cost.

**The safety gate separates them, and it is one rule doing all the work.**
Every safety failure across all three tiers is **A2** — echoing an order
reference into the classification output. Haiku failed it 48 times and Sonnet
49, against Opus's 5.

This is a competing-constraints test, not a knowledge test. The prompt asks
for a verbatim evidence span *and* forbids repeating order references, so on
any message where the identifier sits inside the most natural quote, the model
has to notice the tension and pick a different span. That is what the premium
tier is buying here — not better labels, but tighter instruction-following
when two instructions pull against each other.

Whether that is worth 6.5× the cost is a deployment decision, and it depends
entirely on whether those identifiers flow anywhere that matters. Two things
worth noting before treating the gap as settled:

- A prompt revision might close most of the gap on the cheaper tiers. **This
  was subsequently tested — see "The A2 prompt experiment" below. It largely
  worked, and it changes the recommendation.**
- This is one run of one prompt with no repeated sampling, so there are no
  confidence intervals. The 10× gap in A2 failures is large enough to be
  real; the 0.3-point accuracy spread is not.

**Groundedness was near-saturated under v1** (99.8–100%), so it did not
discriminate there. It became the decisive gate under v2 — see below.

---

## The A2 prompt experiment

`src/run_prompt_experiment.py`. The v1 gap looked like an unstated-precedence
problem rather than a capability gap: v1 gives "quote verbatim" (rule 2) and
"never repeat identifiers" (rule 4) as independent requirements and leaves the
model to notice they collide. **v2 adds one sentence** saying rule 4 outranks
rule 2 when they conflict. Nothing else changes, and v1 is byte-identical to
the original run's prompt, so any delta is attributable to that sentence.

**Scope.** Treatment = all 80 golden rows whose message contains an
identifier. This is the complete population where A2 can fire, not a sample:
the v1 run had zero A2 failures outside it across all three tiers. Control =
100 sampled rows without identifiers, to detect whether the added sentence
degrades accuracy where it was not aimed. Cost: $1.10.

| Tier | A2 fails v1 → v2 | Safety v1 → v2 | Groundedness v2 | Clean & correct v1 → v2 |
|---|---:|---:|---:|---:|
| budget | 48/80 → **5/80** | 40.0% → 93.8% | 69/80 | 40.0% → **80.0%** |
| mid | 49/80 → **4/80** | 38.8% → 95.0% | 80/80 | 38.8% → **95.0%** |
| premium | 5/80 → **0/80** | 93.8% → 100.0% | 80/80 | 93.8% → **100.0%** |

Control accuracy moved −1.0 point on every tier — one row each. That is noise,
not a regression.

### The hypothesis held, but not the whole way

On safety alone the fix is decisive. Haiku's A2 failures drop 48 → 5, landing
exactly on Opus's *v1* score. One sentence recovered nearly the entire gap at
a sixth of the cost, which means the v1 result was **not** measuring a
capability difference — it was measuring which model noticed an ambiguity in
the prompt that should not have been there.

**But the budget tier does not reach parity on clean & correct, because it
satisfies the new rule the wrong way.** Haiku *rewrites the quote* to remove
the identifier — emitting `"wrong address on my order, can you fix it?"` where
the message read `"wrong address on 32435772, can you fix it?"`. That is a
verbatim span which is not verbatim. It trades 43 safety failures for 11
groundedness failures. Sonnet and Opus do what was asked and select a
different genuine span: zero groundedness failures.

So the residual gap is no longer about *spotting* the conflict — all three
tiers now spot it — but about *resolving it without fabricating*.

### What this says about the gates

Scoring safety alone would have declared the budget tier fixed. It is the
second gate that catches a model paying for one rule with another. Two cheap
mechanical checks that constrain each other found something neither would have
found alone, which is a stronger argument for deterministic gates than any of
the individual pass rates.

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
