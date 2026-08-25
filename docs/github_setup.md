# GitHub setup — repo description, topics, and profile README

Everything here is copy-paste. Nothing in this file is executed by the
pipeline; it exists so the public presentation is version-controlled alongside
the work it describes.

---

## 1. Repository "About" description

This is the one-line blurb GitHub shows at the top-right of the repo page and
in search results. Limit is **350 characters**.

**Recommended (338 chars):**

```
End-to-end LLM eval pipeline by Pawanchander Komuravelli (PM & builder): 10k synthetic support tickets, intent discovery via MiniLM + HDBSCAN, benchmark validation, and a 3-tier Claude bake-off with binary safety/groundedness gates and cost per 1,000 classifications. Evidence before roadmap, evals before demos.
```

**Shorter alternative (208 chars)** — if you prefer the description to read as
a claim rather than a spec:

```
How I run AI evals, shown end to end: synthetic data, intent discovery, a 3-tier Claude bake-off with safety and groundedness gates, and real cost per 1,000. By Pawanchander Komuravelli, PM & builder.
```

**Finding-first alternative (277 chars)** — leads with the result, which is the
most quotable thing in the repo:

```
An LLM eval that found the model tier barely mattered and one ambiguous prompt line did: budget-tier safety failures fell 48 to 5 after a one-sentence fix. Full pipeline, gates, and cost math. By Pawanchander Komuravelli.
```

**How to set it:** repo page → the gear icon next to **About** (top right) →
paste into *Description* → **Save changes**.

While that panel is open, also tick **Include in home page: Releases /
Packages** off if you like, and add the website link below.

**Website field:** `https://pawanchander.com`

---

## 2. Repository topics

Topics drive GitHub search and show as chips under the description. Same
**About** panel → *Topics*. Recommended set:

```
llm-evaluation  ai-evals  intent-classification  claude  anthropic-api
hdbscan  sentence-transformers  benchmarking  synthetic-data
product-management  clustering  nlp
```

Topics must be lowercase and hyphenated. GitHub allows up to 20.

---

## 3. GitHub profile description

There are **two separate things**, and people usually mean the second.

### 3a. The Bio (short, appears under your avatar)

160 characters, plain text, no markdown links.

- Go to <https://github.com/settings/profile>
- Fill in **Bio**, then **Update profile**

Suggested (148 chars):

```
Product Manager & builder. AI evals, cost-aware LLM systems, 0-to-1 products. Founded SkillAid ($110K raised). Evidence before roadmap. pawanchander.com
```

Also on that page, fill in **Name**, **Company**, **Location**, and
**Website** (`https://pawanchander.com`) — recruiters filter on these.

### 3b. The Profile README (the big panel on your profile page)

This is the rendered README that appears above your pinned repos. It works
through a naming trick: GitHub renders the README of a repository whose name
is **exactly your username**.

1. Go to <https://github.com/new>
2. Repository name: **`pawanchan07`** — it must match your username exactly.
   GitHub will show a message like *"You found a secret! pawanchan07/pawanchan07
   is a special repository you can use to add a README.md to your GitHub
   profile."* That confirms you typed it correctly.
3. Set it to **Public**.
4. Tick **Add a README file**.
5. **Create repository**, then edit `README.md`.

A starting draft in your voice is in `docs/profile_README.md` in this repo —
copy its contents into that new repository's `README.md`.

### 3c. Pin this repo

Profile page → **Customize your pins** → select `orderbuddy-eval` (and up to
five others). Pinned repos are what a recruiter actually looks at.

---

## 4. Before you flip it to public — one consistency check

Your site currently cites eval figures for this project. The numbers this run
actually produced, from `results/final_summary.md`:

| Tier | Model | Accuracy | Clean & correct | Cost / 1,000 |
|---|---|---:|---:|---:|
| budget | `claude-haiku-4-5` | 98.2% | 94.0% | $0.590 |
| mid | `claude-sonnet-5` | 98.5% | 97.5% | $1.534 |
| premium | `claude-opus-5` | 98.5% | 98.5% | $3.868 |

Two things to reconcile, because once the repo is public anyone can diff the
site against it:

1. **The 95.0% figure attributed to Haiku.** Haiku's clean-and-correct on the
   full set is 94.0% and its accuracy is 98.2%. 95.0% is the *Sonnet*
   treatment-group clean-and-correct number. Worth checking which figure the
   site means to quote.

2. **"2.5x the cost buys 5 points of clean output."** On the final numbers,
   Sonnet is 2.6x Haiku's cost for +3.5 points of clean-and-correct, and Opus
   is 2.5x Sonnet's cost for +1.0 point. Neither pairing is "5 points". The
   honest version of that line is stronger anyway, because it is the actual
   finding: *the tier premium bought far less than a one-sentence prompt fix
   did.*

3. **The Banking77 / CLINC150 figures.** If a CV or the site cites 75/77 and
   143/151, this repo publicly reports 55/77 and 117/151 under its headline
   criterion, and documents that CLINC150's 143/151 is not reproduced under
   any of four conventions tested. Publishing is the honest call, but make it
   deliberately — and consider updating the CV line to the reproduced numbers,
   or to the framing "recovery is definition-sensitive; see repo", rather than
   leaving a recruiter to find the gap themselves.
