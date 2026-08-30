# AI Text Forensics

Local AI-writing detector and humanizer. Scores prose for how machine-generated
it reads, shows the evidence sentence by sentence, then rewrites it to read as
human without changing what it says.

## Why this project

Most AI-writing detectors provide a score without explaining it. 
AI Text Forensics runs locally, highlights sentence-level evidence, and can rewrite text while protecting its meaning and facts.

## Privacy posture

- **Detection never touches the network.** All scoring runs on-device.
- **The Claude rewrite pass is opt-in, per document, off by default.** A badge
  in the UI always shows which mode you're in.
- **No telemetry.** The server binds `127.0.0.1` only. Documents live in a local
  SQLite file you can delete.
- One-time network use: downloading public model weights on first run of
  Phase 2. That's a download, not an upload.

## Setup

```bash
cd backend
uv venv --python 3.12
uv pip install -e ".[dev,models,calibrate,llm,export]"
.venv/bin/python -m spacy download en_core_web_sm

cd ../frontend && npm install
```

## Running the web UI

Two terminals:

```bash
cd backend && .venv/bin/humanizer serve      # API on 127.0.0.1:8420
cd frontend && npm run dev                   # UI  on 127.0.0.1:5173
```

Open **http://127.0.0.1:5173**. Paste text or drop a `.txt/.md/.docx/.pdf`,
hit Analyze for the score and sentence heatmap, then Humanize. Export the
result as `.txt/.md/.docx/.pdf`, or download a Markdown score report.

Both servers bind to loopback only and are unreachable from your network. The
header badge always shows whether the Claude pass is enabled.

## Command line

```bash
.venv/bin/humanizer score essay.docx              # score a document
.venv/bin/humanizer score essay.md --sentences    # + per-sentence heatmap
.venv/bin/humanizer score --text "Some prose..."  # score literal text
cat notes.txt | .venv/bin/humanizer score         # or pipe it
.venv/bin/humanizer info                          # config + component status
```

Reads `.txt`, `.md`, `.docx`, `.pdf`.

## What the score means

| Range  | Verdict                  |
|--------|--------------------------|
| 0–20   | Reads as human           |
| 20–40  | Mostly human             |
| 40–60  | Mixed signals            |
| 60–80  | Likely AI-generated      |
| 80–100 | Very likely AI-generated |

Below ~150 words the statistics are too noisy for a confident verdict, and the
tool says so instead of guessing.

## Responsible use

AI-detection scores are probabilistic indicators, not proof of authorship. 
Do not use a score alone for academic, employment, or disciplinary decisions. 
Review the sentence-level evidence and context before drawing conclusions.

## How detection works

Four tiers of evidence, fused into one score. The two families are
complementary and both are needed:

**Tier A measures sampling signature** — it catches genuine model output even
when the style has been cleaned up. **Tiers B–D measure style** — they catch
machine text that has been paraphrased or hand-edited until its sampling
signature is gone. Text that defeats one routinely trips the other.

Measured on three 300–370 word samples:

| | Real model output | Human imitating AI | Human |
|---|---|---|---|
| Tier A (statistical) | 94.7 | 36.0 | 2.1 |
| Tiers B–D (style) | 36–72 | 65–84 | 3–10 |
| **Final** | **79.0** | **56.8** | **6.2** |

- **Tier A — statistical**: perplexity, per-sentence burstiness,
  Fast-DetectGPT conditional-probability curvature (analytic form — exact, no
  perturbation sampling), Binoculars cross-perplexity ratio. Two local
  Qwen2.5-0.5B models, ~1 s per document once loaded.
- **Tier B — structural**: sentence-length variation, paragraph uniformity,
  opener diversity, syntactic-depth variation, rule-of-three lists,
  "not just X, but Y" constructions.
- **Tier C — lexical**: LLM-marker vocabulary, formulaic stock phrases,
  mechanical discourse connectives, hedging without commitment, vocabulary
  compression.
- **Tier D — mechanical**: punctuation fingerprint, specificity starvation
  (concrete numbers, dates, citations), intro/conclusion redundancy,
  information-density variation.

Every threshold in `detect/rulescore.py` is a documented prior, written as a
(human_typical, machine_typical) pair so a wrong one is easy to spot and fix.
Phase 3 replaces them with weights fit on real data.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 1 | Ingest, segmentation, Tier B/C/D features, rule scorer, CLI | ✅ done |
| 2 | Local statistical detectors (Tier A), ensemble fusion | ✅ done |
| 3 | Calibration harness and score mapping | ✅ harness done — needs human corpus |
| 4 | Local humanizer with adversarial loop | ✅ done |
| 5 | Claude rewrite engine (opt-in) | ✅ done |
| 6 | React UI and export | ✅ done |
| 7 | Calibration against real detectors | pending |

## Humanizing

```bash
humanizer humanize essay.docx --strength aggressive --diff
humanizer humanize essay.md -o rewritten.md
```

Greedy best-first search against our own detector, bounded by two guards that
can veto any rewrite: embedding similarity (meaning must survive) and fact
preservation (no invented or dropped numbers, entities, or citations).

Measured, local engine only, no API:

| Sample | Before | After | Meaning preserved |
|---|---|---|---|
| Real model output | 79.0 | **37.2** | 0.995 |
| Human imitating AI | 56.8 | **32.7** | 0.979 |
| Genuine human | 6.2 | 6.2 (untouched) | 1.000 |

### Optional Claude pass

```bash
cp backend/.env.example backend/.env      # set HUMANIZER_LLM_ENABLED and ANTHROPIC_API_KEY
humanizer humanize essay.docx --claude
humanizer humanize thesis.docx --claude --high-quality   # escalate to Opus 5
```

This is the only feature that sends text off the machine. It is off by default,
asks for confirmation per document, quotes cost first, enforces a hard spend cap
(`--budget`, default $0.15/document), and reports actual token spend afterwards.

Measured on a 370-word machine-written document, Sonnet 5:

| Strength | Cost | Score | Target met |
|---|---|---|---|
| local only | $0.00 | 79.0 → 37.2 | no |
| `--claude --strength light` | ~$0.09 | 79.0 → 32.8 | no |
| `--claude` (moderate) | ~$0.15 | 79.0 → **19.4** | **yes** |
| `--claude --strength aggressive` | ~$0.30 | 79.0 → **15.1** | **yes** |

Moderate is the sweet spot: it clears the <20 target at about $0.15 a document,
so $5 of credit covers roughly 30 documents. Cost scales with paragraph count,
not just word count — the loop makes one call per targeted paragraph per
iteration.

**Claude gets no privileged path.** It produces *candidates*; they face the same
meaning and fact guards as a regex transform and are accepted only if they lower
the real score. A fluent rewrite that invents a statistic is discarded — there's
a test that asserts exactly this, rejecting a plausible-sounding "73% of
managers reported…" with `claude: invents number`.

Two implementation notes: rewriting happens at paragraph scale, because rhythm
is a property of a passage rather than a sentence; and variants come several per
call rather than one per call, because Sonnet 5 and Opus 5 reject non-default
`temperature`, so asking one request for several deliberately-different rewrites
is both cheaper and more varied than repeated sampling.

### Two findings worth knowing

**Lexical de-marking can make text *more* detectable.** Swapping an unusual
marker word for its most obvious synonym is argmax behavior — it walks the text
toward the local probability maximum that Fast-DetectGPT keys on. Measured
directly: a fixed-choice bulk substitution halved the lexical tier while driving
curvature from −0.41 to 2.22, raising the overall score. Two mitigations ship as
a result: substitution choices are varied rather than fixed, and every iteration
is verified against the full scorer (models included) and **rolled back if the
real score rose**, even when the fast style-only proxy reported progress.

**There is a floor, and it is the right floor.** After rewriting, the signals
that stay maxed are concrete-number rate, proper-noun rate, and first-person
presence. Those can only be improved by inventing specifics, which the fact
guard forbids. A document with nothing concrete in it has a detection floor
that no honest rewriter can lower.

## Calibration

The hand-tuned thresholds work, but fitting them on real data works better.

```bash
humanizer calibrate generate -n 40         # machine samples, generated locally
humanizer calibrate add ~/my-essays -l human   # your own writing
humanizer calibrate status                 # composition + problems
humanizer calibrate train                  # fit, cross-validate, report
```

**The corpus is gitignored and never leaves this machine.** Calibrating partly
on your own writing is the point, not a compromise: a detector tuned to a
generic corpus will mis-score an individual voice, while one that has seen
yours learns not to flag the things you actually do.

The trainer refuses to save a model it can't justify. It audits the corpus
first and blocks on fewer than 10 samples per class, warns on class imbalance,
on a single generator, and on registers that appear in only one class. It also
refuses to report "TPR @ 1% FPR" when the corpus is too small to resolve 1% —
with 30 human samples the finest measurable rate is 3.3%, and it says so rather
than printing a more flattering number.

Until a run clears the bar, the ensemble keeps its hand-tuned weights.

**Why the human half can't be auto-generated:** any "human" sample this project
writes for you would in fact be machine text wearing a human label, and
training on it would teach the detector to call machine text human — the exact
inversion of what it's for. Genuine human writing has to come from outside.

## Design constraint

The rewriter will never invent facts, numbers, dates, names, or citations. A
hard guard rejects any rewrite that introduces an entity the original didn't
contain.

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q
```
