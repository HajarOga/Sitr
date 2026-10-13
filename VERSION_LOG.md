# Version log

## 0.2.1 — the "Safe version" column now shows the safe version

**Fixed**

- The Review page's right column was rendering the **original** text: `highlight_block()`
  always emits `text[start:end]`, and the right column passed only the KEEP items, so a
  replaced value was shown as itself (in green) and a removed value was still visible. The
  released text was correct all along — only the preview lied, which is the worst place for
  it, because that is the column a reviewer trusts.
- The right column is now built from the same call the **Generate release** button makes
  (`apply_review(analysis, decisions)`), so it shows what would actually be written: the
  substitute for a REPLACE, a thin mark where a REMOVE took something out, kept values in
  place, and anything still undecided absent from the release. It also updates live as the
  reviewer answers items.

**Changed**

- `released_block_html()` replays the block in reading order instead of highlighting the
  original, because the released string has different offsets. `items_for_block()` lost its
  unused `source` argument; `pending_for_block()` was removed as dead.
- `scripts/check_review_preview.py` is the regression check: it renders both columns and
  asserts the right one equals the released text, that no removed or replaced value is
  visible in it, and that every substitute is.

## 0.2.0 — provider switch: the same pipeline, a faster model

**Added**

- `SAFESHARE2_PROVIDER=api|local|auto`. `api` calls an OpenAI-compatible chat endpoint
  (DeepSeek by default) with `httpx` and `response_format={"type":"json_object"}`;
  `local` is 0.1.0's behaviour. `auto` follows whether a key is configured. The four
  prompts are untouched — both providers are asked identically, and the guards, the
  confidence threshold and the audit log are unchanged.
- `.env` at the project root, read by `app/config.py` (dependency-free loader; a real
  environment variable wins). It holds `DEEPSEEK_API_KEY` and **is committed on purpose**
  for the demo — `.gitignore` keeps the ignore line commented out, to be restored when the
  key is rotated.
- `engine.api_state()` / `_api_generate()`: the API path reports the same state keys as the
  local one (`install_state`, `status`, `loaded_info`), records `tokens_in`/`tokens_out` and
  the endpoint in the call log, and raises `ModelCallFailed` with the HTTP status when the
  call fails. `generate_json` now asks for JSON mode explicitly.
- The dashboard, the API and `setup_model` say which provider is active; `setup_model`
  exits without downloading anything when the API is configured.

**Why**

- A page took ~45s of model time with the 3B model on this laptop and ~3s over the API.
  The trade is stated, not hidden: **with `api` the document text is sent to that endpoint
  in the prompt**. README, `docs/AI.md`, `docs/SETUP.md`, `docs/DEMO.md` and the dashboard's
  Privacy panel were corrected accordingly — 0.1.0's "nothing leaves the machine" is now
  true only of `SAFESHARE2_PROVIDER=local`.

**Verified**

- A real run of the sample document over the API: 13 items, every change citing a rule, the
  guards rejecting the model's non-synthetic substitutes, the release written, the audit
  recording the calls (analyze 983 tokens in 3.25s, summary 47 tokens in 0.67s).
- The dashboard renders in both modes: `Hosted model — deepseek-chat` with the privacy line
  naming the endpoint, or `Model installed — Qwen/Qwen2.5-3B-Instruct`, mps/float16, 5.75 GB.

## 0.1.0 — SafeShare, the AI-core rebuild

A rebuild of the SafeShare prototype with the model at the centre instead of beside the
pipeline.

**Changed from the first prototype**

- The model is now mandatory. `app/core.py` calls `Qwen/Qwen2.5-3B-Instruct` for every
  decision, and `engine.ensure_installed()` raises `ModelNotInstalled` when the weights are
  absent — the dashboard shows the install command and does not offer an analysis. The
  first prototype detected with patterns and decided with `rules.json`, with the model as an
  optional helper.
- The four prompts (`app/llm/prompts.py`) are now the product: analyse, adjudicate, write a
  substitute, write the audit note.
- New `app/guards.py` holds the contract instead of the decision: a cited rule must exist,
  for this receiver and this type; the value must be in the document; a substitute must
  survive validation; a `KEEP` with no rule is never taken on trust; below the threshold a
  person decides.
- New `app/llm/engine.py`: model loading with automatic device/precision, JSON extraction
  that tolerates fences and trailing commas and asks again when the answer is prose, and a
  record of every call (purpose, tokens, seconds, prompt hash).
- The audit log gained `model_calls` and the run's `model`, `model_seconds`, `model_calls`
  and `ai_summary` columns, so the AI's footprint is auditable.
- `python -m scripts.setup_model` is a required installation step: it downloads, verifies
  every shard named in the model index, resumes an interrupted download, and reports the
  device and precision the model will run on.
- Removed from the first prototype: the deterministic detectors (`patterns.py`,
  `names.py`, `detector.py`) and rule retrieval (`retrieval.py`). Keeping them would have
  meant keeping a non-AI path that could silently do the work.

**Kept, deliberately**

- The dashboard: same three pages (Upload, Review, Audit), same layout and colour language.
- `rules.json` as policy-as-data (48 rules, five receivers, one of them with no rules on
  purpose), the validated synthetic-value generator, the structure-preserving DOCX writer,
  the human-review queue, and the audit rule from the first build: masked value plus hash,
  never the original, never the released document.

**Verified**

- 22 tests pass without the model (the contract, the JSON handling, the release, the audit).
- `python -m scripts.smoke_test` runs the whole chain with the model installed: load,
  analyse, guard, write a release, record the audit, render all three dashboard pages.

## 0.3.0 — the adaptation path: what is trained, and how it is measured

The capstone brief requires prompt engineering **and** RAG always, and at least one of
fine-tuning or alignment, with before-and-after numbers. This release adds the missing two
halves — the training code and the measurement — and leaves the running application's
behaviour unchanged.

**Added**

- `evaluation/reference.py` — what the policy expects for an item and a receiver. **Not
  used by the application**; it exists so the training targets and the evaluation agree on
  one definition instead of two.
- `training/make_training_documents.py` — 24 of our own documents with labels, and a hard
  check that no labelled value appears in the evaluation set. The check prints its count
  every run; it is not a promise in a document.
- `training/prepare_data.py` — prompt/completion pairs in exactly the shape the app sends
  (same prompt builder, same policy block), plus two licensed public datasets on request.
- `training/make_preference_pairs.py` — chosen/rejected pairs in five auditable kinds, one
  per way this product could publish somebody's data: `confident-instead-of-asking`,
  `invented-rule`, `policy-overruled`, `leaked-by-omission`, `unsafe-substitute`.
- `training/train_lora.py`, `training/train_dpo.py`, `training/requirements-train.txt`,
  `training/COLAB.md` — QLoRA then DPO on a free T4, with the exact cells to run.
- `evaluation/metrics.py`, `evaluation/evaluate.py` — five metrics, `--live` / `--score` /
  `--compare`. `--score` runs on any machine, so a Colab run can be scored at home.
- `scripts/make_reference_runs.py` — three fixed answers (perfect, plausible-but-wrong,
  silent) used to show that the metrics discriminate, and to keep that under test.
- `docs/MODEL_ADAPTATION.md` — where the generative AI is, the adaptation path mapped to
  files, why both fine-tuning and alignment, the prompt template verbatim, and every
  dataset with its licence.

**Changed**

- `app/config.py` — `SAFESHARE2_ADAPTER`. The engine loads the fine-tuned adapter on top of
  frozen base weights (`app/llm/engine.py`), so the base/adapted comparison is a setting,
  not a code change, and the base weights are never modified.
- `.gitignore` — `predictions/`, `training/lora_adapter/`, `training/dpo_adapter/`: run
  artifacts, not source.

**Verified**

- 42 tests pass without the model, including four that pin the metrics down: a perfect
  answer scores 100%, silence scores a 100% leak rate, and an invented rule is never
  accepted — a plausible answer that redacts everything still leaks every value, because
  no valid rule justifies the removal.
- The harness was run on the fixed test set with three reference answers: perfect →
  structured 100% / citation 100% / agreement 100% / leak 0%; naive (invents a rule) →
  agreement 30% / leak 100%; silent → recall 0% / leak 100%.
- All seven entry points (`--help` on the five training/evaluation modules plus
  `make_reference_runs`) run in the project venv.

**Not yet done, and it is not a code problem**

- No training run has happened yet: the numbers above are from reference answers, not from
  a model. `training/COLAB.md` §4 produces the real before/after table on a free T4, and
  the adapters it saves are what the report should quote.

## 0.3.1 — the model's answer is read in both shapes it actually uses

**Fixed**

- `app/core.py` accepted the model's findings only as `{"findings": [...]}`. The model also
  answers with a **bare array** `[...]`, and that answer was silently discarded: on a
  single-part document it failed loudly, but on a multi-part document the discard was
  recorded as a note and the release went out **with those values still in it**. Nobody
  noticed until a Colab baseline run put a real model behind the evaluation harness, where
  `json.loads(answer)["findings"]` raised `TypeError: list indices must be integers`.
  `items_from()` and `answered()` now read either shape, for both the findings and the second
  look's decisions, while `{"findings": []}` remains a valid "nothing to redact" rather than
  a failure.
- `evaluation/metrics.py` and `evaluation/evaluate.py` had the same class of mistake in the
  other direction: `engine.generate_json()` returns a **parsed payload**, and the harness
  treated it as text, printing the key-count of a dictionary as if it were a character count
  ("1 chars back") and then crashing on `.strip()`. Answers are now stored as JSON text, the
  run reports findings counts and model time, and `parse_answer()` accepts a string, an
  object, or nothing.

**Verified**

- 46 tests pass, including: a bare array becomes a real decision end to end, an empty
  findings list is not treated as a failure, a parsed answer scores identically to its text,
  and no answer at all is scored as a leak rather than raising.
- Both fixes came out of the first real model run, which is the argument for running the
  baseline before trusting any of it.

## Rename to سِتر (Sitr), the validated GPU report, and the merged team state

**Changed**

- The product is presented as **سِتر · Sitr** (formerly SafeShare) in the UI brand, the
  page title, and exported filenames. The repository slug is unchanged so earlier links,
  scripts, and datasets keep working.
- Policy document Q&A (`app/policy/document_qa.py`): the Policy page answers Arabic and
  English questions from the retrieved rules and shows the rule each answer cites.
- `evaluation/full_report.py` runs the second set (20 documents / 87 labels) on the local
  Qwen model with a LoRA adapter on CUDA/BF16/4-bit, and writes a timestamped JSONL plus a
  five-sheet Excel workbook to `evaluation/results/`.
- `evaluation/evaluate_rag.py` adds the retrieval-only policy benchmark, and
  `tests/test_evaluation_report.py` covers the report builder.

**Verified**

- 66 tests pass (62 pipeline, policy and guards; 4 evaluation report).
- The committed second-set artifacts were re-scored from the saved predictions:
  Precision 93.02%, Recall 91.95%, F1 92.49%, action agreement 100% (80/80),
  leak rate 2.86% (1 of 35 protected values), deferred 0%, structured output 98.85%,
  citation validity 97.44% — Qwen2.5-3B-Instruct + LoRA, local, BF16 CUDA.

