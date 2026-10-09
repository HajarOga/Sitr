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

## 0.1.0 — SafeShare2, the AI-core rebuild

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
