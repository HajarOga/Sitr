# The AI in SafeShare

Everything in this file answers one question: **what is the model actually doing here?**

## Which model, and why this one

Two providers, one pipeline — `SAFESHARE2_PROVIDER` picks (see `docs/SETUP.md` §1a):

| provider | model | where it runs | the document |
|---|---|---|---|
| `api` (what this build ships) | `deepseek-chat` at `https://api.deepseek.com` | on DeepSeek's servers | sent in the prompt |
| `local` | `Qwen/Qwen2.5-3B-Instruct` (Apache-2.0) | in this process | never leaves the machine |

Both are asked exactly the same way — the four prompts in `app/llm/prompts.py` do not know
which one answers — so a claim about the contract (must cite a real rule, must return the
schema, low confidence goes to a person) is a claim about both.

### The local model

`Qwen/Qwen2.5-3B-Instruct` (Apache-2.0), and the reason it is the local default:

- **Instruction-tuned, so it follows a schema.** The pipeline needs strict JSON, not
  prose, and needs a model that will say "I am not sure" instead of inventing.
- **3B parameters is the size that runs on a laptop.** On Apple Silicon it loads in fp16
  (~6 GB) and answers in seconds; on a CUDA GPU it is several times faster; on CPU it
  works, slowly. A larger model is a setting, not a rewrite.
- **Strong Arabic.** Half of the sample document is Arabic, including Arabic-Indic digits,
  and this family handles both scripts well.
- **It is chat-template based**, which is what `app/llm/engine.py` uses
  (`tokenizer.apply_chat_template`), so swapping the model does not change any code.

Swapping it:

```bash
SAFESHARE2_MODEL=Qwen/Qwen2.5-7B-Instruct streamlit run app/ui_streamlit.py
SAFESHARE2_DEVICE=cuda SAFESHARE2_DTYPE=bfloat16 ...     # force device and precision
```

Precision is chosen automatically: `bfloat16`/`float16` on CUDA, `float16` on Apple
Silicon, `float32` on CPU. 4-bit loading needs `bitsandbytes` and a CUDA GPU.

### The API provider

`deepseek-chat` over the OpenAI-compatible chat-completions route (`httpx`, no SDK), with
`response_format={"type": "json_object"}` on every structured call. It answers a page in
about two seconds where the 3B model takes tens of seconds on a laptop — the reason this
build points at it. The cost is that the document travels in the prompt; the dashboard
says so, and the audit log still stores only a prompt **hash**, the token counts and the
seconds, never the prompt itself. Any OpenAI-compatible endpoint works:
`SAFESHARE2_API_BASE_URL` + `SAFESHARE2_API_MODEL`.

## The four jobs

The model does not do everything; it does four things, and the fourth only at the end.

### 1. `analyze` — read the document, decide every item

Input: the receiver, the purpose, the receiver's rules as JSON, and the document text.
Output: `{"findings": [{original_text, type, action, replacement, confidence, rule_id, reason}]}`.

The prompt states, and the guards then enforce:

- the policy is the only authority, and a rule id must be copied from it;
- if no rule covers an item, the answer is `ASK_HUMAN` with `rule_id: null`;
- `original_text` must be copied character for character, including Arabic and
  Arabic-Indic digits;
- confidence means "this is that type of data **and** that rule applies";
- a low confidence is a *correct* answer — a wrong confident answer publishes somebody's
  data, a question does not.

Long documents are chunked on line boundaries (never mid-value) and each part is analysed,
with the parts listed in the audit as separate model calls.

### 2. `adjudicate` — a second look at what it was unsure about

Items the model left at `ASK_HUMAN` but with confidence above a floor (0.35) are sent back
in batches of eight **with the text around them** — often the first pass saw a name without
the word that says whose name it is.

This pass can only do one thing: point at a **valid rule for that type and receiver**.
If it does, the policy's action is applied and confidence is raised to at most **0.90**.
If it cites something invalid, or is still unsure, the item stays with a person. The model
cannot promote itself to certain, and it cannot overrule the policy.

### 3. `replacement` — invent a substitute that is obviously fake

Asked for a value with the same shape (digit count, separators, language, name gender) that
cannot be mistaken for real data: reserved-looking test numbers, `example.test` addresses,
invented names. Every substitute then goes through `app/replacement/validators.py`
(shape, checksum, deliberately-not-valid IBAN). A substitute that fails is discarded, the
deterministic generator supplies one, and the dashboard and audit say so.

### 4. `summary` — the audit note

Three sentences at most, numbers only, no names and no values. It is written once per
release and stored with the run, so the audit page shows prose a person can read next to
the rows.

## What the code does around it

| concern | who owns it |
|---|---|
| which rules exist, and for which receiver | `app/rules/rules.json` — never the model |
| the document offsets of a reported value | code (exact and normalised match, digits folded) |
| whether a cited rule is valid, and for the right type | code |
| what happens when the model and the rule disagree | the **rule** wins; the disagreement is recorded |
| whether a substitute is safe to publish | code (validators) |
| below 0.80 confidence | a person decides — always |
| what happens to an unanswered item | it leaves the release |
| what is written into the audit log | masked value + hash; model calls as prompt hash + sizes |

## When it goes wrong

Three failure modes are handled explicitly, because each one is silent otherwise:

1. **The model is not installed.** `engine.ensure_installed()` raises `ModelNotInstalled`,
   the API answers 503, and the dashboard replaces the Analyze button with the exact
   install command. There is no fallback path to hide behind.
2. **The model answers with prose instead of JSON.** `generate_json()` asks again, twice,
   with the bad answer quoted back; if it still fails, the part is reported as unanalysed.
   If *no* part parsed, the call raises `ModelAnswerError` — an empty result is never
   presented as "nothing sensitive found".
3. **The model answers with something the policy does not support.** Refused, in the
   guards, with the reason written down: an invented rule id, a rule for the wrong type, a
   `REPLACE`/`REMOVE` with no rule at all, a `KEEP` with no rule cited, a value that is not
   in the document, an overlapping duplicate.

## Cost, timing and proof

Every call is recorded with its purpose, tokens out, seconds, device and a prompt hash
(`app/audit/schema.sql`, table `model_calls`). The dashboard shows this on the Audit page,
the API exposes it at `/api/audit/model-calls/{run_id}`, and the run row carries the totals.
That is what makes "the AI is in the loop" a checkable statement rather than a claim: you
can see which jobs ran, how long they took and how much they produced.

## What the dashboard no longer shows (and why it is still true)

The upload page used to carry two panels — *The AI* and *Privacy* — that stated, on screen,
which model answers and what happens to a document. They were removed from the interface
on purpose: the product should be read as a product, not as a description of its own
machinery. The facts did not change, so they are recorded here instead.

**Which model answers.** `deepseek-chat` over `https://api.deepseek.com` when a key is
configured; the local `Qwen2.5-3B-Instruct` otherwise. `SAFESHARE2_PROVIDER` decides, and
`auto` follows whether a key is present.

**What the model is given.** Detection, the action for the chosen receiver, the rule
citation, the substitute, the reasons, and the audit summary.

**What leaves the machine.** With the API provider, the document text is part of the
prompt sent to that endpoint. With the local provider, nothing leaves this machine. The
policy search embeds the *policy*, never a document.

**Where uploaded files go.** To `.tmp/` under the project folder, deleted as soon as the
text and structure have been read, unless `SAFESHARE_KEEP_UPLOADS=1` is set.

**What the audit log stores.** Masked values (``1********9``), a hash of each value, the
action, the rule id, the reason, and one row per model call with its purpose, token count
and seconds. It never stores the original value, and never the released document.

**Where to point someone who asks.** This file, plus `docs/SETUP.md` for the
configuration and `README.md` for the licence table and the honest limitations.
