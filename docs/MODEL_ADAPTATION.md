# The adaptation path — where the generative AI is, and what we train

This document answers the four questions the technical-design template asks, and it is
written so that you can defend every line of it in the assessment.

- **Slide 5 — "Where is the generative AI?"** → [§1](#1-where-the-generative-ai-is)
- **Slide 9 — "Adaptation path"** → [§2](#2-the-adaptation-path-in-this-repo)
- **Slide 10 — "Adaptation decision"** → [§3](#3-why-fine-tuning-and-alignment-both)
- **Slide 11 — "Prompt template (version 1)"** → [§4](#4-prompt-template-version-1)
- **Datasets and licences** → [§5](#5-datasets)

---

## 1. Where the generative AI is

The product has three parts. Only the middle one is the generative AI, and being clear
about which is which is the difference between "a GenAI application" and "a script that
calls a model".

| part | what it is | who decides |
|---|---|---|
| **The policy** | 12 written rules per receiver, in `app/rules/rules.json` | the company (that is, you). Data, not code. |
| **The model** | a pre-trained instruct LLM, prompted per document | **the model decides every per-item question**: is this item personal data, what type is it, may this receiver see it, what should it become, what is the reason, how sure am I |
| **The guards** | `app/guards.py`, plain code, no model | nobody: they only *check* the model's answer against the document and the policy |

So the generative work is real work, not decoration:

1. **Reading the document.** Nothing marks the personal data in advance. The model finds
   it, in Arabic, in English, in mixed documents, including Arabic-Indic digits.
2. **Classifying each item** into one of the twelve categories.
3. **Choosing the action** — keep, replace, remove, or hand it to a person — by reading
   the natural-language rule and applying it to this item in this document.
4. **Writing the substitute** — a fake value that keeps the format and the gender.
5. **Writing the reason** in the company's own words, with the rule it came from.
6. **Answering policy questions** ("may the vendor see a phone number?"), grounded in the
   retrieved rules, not from memory.

**What the model is not allowed to do**, and this is the design in one line: it may not
invent a rule, and it may not decide alone. Every answer is checked against the document
(the value must really be there) and against the policy (the rule must exist, for that
receiver, for that type). If it fails either check, the item goes to a human instead of
into the release.

**Why the policy file is not "hard-coded rules":** the file holds *policy*, in sentences,
and the model applies it. There is a test for exactly this distinction —
`test_the_policy_wins_over_the_model` and `test_an_invented_rule_is_never_applied`: when
the model's answer contradicts the policy, the policy wins and the model is not asked
again with a wink. Change a sentence in `rules.json` and the behaviour changes with no
retraining — which is what "the rules are data" buys you.

---

## 2. The adaptation path in this repo

The template's path, mapped to files you can open and run:

| step | where it lives | state |
|---|---|---|
| **Prompt engineering — always** | `app/llm/prompts.py` — the system contract, the answer schema, examples, the policy block | done, and shown in [§4](#4-prompt-template-version-1) |
| **RAG — your documents, vector DB** | `app/policy/policy_index.py` — every rule is embedded into a Chroma collection and retrieved by receiver and question; `app/policy/qa.py` answers policy questions from what was retrieved; `app/rules/store.py` holds the rules themselves | done |
| **Fine-tuning — LoRA / QLoRA** | `training/train_lora.py`, data from `training/prepare_data.py` | **code ready, run on a GPU** |
| **Alignment — DPO** | `training/make_preference_pairs.py` + `training/train_dpo.py` | **code ready, run on a GPU** |
| **Evaluation** | `evaluation/` — five metrics, before/after table | done; `--live` needs a model |

Data is built, never hand-written:

```
python -m training.make_training_documents      # our own documents + labels, 24 by default
python -m training.prepare_data --skip-public   # prompt/completion pairs (+ 2 public datasets if you omit this flag)
python -m training.make_preference_pairs        # chosen/rejected pairs for DPO
python -m training.train_lora                   # QLoRA, T4, saves training/lora_adapter
python -m training.train_dpo                    # alignment, saves training/dpo_adapter
python -m evaluation.evaluate --live --out predictions/base.jsonl
SAFESHARE2_ADAPTER=training/lora_adapter python -m evaluation.evaluate --live --out predictions/lora.jsonl
python -m evaluation.evaluate --compare predictions/base.jsonl predictions/lora.jsonl
```

The exact Colab steps are in [`training/COLAB.md`](../training/COLAB.md).

**The evaluation set is never trained on.** `data/test/` (12 documents, Arabic / English /
mixed) is read by `evaluation/` and by nothing in `training/`. `make_training_documents.py`
refuses to write a training document containing a value that appears in the test set, and
prints the check it ran:

```
overlap with the evaluation set: 0 values  (36 test values, 43 train values)
```

---

## 3. Why fine-tuning and alignment both

The template requires at least one. **If only one is run, it should be alignment** — the
failure this product must not have is a confident wrong decision, and that is a preference,
not a format. We do both, because they teach two different things, and this product needs
both of them:

- **Fine-tuning (LoRA/QLoRA)** teaches the **task and the format**: find the items, answer
  in this exact JSON, use these twelve category names, cite rules this way. That is a
  style the base model does not reliably hold over long Arabic documents, and it is
  cheap to teach.
- **Alignment (DPO)** teaches the **judgement**: when the policy is silent, ask a human
  instead of inventing an answer; admit the substitute you cannot justify; never keep a
  value the policy says must go. These are preferences, not formats — you cannot fix them
  with instructions, which is exactly why the template puts them in a separate step.

The DPO set is built around the four ways this product could publish somebody's data, so
one row of the report corresponds to one failure you can name:

| kind | the rejected answer | the lesson |
|---|---|---|
| `confident-instead-of-asking` | decides an item the policy does not cover, at 0.95 confidence | ask rather than guess |
| `invented-rule` | cites a rule that does not exist for this receiver | the policy is the only authority |
| `policy-overruled` | chooses an action the cited rule contradicts | the rule wins |
| `leaked-by-omission` | reports nothing, or keeps what must go | silence leaks data |
| `unsafe-substitute` | "replaces" a value with the value itself | a substitute must be fake |

### Measured baseline — the first real GPU run

Qwen2.5-3B-Instruct, untouched, with the prompt as it is, on the frozen 12-document test set
(70 labelled values). Measured 2026-10-10, and reproduced after two corrections to the
scoring itself (see `VERSION_LOG.md` 0.3.1 and the commits after it: an exact-string
comparison and then an exact-value comparison each reported a false 100% leak rate).

| metric | baseline |
|---|---|
| structured output | 88.3% |
| citation validity | 96.7% |
| action agreement | 92.0% |
| finding recall | 71.4% |
| **leak rate** | **34.7%** (17 of 49 values that had to go) |

Read it as: the model understands the policy and applies it correctly most of the time, and
still leaks one value in three, because it misses roughly a quarter of the values and fails
completely on two documents (one came back as the whole document with no type at all). For a
product whose purpose is not publishing personal data, 34.7% is disqualifying — that is the
gap the adaptation step exists to close, and the number the "after" columns must beat.

### Measured after fine-tuning (LoRA)

Same 12 documents, same frozen test set, same scoring. Qwen2.5-3B-Instruct plus a LoRA adapter
trained on 108 pairs (3 epochs, 15.6 minutes on a T4, 0.96% of the weights trainable). Training
loss 0.519 → 0.094.

| metric | base | after LoRA |
|---|---|---|
| structured output | 88.3% | **100.0%** |
| citation validity | 96.7% | **100.0%** |
| action agreement | 92.0% | **100.0%** |
| finding recall | 71.4% | **97.1%** |
| **leak rate** | **34.7%** | **2.0%** (1 of 49 values) |

The two documents that previously produced nothing usable now return findings, and the
format failures are gone entirely. The single remaining leak is one value in `kyc-form-ar`,
which the release still sends to a person — the safe default doing its job.

**The honest caveat, and it belongs in the report:** the training documents and the test
documents come from the same generator, so they share a writing style even though they share
no values and no documents (that is checked on every data build). The model learned this
template's task and format; part of the improvement may not carry over to documents from a
different source. Closing that gap is what the two licensed public datasets are for — they
are a different source, and the fine-tuned model has not been measured on them yet.

### The measured case, not the assumed one

Does the prompt alone produce good answers? Mostly yes — which is exactly why the honest
justification is not "quality" but **cost and privacy**, and why the claim is measured
rather than asserted. Measured on our own training set, one request is:

| part of the prompt | size | note |
|---|---|---|
| system instructions | 1,617 characters | identical on every call |
| the policy block for that receiver | 1,632 characters | identical for every document to that receiver |
| the document itself | 798 characters | **the only part that changes** |
| **total** | **~4,048 characters ≈ 1,156 tokens** | **80% of it is standing instructions** |

So every document costs four times its own length in instructions before the model reads a
single word of it. Fine-tuning earns its place by teaching the model the task and the answer
format, which is what allows that fixed part to shrink — and, more commercially, by making
the pipeline runnable on a company machine with no document leaving the building. For a
product whose whole purpose is handling other people's personal data, "it can run offline"
is the argument a buyer cares about.

**Why it has to be Qwen, specifically:** you can only fine-tune a model whose weights you
hold. `deepseek-chat` is a closed hosted model — there is no way to train it, and no adapter
can be attached to an API. Qwen2.5-3B-Instruct is open-weight, small enough to train at
4-bit on the free T4 the brief requires, and it is already the model the application uses
when it runs locally. So the adaptation step is Qwen by necessity, not by preference.

**If the baseline already scores well, that is a finding, not a failure** — and it is the
number the brief asks for either way: "the prompt alone reached X% structured output with a
0% leak rate; adaptation reached Y% and cut the fixed prompt from 3,249 to Z characters."
Report the difference; if there is none, report that honestly and let the cost argument
carry the decision.

**If the assessment asks "why not fine-tuning alone?"** — because fine-tuning on our own
targets would teach the model to answer confidently in *every* case, including the ones
where the right answer is "I do not know". The preference pairs are where the model
learns that `ASK_HUMAN` is a good answer, not a failure.

**If it asks "is fine-tuning even needed here?"** — the honest answer: the product works
without it, because prompt engineering and RAG carry the reasoning and the guards are
deterministic. Fine-tuning earns its place by removing the *format* failures (invalid
JSON, wrong category names, missing keys) and by shortening the prompt, not by making the
safety decisions — and the evaluation measures exactly that difference, per metric, on a
fixed set. The brief requires the step; this is what it buys.

---

## 4. Prompt template (version 1)

Two messages, exactly as they are sent. From `app/llm/prompts.py`:

**System** (exactly as sent, from `app/llm/prompts.py`):

```
You are the release decision engine for a company document-sharing system.
One document is being prepared for one receiver. Your job is to decide, for every piece
of personal or sensitive data in the document, whether this receiver may see it.

The policy below is the ONLY authority. Rules:
1. Never invent a rule. Cite a `rule_id` that appears in the policy, or set it to null.
2. If no rule in the policy covers an item, use action ASK_HUMAN with rule_id null.
3. `original_text` must be copied character for character from the document, including
   Arabic and Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩). Do not translate, reformat or shorten it.
4. Confidence is your honest probability that this is that type of data AND that the
   cited rule applies. Below 0.8 the item goes to a human. Low confidence is a correct,
   useful answer - a wrong confident answer publishes somebody's data, a question does not.
5. Report only real items. Do not list headings, product names, amounts, dates of a
   contract or business figures unless the policy names them as personal data.
6. Allowed types: PERSON_NAME, NATIONAL_ID, PHONE, EMAIL, ADDRESS, BANK_ACCOUNT, IBAN, DATE_OF_BIRTH, PASSPORT_NUMBER, CUSTOMER_ID, EMPLOYEE_ID, COMPANY_IDENTIFIER.
7. Allowed actions: KEEP, REPLACE, REMOVE, ASK_HUMAN (KEEP = this receiver may see it,
   REPLACE = show a fake substitute, REMOVE = take it out, ASK_HUMAN = a person decides).
8. If action is REPLACE you must also give `replacement`, and it must keep the original
   shape while being obviously fake.
9. Answer with JSON only. No markdown fence, no comment, no text before or after.
```

**User** (the placeholders are filled per request):

```
RECEIVER: {receiver_label}
PURPOSE OF SHARING: {purpose}
POLICY RULES (id, type, action, reason):
{rules_block}
If the policy has no rule for a type, that is a missing rule: ASK_HUMAN.

DOCUMENT:
"""
{document_text}
"""

List every piece of data the policy applies to in this document. Return exactly this JSON shape:
{"findings": [{"original_text": "<copied exactly from the document>", "type": "<TYPE>", "action": "<ACTION>", "replacement": <string or null>, "confidence": <0.0-1.0>, "rule_id": "<id from the policy, or null>", "reason": "<one sentence, max 20 words>"}]}
If the document has no such data, return {"findings": []}.
```

`{rules_block}` is a JSON list of the rules for **that receiver only** — for example
`{"rule_id": "VENDOR-PII-001", "type": "PERSON_NAME", "action": "REPLACE", "reason":
"Customer names must not be shared with external vendors."}` — retrieved from the Chroma
index at request time. A rule edited in the Policy page is in the next prompt, with no
restart and no retraining. The four receivers that have a policy have 12 rules each; the
fifth (`CUSTOM`) deliberately has none, so every item is answered `ASK_HUMAN` — a receiver
with no written policy must not receive a guessed decision.

---

## 5. Datasets

| source | what it is | licence | how we use it |
|---|---|---|---|
| our own synthetic documents | 24 training + 12 test documents, Arabic/English/mixed, generated with labelled values | ours | training targets, and the fixed evaluation set |
| `gretelai/gretel-pii-masking-en-v1` | 50,000 English documents with labelled entities | **Apache-2.0** | volume for fine-tuning (optional flag) |
| `mabahboh/sitr-arabic-pii` | 14,288 Arabic documents with labelled entities | **Apache-2.0** | **sampled, 2,500 maximum**, for Arabic coverage |

Checked and rejected: `aniket-curlscape/pii-masking-english-1k` — its licence is declared
only as *"other"*, which is not a licence you can put in a report. `ai4privacy/pii-masking-65k`
— no licence declared, no Arabic; we copied its *taxonomy* of sensitive types, not its
content. Any CC-BY-**NC** dataset is out because the goal is to sell this.

Their label vocabularies are not ours, so labels are mapped onto our twelve categories and
anything unmapped (`vehicle_plate`, `medicine_name`, `ipv4`) is dropped rather than forced
into a category. Ten of the twelve categories are covered by the two sets
(`PERSON_NAME`, `NATIONAL_ID`, `DATE_OF_BIRTH`, `ADDRESS`, `PHONE`, `CUSTOMER_ID`,
`BANK_ACCOUNT`, `EMAIL`, `EMPLOYEE_ID`, `PASSPORT_NUMBER`); `IBAN` and `COMPANY_IDENTIFIER`
come from our own documents.

Every source is named here and in your report; `training/prepare_data.py` prints the list
of what it actually touched, with the licence, every time it runs.
