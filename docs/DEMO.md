# The five-minute demo

Assumes the model is installed (`python -m scripts.setup_model --check` says `installed:
yes`). Run one throwaway analysis before the audience arrives — that is the slow one,
because the model has to load into memory.

## Before you start

```bash
python -m scripts.make_demo_document      # the sample Arabic + English document
streamlit run app/ui_streamlit.py         # open http://localhost:8501
python3 tests/test_safeshare2.py          # 22 tests, in case someone asks
```

Have `docs/AI.md` open in a second tab in case of questions about the prompts.

## The script

**1. The problem (30s).** One company document, several receivers: a vendor, HR, an
analyst. Someone does this by hand today — slow, inconsistent, and afterwards nobody can
say why each piece was removed. Show the sample document on the Upload page.

**2. The AI panel (30s).** Point at the sidebar panel: the model name and where it runs.
Say the sentence that matches the configured provider:

- **`SAFESHARE2_PROVIDER=local`** (the offline mode): *the model runs on this machine — the
  device, the precision and the 6 GB on disk are all visible here — and no document text
  leaves it.*
- **`SAFESHARE2_PROVIDER=api`** (the mode this build ships in, for speed): *the model is
  `deepseek-chat`, called over an API; the document text is sent to that endpoint in the
  prompt, which is the trade we made for speed, and the upload page states it per run.*

Then the sentence that holds either way: *there is no rules-only mode — if the model is
unreachable, the app refuses to analyse a document. A fallback would let us claim "AI"
while a regex did the work.*

**3. Analyze (60–90s).** Choose **External Vendor** — there is no purpose box any more;
the reason for sharing comes from the receiver itself ("Repair or service work that
requires the complaint and invoice details"), and it goes into the prompt and the audit
record the same way. Press **Analyze with the model**, and while it
runs, say what is happening: the receiver's 48 rules go into the prompt as data, the model
reads the document, and for every item it returns a value, a type, an action, the rule it
is citing, a substitute, and its own confidence. Then note the number that appears under
the title — how many model calls and how many seconds — because that number is recorded.

**4. Review (90s).** The side-by-side page. Two things to point out:

- The cards that say **Model was unsure**. Open one: the value, what the model proposed,
  the rule it cited, and the sentence explaining *why you are being asked*. This is the
  honest part of the system — Arabic free-text names and values without a clear rule
  arrive as questions, not decisions.
- The expander **What the guards did to the model's answer**. It lists everything the code
  refused: an invented rule id, a rule that was about a different type, a substitute that
  failed validation, a value that was not actually in the document. Say the sentence:
  *the model decides, the policy is the authority, and the code is what stops a
  confident-sounding mistake from reaching a document.*

**5. The safe version (60s).** Answer the questions (Keep / Replace / Remove — or use the
toggle for speed, and note that even then each answer is recorded as a human decision).
Press **Generate release**. Show: the download button, the count of applied changes, and
the model-written audit note. Point out that anything unanswered is *left out* of the
release, never left in.

**6. Audit (30s).** The third page. Rows by receiver, then the model-call table: purpose,
tokens, seconds, prompt hash. *This is how the audit proves the AI was in the loop, without
storing a copy of the document.*

## Questions you will be asked

**"Is it really the AI, or is it rules?"** Open `app/llm/prompts.py` on screen: four
prompts, and the model is the one producing every decision. Then open `app/guards.py`: the
code that decides what is allowed to stick. The split is the design.

**"What if the model gets it wrong?"** Three answers, in order: the confidence threshold
(0.80) sends uncertain items to a person; the guards refuse anything that contradicts the
policy; and an unanswered item leaves the release. A wrong guess is not the failure mode —
a silent wrong guess is, and that is what those three mechanisms prevent.

**"Why a 3B model?"** Because it runs on the laptop in front of you, in fp16, with no
network. A larger model is one setting (`SAFESHARE2_MODEL=`) and more memory.

**"What is in the prompt?"** The receiver, the purpose, the receiver's rules as JSON, the
document text, and the output schema. Then say the honest part: the prompt *is* the
quality of this system, which is why it lives in one reviewable file.

**"What are the limits?"** Say them before you are asked: a 3B model misses things a larger
one would catch, especially in dense Arabic free text; PDF layout is not preserved on the
way out; everything here is synthetic and has never been run against a real company
document.
