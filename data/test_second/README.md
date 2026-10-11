# The second test set — hand-written, prose style

**What it is.** Twenty documents written by hand for `scripts/make_second_test_set.py`: **five
per receiver**, ten Arabic and ten English, **87 labelled values** in total. They are ordinary
business paperwork — complaint emails, service-visit threads, HR memos and onboarding notes,
analyst summaries, support call logs and chat transcripts.

**Why it exists.** The frozen set in `data/test/` and the training documents were written by the
same generator, so they share a house style: labelled lines, one value per line. That measures
whether the model learned the task, but not whether it survives a document that does not look
like its training data. These eight documents put the personal data **inside sentences** instead.

**How it differs from `data/test/` and `data/test_no_rules/`:**

| | frozen set | no-rule set | this set |
|---|---|---|---|
| written by | generator | same documents, receiver `CUSTOM` | hand, in prose |
| values overlap training | no | no | no |

**How to run it**

```
python -m evaluation.evaluate --live --test-dir data/test_second --out predictions/second.jsonl
python -m evaluation.evaluate --score predictions/second.jsonl
```

**What it can and cannot prove.** It can show whether the model's behaviour holds up when the
document style changes — which the frozen set cannot, since it shares the generator's style with
the training data. It cannot prove robustness against documents from a genuinely different
source: these were still written by the same team, in the same language pair, about the same
kinds of business paperwork.

**The frozen set is untouched.** This set is separate on purpose, so the 12-document numbers stay
comparable. Every value here was checked against `data/train/`, `data/test/` and
`data/test_no_rules/` — the build script refuses to write anything if a value collides, and it
prints that check on every run.

**Expected actions are not hand-typed.** They are read from `app/rules/rules.json` at scoring
time, so the labels cannot drift away from the policy.
