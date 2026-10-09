# The pipeline, and where the model sits in it

```
Document
   │
   ├─ read            PDF / DOCX / TXT → blocks with exact character offsets
   │                  (upload deleted immediately after reading)
   ▼
policy ────────────► the receiver's rules, as JSON, straight into the prompt
   │
   ▼
MODEL  analyze       reads the document; for every item returns
   │                 value, type, action, cited rule, substitute, confidence, reason
   ▼
guards               valid type? valid action? rule exists, for this receiver, for this
   │                 type? value actually in the document? substitute safe? 
   │                 → otherwise: downgrade to ASK_HUMAN, or drop, and say why
   ▼
MODEL  adjudicate    second look at everything it was unsure about, with surrounding text
   │                 (may only point at a valid rule; confidence capped at 0.90)
   ▼
MODEL  replacement   a substitute for each REPLACE — validated, repaired if unsafe
   │
   ▼
human review         below 0.80 confidence, or no valid rule: a person decides.
   │                 the reviewer's answer is final
   ▼
document             the change is written back with python-docx, structure preserved
   │                 an item still unanswered is LEFT OUT — never left in
   ▼
audit                one row per decision: masked value, hash, rule, reason, who decided
   │                 one row per model call: purpose, tokens, seconds, prompt hash
   ▼
MODEL  summary       three sentences for the report
```

## Who does what, and why

| stage | who | why |
|---|---|---|
| Finding personal data in free text | **the model** | names and context cannot be written as patterns; this is the judgement the project is about |
| Finding exact values (IDs, IBANs, phones) | **the model** | it reports them as items like any other; the *validation* of a substitute is still arithmetic |
| Which rules exist, and for whom | **code + `rules.json`** | policy is data; the model is given it and may not invent it |
| Whether a cited rule is real and applies | **code** | an invented rule id must never reach a document |
| The action when the model and the rule disagree | **the rule** | the model's job is to find the rule, not to overrule it |
| Generating the substitute | **the model**, validated by **code** | format-preserving fakes are a generation task; "is this safe to publish" is not |
| Locating the value in the document | **code** | offsets must be exact, and Arabic-Indic digits need folding |
| Deciding to ask a person | **code** | the threshold is a policy decision, not a model opinion |
| The release document and the audit log | **code** | deterministic, reproducible, append-only |

## The order that matters

Changes are applied **right to left inside each block**, so an earlier change cannot shift
the offsets of a later one. Items are applied per block, which is why a table cell and a
paragraph can each hold their own version of the same value.

## The two hard defaults

1. **Below 0.80 confidence → a person.** `SAFESHARE2_ASK_THRESHOLD` sets it. The item is
   shown with the model's proposal, its rule, its confidence and a plain reason for why it
   was questioned.
2. **An unanswered item is left out of the release.** The reviewer can keep, replace,
   remove, or reject the proposal (which leaves it open). Keeping the original text is
   always an explicit human choice — it is never what happens by default, and never what
   happens because of a timeout.

## Failure behaviour

| what happens | what the system does |
|---|---|
| the model is not installed | stops. `ModelNotInstalled`; the API answers 503; the dashboard shows the install command |
| the model returns prose, not JSON | asks again twice, quoting the bad answer back |
| nothing parsed in any chunk | raises `ModelAnswerError` — an empty result is never shown as "nothing found" |
| the model cites a rule that does not exist | the item goes to a person, and the guard note says exactly that |
| the model proposes KEEP with no rule | goes to a person: keeping is what leaks, so it is never taken on trust |
| the model's substitute fails validation | the deterministic generator supplies one; the repair is recorded on the item and in the notes |
| the document is longer than the chunk size | split on line boundaries, each part analysed, all parts reported |

Everything in the left column above is covered by a test in `tests/test_safeshare2.py`.
