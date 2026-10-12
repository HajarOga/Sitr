# SafeShare — Independent Human Judge Test Set

## Purpose
This is a newly authored synthetic evaluation set intended as an independent alternative to the uploaded `test_second` files. It contains 20 prose-style documents and paired `.gold.json` reference files.

## Split
- Easy: 6 documents
- Medium: 7 documents
- Hard: 7 documents
- Arabic: 10 documents
- English: 10 documents
- Receivers: ANALYST, CUSTOMER_SUPPORT, EXTERNAL_VENDOR, INTERNAL_HR
- Labelled spans: 87

## How difficulty was assigned
Difficulty is a provisional, rule-based design label, not a model-measured score:
- Easy (6): 2–4 labelled spans; short, direct context; familiar types such as PERSON_NAME, CUSTOMER_ID, PHONE, EMAIL; relatively little ambiguity.
- Medium (7): usually 4–5 labelled spans; values embedded in natural prose; mixed formats (e.g. Arabic-Indic phone digits, addresses, company identifiers); moderate context variation.
- Hard (7): 5–8 labelled spans and/or multiple higher-risk types together (e.g. NATIONAL_ID, DATE_OF_BIRTH, PASSPORT_NUMBER, BANK_ACCOUNT, IBAN, ADDRESS); denser documents and more chances to miss a span or misclassify it.
- Language and receiver are balanced across the set overall. Difficulty is not assigned based on whether a model succeeds.

## Files and labels
Each document has a `.txt` file and a matching `.gold.json`. `start` and `end` are zero-based Python string offsets, with `end` exclusive. The gold spans identify candidate sensitive values. Expected actions are not hard-coded; they should be derived from the current project policy (`app/rules/rules.json`) during scoring.

## Validation performed
- All 20 documents have a matching gold file.
- All 87 span strings were checked against their exact text offsets.
- The set uses synthetic, fabricated content. Do not replace with real personal data.
- This validation confirms file pairing and span alignment only; it does not by itself prove every label is policy-correct or that model performance is good.

## Suggested use
Copy this directory to `data/test_human_judge` in the project (keep the original `data/test_second` unchanged). Run the project's evaluator against the new directory using the repo's documented command, after confirming the policy rules and provider configuration. Running `--live` may incur API costs.
