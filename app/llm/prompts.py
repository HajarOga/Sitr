"""The prompts - in this project they are the product.

The model is the decision engine here, so these strings are not decoration around a
rule engine; they are what decides what a vendor may see and what an HR team may see.
Four jobs, four prompts:

1. `analyze_messages()`      - read the document, find the personal data, propose an
                               action for each item, cite the rule.
2. `adjudicate_messages()`   - a second, focused look at the items the model itself was
                               unsure about, with more of the surrounding text.
3. `replacement_messages()`  - invent a format-preserving, obviously fake substitute.
4. `summary_messages()`      - describe the run for the audit report.

Two rules run through all of them, and both are then enforced in code (app/guards.py):

* the policy is the only authority - never invent a rule, cite one or say none covers it;
* the text must be copied exactly from the document, or the finding cannot be applied
  to anything and is thrown away.

Keeping the prompts here means the wording can be reviewed, versioned and diffed like
any other part of the system - which is what the build brief asks for when it says the
model must not be one giant prompt hidden inside the code.
"""
from __future__ import annotations

import json

from ..pii_detection.schema import Action, PiiType

TYPES = [t.value for t in PiiType]
ACTIONS = [a.value for a in Action]

FINDINGS_SCHEMA = (
    '{"findings": [{"original_text": "<copied exactly from the document>", '
    '"type": "<TYPE>", "action": "<ACTION>", "replacement": <string or null>, '
    '"confidence": <0.0-1.0>, "rule_id": "<id from the policy, or null>", '
    '"reason": "<one sentence, max 20 words>"}]}'
)

DECISIONS_SCHEMA = (
    '{"decisions": [{"id": "<the id you were given>", "type": "<TYPE>", '
    '"action": "<ACTION>", "replacement": <string or null>, "confidence": <0.0-1.0>, '
    '"rule_id": "<id from the policy, or null>", "reason": "<one sentence>"}]}'
)

BASE_RULES = f"""You are the release decision engine for a company document-sharing system.
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
6. Allowed types: {", ".join(TYPES)}.
7. Allowed actions: {", ".join(ACTIONS)} (KEEP = this receiver may see it,
   REPLACE = show a fake substitute, REMOVE = take it out, ASK_HUMAN = a person decides).
8. If action is REPLACE you must also give `replacement`, and it must keep the original
   shape while being obviously fake.
9. Answer with JSON only. No markdown fence, no comment, no text before or after."""


def policy_block(receiver_label: str, purpose: str, rules: list[dict]) -> str:
    return (f"RECEIVER: {receiver_label}\n"
            f"PURPOSE OF SHARING: {purpose or 'not stated'}\n"
            f"POLICY RULES (id, type, action, reason):\n"
            f"{json.dumps(rules, ensure_ascii=False, indent=1)}\n"
            f"If the policy has no rule for a type, that is a missing rule: ASK_HUMAN.")


def analyze_messages(document_text: str, receiver_label: str, purpose: str,
                     rules: list[dict]) -> list[dict]:
    """Job 1: the whole document, one pass."""
    return [
        {"role": "system", "content": BASE_RULES},
        {"role": "user", "content":
            f"{policy_block(receiver_label, purpose, rules)}\n\n"
            f"DOCUMENT:\n\"\"\"\n{document_text}\n\"\"\"\n\n"
            f"List every piece of data the policy applies to in this document. "
            f"Return exactly this JSON shape:\n{FINDINGS_SCHEMA}\n"
            f"If the document has no such data, return {{\"findings\": []}}."},
    ]


def adjudicate_messages(items: list[dict], receiver_label: str,
                        rules: list[dict]) -> list[dict]:
    """Job 2: a second, focused look at the items the model was unsure about.

    Each item carries the text around it, because the first pass may have seen a name
    without the word that tells us whose name it is.
    """
    listing = json.dumps(items, ensure_ascii=False, indent=1)
    return [
        {"role": "system", "content": BASE_RULES},
        {"role": "user", "content":
            f"{policy_block(receiver_label, 'second look at uncertain items', rules)}\n\n"
            f"These items were found in the document but were not decided with "
            f"confidence, so a person is about to be asked. Look at the surrounding "
            f"text and decide each one properly. Lower your confidence rather than "
            f"guessing: an item you cannot place stays ASK_HUMAN.\n\n"
            f"ITEMS (id, value, type, proposed action, confidence, why it was questioned):\n"
            f"{listing}\n\n"
            f"Return exactly this JSON shape:\n{DECISIONS_SCHEMA}"},
    ]


def replacement_messages(original_text: str, pii_type: str, language: str,
                         examples: list[str]) -> list[dict]:
    """Job 3: invent a substitute that is the right shape and obviously not real."""
    return [
        {"role": "system", "content":
            "You invent stand-in values for a document-release tool. The substitute must "
            "keep the shape of the original (same number of digits, same separators, same "
            "language, same name gender and structure) and must be obviously fake: use "
            "reserved-looking test values, example.test addresses or clearly invented "
            "names. It must never be a real person's data and never pass a real checksum "
            "(for example an ID whose check digit is wrong, or an IBAN ending in "
            "0000 0000). Answer with JSON only: {\"replacement\": \"<value>\"}."},
        {"role": "user", "content":
            f"Type: {pii_type}\nLanguage: {language}\nOriginal: {original_text!r}\n"
            f"Examples of the house style: {', '.join(examples) if examples else 'none'}"},
    ]


def summary_messages(receiver_label: str, purpose: str, counts: dict, total: int,
                     pending: int, changed: list[dict]) -> list[dict]:
    """Job 4: the paragraph that goes into the audit report."""
    highlights = json.dumps(
        [{"type": c.get("type"), "action": c.get("final_action"),
          "rule": c.get("rule_id")} for c in changed[:12]], ensure_ascii=False)
    return [
        {"role": "system", "content":
            "You write a short audit note about a document release. At most three "
            "sentences, plain language, no bullet points, no advice. State only the "
            "numbers and items you are given and do not name any person or value. "
            "Answer with JSON only: {\"summary\": \"...\"}."},
        {"role": "user", "content":
            f"Receiver: {receiver_label}\nPurpose: {purpose or 'not stated'}\n"
            f"Items found: {total}\nReplaced: {counts.get('REPLACE', 0)}\n"
            f"Removed: {counts.get('REMOVE', 0)}\nKept: {counts.get('KEEP', 0)}\n"
            f"Waiting for a person: {pending}\n"
            f"Items: {highlights}"},
    ]


# --------------------------------------------------------------- job 5: policy Q&A

POLICY_QA_SYSTEM = """You answer questions about a company's data-sharing policy for a
colleague who is preparing a document for a receiver.

You are given a short list of POLICY RULES retrieved from the policy library. They are
the only policy you know. The rules that follow are:

1. Answer only from the retrieved rules. If they do not settle the question, say so
   plainly and say what is missing. Never invent a rule, a rule id, a clause number or a
   practice that is not in the retrieved text.
2. Copy the rule id exactly as it appears in the retrieved rules and put it in the
   sentence it supports, for example "a phone number must be removed (VENDOR-PII-003)".
   Do not retype it from memory and do not change its spelling. A sentence that no rule
   supports does not belong in the answer.
3. If two rules disagree, say so instead of choosing. A person resolves that.
4. If the question is about a receiver with no retrieved rules, say that the policy is
   silent for that receiver and that such an item reaches a person as ASK_HUMAN.
5. Answer in the language of the question - Arabic question, Arabic answer. Be brief:
   three sentences at most, or a short list when the question asks for several rules.
6. This is guidance for a person, not a decision. Do not tell the reader that an item is
   safe to release; say what the policy says and let them decide.
"""


def policy_qa_messages(question: str, hits: list[dict]) -> list[dict]:
    """Job 5: answer a policy question from retrieved rules, and cite them.

    This prompt sits outside the release pipeline on purpose. It reads the policy; it
    never sees a document, and nothing it produces is applied to one.
    """
    grounded = [{"rule_id": hit.get("rule_id"), "receiver": hit.get("receiver"),
                 "type": hit.get("pii_type"), "action": hit.get("action"),
                 "reason": hit.get("reason"), "citation": hit.get("citation", "")}
                for hit in hits]
    return [
        {"role": "system", "content": POLICY_QA_SYSTEM},
        {"role": "user", "content":
            f"QUESTION:\n{question.strip()}\n\n"
            f"RETRIEVED POLICY RULES (this is all the policy you have):\n"
            f"{json.dumps(grounded, ensure_ascii=False, indent=1)}\n\n"
            f"Return exactly this JSON shape:\n"
            f'{{"answer": "<the answer, in the language of the question>", '
            f'"citations": ["<rule ids you actually used>"]}}\n'
            f"If the retrieved rules do not answer the question, say so in `answer` and "
            f"leave `citations` empty."},
    ]
