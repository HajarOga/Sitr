"""Policy tooling: the vector index over the rules, and the safe editing path.

Nothing in this package decides anything about a document. `policy_index` makes the
rules *findable* by meaning (so a person can ask a question in Arabic or English) and
`policy_editor` is the only code allowed to write `rules.json`, with validation and an
audit trail. The release pipeline keeps reading the same library it always did.
"""
from .policy_editor import PolicyEdit, PolicyEditError, PolicyEditor  # noqa: F401
from .policy_index import (HashingEmbedder, PolicyIndex, PolicyIndexError,  # noqa: F401
                           SentenceTransformerEmbedder, get_index, reset_cache)

__all__ = ["PolicyEdit", "PolicyEditError", "PolicyEditor", "HashingEmbedder",
           "PolicyIndex", "PolicyIndexError", "SentenceTransformerEmbedder",
           "get_index", "reset_cache"]
