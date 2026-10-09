"""The policy index: Chroma over the rules, so the policy can be asked a question.

What this is for
----------------
The decision path does **not** use this. `app/core.py` pastes the receiver's whole rule
set into the prompt (12 short rules, ~1.8k characters) and the guards enforce the
contract. That stays exactly as it is.

This module exists for two other jobs:

1. **Questions.** A reviewer asks "may a vendor see a phone number?" - in Arabic or
   English - and gets back the rules that actually answer it, with their ids and
   citations. Retrieval is advisory: a person reads the rule text and decides.
2. **Coverage.** When the policy grows past the point where it fits in a prompt, the same
   index is what makes rule retrieval possible. Two guardrails are mandatory there and
   are stated here so a future change cannot miss them: retrieval may only ever *add*
   candidate rules to a prompt (never remove an item or satisfy a guard), and a retrieval
   miss must mean ASK_HUMAN, never KEEP.

How the index stays honest
--------------------------
Chroma collections are keyed by name, and this module puts the **embedder identity and a
fingerprint of the policy** into that name (see `collection_name()`). Two consequences,
both deliberate:

* a different embedding model or an edited policy can never be mixed into the same
  collection - the query simply moves to a new collection, so there is no stale-vector
  failure mode to reason about;
* vectors are always comparable, because everything in one collection came from one
  embedder and one policy.

The cost is an orphaned collection per policy edit. `cleanup()` removes them, the setup
script calls it, and the whole folder is disposable: `data/index/` is derived data and is
in `.gitignore`. The authority is always `app/rules/rules.json`.

Why an injected embedder
------------------------
`sentence-transformers` and its weights are a setup step (see `scripts/setup_model.py`).
Tests must not need either, so every entry point takes an `embedder` and the default is
built lazily. A test passes `HashingEmbedder` and gets a real Chroma collection with no
download. The hashing embedder is plumbing-only and is never the default.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import config
from ..rules.store import Rule, RulesLibrary, library as default_library


class PolicyIndexError(RuntimeError):
    """Chroma is missing, or the index cannot be built or read."""


# --------------------------------------------------------------------- embedders


class SentenceTransformerEmbedder:
    """The real embedder: a small multilingual model, run locally, never a hosted API.

    `paraphrase-multilingual-MiniLM-L12-v2` is ~470 MB and covers Arabic and English,
    which is what these documents and this policy are written in.
    """

    is_deterministic = False

    def __init__(self, model_name: str | None = None, cache_dir: str | Path | None = None,
                 offline: bool | None = None):
        self.model_name = model_name or config.EMBED_MODEL
        self.cache_dir = Path(cache_dir or config.EMBED_CACHE)
        self.offline = config.EMBED_OFFLINE if offline is None else offline
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:                                  # pragma: no cover
            raise PolicyIndexError(
                "the policy index needs `sentence-transformers` and Chroma. Install the "
                "prerequisites with:\n\n    python -m scripts.setup_model\n\n"
                "or `pip install -r requirements.txt`") from exc
        try:
            self._model = SentenceTransformer(self.model_name,
                                              cache_folder=str(self.cache_dir),
                                              local_files_only=self.offline)
        except Exception as exc:                                    # noqa: BLE001
            raise PolicyIndexError(
                f"could not load the embedding model {self.model_name!r}: {exc}. "
                f"Run `python -m scripts.setup_model` to download it, or unset "
                f"SAFESHARE2_EMBED_OFFLINE.") from exc
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        # Normalised at encode time, so Chroma's cosine distance is a plain similarity.
        vectors = model.encode(list(texts), normalize_embeddings=True,
                               show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]

    def __call__(self, input: Sequence[str]) -> list[list[float]]:    # noqa: A002
        """Chroma's embedding-function protocol. We pass vectors explicitly instead, so
        the same embedder object is used for indexing and querying by construction."""
        return self.encode(input)

    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def is_installed(self) -> bool:
        """Is the model on this machine? Filesystem only: no import, no network."""
        if "/" in self.model_name:
            folder = self.cache_dir / ("models--" + self.model_name.replace("/", "--"))
            return folder.is_dir() and any(folder.rglob("*.safetensors"))
        return (self.cache_dir / self.model_name).is_dir()

    def install_state(self) -> dict:
        return {"embedder": self.name(), "model": self.model_name,
                "installed": self.is_installed(), "cache_dir": str(self.cache_dir),
                "offline_only": self.offline}


class HashingEmbedder:
    """A dependency-free stand-in for tests and offline smoke runs.

    Character-trigram hashing into a fixed-width vector: a real bag-of-ngrams ranking,
    enough to prove the plumbing and keep the suite download-free, but it is not a
    semantic model and it is never the default. It still needs the same words to rank
    well, which is exactly why the test that uses it asserts on plumbing, not on quality.
    """

    is_deterministic = True
    model_name = "hashing-trigram-384"

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        folded = " ".join(text.lower().split())
        vector = [0.0] * self.dimensions
        grams = [folded[i:i + 3] for i in range(max(len(folded) - 2, 1))]
        for gram in grams:
            digest = hashlib.sha256(gram.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def __call__(self, input: Sequence[str]) -> list[list[float]]:    # noqa: A002
        return self.encode(input)

    def name(self) -> str:
        return f"hashing:{self.model_name}"

    def is_installed(self) -> bool:
        return True

    def install_state(self) -> dict:
        return {"embedder": self.name(), "model": self.model_name, "installed": True,
                "cache_dir": "", "offline_only": True}


# ------------------------------------------------------------------ rule documents


def rule_document(rule: Rule, receivers: dict | None = None) -> str:
    """What actually gets embedded for one rule.

    The meaning of a rule is its type, its action and its reason; the receiver is part of
    the key, so it is embedded too and also stored as metadata for filtering.
    """
    label = ""
    if receivers and rule.receiver in receivers:
        label = getattr(receivers[rule.receiver], "label", "") or str(receivers[rule.receiver])
    parts = [f"PII type: {rule.pii_type}", f"Action: {rule.action}",
             f"Receiver: {label or rule.receiver}", f"Policy reason: {rule.reason}"]
    if rule.citation:
        parts.append(f"Policy source: {rule.citation}")
    return "\n".join(parts)


def rule_hash(rule: Rule, receivers: dict | None = None) -> str:
    return hashlib.sha256(rule_document(rule, receivers).encode("utf-8")).hexdigest()[:16]


def _metadata(rule: Rule, receivers: dict | None, embedder_name: str) -> dict:
    return {"rule_id": rule.rule_id, "receiver": rule.receiver,
            "pii_type": rule.pii_type, "action": rule.action, "reason": rule.reason,
            "citation": rule.citation, "content_hash": rule_hash(rule, receivers),
            "embedder": embedder_name}


def policy_fingerprint(lib: RulesLibrary, embedder_name: str) -> str:
    """Everything that would make an existing vector wrong: the policy, and the embedder."""
    payload = "\n".join(f"{r.rule_id}|{r.receiver}|{r.pii_type}|{r.action}|{r.reason}|"
                        f"{r.citation}" for r in lib.rules)
    digest = hashlib.sha256(f"{embedder_name}\n{payload}".encode("utf-8")).hexdigest()
    return digest[:12]


# ------------------------------------------------------------------------ client


def chroma_client(persist: bool = True, path: str | Path | None = None):
    """A persistent client by default; `persist=False` gives an in-memory one for tests."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise PolicyIndexError(
            "Chroma is a prerequisite of this build and is not installed. Install the "
            "prerequisites with:\n\n    python -m scripts.setup_model\n\n"
            "or `pip install -r requirements.txt`") from exc
    settings = Settings(anonymized_telemetry=False)
    if not persist:
        return chromadb.EphemeralClient(settings=settings)
    directory = Path(path or config.CHROMA_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(directory), settings=settings)


# ------------------------------------------------------------------- the index


class PolicyIndex:
    """Chroma, plus the small amount of logic that keeps it level with the policy."""

    def __init__(self, embedder=None, client=None, library: RulesLibrary | None = None,
                 collection_name: str | None = None, persist: bool = True,
                 path: str | Path | None = None):
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.library = library or default_library()
        self.persist = persist
        self._client = client
        self._base_name = collection_name or config.CHROMA_COLLECTION

    # -------------------------------------------------------------- housekeeping
    @property
    def client(self):
        if self._client is None:
            self._client = chroma_client(persist=self.persist)
        return self._client

    def collection_name(self) -> str:
        """Base name + embedder + policy fingerprint: a new policy is a new collection."""
        slug = re.sub(r"[^a-z0-9]+", "-",
                      self.embedder.name().lower().split(":")[-1].split("/")[-1]).strip("-")
        return f"{self._base_name}__{slug[:28]}__" \
               f"{policy_fingerprint(self.library, self.embedder.name())}"

    @property
    def collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name(), metadata={"hnsw:space": "cosine"})

    def collections(self) -> list[str]:
        """Every collection this app owns, including the ones older policies left behind."""
        try:
            names = [getattr(item, "name", str(item)) for item in self.client.list_collections()]
        except Exception:                                            # noqa: BLE001
            return []
        return sorted(name for name in names if name.startswith(self._base_name))

    def cleanup(self, keep_current: bool = True) -> list[str]:
        """Delete collections for policies or embedders that are no longer in use."""
        current = self.collection_name() if keep_current else None
        removed = []
        for name in self.collections():
            if name == current:
                continue
            try:
                self.client.delete_collection(name)
                removed.append(name)
            except Exception:                                        # noqa: BLE001
                continue
        return removed

    def reset(self) -> None:
        """Drop the current collection only. The policy file is never touched."""
        try:
            self.client.delete_collection(self.collection_name())
        except Exception:                                            # noqa: BLE001
            pass

    def _call(self, operation: str, *args, **kwargs):
        """Run one collection operation, recovering once from a stale handle.

        Chroma hands back a collection object that can go stale - most often while the
        app is updating the policy, and occasionally on a freshly created store, where a
        brand-new collection is briefly reported as missing. Either way the recovery is
        the same and harmless: drop the handle, get the collection again, retry once.
        Otherwise a policy edit would fail for a reason that has nothing to do with policy.
        """
        try:
            return getattr(self.collection, operation)(*args, **kwargs)
        except Exception as exc:                                     # noqa: BLE001
            if "does not exist" not in str(exc).lower():
                raise
            try:
                self.client.delete_collection(self.collection_name())
            except Exception:                                        # noqa: BLE001
                pass
            return getattr(self.collection, operation)(*args, **kwargs)

    # ---------------------------------------------------------------- sync/build
    def sync(self, rebuild: bool = False) -> dict:
        """Make the current collection match `rules.json` exactly. Idempotent and cheap.

        Because the collection name carries the policy fingerprint, an edited policy
        lands on an empty collection and this method simply fills it; a re-run with no
        changes embeds nothing.
        """
        lib = self.library
        embedder_name = self.embedder.name()
        if rebuild:
            self.reset()

        wanted = {rule.rule_id: rule for rule in lib.rules}
        try:
            existing = self._call("get", include=["metadatas"])
        except Exception as exc:                                     # noqa: BLE001
            raise PolicyIndexError(f"could not read the policy index: {exc}") from exc
        have = {meta.get("rule_id"): meta
                for meta in (existing.get("metadatas") or [])
                if meta.get("rule_id")}

        # The collection name encodes the policy and the embedder, so `have` can only be
        # wrong if a run was interrupted mid-build. Rather than trust a partial index,
        # rebuild it: nothing here is expensive enough to justify a clever recovery.
        if len(have) != len(wanted):
            self.reset()
            have = {}

        to_add, to_update = [], []
        for rule_id, rule in wanted.items():
            current = have.get(rule_id)
            if current is None:
                to_add.append(rule)
            elif current.get("content_hash") != rule_hash(rule, lib.receivers):
                to_update.append(rule)
        to_remove = [rule_id for rule_id in have if rule_id not in wanted]

        for group in (to_add, to_update):
            if not group:
                continue
            texts = [rule_document(rule, lib.receivers) for rule in group]
            self._call("upsert",
                       ids=[rule.rule_id for rule in group],
                       documents=texts,
                       embeddings=self.embedder.encode(texts),
                       metadatas=[_metadata(rule, lib.receivers, embedder_name)
                                  for rule in group])
        if to_remove:
            self._call("delete", ids=to_remove)

        return {"rules": len(wanted), "embedded": len(to_add) + len(to_update),
                "added": len(to_add), "updated": len(to_update),
                "removed": len(to_remove), "rebuilt": bool(rebuild),
                "embedder": embedder_name, "rules_version": lib.version,
                "collection": self.collection_name(),
                "orphaned": len(self.collections()) - 1}

    # -------------------------------------------------------------------- query
    def query(self, text: str, receiver: str | None = None, pii_type: str | None = None,
              top_k: int | None = None) -> list[dict]:
        """The rules that answer `text`, best first.

        The receiver/type filters are Chroma metadata filters, so a filter that matches
        nothing returns nothing - it never quietly falls back to the whole policy, because
        "no rule covers this" has to stay visible.
        """
        question = (text or "").strip()
        if not question:
            return []
        # Chroma wants exactly one top-level operator, so two filters become one $and.
        clauses = []
        if receiver:
            clauses.append({"receiver": receiver})
        if pii_type:
            clauses.append({"pii_type": pii_type})
        if not clauses:
            where = None
        elif len(clauses) == 1:
            where = clauses[0]
        else:
            where = {"$and": clauses}
        self.sync()
        try:
            found = self._call(
                "query",
                query_embeddings=self.embedder.encode([question]),
                n_results=max(1, top_k or config.QA_TOP_K),
                where=where, include=["metadatas", "documents", "distances"])
        except Exception as exc:                                     # noqa: BLE001
            raise PolicyIndexError(f"the policy index could not be queried: {exc}") from exc

        ids = (found.get("ids") or [[]])[0]
        metadatas = (found.get("metadatas") or [[]])[0]
        documents = (found.get("documents") or [[]])[0]
        distances = (found.get("distances") or [[]])[0]
        out = []
        for position, rule_id in enumerate(ids):
            meta = metadatas[position] if position < len(metadatas) else {}
            distance = distances[position] if position < len(distances) else None
            out.append({"rule_id": rule_id, "receiver": meta.get("receiver", ""),
                        "pii_type": meta.get("pii_type", ""),
                        "action": meta.get("action", ""), "reason": meta.get("reason", ""),
                        "citation": meta.get("citation", ""),
                        "document": documents[position] if position < len(documents) else "",
                        "distance": None if distance is None else round(float(distance), 4),
                        "relevance": (None if distance is None
                                      else round(max(0.0, 1.0 - float(distance)), 4))})
        return out

    # ------------------------------------------------------------------- status
    def status(self) -> dict:
        embedder = self.embedder.install_state()
        try:
            count = self.collection.count()
            error = ""
        except Exception as exc:                                     # noqa: BLE001
            count, error = 0, str(exc)
        expected = len(self.library.rules)
        return {"collection": self.collection_name(), "documents": count,
                "rules_in_policy": expected, "in_sync": count == expected and not error,
                "indexed_version": self.library.version,
                "persist_dir": str(config.CHROMA_DIR) if self.persist else "in-memory",
                "collections": len(self.collections()), "error": error, **embedder}


# --------------------------------------------------------------- process-wide cache


@lru_cache(maxsize=4)
def _cached_index(embedder_kind: str, persist: bool, collection_name: str) -> PolicyIndex:
    embedder = HashingEmbedder() if embedder_kind == "hashing" else SentenceTransformerEmbedder()
    return PolicyIndex(embedder=embedder, persist=persist, collection_name=collection_name)


def get_index(embedder_kind: str | None = None, persist: bool = True) -> PolicyIndex:
    """The app's index. Streamlit reruns constantly, so it is cached per process."""
    kind = embedder_kind or ("hashing" if config.EMBED_MODEL == "hashing" else "st")
    return _cached_index(kind, persist, config.CHROMA_COLLECTION)


def reset_cache() -> None:
    _cached_index.cache_clear()


# ------------------------------------------------------------- question answering


def asks_about_receiver(question: str, lib: RulesLibrary | None = None) -> str | None:
    """A small, transparent heuristic: does the question name a receiver?

    Deliberately not a model call - it only narrows a metadata filter, and getting it
    wrong shows up as a filtered result the person can widen with one click.
    """
    lib = lib or default_library()
    folded = " ".join((question or "").lower().split())
    for key, receiver in lib.receivers.items():
        candidates = {key.lower(), key.lower().replace("_", " "), receiver.label.lower()}
        candidates.update(part for part in receiver.label.lower().split() if len(part) > 3)
        if any(candidate and candidate in folded for candidate in candidates):
            return key
    if "vendor" in folded or "مورد" in folded:
        return "EXTERNAL_VENDOR"
    if "hr" in folded.split() or "employee" in folded or "موظف" in folded:
        return "INTERNAL_HR"
    if "analyst" in folded or "تحليل" in folded:
        return "ANALYST"
    if "support" in folded or "دعم" in folded or "عميل" in folded:
        return "CUSTOMER_SUPPORT"
    return None


def asks_about_type(question: str) -> str | None:
    """Does the question name one of the PII types, in English or Arabic?"""
    from ..pii_detection.schema import PiiType

    folded = " ".join((question or "").lower().split())
    words = {
        "PERSON_NAME": ("name", "اسم", "الاسم", "أسماء"),
        "NATIONAL_ID": ("national id", "iqama", "هوية", "الهوية"),
        "PHONE": ("phone", "mobile", "جوال", "الجوال", "هاتف"),
        "EMAIL": ("email", "e-mail", "بريد", "البريد"),
        "ADDRESS": ("address", "عنوان", "العنوان"),
        "BANK_ACCOUNT": ("bank account", "account number", "حساب", "الحساب"),
        "IBAN": ("iban", "ايبان", "آيبان"),
        "DATE_OF_BIRTH": ("birth", "dob", "ميلاد", "الميلاد"),
        "PASSPORT_NUMBER": ("passport", "جواز", "الجواز"),
        "CUSTOMER_ID": ("customer no", "customer id", "رقم العميل"),
        "EMPLOYEE_ID": ("employee id", "employee no", "الرقم الوظيفي"),
        "COMPANY_IDENTIFIER": ("cr number", "vat", "commercial registration", "سجل", "ضريبة"),
    }
    for pii_type in PiiType:
        for needle in words.get(pii_type.value, ()):
            if needle in folded:
                return pii_type.value
    return None


def citations(hits: Iterable[dict]) -> list[dict]:
    """The compact citation list the UI renders under an answer."""
    return [{"rule_id": hit["rule_id"], "action": hit["action"], "pii_type": hit["pii_type"],
             "receiver": hit["receiver"], "citation": hit.get("citation", ""),
             "relevance": hit.get("relevance")} for hit in hits]
