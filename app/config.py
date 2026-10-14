"""Sitr (سِتر) - settings.

The difference from the first prototype, in one line: **the model is mandatory**.
Here there is no rules-only fallback. Every finding, every action and every
replacement is proposed by a language model - `Qwen/Qwen2.5-3B-Instruct` running
locally, or a hosted model through an OpenAI-compatible API (DeepSeek by default) -
and the code around it only enforces the contract (a cited rule must exist, a
replacement must be shaped right, a low-confidence item goes to a human). Which one
is used is `SAFESHARE2_PROVIDER`; either way, if the model is not reachable the
application does not quietly degrade - it stops and says so.

Everything below can be overridden with an environment variable, so the same clone runs
on a laptop and on a GPU box without a code change.

    SAFESHARE2_PROVIDER       api | local | auto  (default auto = api when a key is set)
    SAFESHARE2_API_BASE_URL   hosted, OpenAI-compatible endpoint (default DeepSeek)
    SAFESHARE2_API_KEY        key for that endpoint (falls back to DEEPSEEK_API_KEY)
    SAFESHARE2_API_MODEL      which model to send there (default deepseek-chat)
    SAFESHARE2_API_TIMEOUT    seconds to wait for one answer (default 120)
    SAFESHARE2_MODEL          Hugging Face model id  (default Qwen/Qwen2.5-3B-Instruct)
    SAFESHARE2_DTYPE          auto | float16 | bfloat16 | float32
    SAFESHARE2_DEVICE         auto | cuda | mps | cpu
    SAFESHARE2_LOAD_IN_4BIT   auto | true | false  (auto uses NF4 on CUDA GPUs <= 10 GiB)
    SAFESHARE2_ALLOW_DOWNLOAD 1 = let the app fetch the model itself (default 0: use
                              `python -m scripts.setup_model` instead)
    SAFESHARE2_ASK_THRESHOLD  below this confidence a person decides (default 0.80)
    SAFESHARE2_ADJUDICATE     1 = give the model a second, focused look at uncertain
                              items (default 1, because the model is the decision maker
                              here)
    SAFESHARE2_SUMMARY        1 = let the model write the run summary (default 1)
    SAFESHARE2_TMP            temporary folder for uploads
    SAFESHARE2_DB             SQLite audit log
    SAFESHARE2_EMBED_MODEL    sentence-transformers model used for the policy index
    SAFESHARE2_EMBED_OFFLINE  1 = never download the embedding model (fail instead)
    SAFESHARE2_CHROMA_DIR     Chroma's persistent folder
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from `.env` into the environment.

    A real environment variable always wins, so the same clone can be pointed somewhere
    else from the shell without editing the file. Kept dependency-free (no
    python-dotenv) so a fresh clone needs nothing beyond requirements.txt.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'").strip('"')


_load_dotenv(ROOT / ".env")


# -------------------------------------------------------------------- where the model
# `local` runs Qwen on this machine and no document text leaves it. `api` sends the
# document to a hosted, OpenAI-compatible chat endpoint - DeepSeek by default - which is
# much faster than a 3B model on a laptop, at the cost of the document leaving the
# machine. `auto` picks `api` when a key is configured and `local` otherwise.
PROVIDER = os.getenv("SAFESHARE2_PROVIDER", "auto").strip().lower()

API_BASE_URL = os.getenv("SAFESHARE2_API_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = (os.getenv("SAFESHARE2_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "").strip()
API_MODEL = os.getenv("SAFESHARE2_API_MODEL", "deepseek-chat")
API_TIMEOUT = float(os.getenv("SAFESHARE2_API_TIMEOUT", "120"))


def resolve_provider() -> str:
    """`api` or `local`. An explicit setting wins; `auto` follows the key."""
    if PROVIDER in ("api", "local"):
        return PROVIDER
    return "api" if API_KEY else "local"

# ------------------------------------------------------------------------- the model
MODEL_NAME = os.getenv("SAFESHARE2_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MODEL_DTYPE = os.getenv("SAFESHARE2_DTYPE", "auto").lower()
MODEL_DEVICE = os.getenv("SAFESHARE2_DEVICE", "auto").lower()
LOAD_IN_4BIT = os.getenv("SAFESHARE2_LOAD_IN_4BIT", "auto").strip().lower()
ALLOW_DOWNLOAD = os.getenv("SAFESHARE2_ALLOW_DOWNLOAD", "0") == "1"

# The fine-tuned adapter from training/train_lora.py (or training/train_dpo.py).
# Empty means: run the base model exactly as it came.
ADAPTER = os.getenv("SAFESHARE2_ADAPTER", "").strip() or None

# The name that goes into the audit log, the dashboard and the run summary: whichever
# model is actually deciding.
ACTIVE_MODEL = (API_MODEL if resolve_provider() == "api" else
                f"{MODEL_NAME} + LoRA ({Path(ADAPTER).name})" if ADAPTER else MODEL_NAME)

# The model is not optional in this project. Kept as a setting so the failure is
# explicit and testable rather than assumed.
REQUIRE_MODEL = os.getenv("SAFESHARE2_REQUIRE_MODEL", "1") == "1"

MAX_NEW_TOKENS = int(os.getenv("SAFESHARE2_MAX_NEW_TOKENS", "1400"))
TEMPERATURE = float(os.getenv("SAFESHARE2_TEMPERATURE", "0.0"))   # greedy: reproducible

# A document is sent to the model whole when it fits; longer documents are analysed
# block by block so nothing is silently dropped.
MAX_DOC_CHARS = int(os.getenv("SAFESHARE2_MAX_DOC_CHARS", "9000"))
CHUNK_OVERLAP_LINES = int(os.getenv("SAFESHARE2_CHUNK_OVERLAP", "2"))

# ------------------------------------------------------------------------ decisions
ASK_THRESHOLD = float(os.getenv("SAFESHARE2_ASK_THRESHOLD", "0.80"))

# Below this the item is not even a candidate for the model's own second look: it is
# simply handed to a person.
CONFIDENCE_FLOOR = float(os.getenv("SAFESHARE2_CONFIDENCE_FLOOR", "0.35"))
# The most a second look may lift confidence to. A model cannot promote itself to
# "certain"; only a cited rule plus high self-confidence gets this far.
ADJUDICATED_CEILING = float(os.getenv("SAFESHARE2_ADJUDICATED_CEILING", "0.90"))

ADJUDICATE = os.getenv("SAFESHARE2_ADJUDICATE", "1") == "1"
SUMMARY = os.getenv("SAFESHARE2_SUMMARY", "1") == "1"

# ------------------------------------------------------- policy index (Chroma + RAG)
# The policy is small enough to paste into a prompt whole (see core._rules_for_prompt).
# The vector index exists so a *person* can ask the policy a question in Arabic or
# English and get the rules that answer it - it is a reading aid, not a decision path.
EMBED_MODEL = os.getenv("SAFESHARE2_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_OFFLINE = os.getenv("SAFESHARE2_EMBED_OFFLINE", "0") == "1"
CHROMA_DIR = Path(os.getenv("SAFESHARE2_CHROMA_DIR", str(ROOT / "data" / "index")))
CHROMA_COLLECTION = os.getenv("SAFESHARE2_CHROMA_COLLECTION", "safeshare2-policy")
EMBED_CACHE = Path(os.getenv("SAFESHARE2_EMBED_CACHE", str(ROOT / "data" / "index" / "models")))
QA_TOP_K = int(os.getenv("SAFESHARE2_QA_TOP_K", "6"))

# -------------------------------------------------------------------- files & paths
RULES_PATH = ROOT / "app" / "rules" / "rules.json"
DATA_DIR = ROOT / "data"
TMP_DIR = Path(os.getenv("SAFESHARE2_TMP", str(ROOT / ".tmp")))
DB_PATH = Path(os.getenv("SAFESHARE2_DB", str(ROOT / ".tmp" / "audit.sqlite3")))
KEEP_UPLOADS = os.getenv("SAFESHARE2_KEEP_UPLOADS", "0") == "1"

RECEIVERS = {
    "INTERNAL_HR": "Internal HR",
    "EXTERNAL_VENDOR": "External Vendor",
    "ANALYST": "Analyst",
    "CUSTOMER_SUPPORT": "Customer Support",
    "CUSTOM": "Custom receiver",
}
