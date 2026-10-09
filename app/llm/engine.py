"""The model layer: get a model, talk to it, and record what it cost.

This is the part the first prototype did not have. Everything here is about the model
being **present**:

* `install_state()` tells you whether the model is really usable - not "the folder
  exists" but "every shard named in the index is present and non-empty" for the local
  model, or "a key is configured" for the hosted one. An interrupted download looks
  exactly like a finished one until you check this.
* `load()` caches one copy of the local model per process (a no-op for the API).
* `generate()` talks to whichever provider is configured - the local model through its
  own chat template, or an OpenAI-compatible HTTP endpoint - so switching is a settings
  change rather than a rewrite.
* `generate_json()` is the contract with the rest of the application: the model answers
  in JSON, and if it does not, we ask again before giving up. A model that returns
  prose is a real failure mode and is handled here, once, instead of in every caller.
* `calls()` returns the metadata of every model call in this process - seconds, sizes,
  tokens - so the dashboard can show that the AI is genuinely in the loop, and the
  audit log can record it.

**Where the document goes** is `SAFESHARE2_PROVIDER` (see `config.py`). With `local`
nothing leaves this machine. With `api` the document text is sent to the configured
endpoint (DeepSeek by default) in the prompt - that is the trade for the speed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from .. import config

SETUP_COMMAND = "python -m scripts.setup_model"


class ModelNotInstalled(RuntimeError):
    """Raised instead of quietly falling back to something that is not the AI."""

    def __init__(self, message: str, *, model: str = "", missing: list[str] | None = None):
        super().__init__(message)
        self.model = model or config.ACTIVE_MODEL
        self.missing = missing or []
        self.command = SETUP_COMMAND


class ModelCallFailed(RuntimeError):
    """The model was reachable in principle but this call did not come back usable."""


def provider() -> str:
    """`api` or `local` - whichever the configuration resolves to right now."""
    return config.resolve_provider()


def active_model() -> str:
    """The model that is doing the deciding, for the audit log and the dashboard."""
    return config.ACTIVE_MODEL


# ----------------------------------------------------------------- source of truth


def hf_cache_dir(model_name: str | None = None) -> Path:
    home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    return home / "hub" / ("models--" + (model_name or config.MODEL_NAME).replace("/", "--"))


def snapshot_dirs(model_name: str | None = None) -> list[Path]:
    base = hf_cache_dir(model_name) / "snapshots"
    try:
        return sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return []


def model_path(model_name: str | None = None) -> Path | None:
    """The newest local snapshot of the model, or None when it is not downloaded."""
    snapshots = snapshot_dirs(model_name)
    return snapshots[-1] if snapshots else None


def _weight_files(snapshot: Path) -> tuple[list[str], list[str]]:
    """(required, missing) weight filenames for one snapshot."""
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            required = sorted(set(json.loads(index.read_text(encoding="utf-8"))
                                  .get("weight_map", {}).values()))
        except (json.JSONDecodeError, OSError):
            required = []
    else:
        required = [p.name for p in sorted(snapshot.glob("*.safetensors"))]
    missing = [name for name in required
               if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0]
    return required, missing


def api_state() -> dict:
    """The hosted provider's install state: a configured key is what makes it usable.

    Same keys as the local one, so every caller - the dashboard, the API, the smoke test,
    the setup script - keeps working without asking which provider it is talking to.
    """
    key = config.API_KEY
    return {
        "provider": "api",
        "model": config.API_MODEL,
        "installed": bool(key),
        "path": config.API_BASE_URL,
        "weights": [],
        "missing_files": [],
        "tokenizer": True,
        "size_gb": 0.0,
        "partial": False,
        "cache_dir": config.API_BASE_URL,
        "setup_command": "put DEEPSEEK_API_KEY in .env",
        "download_allowed": False,
        "base_url": config.API_BASE_URL,
        "api_key_set": bool(key),
        # last four only - enough to tell which key is loaded, not enough to use it
        "api_key_hint": f"…{key[-4:]}" if key else "",
        "device": "api",
        "dtype": "remote",
    }


def install_state() -> dict:
    """Is the model really usable, and on what device would it run?

    Cheap and safe to call from the UI: it reads the filesystem only and never imports
    torch or touches the network. For the API provider it reports whether a key is set.
    """
    if provider() == "api":
        return api_state()

    snapshot = model_path()
    required: list[str] = []
    missing: list[str] = []
    size_gb = 0.0
    if snapshot is not None:
        required, missing = _weight_files(snapshot)
        for path in snapshot.glob("*.safetensors"):
            try:
                size_gb += path.stat().st_size / 1_073_741_824
            except OSError:
                continue
    tokenizer_present = bool(snapshot is not None and (snapshot / "tokenizer.json").is_file())
    installed = bool(snapshot is not None and required and not missing and tokenizer_present)
    return {
        "provider": "local",
        "model": config.MODEL_NAME,
        "installed": installed,
        "path": str(snapshot) if snapshot else "",
        "weights": required,
        "missing_files": missing,
        "tokenizer": tokenizer_present,
        "size_gb": round(size_gb, 2),
        "partial": bool(snapshot is not None and (missing or not required)),
        "cache_dir": str(hf_cache_dir()),
        "setup_command": SETUP_COMMAND,
        "download_allowed": config.ALLOW_DOWNLOAD,
    }


def ensure_installed() -> dict:
    """Stop the application if the AI is not available. There is no rules-only mode."""
    state = install_state()
    if state["installed"]:
        return state

    if state.get("provider") == "api":
        raise ModelNotInstalled(
            f"SafeShare2 is set to use the hosted {config.API_MODEL} API at "
            f"{config.API_BASE_URL}, but no key is configured. Put one in the `.env` "
            f"file at the project root:\n\n    DEEPSEEK_API_KEY=<your key>\n\n"
            f"or run the local model instead with SAFESHARE2_PROVIDER=local.",
            model=config.API_MODEL)

    if state["partial"]:
        detail = ("the download is incomplete"
                  + (f" (missing: {', '.join(state['missing_files'][:3])})"
                     if state["missing_files"] else ""))
    elif state["path"]:
        detail = "the cached copy has no weights"
    else:
        detail = "no local copy was found"

    raise ModelNotInstalled(
        f"SafeShare2 needs {config.MODEL_NAME} and it is not usable on this machine: "
        f"{detail}. Install it with:\n\n    {SETUP_COMMAND}\n\n"
        f"(downloads once, about 6 GB, into {state['cache_dir']})",
        missing=state["missing_files"])


# ----------------------------------------------------------------------- device setup


def resolve_device() -> str:
    if provider() == "api":
        return "api"
    if config.MODEL_DEVICE != "auto":
        return config.MODEL_DEVICE
    try:
        import torch
    except ImportError:                                   # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str) -> str:
    if device == "api":
        return "remote"
    if config.MODEL_DTYPE != "auto":
        return config.MODEL_DTYPE
    if device == "cuda":
        import torch
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if device == "mps":
        return "float16"          # half precision: ~6 GB instead of ~12 GB
    return "float32"


def _torch_dtype(name: str):
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


@lru_cache(maxsize=1)
def _load_local():
    """Load the tokenizer and the local model once (one copy per process)."""
    state = ensure_installed()
    local_only = not config.ALLOW_DOWNLOAD

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device()
    dtype_name = resolve_dtype(device)
    dtype = _torch_dtype(dtype_name)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME,
                                             local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, local_files_only=local_only, dtype=dtype)
    if device != "cpu":
        model = model.to(device)
    model.eval()
    _LOADED.update({"provider": "local", "model": config.MODEL_NAME, "device": device,
                    "dtype": dtype_name, "path": state["path"]})
    return model, tokenizer


def load():
    """Prepare the model, whichever provider is configured.

    The one entry point the dashboard, the smoke test and the live run call, so none of
    them has to ask where the model runs. Returns (model, tokenizer) for the local
    model, and (None, None) for the API - there is nothing to load there.
    """
    if provider() == "api":
        ensure_installed()
        _LOADED.update({"provider": "api", "model": config.API_MODEL, "device": "api",
                        "dtype": "remote", "path": config.API_BASE_URL})
        return None, None
    return _load_local()


def loaded_info() -> dict:
    """What the model is, without forcing a load."""
    if _LOADED:
        return dict(_LOADED)
    device = resolve_device()
    return {"provider": provider(), "model": active_model(), "device": device,
            "dtype": resolve_dtype(device), "path": install_state()["path"]}


def status() -> dict:
    state = install_state()
    state.update(loaded_info())
    state.setdefault("provider", provider())
    state["setup_needed"] = not state["installed"]
    return state


# -------------------------------------------------------------------------- inference

_LOADED: dict = {}
_CALLS: list[dict] = []


def calls() -> list[dict]:
    return list(_CALLS)


def reset_calls() -> None:
    _CALLS.clear()


CHAT_COMPLETIONS = "/chat/completions"


def _api_generate(messages: list[dict], *, purpose: str, max_new_tokens: int | None,
                  temperature: float | None, json_mode: bool) -> str:
    """One turn of chat at an OpenAI-compatible endpoint (DeepSeek by default).

    The document text travels in the prompt, so this is the one path in the project that
    sends anything off the machine - see the note at the top of this module.
    """
    import httpx

    if not config.API_KEY:
        ensure_installed()                       # raises with the .env instructions

    temperature = config.TEMPERATURE if temperature is None else temperature
    payload: dict[str, Any] = {
        "model": config.API_MODEL,
        "messages": [{"role": str(m.get("role") or "user"),
                      "content": str(m.get("content") or "")} for m in messages],
        "max_tokens": max_new_tokens or config.MAX_NEW_TOKENS,
        "temperature": temperature,
        "stream": False,
    }
    if json_mode:
        # The API can be told to return JSON, on top of the prompts already asking for it.
        payload["response_format"] = {"type": "json_object"}

    started = time.time()
    try:
        response = httpx.post(
            config.API_BASE_URL + CHAT_COMPLETIONS,
            headers={"Authorization": f"Bearer {config.API_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=config.API_TIMEOUT)
    except httpx.HTTPError as exc:
        raise ModelCallFailed(
            f"the {config.API_MODEL} API at {config.API_BASE_URL} could not be reached: "
            f"{exc}") from exc
    seconds = time.time() - started

    if response.status_code != 200:
        raise ModelCallFailed(
            f"the {config.API_MODEL} API returned HTTP {response.status_code}: "
            f"{response.text[:300]}")

    try:
        body = response.json()
        text = (body["choices"][0]["message"]["content"] or "").strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ModelCallFailed(
            f"the {config.API_MODEL} API returned something unexpected: "
            f"{response.text[:300]}") from exc

    usage = body.get("usage") or {}
    prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
    _CALLS.append({
        "purpose": purpose,
        "model": config.API_MODEL,
        "provider": "api",
        "prompt_chars": len(prompt_text),
        "output_chars": len(text),
        "tokens_in": int(usage.get("prompt_tokens") or 0),
        "tokens_out": int(usage.get("completion_tokens") or 0),
        "seconds": round(seconds, 2),
        # a hash, not the prompt: the prompt contains document text
        "prompt_sha": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12],
        "device": "api",
    })
    return text


def generate(messages: list[dict], *, purpose: str = "generate",
             max_new_tokens: int | None = None,
             temperature: float | None = None,
             json_mode: bool = False) -> str:
    """One turn of chat, at whichever provider is configured. Text out, stripped."""
    if provider() == "api":
        return _api_generate(messages, purpose=purpose, max_new_tokens=max_new_tokens,
                             temperature=temperature, json_mode=json_mode)

    import torch

    model, tokenizer = _load_local()
    device = next(model.parameters()).device
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    temperature = config.TEMPERATURE if temperature is None else temperature

    started = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=temperature > 0,
            temperature=temperature or 1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    seconds = time.time() - started

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    _CALLS.append({
        "purpose": purpose,
        "model": config.MODEL_NAME,
        "provider": "local",
        "prompt_chars": len(prompt),
        "output_chars": len(text),
        "tokens_out": int(new_tokens.shape[0]),
        "seconds": round(seconds, 2),
        "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
        "device": str(device),
    })
    return text


# ------------------------------------------------------------------- JSON, carefully

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _balanced_candidates(text: str):
    """Every top-level {...} or [...] block in the text, longest first."""
    for opener, closer in (("{", "}"), ("[", "]")):
        depth, start = 0, None
        for index, char in enumerate(text):
            if char == opener:
                if depth == 0:
                    start = index
                depth += 1
            elif char == closer and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:index + 1]


def _repair(raw: str) -> str:
    fixed = raw.strip().rstrip(",")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)              # trailing commas
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    return fixed


def extract_json(text: str | None) -> Any | None:
    """Parse the model's answer into Python, accepting the usual sloppiness."""
    if not text:
        return None
    candidates: list[str] = []
    for block in _FENCE.findall(text):
        candidates.extend(_balanced_candidates(block))
    candidates.extend(_balanced_candidates(text))
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        for attempt in (candidate, _repair(candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    return None


JSON_NUDGE = ("Your previous answer was not valid JSON and could not be parsed. "
              "Reply with valid JSON only - no explanation, no markdown fence, no "
              "comment before or after it.")


def generate_json(messages: list[dict], *, purpose: str = "json",
                  max_new_tokens: int | None = None, retries: int = 2) -> tuple[Any, dict]:
    """Ask, parse, and ask again if the answer was not machine-readable.

    Returns (payload, meta). `payload` is None only when every attempt failed - callers
    treat that as "no answer", never as "no findings".
    """
    conversation = list(messages)
    meta: dict = {"purpose": purpose, "attempts": 0, "parsed": False, "seconds": 0.0}
    text = ""
    for attempt in range(retries + 1):
        text = generate(conversation, purpose=purpose, max_new_tokens=max_new_tokens,
                        json_mode=True)
        meta["attempts"] = attempt + 1
        if _CALLS:
            meta["seconds"] = round(meta["seconds"] + _CALLS[-1]["seconds"], 2)
        payload = extract_json(text)
        if payload is not None:
            meta["parsed"] = True
            meta["raw"] = text[:2000]
            return payload, meta
        if attempt < retries:
            conversation = conversation + [
                {"role": "assistant", "content": text[:600]},
                {"role": "user", "content": JSON_NUDGE},
            ]
    meta["raw"] = text[:2000]
    return None, meta
