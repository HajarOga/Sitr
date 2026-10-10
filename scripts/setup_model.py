"""Install the model. This is the setup step this project cannot skip.

    python -m scripts.setup_model            # download, verify, report
    python -m scripts.setup_model --check    # is it installed? (no network)
    python -m scripts.setup_model --json     # machine-readable status
    python -m scripts.setup_model --index    # (re)build the Chroma policy index

Three prerequisites, not one:

1. **The decision model** (`Qwen/Qwen2.5-3B-Instruct`, ~6 GB). Every release decision
   comes from it, so "the model is missing" is not a degraded mode, it is a stop. A fresh
   clone contains no weights - they are deliberately not in git.
2. **Chroma and the multilingual embedder** (`sentence-transformers`, ~470 MB). They make
   the *policy* searchable, which is what the Policy page's questions and the rule editor
   use. The release pipeline does not depend on them, but this build ships them as part of
   the install so a new machine gets the whole system in one step.
3. **The policy index itself** - derived data, rebuilt at will from `rules.json`.

Why this exists as its own step: it downloads gigabytes and writes to your cache, which a
web app should never do silently on import. The sequence on a new machine is:

    pip install -r requirements.txt
    python -m scripts.setup_model
    streamlit run app/ui_streamlit.py

Two practical notes, both learned the hard way:

* The download runs through plain HTTP by default (`HF_HUB_DISABLE_XET=1`). The Xet
  transfer path can stall at a few percent with no error, which looks exactly like a
  slow model. Pass `--xet` to use it anyway.
* `install_state()` checks that every shard named in the model index is present and
  non-empty, so an interrupted download is reported as incomplete instead of being
  mistaken for a working install. The embedder is verified the same way, by loading it
  once and encoding a sentence - a folder that exists is not proof.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config                                        # noqa: E402
from app.llm import engine                                    # noqa: E402
from app.policy import policy_index                           # noqa: E402


def human(size_gb: float) -> str:
    return f"{size_gb:.2f} GB"


def free_space_gb(path: Path) -> float:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / 1_073_741_824


def device_plan() -> dict:
    """What will happen when the model runs here. No torch import unless available."""
    device = engine.resolve_device()
    try:
        dtype = engine.resolve_dtype(device)
    except ImportError:                                       # pragma: no cover
        dtype = "unknown"
    return {"device": device, "dtype": dtype}


def embedder_state() -> dict:
    """Chroma plus the embedding model: is the policy searchable on this machine?

    Cheap: filesystem and imports only, no model load and no network.
    """
    embedder = policy_index.SentenceTransformerEmbedder()
    state = embedder.install_state()
    try:
        import chromadb                                     # noqa: F401
        state["chroma_installed"] = True
        state["chroma_version"] = chromadb.__version__
    except ImportError:
        state["chroma_installed"] = False
        state["chroma_version"] = ""
    state["ready"] = bool(state["installed"] and state["chroma_installed"])
    state["chroma_dir"] = str(config.CHROMA_DIR)
    return state


def index_state(verbose: bool = False) -> dict:
    """What the policy index holds right now, without building it."""
    try:
        index = policy_index.get_index()
        if verbose is True:                     # build it for real
            index.sync()
        status = index.status()
    except policy_index.PolicyIndexError as exc:
        return {"ready": False, "error": str(exc)}
    status["ready"] = status.get("in_sync", False)
    return status


def report(state: dict, *, verbose: bool = True) -> None:
    plan = device_plan()
    if verbose:
        print(f"model      : {state['model']}")
        print(f"installed  : {'yes' if state['installed'] else 'NO'}")
        print(f"cache      : {state['cache_dir']}")
        if state["path"]:
            print(f"snapshot   : {state['path']}")
        if state["weights"]:
            print(f"weights    : {len(state['weights'])} shard(s), {state['weights']}")
        if state["missing_files"]:
            print(f"missing    : {state['missing_files']}")
        print(f"on disk    : {human(state['size_gb'])}")
        print(f"will run on: {plan['device']} in {plan['dtype']}")
        print(f"free disk  : {human(free_space_gb(Path(state['cache_dir'])))}")


def report_embedder(state: dict, *, verbose: bool = True) -> None:
    if not verbose:
        return
    print(f"\npolicy search (Chroma + embeddings)")
    print(f"chroma     : {'yes, ' + state['chroma_version'] if state['chroma_installed'] else 'NO'}")
    print(f"embedder   : {state['model']}")
    print(f"downloaded : {'yes' if state['installed'] else 'NO'}")
    print(f"cache      : {state['cache_dir'] or '(not set)'}")
    print(f"index dir  : {state['chroma_dir']}")
    print(f"usable     : {'yes' if state['ready'] else 'NO'}")


def install_embedder(*, assume_yes: bool = False) -> int:
    """Download and *verify* the multilingual embedding model.

    Verification is a real encode, not a folder check: a half-downloaded sentence-
    transformers model looks installed right up until the first question.
    """
    state = embedder_state()
    if not state["chroma_installed"]:
        print("Chroma is a prerequisite of this build and is not importable. Install the "
              "Python dependencies first:\n\n    pip install -r requirements.txt\n")
        return 5
    if state["installed"]:
        return 0

    print(f"The policy index needs the embedding model {config.EMBED_MODEL} (~470 MB).")
    print(f"It is downloaded once into:\n  {state['cache_dir']}")
    if not assume_yes:
        answer = input("\nContinue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled. The release pipeline works without it; only the Policy "
                  "page's search needs it.")
            return 1

    embedder = policy_index.SentenceTransformerEmbedder(offline=False)
    print("\ndownloading... (safe to interrupt)")
    try:
        embedder.encode(["verification sentence"])
    except Exception as exc:                                    # noqa: BLE001
        print(f"\nthe embedding model could not be installed: {exc}")
        return 3
    print("verified: the embedder loaded and encoded a sentence.")
    return 0


def build_index(*, rebuild: bool = False) -> int:
    """Build (or rebuild) the Chroma collection from `rules.json`, then tidy up."""
    try:
        index = policy_index.get_index()
    except policy_index.PolicyIndexError as exc:
        print(exc)
        return 5
    try:
        report = index.sync(rebuild=rebuild)
        removed = index.cleanup()
    except policy_index.PolicyIndexError as exc:
        print(f"the policy index could not be built: {exc}")
        return 3
    print(f"policy index: {report['rules']} rule(s), {report['embedded']} embedded, "
          f"{report['removed']} removed"
          + (f", rebuilt" if report["rebuilt"] else ""))
    print(f"collection : {report['collection']}")
    if removed:
        print(f"cleaned up : {len(removed)} collection(s) left by older policy versions")
    return 0


def install_model(*, assume_yes: bool = False) -> int:
    """Download the decision model, verifying every shard afterwards."""
    state = engine.install_state()
    if state["installed"]:
        return 0

    needed = max(0.0, 7.0 - state["size_gb"])
    free = free_space_gb(Path(state["cache_dir"]))
    report(state)
    print(f"\nThis will download {config.MODEL_NAME} (~6 GB; about {needed:.1f} GB still "
          f"to fetch) into:\n  {state['cache_dir']}\n"
          f"Free space there: {human(free)}")

    if free < 8:
        print("\nNot enough free disk space: about 8 GB is needed. Free some space and "
              "run this again.")
        return 2

    if not assume_yes:
        answer = input("\nContinue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled. Nothing was downloaded.")
            return 1

    from huggingface_hub import snapshot_download

    print("\ndownloading... (safe to interrupt: it resumes where it stopped)")
    try:
        path = snapshot_download(
            config.MODEL_NAME,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    except Exception as exc:                                  # noqa: BLE001
        print(f"\nthe download failed: {exc}\n"
              f"Check the connection and run it again - it resumes rather than "
              f"restarting. If it stalls at a few percent, the Xet backend is the usual "
              f"cause; this script already disables it by default.")
        return 3

    state = engine.install_state()
    report(state)
    if not state["installed"]:
        print("\nThe download finished but the install is still incomplete - run this "
              "script again to fetch what is missing.")
        return 4

    print(f"\nDone. {config.MODEL_NAME} is installed.")
    print("snapshot:", path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the SafeShare model")
    parser.add_argument("--check", action="store_true", help="report only, no download")
    parser.add_argument("--json", action="store_true", help="print status as JSON")
    parser.add_argument("--xet", action="store_true",
                        help="use the Xet transfer backend (default: plain HTTP)")
    parser.add_argument("--yes", action="store_true", help="do not ask before downloading")
    parser.add_argument("--index", action="store_true",
                        help="(re)build the Chroma policy index and exit")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="drop and rebuild the policy index from rules.json")
    args = parser.parse_args()

    if not args.xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    model_state = engine.install_state()

    if engine.provider() == "api" and not (args.json or args.check or args.index
                                           or args.rebuild_index):
        print(f"SAFESHARE2_PROVIDER=api: the model runs at {config.API_BASE_URL} "
              f"({config.API_MODEL}), so there is nothing to download.\n"
              f"  run the local model instead : SAFESHARE2_PROVIDER=local\n"
              f"  build the policy index only  : python -m scripts.setup_model --index")
        return 0

    if args.json:
        print(json.dumps({"model": model_state, "embedder": embedder_state(),
                          "index": index_state()}, indent=1, ensure_ascii=False))
        return 0 if model_state["installed"] else 1

    if args.rebuild_index:
        return build_index(rebuild=True)
    if args.index:
        return build_index(rebuild=False)

    if args.check:
        embedder = embedder_state()
        report(model_state)
        report_embedder(embedder)
        index = index_state()
        print(f"index      : {index.get('documents', 0)} rule(s) "
              f"{'(in sync)' if index.get('ready') else '(not built yet)'}")
        return 0 if (model_state["installed"] and embedder["ready"]) else 1

    code = install_model(assume_yes=args.yes)
    if code not in (0,):
        return code

    embedder_code = install_embedder(assume_yes=args.yes)
    print()
    report(engine.install_state())
    report_embedder(embedder_state())

    if embedder_code == 0:
        index_code = build_index(rebuild=False)
    else:
        index_code = 0
        print("\nPolicy search is not ready, so the index was not built. The release "
              "pipeline is unaffected. Finish with:\n"
              "    python -m scripts.setup_model --index")

    plan = device_plan()
    print(f"\nDone. {config.MODEL_NAME} will run on {plan['device']} in {plan['dtype']}.")
    print("\nNext:\n  streamlit run app/ui_streamlit.py\n  python -m scripts.smoke_test")
    return 0 if index_code == 0 else index_code


if __name__ == "__main__":
    raise SystemExit(main())
