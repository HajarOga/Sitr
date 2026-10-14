"""Publish the trained Sitr adapters to the Hugging Face Hub.

Run this where the adapters exist - the Colab runtime that trained them:

    !python training/upload_adapters_to_hf.py --repo sitr-qwen2.5-3b-lora-dpo

It refuses to publish an adapter folder that has no adapter_config.json, so an
empty folder can never be mistaken for a successful upload.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTERS = [
    ("lora_adapter", "Supervised fine-tuning (QLoRA, 4-bit, T4)"),
    ("dpo_adapter", "Preference alignment (DPO), trained on top of the LoRA adapter"),
]

CARD = """---
base_model: {base}
library_name: peft
license: mit
language:
  - ar
  - en
tags:
  - peft
  - lora
  - dpo
  - arabic
  - pii
  - data-protection
---

# Sitr (سِتر) - Qwen2.5-3B adapters

Adapters for **{base}** that turn the base model into the decision model of
**Sitr**: it reads a business document, finds the personal data in it, and
decides - per receiver - whether each value is kept, replaced, removed, or
should be asked about by a human, citing the rule it applied.

Arabic-first, English included. Built for the Tuwaiq Academy Generative AI
Application Engineering bootcamp capstone.

## The adapters in this repository

{adapters}

If both are present, use `dpo_adapter` for the demo: it is the aligned model and it
has learned to ask a human instead of guessing when the policy is silent.

## Measured on a frozen 12-document test set (never trained on)

| metric | base | + LoRA | + DPO |
|---|---|---|---|
| structured output | 88.3% | 100.0% | 100.0% |
| citation validity | 96.7% | 100.0% | 100.0% |
| action agreement | 92.0% | 100.0% | 98.5% |
| finding recall | 71.4% | 97.1% | 94.3% |
| **leak rate** | 34.7% | **2.0%** | 4.1% |

On a companion set where the receiver has **no policy at all** (correct answer:
ask a human about every value):

| metric | base | + LoRA | + DPO |
|---|---|---|---|
| action agreement | 0.0% | 100.0% | 100.0% |
| leak rate | 100.0% | 0.0% | 0.0% |
| deferred to a person | 0.0% | 100.0% | 100.0% |

Honest caveat: the training and test documents come from the same generator, so
they share a writing style. The numbers measure the adaptation, not robustness
against documents from another source.

## How to run it

```bash
pip install -r requirements.txt
hf download {repo} --local-dir training        # base model downloads on first run
export SAFESHARE2_PROVIDER=local
export SAFESHARE2_ADAPTER=training/dpo_adapter
streamlit run app/ui_streamlit.py
```

The adapter only works with the local open-weight model. It does not attach to a
hosted API.

## Licence and credits

Base model: {base_lic}. Adapter: MIT, same as the Sitr project.
Team: Amal, Hajar, Ruba, Sana - Tuwaiq Academy, Generative AI Application
Engineering bootcamp, October 2026.
"""


def find_adapter(folder: Path, drive: Path | None) -> Path | None:
    """Return a usable adapter folder, or None with a clear reason printed."""
    if (folder / "adapter_config.json").is_file():
        return folder
    if drive is not None:
        restored = drive / folder.name
        if (restored / "adapter_config.json").is_file():
            shutil.copytree(restored, folder, dirs_exist_ok=True)
            print(f"restored {folder.name} from Drive")
            return folder
    print(f"skipping {folder.name}: no adapter_config.json there")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="sitr-qwen2.5-3b-lora-dpo",
                    help="repository name under your account")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--drive", default=None,
                    help="Drive folder holding the adapters, e.g. /content/drive/MyDrive/safeshare_adapters")
    ap.add_argument("--train-dir", default="training")
    args = ap.parse_args()

    train = Path(args.train_dir)
    drive = Path(args.drive) if args.drive else None
    found: list[tuple[str, Path, str]] = []
    for name, note in ADAPTERS:
        path = find_adapter(train / name, drive)
        if path is not None:
            found.append((name, path, note))
    if not found:
        print("\nnothing to publish.")
        print("  a disconnected Colab runtime wipes /content, and the adapter with it.")
        print("  re-run the training cell (training/COLAB.md), or pass")
        print("  --drive /content/drive/MyDrive/<folder that holds the adapters>")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder, whoami
    except ImportError:
        print("huggingface_hub is not installed here:  pip install huggingface_hub")
        sys.exit(1)

    user = whoami()["name"]
    repo_id = args.repo if "/" in args.repo else f"{{user}}/{{args.repo}}"
    create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)
    print(f"repository: https://huggingface.co/{{repo_id}}")

    api = HfApi()
    rows = "\n".join(f"| `{name}` | {note} |" for name, _, note in found)
    card = CARD.format(base=BASE_MODEL, base_lic="Apache-2.0", repo=repo_id,
                       adapters="| folder | what it is |\n|---|---|\n" + rows)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")

    for name, path, note in found:
        files = sorted(p.name for p in path.iterdir() if p.is_file())
        size = sum(p.stat().st_size for p in path.iterdir() if p.is_file()) / 1e6
        print(f"uploading {{name}} ({{size:.1f}} MB, {{len(files)}} files) - {{note}}")
        upload_folder(folder_path=str(path), path_in_repo=name, repo_id=repo_id,
                      repo_type="model", commit_message=f"Add {{name}} ({note})")

    print("\ndone. give your teammates this:")
    print(f"  hf download {{repo_id}} --local-dir training")
    print("  export SAFESHARE2_PROVIDER=local")
    print("  export SAFESHARE2_ADAPTER=training/dpo_adapter")


if __name__ == "__main__":
    main()
