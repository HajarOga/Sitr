# Setting SafeShare2 up

Two audiences: the machine you are on now, and a fresh machine (or a GPU box) where this
is cloned from GitHub. The steps are the same; the notes underneath explain the choices.

## 1. The short version

```bash
git clone <YOUR-REPO-URL> safeshare2 && cd safeshare2
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # Windows: copy .env.example .env
# put your own key in .env: DEEPSEEK_API_KEY=<key>
python3 tests/test_safeshare2.py                        # the contract tests, no model needed
streamlit run app/ui_streamlit.py                       # http://localhost:8501
```

That is the whole setup **when the model is the hosted API**. The key is not in the
repository — copy the template and add your own. On Windows use `py` instead of `python3`
if `python3` is not recognised, and run these in PowerShell opened in the project folder
(right-click the folder in Explorer -> *Open in Terminal*).

Downloaded the repository as a ZIP instead of cloning it? You get the same files with two
differences worth knowing: there is no `.git` folder, so there is no history and no
`git pull` to update it, and there is no `data/index`, so the first question on the Policy
page downloads the policy embedder (about 470 MB, once) and builds the index.
To run the model on this machine instead (`SAFESHARE2_PROVIDER=local`), add the two steps
that need the weights:

```bash
python -m scripts.setup_model                           # the model: required, once, ~6 GB
python -m scripts.smoke_test                            # end to end, with the model
```

## 1a. Which provider, and why it matters

| `SAFESHARE2_PROVIDER` | where the model runs | the document | needs |
|---|---|---|---|
| `api` | `SAFESHARE2_API_BASE_URL` (DeepSeek) | **sent there in the prompt** | a key in `.env` |
| `local` | in this process (Qwen2.5-3B-Instruct) | never leaves the machine | ~6 GB weights |
| `auto` (default) | `api` if a key is set, else `local` | per the resolved mode | per the resolved mode |

The API is roughly an order of magnitude faster than the 3B model on a laptop (seconds
instead of tens of seconds per page), and it is the mode this build is set to. The local
model is the offline, nothing-leaves-the-machine mode. Both are the same pipeline: the
model still has to cite a real rule, and the guards still validate every substitute.

The key lives in `.env` at the project root and is read by `app/config.py` (`python -c
"from app import config; print(config.API_KEY[-4:])"` shows which one is loaded — never
print the whole key). A real environment variable overrides the file.

## 2. Prerequisites

| | minimum | notes |
|---|---|---|
| Python | 3.10 (3.13 tested) | `python3 --version` |
| Disk | ~8 GB free | the model (~6 GB) plus the Hugging Face cache |
| RAM | 16 GB on Apple Silicon | the model loads in fp16, about 6 GB resident |
| GPU | optional | a CUDA GPU is several times faster; 8 GB+ VRAM runs the 3B model in bf16/fp16 |
| Network | once | only for the model download; afterwards it runs offline |

## 3. The model step in detail

```bash
python -m scripts.setup_model            # asks before downloading
python -m scripts.setup_model --yes      # no prompt, for scripts and CI
python -m scripts.setup_model --check    # report only; exit code 0 = installed
python -m scripts.setup_model --json     # machine-readable, for a bootstrap script
```

It downloads `Qwen/Qwen2.5-3B-Instruct` into the standard Hugging Face cache
(`~/.cache/huggingface/hub/`), which every project on the machine shares. Nothing is
written into the repository, so the clone stays small and `git status` stays clean.

Installation is **verified, not assumed**: the script re-reads the model index and checks
that every shard it names exists and is non-empty, plus the tokenizer. An interrupted
download fails that check and is reported as incomplete, and running the script again
resumes it — downloads are never restarted from zero.

To put the weights somewhere else (a shared disk, for example):

```bash
export HF_HOME=/path/to/big-disk/huggingface
python -m scripts.setup_model
```

## 4. Hardware: what to expect

| machine | device / precision (automatic) | one page of text |
|---|---|---|
| Apple Silicon (M-series) | `mps` in `float16` | seconds per model call, visible in the dashboard |
| CUDA GPU (T4 and up) | `cuda` in `bfloat16` or `float16` | a few times faster |
| CPU only | `cpu` in `float32` | usable for a test, slow for a demo |

Override when you know better than the automatic choice:

```bash
SAFESHARE2_DEVICE=cuda SAFESHARE2_DTYPE=bfloat16 streamlit run app/ui_streamlit.py
```

A 4-bit GPU load (needs `bitsandbytes`, CUDA only) is the way to fit the model on a small
GPU; install it and set `SAFESHARE2_DTYPE=4bit` when that path is added to `engine.py`.

## 5. Settings that matter

| variable | default | what it does |
|---|---|---|
| `SAFESHARE2_PROVIDER` | `auto` | `api` / `local` / `auto` (see §1a) |
| `SAFESHARE2_API_BASE_URL` | `https://api.deepseek.com` | the OpenAI-compatible endpoint |
| `DEEPSEEK_API_KEY` (or `SAFESHARE2_API_KEY`) | — | the key; without it `auto` stays local |
| `SAFESHARE2_API_MODEL` | `deepseek-chat` | which model to send there |
| `SAFESHARE2_API_TIMEOUT` | `120` | seconds to wait for one answer |
| `SAFESHARE2_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | which local model (ignored by the API provider) |
| `SAFESHARE2_DEVICE` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `SAFESHARE2_DTYPE` | `auto` | `auto` / `float16` / `bfloat16` / `float32` |
| `SAFESHARE2_ASK_THRESHOLD` | `0.80` | below this, a person decides |
| `SAFESHARE2_ADJUDICATE` | `1` | the model's second look at uncertain items |
| `SAFESHARE2_SUMMARY` | `1` | the model writes the audit note |
| `SAFESHARE2_ALLOW_DOWNLOAD` | `0` | `1` lets the app download the model itself |
| `SAFESHARE2_MAX_DOC_CHARS` | `9000` | chunk size before a document is split |
| `SAFESHARE2_TMP` | `./.tmp` | where uploads are read and releases are written |
| `SAFESHARE2_DB` | `./.tmp/audit.sqlite3` | the audit log |

## 6. Verify the install

```bash
python3 tests/test_safeshare2.py     # 22 tests: the contract, no model required
python -m scripts.smoke_test         # model loads, a document is analysed, a release is
                                     # written, the audit records it, the dashboard renders
```

`smoke_test` prints **skipped** for anything that needs the model when the model is absent
— skipped is not passed, and the summary line says which is which.

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| Dashboard: "The model is not reachable" with `SAFESHARE2_PROVIDER=api` | no key. Put `DEEPSEEK_API_KEY=<key>` in `.env` at the project root, or set `SAFESHARE2_PROVIDER=local` to use the local model |
| `the deepseek-chat API returned HTTP 401` | the key is wrong, revoked, or has no balance. Check it at the provider's console and update `.env` |
| `the deepseek-chat API returned HTTP 429` | rate limit or quota. Wait, or run a smaller document, or switch to `local` |
| `the deepseek-chat API at ... could not be reached` | no network / a proxy. `SAFESHARE2_API_TIMEOUT` (default 120s) is per call |
| Dashboard: "The model is not installed" | run `python -m scripts.setup_model` (local provider) |
| The download stalls at a few percent with no error | the Xet transfer backend. `setup_model` already disables it (`HF_HUB_DISABLE_XET=1`); pass `--xet` only if your network prefers it |
| `OSError: ... does not appear to have a file named model-00001-of-00002.safetensors` | an interrupted download. Check with `--check` and re-run `setup_model` to resume |
| The first analysis takes minutes | the model is loading into memory; run one throwaway analysis after starting the dashboard. On CPU expect it to stay slow |
| `ModuleNotFoundError: app` | run from the repository root |
| `ModuleNotFoundError: fastapi`/`streamlit` from a script or a background shell | that shell resolved a different interpreter. Use the explicit path (`/usr/local/bin/python3 -m scripts.smoke_test`) or activate the venv |
| A test run says `skipped: the model is installed` | that is the intended behaviour on a machine without the weights |
| `OutOfMemoryError` / the machine swaps heavily | another app is holding memory, or use a smaller model: `SAFESHARE2_MODEL=Qwen/Qwen2.5-1.5B-Instruct` |

Print the environment when asking for help:

```bash
python -c "import sys; print(sys.version)"
python -c "import torch, transformers, streamlit; print(torch.__version__, transformers.__version__, streamlit.__version__)"
python -m scripts.setup_model --json
```

## 8. Pushing to GitHub

`.gitignore` already excludes the weights (`*.safetensors`, `models/`), the runtime folders
(`.tmp/`), the audit database and `.env`. There are no secrets in this project and no
machine-specific paths in code.

```bash
git add . && git commit -m "SafeShare2: purpose-bound release with a local model at the core"
git remote add origin git@github.com:<account>/safeshare2.git
git push -u origin main
```

Anyone who clones it then needs exactly one extra step beyond `pip install`:
`python -m scripts.setup_model`.
