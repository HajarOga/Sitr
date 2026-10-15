# سِتر · Sitr

> Formerly **SafeShare** — the name used in earlier submissions. The repository is now
> `HajarOga/sitr`; GitHub redirects the previous URLs.

Sitr analyses a document for personal information, applies receiver-specific
policy, and prepares a safer version for review and release. Rules and code enforce
the release contract; the model proposes findings and actions.

## Requirements

- Python 3.10 or newer (the project is tested with Python 3.13).
- A supported local model installation for local inference, or an OpenAI-compatible
  API key for hosted inference.
- A CUDA-compatible NVIDIA GPU is optional for local inference. CPU and Apple
  Silicon (`mps`) are also supported by the model engine when available.
- `requirements.txt` also installs what the evaluation reports and the adapter path
  need: `openpyxl` (Excel reports), `plotly` (charts), `peft` (loading a LoRA adapter),
  and the pinned `bitsandbytes` (4-bit CUDA loading).

The application uses Streamlit. The optional HTTP API uses FastAPI and Uvicorn.
Document support includes TXT, Markdown, DOCX, and PDF input; releases are written as
DOCX or TXT.

## Install

Run commands from the project root.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, create and activate the virtual environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` for the provider you intend to use. The checked-in template explicitly
selects the API provider; enter your own API key there, or switch to `local`. Never
put a real key in source control. `.env` is ignored by Git.

## Choose an inference provider

`SAFESHARE2_PROVIDER` accepts `api`, `local`, or `auto`.

| Value | Model location | Document privacy |
|---|---|---|
| `api` | Configured OpenAI-compatible endpoint (DeepSeek by default) | Document text is sent to that endpoint in the prompt. |
| `local` | Local Hugging Face model | Document text stays on this machine. |
| `auto` | API when an API key is configured; otherwise local | Follows the selected provider. |

For API mode, set `SAFESHARE2_API_BASE_URL`, `SAFESHARE2_API_MODEL`, and either
`SAFESHARE2_API_KEY` or `DEEPSEEK_API_KEY`. For local mode, the default base model is
`Qwen/Qwen2.5-3B-Instruct`. A LoRA adapter is optional and is selected with
`SAFESHARE2_ADAPTER`, pointing to a directory containing the adapter files. The
adapter is not included in a fresh clone and is not modified by setup or evaluation.

The `.env.example` file contains the supported settings and safe defaults. Environment
variables override values in `.env`.

## Install and configure a local model

Install the project dependencies, then run the model setup tool from the project root:

```bash
python -m scripts.setup_model
python -m scripts.setup_model --check
```

The setup tool downloads the configured model to the Hugging Face cache and can install
the local policy-search embedder. The model weights and downloaded policy index are not
stored in Git. Use `python -m scripts.setup_model --help` for options such as rebuilding
the policy index.

### GPU acceleration

The engine can use CUDA, Apple Metal (`mps`), or CPU according to
`SAFESHARE2_DEVICE` and available hardware. On a CUDA-capable NVIDIA system, set local
mode and select CUDA in `.env`:

```dotenv
SAFESHARE2_PROVIDER=local
SAFESHARE2_DEVICE=cuda
SAFESHARE2_DTYPE=bfloat16
SAFESHARE2_LOAD_IN_4BIT=auto
```

`SAFESHARE2_LOAD_IN_4BIT=auto` selects NF4 quantization on CUDA GPUs with 10 GiB VRAM
or less; `true` and `false` explicitly enable or disable it. NF4 requires the pinned
`bitsandbytes` dependency. BF16 is only used when supported by the GPU; `auto` chooses
an available dtype.

The `torch` version is pinned in `requirements.txt`, but CUDA wheel availability depends
on the operating system, Python version, and NVIDIA driver. After installation, verify
the actual PyTorch build and device before running:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -m scripts.setup_model --check
```

Install a PyTorch build compatible with the project pin, Python version, and installed
NVIDIA driver if CUDA is not available. The full-report evaluator intentionally refuses
CPU inference or an unconfirmed model/adapter configuration.

## Run Sitr

Launch the Streamlit application from the project root:

```bash
streamlit run app/ui_streamlit.py
```

The dashboard provides Upload, Review, Audit log, and Policy pages. Analysis runs only
after the user starts it. Progress and timing information reflect actual pipeline stages
and completed document chunks.

The optional HTTP API is:

```bash
uvicorn app.main:app --reload
```

Its health endpoint is available at `http://127.0.0.1:8000/api/health`.

## How policy and search work

- **Rules:** `app/rules/rules.json` holds the rules for each receiver. For each
  document, the receiver's rules are read from this file and placed in the model prompt.
- **Policy search:** the rules are also embedded in a Chroma index. The Policy page uses
  it to answer Arabic and English questions and shows the rule behind each answer.
  Editing a rule updates the file, the version and the index.
- **Document assistant:** answers questions about the released safe copy and summarizes it.

## Tests and smoke check

Run the complete test suite:

```bash
python -m pytest -q
```

The suite currently holds **66 tests** — 62 for the pipeline, policy and guards, and 4 for
the evaluation reports.

The smoke test exercises the model-backed analysis path when the configured model is
available:

```bash
python -m scripts.smoke_test
```

Model-dependent checks can be reported as skipped when the required local model or
provider is unavailable; a skip is not a passing inference test.

## Evaluation

The standard evaluation CLI can run a dataset, score saved predictions, or compare
prediction files:

```bash
python -m evaluation.evaluate --live --test-dir data/test_second --out predictions/run.jsonl
python -m evaluation.evaluate --score predictions/run.jsonl
python -m evaluation.evaluate --compare predictions/base.jsonl predictions/adapter.jsonl
```

The policy RAG benchmark is a separate retrieval-only evaluation. It exercises English
and Arabic questions at several Top-K values, does not invoke the answer model, and
writes a uniquely timestamped Excel report under `evaluation/results/`:

```bash
python -m evaluation.evaluate_rag
```

The detailed report runner defaults to `data/test_second` (20 documents and 87 labels)
and writes uniquely timestamped Excel and JSONL files to `evaluation/results/`:

```bash
python -m evaluation.full_report
```

For the Human Judge dataset, direct output to `testing/`:

```bash
python -m evaluation.full_report --test-dir data/test_human_judge --output-dir testing --dataset-name test_human_judge --expected-documents 20 --expected-labels 87
```

A validated run from this evaluator is committed under `evaluation/results/` as a
timestamped predictions file and Excel workbook, so the published numbers can be
re-derived without a GPU.

The detailed report runner requires the local Qwen2.5-3B-Instruct model, a configured
LoRA adapter, CUDA, BF16 support, and the validated 4-bit load. It checks these
requirements before counting a run as valid. Failed documents are reported separately
and are not scored as successful documents. It does not alter the datasets, model
weights, adapter, or earlier results.

## Project layout

```text
app/                 Streamlit UI, optional API, pipeline, policy, model, and audit
data/                Synthetic documents, training data, and evaluation fixtures
docs/                Setup, pipeline, AI, demo, and model-adaptation documentation
evaluation/          Evaluation CLI, metrics, reference policy, and detailed reports
notebooks/            Optional training notebook
scripts/              Model setup, smoke checks, and dataset utilities
tests/                Automated test suite
training/             Training-data preparation and LoRA/DPO scripts
```

Local runtime data, the virtual environment, `.env`, downloaded models and adapters,
and generated evaluation reports are excluded from Git. Source datasets and their
labels remain project files and are not changed by evaluation.

## Privacy and release safeguards

- In `local` mode, document inference runs on the machine. In `api` mode, document text
  is sent to the configured endpoint.
- Uploads are temporary by default. `SAFESHARE2_KEEP_UPLOADS=1` retains them for
  debugging.
- The local audit records masked values and hashes rather than original PII values;
  model-call telemetry records metadata, not prompt text.
- Policy rules are validated in code. Invalid findings or unresolved human-review
  decisions are not silently treated as safe releases.

For the demo script and the questions to expect, see [docs/DEMO.md](docs/DEMO.md).
For implementation details, see [docs/PIPELINE.md](docs/PIPELINE.md),
[docs/AI.md](docs/AI.md), [docs/SETUP.md](docs/SETUP.md), and
[training/COLAB.md](training/COLAB.md).

## Previously published results

The following baseline was reported on the frozen 12-document set in `data/test/`.
It is retained for project history and is separate from the current detailed reports
and retrieval-only RAG benchmark described above.

| Metric | Base model | + LoRA | + LoRA + DPO |
|---|---:|---:|---:|
| Structured output | 88.3% | 100.0% | 100.0% |
| Citation validity | 96.7% | 100.0% | 100.0% |
| Action agreement | 92.0% | 100.0% | 98.5% |
| Finding recall | 71.4% | 97.1% | 94.3% |
| Leak rate | 34.7% | 2.0% | 4.1% |

The published policy retrieval result was recall@1 91.7% and recall@3 100% across
48 receiver-filtered questions. On `data/test_no_rules`, adapted models deferred all
cases for human review, with no leaked values. See
[docs/MODEL_ADAPTATION.md](docs/MODEL_ADAPTATION.md) for the original method and caveats.

## Data sources and licences

| Source | Use | Licence |
|---|---|---|
| Team-authored synthetic documents | Receiver-specific decisions and test material | Authored by the team |
| `gretelai/gretel-pii-masking-en-v1` | English PII spans for detection | Apache-2.0 |
| `mabahboh/sitr-arabic-pii` | Arabic and English PII spans | Apache-2.0 |
| `Qwen/Qwen2.5-3B-Instruct` | Pre-trained base model | Apache-2.0 |

The project previously checked and did not use `aniket-curlscape/pii-masking-english-1k`
(licence listed as “other”) or `ai4privacy/*` (custom two-tier licence). The policy
library in `app/rules/rules.json` is team-authored.

