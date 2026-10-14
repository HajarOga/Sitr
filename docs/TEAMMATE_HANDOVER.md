# سِتر (Sitr) — handover for the teammate with the GPU

You're setting up **سِتر (Sitr)** — formerly SafeShare — our capstone app: it reads a document, finds the personal
data in it, and decides — per receiver — what is kept, replaced, removed, or asked about by a
human, citing the rule it applied. The decision model is Qwen2.5-3B-Instruct, adapted with
LoRA (supervised fine-tuning) and DPO (preference alignment).

Repo: **https://github.com/HajarOga/sitr**

---

## 1 · What you need

- Python 3.10 or newer
- An NVIDIA GPU with **8 GB+ VRAM** is ideal. CPU works but is slow (minutes per document)
- ~15 GB free disk: the base model is ~6 GB, plus the environment
- Internet for the first run only (to download the base model)

## 2 · Set up

```bash
git clone https://github.com/HajarOga/sitr
cd sitr
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r training/requirements-train.txt   # needed to train
```

## 3 · Get the model — two cases

**Case A — the adapters are on Hugging Face** (Hajar will send the repo link):

```bash
pip install huggingface_hub
hf download <user>/sitr-qwen2.5-3b-lora-dpo --local-dir training
```

Then `training/lora_adapter/` and `training/dpo_adapter/` exist, each containing
`adapter_config.json` and `adapter_model.safetensors`.

**Case B — the adapters have to be rebuilt** (the Colab runtime was wiped before they were
saved). Train them here instead — a GPU makes this ~20 min + ~50 min. The exact sequence,
including the data-build flags, is in **`training/COLAB.md`**:

```bash
python -m training.make_training_documents     # 24 documents + labels, prints the overlap check
python -m training.prepare_data                # writes the supervised fine-tuning pairs
python -m training.train_lora                  # ~20 min on a T4-class GPU
python -m training.make_preference_pairs       # builds the chosen/rejected pairs
python -m training.train_dpo --low-memory      # ~50 min; starts from training/lora_adapter
```

Then **publish them immediately**, so they can never be lost the way the Colab ones were:

```bash
pip install huggingface_hub
hf auth login                                  # paste a token with WRITE access
python training/upload_adapters_to_hf.py       # creates the HF repo + model card, uploads both
```

It prints the repo link. Send that link to the team.

## 4 · Run the app

```bash
export SAFESHARE2_PROVIDER=local
export SAFESHARE2_ADAPTER=training/dpo_adapter     # the aligned model — use this one
python -m streamlit run app/ui_streamlit.py
```

Windows PowerShell uses `$env:SAFESHARE2_PROVIDER = "local"` instead of `export`.

Opens at **http://localhost:8501**. First document takes 30–60 s while the model loads into
memory; after that it's ~10–30 s each.

**Demo script:** Upload → choose the receiver **External Vendor** → **Analyze** → **Review** →
**Generate release** → **Audit log**. Then repeat the same document with a different receiver
to show that the same personal data gets a different decision — that's the point of the app.

Sample documents are in `data/test/` (e.g. `complaint-en`, `kyc-form-ar`).

## 5 · If there's no GPU available

Use the API instead — no model download, no adapter:

```bash
# in .env:  DEEPSEEK_API_KEY=<a working key>
python -m streamlit run app/ui_streamlit.py
```

The app runs the same way; it just sends the document to the API instead of a local model.
Be aware when choosing: the document text leaves the machine in API mode, which is exactly
what the offline model avoids.

## 6 · Two things that will bite you

- **The adapter only works with the local model.** It cannot attach to the API. Don't set
  `SAFESHARE2_ADAPTER` together with API mode.
- **A disconnected Colab runtime wipes `/content`** and takes the trained adapter with it.
  That is what happened to ours. Always publish to the Hub immediately after training.
