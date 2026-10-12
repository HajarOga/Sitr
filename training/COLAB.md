# Running the training on a free Colab GPU

The application runs on a laptop with no GPU. **Training does not** — LoRA on a 3B model
needs a GPU, and the brief's constraint is that it must fit a free one.

## The fast path

`notebooks/train_on_colab.ipynb` is this whole document as runnable cells, in the order
that matters: **it measures the baseline before it trains anything**, so the "before"
column exists even if training fails or the session dies.

Upload it to Colab (File → Upload notebook), set the runtime to T4, and run top to bottom.

**The repository is private**, so the first cell needs permission to clone it. Two options,
both written into the cell:

- **Recommended:** add a GitHub token as a Colab secret named `GITHUB_TOKEN` — sidebar key
  icon, then GitHub → Settings → Developer settings → Personal access tokens → Fine-grained
  → repository access: only `safeshare2` → Permissions: Contents = Read-only.
- Or make the repository public once. Its history holds no secrets (`.env` was never
  committed; the only `sk-` ever added was a placeholder), but keeping coursework private
  is the safer choice while it is being marked.

The sections below are the same steps spelled out, for reading.

---

## 0. Get the code and the libraries

```python
!git clone https://github.com/HajarOga/safeshare2.git
%cd safeshare2
!pip install -q -r requirements.txt -r training/requirements-train.txt
```

**Expected warning, not an error:** this step prints `dependency resolver` messages naming
`google-colab`, `cudf`, `dask`, `torchvision` or `pandas`. Colab ships its own versions of
those libraries and the install moved some of them. The pandas and opentelemetry ones are
harmless.

**The torchvision one is not harmless** — it is a real trap, and it has bitten this project
once already. Colab pairs `torch` with a matching `torchvision`; installing our requirements
(which pin `torch`) leaves the two out of step, and an out-of-step torchvision makes
transformers fail to import **any** model:

```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'Qwen2ForCausalLM'
```

Every document then fails and the evaluation reports zero results without saying why. Nothing
in this project touches images, so the fix is to remove the package:

```
!pip uninstall -y torchvision
```

The notebook does this in the install step. If you are running the steps by hand, do it before
the first model call.

**The second trap is `torchao`.** If Colab's copy is older than peft expects, *loading a trained
adapter* fails with:

```
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
             but only versions above 0.16.0 are supported
```

Nothing in this project quantises with torchao, so remove it as well. The training itself is
unaffected — the adapter on disk stays valid, so this costs one pip command, not a retrain:

```
!pip uninstall -y torchao
```

Check the GPU is really there before spending time:

```python
!nvidia-smi
```

You want to see `Tesla T4` and about 15 GB. If it says no GPU, the runtime type is wrong.

---

## 1. Build the data (no GPU needed, a minute)

```python
!python -m training.make_training_documents      # 24 documents + labels, and it prints
                                                 # the overlap check against the test set
!python -m training.prepare_data --skip-public   # prompt/completion pairs
!python -m training.make_preference_pairs        # chosen/rejected pairs for the alignment
```

Leave `--skip-public` off if you want the two public datasets in the mix; that step
downloads from Hugging Face and needs `datasets` (installed above):

- `gretelai/gretel-pii-masking-en-v1` — 50,000 English documents, **Apache-2.0**
- `mabahboh/sitr-arabic-pii` — 14,288 Arabic documents, **Apache-2.0**, sampled at 2,500

Their labels are not our twelve categories, so `prepare_data.py` maps them and drops what
does not fit. Run it once with small limits to see the counts before a long run:

```python
!python -m training.prepare_data --english-limit 60 --arabic-limit 60
```

Look at what it built before training on it — one row, either side:

```python
import json
print(json.loads(open('data/train/preference_pairs.jsonl').readline())['rejected'][:600])
```

---

## 2. Fine-tune (LoRA / QLoRA) — about 20 minutes on a T4

```python
!python -m training.train_lora
```

It prints the trainable-parameter count first — expect roughly **1.7% of the model**, the
same figure the LoRA lab produced — and saves `training/lora_adapter`.

---

## 3. Align (DPO) — about 25 minutes on a T4

```python
!python -m training.train_dpo
```

It starts from `training/lora_adapter` when that exists, and saves `training/dpo_adapter`.

---

## 4. Measure all three states on the same test set

This is the before/after evidence for the report. Run the three in order; each writes a
predictions file, and the last cell prints the comparison table.

```python
# the base model, untouched
!SAFESHARE2_PROVIDER=local SAFESHARE2_ALLOW_DOWNLOAD=1 python -m evaluation.evaluate --live --out predictions/base.jsonl

# after fine-tuning
!SAFESHARE2_PROVIDER=local SAFESHARE2_ADAPTER=training/lora_adapter python -m evaluation.evaluate --live --out predictions/lora.jsonl

# after alignment
!SAFESHARE2_PROVIDER=local SAFESHARE2_ADAPTER=training/dpo_adapter python -m evaluation.evaluate --live --out predictions/dpo.jsonl

!python -m evaluation.evaluate --compare predictions/base.jsonl predictions/lora.jsonl predictions/dpo.jsonl
```

Copy the table out of Colab **and keep the three `.jsonl` files** — they download from the
file browser, and the scoring runs anywhere afterwards, including a laptop with no GPU:

```python
!python -m evaluation.evaluate --score predictions/lora.jsonl
```

So if Colab disconnects after training, nothing is lost: the adapters and the predictions
files are the only things worth saving. Download `training/lora_adapter` and
`training/dpo_adapter` too — they are tens of MB, not gigabytes.

---

## Save the adapter before anything else

A Colab runtime is a **fresh machine every session**. If it disconnects, `/content` is wiped:
the code, the 6 GB model, the training data and the adapter all vanish. This has already cost
one 21-minute training run that had completed successfully.

So, in order of preference:

1. **Do the training and the measurement in one cell, back to back.** A running cell keeps the
   session alive, and there is no idle gap for Google to reclaim.
2. **Download the adapter the moment training ends** — `files.download` works whether or not
   Drive is mounted:
   ```python
   import shutil
   shutil.make_archive('/content/adapter', 'zip', '.', 'training/lora_adapter')
   from google.colab import files
   files.download('/content/adapter.zip')
   ```
3. Only rely on Drive if the mount actually succeeded. The notebook prints *"adapter saved to
   Drive"* in older copies even when the copy failed — check for the file, do not trust the
   message.

## Notes that save time

- **Free Colab disconnects.** Save the adapter as soon as training ends, and keep the
  notebook open in a tab; the runtime is dropped after idle time.
- **The first model download is ~6 GB** for a 3B model at 4-bit plus the tokenizer. It
  happens once per session.
- **Out of memory?** Lower `--max-len` to 1536, or `--batch-size 1 --grad-accum 16`. The
  prompt carries the whole policy block, so trimming it too far starts costing accuracy —
  check the evaluation numbers, do not assume.
- **The `CUSTOM` receiver has no rules**, so every item comes back `ASK_HUMAN`. That is
  correct behaviour, and it is why the training data includes it: a receiver with no
  written policy must not get a guessed decision.
- **Nothing here trains on the evaluation set.** `data/test/` is read by `evaluation/`
  only, and `make_training_documents.py` refuses values that appear in it. The check runs
  every time and prints its result.
