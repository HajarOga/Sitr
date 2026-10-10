# Running the training on a free Colab GPU

The application runs on a laptop with no GPU. **Training does not** — LoRA on a 3B model
needs a GPU, and the brief's constraint is that it must fit a free one. Follow these steps
in order; each block is one Colab cell.

Set the runtime first: **Runtime → Change runtime type → T4 GPU**.

---

## 0. Get the code and the libraries

```python
!git clone https://github.com/HajarOga/safeshare2.git
%cd safeshare2
!pip install -q -r requirements.txt -r training/requirements-train.txt
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
downloads from Hugging Face and needs `datasets` (installed above).

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
