"""Fine-tune the decision model with LoRA / QLoRA - run this on a free GPU.

    python -m training.train_lora                       # Qwen2.5-3B, 4-bit (QLoRA)
    python -m training.train_lora --base Qwen/Qwen2.5-0.5B-Instruct --no-four-bit

This is the fine-tuning step of the adaptation path: **LoRA / QLoRA - task, format,
domain**. It teaches the model this application's task in this application's output
format, and nothing else. The policy is not learned - it is still handed to the model in
the prompt at run time, rule by rule - so a policy change never needs retraining.

Not runnable on a laptop CPU in reasonable time. On a free Colab T4:

    !pip install -q -r requirements.txt peft trl bitsandbytes
    !python -m training.make_training_documents
    !python -m training.prepare_data --skip-public
    !python -m training.train_lora

Two decisions worth knowing:

* **Only the answer is trained on.** Prompt tokens are set to -100, so the model is not
  rewarded for copying the document and the policy back at us - the loss is entirely on
  the JSON it should produce.
* **Only the adapter is saved** (a few tens of MB). The base weights are untouched, which
  is what makes this LoRA rather than a full fine-tune, and why the demo can swap between
  the base model and the adapted one for a before/after comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IGNORE = -100


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run first:\n"
                         f"  python -m training.make_training_documents\n"
                         f"  python -m training.prepare_data --skip-public")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for Sitr")
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct",
                        help="any Hugging Face chat model; the app reads the same name "
                             "from SAFESHARE2_MODEL")
    parser.add_argument("--train", default=str(ROOT / "data" / "train" / "sft.jsonl"))
    parser.add_argument("--validation", default=str(ROOT / "data" / "validation" / "sft.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "training" / "lora_adapter"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048,
                        help="prompt + answer. The policy block is long, so keep this "
                             "generous on a T4 (2048 costs about 10 GB at 4-bit)")
    parser.add_argument("--no-four-bit", action="store_true",
                        help="plain fp16 LoRA (only for a model that fits without 4-bit)")
    args = parser.parse_args()

    import torch                                                     # noqa: PLC0415
    from datasets import Dataset                                     # noqa: PLC0415
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import (AutoModelForCausalLM, AutoTokenizer,    # noqa: PLC0415
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    rows = load_rows(Path(args.train))
    print(f"{len(rows)} training rows from {args.train}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: dict = {"device_map": "auto", "torch_dtype": torch.float16}
    if not args.no_four_bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    if not args.no_four_bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    def tokenise(row: dict) -> dict:
        """Prompt masked, answer trained on."""
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                               add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(row["completion"] + tokenizer.eos_token,
                               add_special_tokens=False)["input_ids"]
        # keep the whole policy block: trim the answer first, then the prompt tail
        budget = args.max_len
        if len(answer_ids) >= budget:
            answer_ids = answer_ids[:budget // 2]
        if len(prompt_ids) + len(answer_ids) > budget:
            prompt_ids = prompt_ids[:budget - len(answer_ids)]
        ids = prompt_ids + answer_ids
        return {"input_ids": ids, "labels": [IGNORE] * len(prompt_ids) + answer_ids,
                "attention_mask": [1] * len(ids)}

    dataset = Dataset.from_list(rows).map(
        tokenise, remove_columns=["messages", "completion", "document", "receiver", "source"])

    config = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                        bias="none", task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Mixed precision has to match the parameters' own dtype, or torch's fp16 gradient scaler
    # fails in its first clip-grad-norm step with
    #   NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented
    #   for 'BFloat16'
    # A 4-bit load keeps the checkpoint's bfloat16, and a T4 has no native bf16, so the
    # trainable parameters are moved to fp16 rather than emulating bf16 arithmetic.
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    trainable_dtype = (trainable[0][1].dtype if trainable
                       else next(model.parameters()).dtype)
    if trainable_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print(f"trainable parameters are {trainable_dtype} and this card has no native bf16: "
              f"moving them to fp16")
        for _, parameter in trainable:
            parameter.data = parameter.data.to(torch.float16)
        trainable_dtype = torch.float16
    use_bf16 = trainable_dtype == torch.bfloat16
    print(f"training precision: {'bf16' if use_bf16 else 'fp16' if trainable_dtype == torch.float16 else 'none'}")

    class Collator:
        def __call__(self, features):
            length = max(len(f["input_ids"]) for f in features)
            batch = {"input_ids": [], "labels": [], "attention_mask": []}
            for feature in features:
                gap = length - len(feature["input_ids"])
                batch["input_ids"].append(feature["input_ids"] + [tokenizer.pad_token_id] * gap)
                batch["labels"].append(feature["labels"] + [IGNORE] * gap)
                batch["attention_mask"].append(feature["attention_mask"] + [0] * gap)
            return {key: torch.tensor(value) for key, value in batch.items()}

    arguments = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=10,
        fp16=trainable_dtype == torch.float16, bf16=use_bf16,
        logging_steps=5, save_strategy="epoch", save_total_limit=1,
        report_to="none", remove_unused_columns=False, seed=42,
        gradient_checkpointing=True)

    trainer = Trainer(model=model, args=arguments, train_dataset=dataset,
                      data_collator=Collator())
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nadapter saved -> {args.out}")
    print("evaluate before and after with:")
    print("  python -m evaluation.evaluate --live --out predictions_base.jsonl")
    print(f"  SAFESHARE2_ADAPTER={args.out} python -m evaluation.evaluate --live "
          f"--out predictions_lora.jsonl")
    print("  python -m evaluation.evaluate --compare predictions_base.jsonl predictions_lora.jsonl")


if __name__ == "__main__":
    main()
