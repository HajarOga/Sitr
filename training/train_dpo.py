"""Align the model with DPO - run this on a free GPU, after train_lora.py.

    python -m training.train_dpo
    python -m training.train_dpo --adapter training/lora_adapter --out training/dpo_adapter

This is the alignment step of the adaptation path: **DPO - preferences, tone, safety**.
Fine-tuning (the step before this) taught the model this app's task and format. This step
teaches it the judgement: when the policy is silent, ask a person instead of answering;
cite the rule that exists rather than a rule that sounds right; never keep a value the
policy says must go.

The pairs come from `training/make_preference_pairs.py`, and every pair carries the
`kind` of mistake it is correcting - so the alignment set can be audited, and a specific
weakness (say, invented rule ids) can be trained on its own.

Read the table in `docs/MODEL_ADAPTATION.md` §3 before choosing the numbers here: DPO is
sensitive to beta, and the useful range for this task is narrow.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def length_report(rows: list[dict], tokenizer, max_len: int) -> tuple[list[int], list[str]]:
    """How long each preference pair is, in tokens: the prompt plus the longer answer.

    Worth measuring before loading the model, because the failure mode is invisible otherwise.
    A pair longer than the budget is dropped by the trainer ("Dropping fully truncated
    examples"), and if *every* pair is dropped the run dies deep inside the trainer with an
    error that says nothing about token budgets. The policy block alone is about 1,200 tokens,
    which is most of a small budget before any answer is added.
    """
    totals, too_long = [], []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                               add_generation_prompt=True)
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        chosen = len(tokenizer(row["chosen"], add_special_tokens=False)["input_ids"])
        rejected = len(tokenizer(row["rejected"], add_special_tokens=False)["input_ids"])
        total = prompt_tokens + max(chosen, rejected)
        totals.append(total)
        if total > max_len:
            too_long.append(f"{row.get('document', '?')}/{row.get('receiver', '?')}: {total}")
    return totals, too_long


def supported_kwargs(config_class, kwargs: dict) -> tuple[dict, dict]:
    """Keep only the arguments this installed version of the trainer accepts.

    Trainer configuration APIs drift between releases, and a single renamed argument stops a
    half-hour GPU run at the last line before it starts - `warmup_ratio` was accepted by one
    version of TRL's DPOConfig and not by the next. So the settings are offered as a dictionary
    and filtered against the class actually installed, and anything dropped is reported rather
    than silently ignored.
    """
    import inspect                                                   # noqa: PLC0415
    parameters = inspect.signature(config_class.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs), {}                       # accepts anything: pass it all through
    kept = {key: value for key, value in kwargs.items() if key in parameters}
    dropped = {key: value for key, value in kwargs.items() if key not in parameters}
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO alignment for SafeShare")
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=str(ROOT / "training" / "lora_adapter"),
                        help="the LoRA adapter from train_lora.py, if it exists")
    parser.add_argument("--pairs", default=str(ROOT / "data" / "train" / "preference_pairs.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "training" / "dpo_adapter"))
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--low-memory", action="store_true",
                        help="compute the reference model's log-probabilities up front and "
                             "free it, halving peak memory during training at the cost of a "
                             "pre-pass over the data. This is the setting to reach for when a "
                             "T4 runs out of memory in DPO")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-prompt-len", type=int, default=1536,
                        help="the prompt carries the whole policy block; the answer needs "
                             "the rest of the budget")
    args = parser.parse_args()

    import torch                                                     # noqa: PLC0415
    from datasets import Dataset                                     # noqa: PLC0415
    from peft import LoraConfig, PeftModel                           # noqa: PLC0415
    from transformers import (AutoModelForCausalLM, AutoTokenizer,    # noqa: PLC0415
                              BitsAndBytesConfig)
    from trl import DPOConfig, DPOTrainer                            # noqa: PLC0415

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise SystemExit(f"{pairs_path} not found. Run first:\n"
                         f"  python -m training.make_preference_pairs")
    rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit(f"{pairs_path} is empty")

    kinds = Counter(row.get("kind", "?") for row in rows)
    print(f"{len(rows)} preference pairs:")
    for kind, count in kinds.most_common():
        print(f"  {count:>4}  {kind}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fail before the model is loaded if the budget cannot hold the data.
    totals, too_long = length_report(rows, tokenizer, args.max_len)
    median = sorted(totals)[len(totals) // 2]
    print(f"pair length: median {median} tokens, longest {max(totals)}, "
          f"budget --max-len {args.max_len}")
    if len(too_long) == len(rows):
        raise SystemExit(
            f"every one of the {len(rows)} pairs is longer than --max-len {args.max_len} "
            f"(median {median} tokens), so the trainer would drop all of them and fail with "
            f"an error about the dataset instead. This is the fix:\n\n"
            f"    python -m training.train_dpo --max-len {max(totals) + 256}\n")
    if too_long:
        print(f"warning: {len(too_long)} of {len(rows)} pairs are over budget and will be "
              f"truncated, for example {too_long[0]}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    if Path(args.adapter).exists():
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print(f"starting from the fine-tuned adapter: {args.adapter}")
    else:
        from peft import get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                            "up_proj", "down_proj"]))
        print("no adapter found - aligning the base model directly")

    def render(row: dict) -> dict:
        """DPO wants the prompt, the preferred answer and the rejected answer."""
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                               add_generation_prompt=True)
        return {"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]}

    dataset = Dataset.from_list([render(row) for row in rows])

    settings = {
        "output_dir": args.out, "beta": args.beta, "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum, "learning_rate": args.lr,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.1, "warmup_steps": 10,
        "fp16": True, "bf16": False, "logging_steps": 5, "save_strategy": "epoch",
        "save_total_limit": 1, "report_to": "none",
        "max_length": args.max_len, "max_prompt_length": args.max_prompt_len,
        "gradient_checkpointing": True, "remove_unused_columns": False, "seed": 42,
        # only applied when asked for, and only if this TRL knows the argument
        "precompute_ref_log_probs": bool(args.low_memory),
    }
    kept, dropped = supported_kwargs(DPOConfig, settings)
    if dropped:
        print("this version of TRL does not accept, so these were left out:")
        for key in sorted(dropped):
            print(f"  {key}")
    config = DPOConfig(**kept)

    trainer = DPOTrainer(model=model, args=config, train_dataset=dataset,
                         processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\naligned adapter saved -> {args.out}")
    print("then measure all three states on the same test set:")
    print("  python -m evaluation.evaluate --live --out predictions/base.jsonl")
    print("  SAFESHARE2_ADAPTER=training/lora_adapter python -m evaluation.evaluate --live "
          "--out predictions/lora.jsonl")
    print("  SAFESHARE2_ADAPTER=training/dpo_adapter  python -m evaluation.evaluate --live "
          "--out predictions/dpo.jsonl")
    print("  python -m evaluation.evaluate --compare predictions/base.jsonl "
          "predictions/lora.jsonl predictions/dpo.jsonl")


if __name__ == "__main__":
    main()
