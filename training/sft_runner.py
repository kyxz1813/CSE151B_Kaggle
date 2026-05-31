import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA SFT runner for the CSE151B Qwen math competition model.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--output-dir", default="results/sft/qwen3_4b_math_sft")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=414)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_tokenizer(model_id, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(args):
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16
    kwargs = {"cache_dir": args.cache_dir, "trust_remote_code": True, "device_map": "auto", "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    return model


def build_lora_config(args):
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def build_sft_config(args):
    try:
        from trl import SFTConfig
        kwargs = {
            "output_dir": args.output_dir,
            "max_length": args.max_seq_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "bf16": bool(args.bf16 and torch.cuda.is_available()),
            "fp16": bool((not args.bf16) and torch.cuda.is_available()),
            "seed": args.seed,
            "report_to": "none",
            "save_total_limit": 2,
        }
        if args.eval_jsonl:
            kwargs["eval_strategy"] = "steps"
        try:
            return SFTConfig(**kwargs)
        except TypeError:
            if "eval_strategy" in kwargs:
                kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
            if "max_length" in kwargs:
                kwargs["max_seq_length"] = kwargs.pop("max_length")
            return SFTConfig(**kwargs)
    except Exception:
        from transformers import TrainingArguments
        return TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            evaluation_strategy="steps" if args.eval_jsonl else "no",
            bf16=bool(args.bf16 and torch.cuda.is_available()),
            fp16=bool((not args.bf16) and torch.cuda.is_available()),
            seed=args.seed,
            report_to="none",
            save_total_limit=2,
        )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_id, cache_dir=args.cache_dir)
    model = load_model(args)
    peft_config = build_lora_config(args)
    data_files = {"train": args.train_jsonl}
    if args.eval_jsonl:
        data_files["validation"] = args.eval_jsonl
    dataset = load_dataset("json", data_files=data_files)
    sft_config = build_sft_config(args)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    run_info = vars(args)
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print(json.dumps(run_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
