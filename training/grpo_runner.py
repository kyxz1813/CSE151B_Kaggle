import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOTrainer

from prompting.models import problem_from_record
from prompting.prompt_chain import build_prompt_chain
from training.rewarding import combined_reward_func, format_reward_func


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_grpo_dataset(records, strategy_name="baseline3_adaptive_rules"):
    chain = build_prompt_chain(strategy_name=strategy_name)
    rows = []
    for record in records:
        if record.get("answer") is None:
            continue
        spec = chain.build_spec(problem_from_record(record))
        rows.append({
            "prompt": spec.to_messages(),
            "record": json.dumps(record, ensure_ascii=False),
            "id": str(record.get("id")),
            "answer": json.dumps(record.get("answer"), ensure_ascii=False),
            "question": record.get("question"),
            "options": json.dumps(record.get("options"), ensure_ascii=False),
            "primary_category": record.get("primary_category") or "unknown",
        })
    return Dataset.from_list(rows)


def build_lora_config(args):
    return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def build_grpo_config(args):
    try:
        from trl import GRPOConfig
        return GRPOConfig(output_dir=args.output_dir, max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length, per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, num_generations=args.num_generations, num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate, logging_steps=args.logging_steps, save_steps=args.save_steps, bf16=bool(args.bf16 and torch.cuda.is_available()), fp16=bool((not args.bf16) and torch.cuda.is_available()), report_to="none", seed=args.seed)
    except Exception:
        from transformers import TrainingArguments
        return TrainingArguments(output_dir=args.output_dir, per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, num_train_epochs=args.num_train_epochs, learning_rate=args.learning_rate, logging_steps=args.logging_steps, save_steps=args.save_steps, bf16=bool(args.bf16 and torch.cuda.is_available()), fp16=bool((not args.bf16) and torch.cuda.is_available()), report_to="none", seed=args.seed)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO runner scaffold for CSE151B math reasoning.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", default="results/grpo/qwen3_4b_math_grpo")
    parser.add_argument("--strategy-name", default="baseline3_adaptive_rules")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reward-mode", choices=["combined", "format"], default="combined")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=414)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.records_jsonl)
    if args.limit:
        records = records[:args.limit]
    train_dataset = build_grpo_dataset(records, strategy_name=args.strategy_name)
    reward_funcs = [format_reward_func] if args.reward_mode == "format" else [combined_reward_func]
    trainer = GRPOTrainer(model=args.model_id, reward_funcs=reward_funcs, args=build_grpo_config(args), train_dataset=train_dataset, peft_config=build_lora_config(args))
    trainer.train()
    trainer.save_model(args.output_dir)
    run_info = vars(args)
    run_info["n_records"] = len(records)
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print(json.dumps(run_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
