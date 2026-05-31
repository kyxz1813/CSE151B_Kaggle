import argparse
import json
from pathlib import Path
from pprint import pprint

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from baseline.datasets import ProblemSet
from baseline.generation import GenerationConfig
from baseline.baseline2_runner import run_baseline3_problem_set
from baseline.modeling import ModelConfig, ModelBundle


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_problem_set(path, name):
    return ProblemSet(name, read_jsonl(path))


def load_adapter_bundle(args):
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = {"trust_remote_code": True, "device_map": "auto", "cache_dir": args.cache_dir, "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.base_model_id, **kwargs)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    config = ModelConfig(model_id=args.base_model_id, backend="transformers", cache_dir=args.cache_dir, gpu_id=args.gpu_id, max_input_tokens=args.max_input_tokens, max_model_len=args.max_input_tokens, dtype="bfloat16" if args.bf16 else "float16", torch_dtype="bfloat16" if args.bf16 else "float16", load_in_4bit=args.load_in_4bit, device_map="auto", low_cpu_mem_usage=True)
    return ModelBundle(config=config, tokenizer=tokenizer, model=model)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PEFT adapter using existing Baseline 3 runner.")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--output-dir", default="results/sft_eval")
    parser.add_argument("--name", default="sft_eval")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strategy-name", default="baseline3_adaptive_rules")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_set = load_problem_set(args.records_jsonl, args.name)
    model_bundle = load_adapter_bundle(args)
    gen = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, do_sample=True)
    retry_gen = GenerationConfig(max_new_tokens=1024, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, do_sample=False)
    result = run_baseline3_problem_set(problem_set=problem_set, model_bundle=model_bundle, generation_config=gen, retry_generation_config=retry_gen, batch_size=args.batch_size, limit=args.limit, score=args.score, strategy_name=args.strategy_name, report_label=f"sft_adapter_{args.strategy_name}", output_jsonl_path=output_dir / f"{args.name}_results.jsonl", debug_jsonl_path=output_dir / f"{args.name}_debug.jsonl", report_json_path=output_dir / f"{args.name}_report.json", show_progress=True)
    pprint(result.report["summary"])
    pprint({"output_jsonl_path": result.report["output_jsonl_path"], "debug_jsonl_path": result.report["debug_jsonl_path"], "report_json_path": result.report["report_json_path"]})


if __name__ == "__main__":
    main()
