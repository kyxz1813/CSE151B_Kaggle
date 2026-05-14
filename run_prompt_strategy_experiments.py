import argparse
from pathlib import Path
from pprint import pprint

from baseline.baseline2_runner import run_baseline2_problem_set
from baseline.category_tagging import tag_problem_set_with_qwen
from baseline.generation import GenerationConfig
from run_baseline2 import (
    DryRunModelBundle,
    install_dry_run_generator,
    load_problem_set,
)


STRATEGY_LABELS = {
    "baseline": "baseline_weakest_prompt_chain",
    "baseline2": "baseline2_prompt_format",
    "baseline3": "baseline3_qwen_category_prompts",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt-chain strategy experiments.")
    parser.add_argument("--split", choices=["train", "val", "public", "private"], default="val")
    parser.add_argument("--public-data-path", default="data/public.jsonl")
    parser.add_argument("--private-data-path", default="data/private.jsonl")
    parser.add_argument("--output-dir", default="results/prompt_strategy_experiments")
    parser.add_argument("--comparison-csv", default="results/experiment_comparison.csv")
    parser.add_argument("--strategies", nargs="+", default=["baseline", "baseline2", "baseline3"])
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=414)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_model_bundle(args):
    if args.dry_run:
        install_dry_run_generator()
        return DryRunModelBundle()

    from baseline.modeling import ModelConfig, load_model

    model_config = ModelConfig(
        model_id=args.model_id,
        backend=args.backend,
        cache_dir=args.cache_dir,
        gpu_id=args.gpu_id,
        max_input_tokens=args.max_input_tokens,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        torch_dtype="bfloat16",
        load_in_4bit=False,
        device_map="auto",
        low_cpu_mem_usage=True,
        gpu_memory_utilization=0.85,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
        reuse_loaded=True,
    )
    return load_model(model_config)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_set = load_problem_set(args)
    score = args.score if args.score is not None else args.split != "private" and not args.dry_run
    model_bundle = load_model_bundle(args)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        do_sample=not args.no_sample,
    )

    tagged_problem_set = None
    category_tags_path = output_dir / f"{args.split}_category_tags.jsonl"

    results = {}
    for strategy_name in args.strategies:
        if strategy_name == "baseline3":
            if tagged_problem_set is None:
                tagged_problem_set, _ = tag_problem_set_with_qwen(
                    problem_set=problem_set,
                    model_bundle=model_bundle,
                    generation_config=GenerationConfig(
                        max_new_tokens=128,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        min_p=0.0,
                        do_sample=False,
                    ),
                    batch_size=args.batch_size,
                    output_jsonl_path=category_tags_path,
                    limit=args.limit,
                )
            run_problem_set = tagged_problem_set
            run_limit = None
        else:
            run_problem_set = problem_set
            run_limit = args.limit

        strategy_dir = output_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        result = run_baseline2_problem_set(
            problem_set=run_problem_set,
            model_bundle=model_bundle,
            generation_config=generation_config,
            batch_size=args.batch_size,
            limit=run_limit,
            score=score,
            strategy_name=strategy_name,
            report_label=STRATEGY_LABELS.get(strategy_name, strategy_name),
            output_jsonl_path=strategy_dir / f"{args.split}_results.jsonl",
            debug_jsonl_path=strategy_dir / f"{args.split}_debug.jsonl",
            submission_csv_path=strategy_dir / "submission.csv" if args.split == "private" else None,
            report_json_path=strategy_dir / f"{args.split}_report.json",
            comparison_csv_path=args.comparison_csv,
            experiment_name=f"prompt_chain_{strategy_name}",
            split_name=args.split,
        )
        results[strategy_name] = {
            "summary": result.report["summary"],
            "formatting": result.report["formatting"],
            "report_json_path": result.report["report_json_path"],
        }

    pprint({
        "strategies": results,
        "category_tags_path": str(category_tags_path) if "baseline3" in args.strategies else None,
        "comparison_csv": args.comparison_csv,
    })


if __name__ == "__main__":
    main()
