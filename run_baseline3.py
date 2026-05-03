import json
from pathlib import Path
from pprint import pprint

from baseline.baseline2_runner import run_baseline3_problem_set
from baseline.generation import GenerationConfig
from run_baseline2 import (
    DryRunModelBundle,
    install_dry_run_generator,
    load_problem_set,
    parse_args,
)


def main():
    args = parse_args(
        description="Run Baseline 3 category-prompt inference.",
        default_output_dir="results/baseline3_category_prompts",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_set = load_problem_set(args)
    score = args.score if args.score is not None else args.split != "private" and not args.dry_run

    if args.dry_run:
        install_dry_run_generator()
        model_bundle = DryRunModelBundle()
    else:
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
        model_bundle = load_model(model_config)

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

    result = run_baseline3_problem_set(
        problem_set=problem_set,
        model_bundle=model_bundle,
        generation_config=generation_config,
        batch_size=args.batch_size,
        limit=args.limit,
        score=score,
        output_jsonl_path=output_dir / f"{args.split}_results.jsonl",
        debug_jsonl_path=output_dir / f"{args.split}_debug.jsonl",
        submission_csv_path=output_dir / "submission.csv" if args.split == "private" else None,
        report_json_path=output_dir / f"{args.split}_report.json",
    )

    printed_report = {
        "summary": result.report["summary"],
        "formatting": result.report["formatting"],
        "category_counts": result.report["category_counts"],
        "category_summary": result.report["category_summary"],
        "output_jsonl_path": result.report["output_jsonl_path"],
        "debug_jsonl_path": result.report["debug_jsonl_path"],
        "submission_csv_path": result.report["submission_csv_path"],
    }

    baseline2_report_path = Path("results") / "baseline2_prompt_format" / f"{args.split}_report.json"
    if baseline2_report_path.exists():
        with open(baseline2_report_path, "r", encoding="utf-8") as f:
            baseline2_report = json.load(f)
        printed_report["baseline2_summary"] = baseline2_report.get("summary")
        printed_report["baseline2_formatting"] = baseline2_report.get("formatting")

    pprint(printed_report)


if __name__ == "__main__":
    main()
