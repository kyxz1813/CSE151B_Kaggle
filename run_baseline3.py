import json
from pathlib import Path
from pprint import pprint

from baseline.category_tagging import tag_problem_set_with_qwen
from baseline.baseline2_runner import run_baseline3_problem_set
from baseline.generation import GenerationConfig
from baseline.runner import write_report
from run_baseline2 import (
    DryRunModelBundle,
    install_dry_run_generator,
    load_problem_set,
    parse_args,
)


def add_baseline3_args(parser):
    parser.add_argument("--category-tags-path", default=None)
    parser.add_argument("--reuse-category-tags", action="store_true")
    parser.add_argument("--category-tag-batch-size", type=int, default=None)
    parser.add_argument("--category-tag-max-new-tokens", type=int, default=128)


def main():
    args = parse_args(
        description="Run Baseline 3 category-prompt inference.",
        default_output_dir="results/baseline3_category_prompts",
        extra_args_fn=add_baseline3_args,
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

    category_tags_path = Path(args.category_tags_path) if args.category_tags_path else output_dir / f"{args.split}_category_tags.jsonl"
    existing_tags_path = category_tags_path if args.reuse_category_tags and category_tags_path.exists() else None
    category_generation_config = GenerationConfig(
        max_new_tokens=args.category_tag_max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        do_sample=False,
    )
    tagged_problem_set, category_tag_rows = tag_problem_set_with_qwen(
        problem_set=problem_set,
        model_bundle=model_bundle,
        generation_config=category_generation_config,
        batch_size=args.category_tag_batch_size or args.batch_size,
        output_jsonl_path=category_tags_path,
        existing_tags_path=existing_tags_path,
        limit=args.limit,
    )

    report_json_path = output_dir / f"{args.split}_report.json"
    result = run_baseline3_problem_set(
        problem_set=tagged_problem_set,
        model_bundle=model_bundle,
        generation_config=generation_config,
        batch_size=args.batch_size,
        limit=None,
        score=score,
        output_jsonl_path=output_dir / f"{args.split}_results.jsonl",
        debug_jsonl_path=output_dir / f"{args.split}_debug.jsonl",
        submission_csv_path=output_dir / "submission.csv" if args.split == "private" else None,
        report_json_path=report_json_path,
        comparison_csv_path=args.comparison_csv,
        experiment_name="baseline3_qwen_category_prompts",
        split_name=args.split,
    )
    result.report["category_tags_path"] = str(category_tags_path)
    result.report["category_tag_count"] = len(category_tag_rows)
    write_report(result.report, report_json_path)

    printed_report = {
        "summary": result.report["summary"],
        "formatting": result.report["formatting"],
        "category_counts": result.report["category_counts"],
        "category_summary": result.report["category_summary"],
        "category_tags_path": str(category_tags_path),
        "category_tag_count": len(category_tag_rows),
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
