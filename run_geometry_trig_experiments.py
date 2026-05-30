import argparse
import csv
import importlib
import json
from pathlib import Path
from pprint import pprint

from baseline.category_experiments import (
    category_counts,
    run_category_strategy_grid,
)
from baseline.category_rules import prepare_and_save_rule_annotations
from baseline.datasets import ProblemSet, load_public_splits
from baseline.generation import GenerationConfig
from baseline.rule_guidance import register_default_ved_guidance
from baseline.ved_category_analysis import (
    build_failure_analysis_rows,
    save_failure_analysis,
)
from run_baseline2 import DryRunModelBundle, install_dry_run_generator


CATEGORY = "geometry_trig"

STRATEGIES = [
    "baseline3",
    "baseline3_adaptive_rules",
    "geometry_trig_frq",
]


def read_jsonl(path):
    rows = []
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_tag_map(path):
    if not path:
        return {}

    path = Path(path)

    if not path.exists():
        print(f"Warning: category-tags file not found: {path}")
        return {}

    return {
        str(row.get("id")): row
        for row in read_jsonl(path)
    }


def apply_category_tags(problem_set, tags_path):
    tag_map = load_tag_map(tags_path)
    records = []

    for record in problem_set.records:
        tag = tag_map.get(str(record.get("id")), {})

        primary_category = (
            tag.get("primary_category")
            or record.get("primary_category")
            or "general_math"
        )

        next_record = dict(record)
        next_record["primary_category"] = primary_category
        next_record["qwen_categories"] = (
            tag.get("qwen_categories")
            or [primary_category]
        )
        next_record["category_tag_raw_output"] = tag.get(
            "category_tag_raw_output",
            "",
        )
        next_record["category_tag_parse_ok"] = tag.get(
            "category_tag_parse_ok",
            False,
        )
        next_record["category_tag_source"] = tag.get(
            "category_tag_source",
            "loaded_tags",
        )

        records.append(next_record)

    return ProblemSet(
        f"{problem_set.name}_tagged",
        records,
    )


def register_geometry_trig_work():
    module_name = "baseline.category_work.geometry_trig"

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing baseline/category_work/geometry_trig.py. "
            "Add the geometry_trig category file before running this script."
        ) from exc

    if not hasattr(module, "register_all"):
        raise AttributeError(
            f"{module_name} must define register_all()."
        )

    module.register_all()
    register_default_ved_guidance()
    print(f"Registered: {module_name}")


def strategy_rationales():
    return {
        "baseline3": (
            "Shared Baseline 3 category prompt used as the comparison baseline."
        ),
        "baseline3_adaptive_rules": (
            "Baseline 3 prompt with deterministic derived-rule guidance injected "
            "to reduce geometry and trigonometry reasoning and formatting failures."
        ),
        "geometry_trig_frq": (
            "Geometry and trigonometry free-response workflow designed to reduce "
            "errors in angle units, quadrant signs, periodic trig solutions, "
            "formula selection, multi-answer ordering, and final-answer formatting."
        ),
    }


def strategy_reasons():
    return {
        "baseline3": (
            "Provide a shared reference prompt for comparison."
        ),
        "baseline3_adaptive_rules": (
            "Use derived rules to reduce category-specific mistakes."
        ),
        "geometry_trig_frq": (
            "Improve free-response reliability by checking the number of blanks, "
            "required answer type, formulas, units, signs, order, and precision."
        ),
    }


def write_experiment_log(strategy_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "geometry_trig_experiment_log.csv"
    rationales = strategy_rationales()
    reasons = strategy_reasons()

    baseline_result = strategy_results.get("baseline3")
    baseline_summary = (
        (baseline_result.report or {}).get("summary", {})
        if baseline_result is not None
        else {}
    )
    baseline_acc = baseline_summary.get("overall_acc")

    rows = []

    for strategy_name, result in strategy_results.items():
        summary = (result.report or {}).get("summary", {})
        after_acc = summary.get("overall_acc")

        accuracy_delta = None
        if after_acc is not None and baseline_acc is not None:
            accuracy_delta = after_acc - baseline_acc

        rows.append({
            "category": CATEGORY,
            "strategy_name": strategy_name,
            "change_made": rationales.get(strategy_name, strategy_name),
            "why_it_was_made": reasons.get(strategy_name, ""),
            "accuracy_before": baseline_acc,
            "accuracy_after": after_acc,
            "accuracy_delta_vs_baseline3": accuracy_delta,
            "n_scored": summary.get("n_scored"),
            "n_correct": summary.get("n_correct"),
            "free_form_acc": summary.get("free_form_acc"),
            "mcq_acc": summary.get("mcq_acc"),
        })

    fieldnames = [
        "category",
        "strategy_name",
        "change_made",
        "why_it_was_made",
        "accuracy_before",
        "accuracy_after",
        "accuracy_delta_vs_baseline3",
        "n_scored",
        "n_correct",
        "free_form_acc",
        "mcq_acc",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_failure_analysis(strategy_results, output_dir):
    artifacts = {}

    for strategy_name, result in strategy_results.items():
        rows = build_failure_analysis_rows(
            result.scored_rows,
            categories=(CATEGORY,),
        )

        strategy_dir = (
            Path(output_dir)
            / CATEGORY
            / "failure_analysis"
            / strategy_name
        )

        artifacts[strategy_name] = save_failure_analysis_rows(
            rows=rows,
            strategy_dir=strategy_dir,
            strategy_name=strategy_name,
        )

    return artifacts


def save_failure_analysis_rows(rows, strategy_dir, strategy_name):
    return save_failure_analysis(
        rows,
        output_dir=strategy_dir,
        prefix=f"{CATEGORY}_{strategy_name}",
    )


def add_args(parser):
    parser.add_argument(
        "--split",
        choices=["train", "val", "public"],
        default="public",
    )
    parser.add_argument(
        "--public-data-path",
        default="data/public.jsonl",
    )
    parser.add_argument(
        "--category-tags-path",
        default=(
            "results/baseline3_category_tagging/"
            "public_category_tags.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/geometry_trig_experiments",
    )
    parser.add_argument(
        "--comparison-csv",
        default="results/geometry_trig_experiment_comparison.csv",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=414,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--score",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-4B-Thinking-2507",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default="vllm",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run geometry_trig Baseline 3 strategy experiments."
        )
    )

    add_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_geometry_trig_work()

    splits = load_public_splits(
        args.public_data_path,
        val_frac=args.val_frac,
        seed=args.split_seed,
    )

    problem_set = apply_category_tags(
        splits[args.split],
        args.category_tags_path,
    )

    print("Category counts before filtering:")
    pprint(category_counts(problem_set))

    problem_set = prepare_and_save_rule_annotations(
        problem_set=problem_set,
        categories=(CATEGORY,),
        output_jsonl_path=(
            output_dir / f"{args.split}_{CATEGORY}_rule_annotations.jsonl"
        ),
        output_one_hot_csv_path=(
            output_dir / f"{args.split}_{CATEGORY}_rule_one_hot.csv"
        ),
        name=f"{problem_set.name}_{CATEGORY}_rules",
    )

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
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
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

    print()
    print("=" * 80)
    print(f"Running category: {CATEGORY}")
    print(f"Strategies: {STRATEGIES}")
    print("=" * 80)

    strategy_results = run_category_strategy_grid(
        problem_set=problem_set,
        category=CATEGORY,
        strategy_names=STRATEGIES,
        model_bundle=model_bundle,
        generation_config=generation_config,
        experiment_name=f"{args.split}_{CATEGORY}_strategy_grid",
        batch_size=args.batch_size,
        limit=args.limit,
        score=args.score,
        output_dir=output_dir,
        comparison_csv_path=args.comparison_csv,
        show_progress=True,
    )

    failure_artifacts = save_failure_analysis(
        strategy_results,
        output_dir,
    )

    experiment_log_path = write_experiment_log(
        strategy_results,
        output_dir,
    )

    summary = {
        "split": args.split,
        "category": CATEGORY,
        "category_counts": category_counts(problem_set),
        "strategies": STRATEGIES,
        "experiment_log_path": str(experiment_log_path),
        "comparison_csv": args.comparison_csv,
        "failure_artifacts": failure_artifacts,
    }

    summary_path = (
        output_dir / f"{args.split}_{CATEGORY}_experiment_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("Experiment complete")
    print("=" * 80)

    pprint(summary)


if __name__ == "__main__":
    main()
