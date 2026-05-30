import argparse
import csv
import json
from pathlib import Path
from pprint import pprint

from baseline.category_experiments import (
    category_counts,
    run_category_experiment,
    run_category_strategy_grid,
)
from baseline.category_rules import prepare_and_save_rule_annotations
from baseline.datasets import ProblemSet, load_public_splits
from baseline.generation import GenerationConfig
from baseline.rule_guidance import register_default_ved_guidance
from baseline.ved_category_analysis import (
    VED_CATEGORIES,
    build_failure_analysis_rows,
    save_failure_analysis,
)
from run_baseline2 import DryRunModelBundle, install_dry_run_generator

import baseline.category_work.calculus as calculus_work
import baseline.category_work.general_math as general_math_work


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
        return {}

    return {str(row.get("id")): row for row in read_jsonl(path)}


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
        next_record["qwen_categories"] = [primary_category]
        next_record["category_tag_raw_output"] = tag.get("category_tag_raw_output", "")
        next_record["category_tag_parse_ok"] = tag.get("category_tag_parse_ok", False)
        next_record["category_tag_source"] = tag.get("category_tag_source", "loaded_tags")
        records.append(next_record)

    return ProblemSet(f"{problem_set.name}_tagged", records)


def register_ved_category_work():
    calculus_work.register_all()
    general_math_work.register_all()
    register_default_ved_guidance()


def strategy_grid():
    return {
        "calculus": [
            "baseline3_adaptive_rules",
            "calculus_v1_structured",
            "calculus_limit_asymptotic",
            "calculus_integral",
            "calculus_derivative_extrema",
            "calculus_differential_equation",
            "baseline3",
        ],
        "general_math": [
            "baseline3_adaptive_rules",
            "general_math_v1_structured",
            "general_math_mcq_verifier",
            "general_math_multi_answer",
            "general_math_text_or_unit",
            "baseline3",
        ],
    }


def write_experiment_log(results_by_category, output_dir):
    output_dir = Path(output_dir)
    path = output_dir / "ved_experiment_log.csv"
    rows = []

    rationales = {
        "baseline3": "Shared Baseline 3 category prompt.",
        "baseline3_adaptive_rules": "Baseline 3 prompt with deterministic derived-rule guidance injected.",
        "calculus_v1_structured": "Calculus structured prompt with subtype detection and verification checklist.",
        "calculus_limit_asymptotic": "Limit-focused workflow for asymptotics, substitutions, and one-sided behavior.",
        "calculus_integral": "Integral-focused workflow with method selection, bounds, and antiderivative verification.",
        "calculus_derivative_extrema": "Derivative/extrema workflow with tangent, rate, critical point, and endpoint checks.",
        "calculus_differential_equation": "Differential-equation workflow with condition-based constant solving.",
        "general_math_v1_structured": "General fallback prompt with answer-type parsing and final answer checks.",
        "general_math_mcq_verifier": "General multiple-choice verifier that maps computed results to one option letter.",
        "general_math_multi_answer": "General multi-answer workflow that enforces blank count and answer order.",
        "general_math_text_or_unit": "General text/unit workflow with wording, unit, percent, and rate checks.",
    }

    for category, strategy_results in results_by_category.items():
        baseline_summary = (strategy_results.get("baseline3").report or {}).get("summary", {})
        baseline_acc = baseline_summary.get("overall_acc")

        for strategy_name, result in strategy_results.items():
            summary = (result.report or {}).get("summary", {})
            after_acc = summary.get("overall_acc")
            rows.append({
                "category": category,
                "strategy_name": strategy_name,
                "change_made": rationales.get(strategy_name, strategy_name),
                "why_it_was_made": "Reduce category-specific answer, formatting, and mapping failures found during validation.",
                "accuracy_before": baseline_acc,
                "accuracy_after": after_acc,
                "accuracy_delta_vs_baseline3": (
                    after_acc - baseline_acc
                    if after_acc is not None and baseline_acc is not None
                    else None
                ),
                "n_scored": summary.get("n_scored"),
                "n_correct": summary.get("n_correct"),
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
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


def save_grid_failure_analysis(results_by_category, output_dir):
    artifacts = {}
    for category, strategy_results in results_by_category.items():
        for strategy_name, result in strategy_results.items():
            rows = build_failure_analysis_rows(
                result.scored_rows,
                categories=(category,),
            )
            strategy_dir = Path(output_dir) / category / "failure_analysis" / strategy_name
            artifacts[f"{category}:{strategy_name}"] = save_failure_analysis(
                rows,
                output_dir=strategy_dir,
                prefix=f"{category}_{strategy_name}",
            )
    return artifacts


def add_args(parser):
    parser.add_argument("--split", choices=["train", "val", "public"], default="val")
    parser.add_argument("--public-data-path", default="data/public.jsonl")
    parser.add_argument("--category-tags-path", default="results/baseline3_category_tagging/public_category_tags.jsonl")
    parser.add_argument("--output-dir", default="results/ved_category_experiments")
    parser.add_argument("--comparison-csv", default="results/ved_category_experiment_comparison.csv")
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=414)
    parser.add_argument("--limit-per-category", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--no-sample", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")


def main():
    parser = argparse.ArgumentParser(description="Run Ved category Baseline 3 experiments.")
    add_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_ved_category_work()

    splits = load_public_splits(
        args.public_data_path,
        val_frac=args.val_frac,
        seed=args.split_seed,
    )
    problem_set = apply_category_tags(splits[args.split], args.category_tags_path)
    problem_set = prepare_and_save_rule_annotations(
        problem_set=problem_set,
        categories=VED_CATEGORIES,
        output_jsonl_path=output_dir / f"{args.split}_ved_rule_annotations.jsonl",
        output_one_hot_csv_path=output_dir / f"{args.split}_ved_rule_one_hot.csv",
        name=f"{problem_set.name}_ved_rules",
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

    results_by_category = {}
    for category, strategies in strategy_grid().items():
        results_by_category[category] = run_category_strategy_grid(
            problem_set=problem_set,
            category=category,
            strategy_names=strategies,
            model_bundle=model_bundle,
            generation_config=generation_config,
            experiment_name=f"{args.split}_{category}_ved_strategy_grid",
            batch_size=args.batch_size,
            limit=args.limit_per_category,
            score=args.score,
            output_dir=output_dir,
            comparison_csv_path=args.comparison_csv,
            show_progress=True,
        )

    failure_artifacts = save_grid_failure_analysis(results_by_category, output_dir)
    experiment_log_path = write_experiment_log(results_by_category, output_dir)

    summary = {
        "split": args.split,
        "category_counts": category_counts(problem_set),
        "strategy_grid": strategy_grid(),
        "experiment_log_path": str(experiment_log_path),
        "comparison_csv": args.comparison_csv,
        "failure_artifacts": failure_artifacts,
    }

    with open(output_dir / f"{args.split}_ved_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pprint(summary)


if __name__ == "__main__":
    main()
