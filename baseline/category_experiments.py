import csv
import json
import time
from pathlib import Path

from prompting.strategies import BASELINE3_CATEGORIES

from .baseline2_runner import run_baseline2_problem_set
from .category_harnesses import get_harness, apply_harness, summarize_harness
from .datasets import ProblemSet, save_jsonl
from .runner import save_submission_csv, write_report
from .scoring import summarize_results


def _safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "_")


def filter_problem_set_by_category(problem_set, category, name=None):
    records = [
        record for record in problem_set.records
        if (record.get("primary_category") or "general_math") == category
    ]

    return ProblemSet(
        name or f"{problem_set.name}_{category}",
        records,
    )


def category_counts(problem_set):
    counts = {}

    for record in problem_set.records:
        category = record.get("primary_category") or "general_math"
        counts[category] = counts.get(category, 0) + 1

    return dict(sorted(counts.items()))


def append_comparison_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists()

    fieldnames = sorted(row.keys())

    if exists:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []

        fieldnames = sorted(set(old_fields) | set(fieldnames))

        rows = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for old_row in reader:
                rows.append(old_row)

        rows.append(row)

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in rows:
                writer.writerow(item)

    else:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


def build_category_comparison_row(
    experiment_name,
    category,
    strategy_name,
    result,
    output_dir=None,
):
    report = result.report or {}
    summary = report.get("summary") or result.summary or {}
    formatting = report.get("formatting") or {}
    harness = report.get("harness") or {}
    problem_set = report.get("problem_set") or {}

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": experiment_name,
        "category": category,
        "strategy_name": strategy_name,
        "problem_set_name": problem_set.get("name"),
        "n": problem_set.get("n"),
        "n_scored": summary.get("n_scored"),
        "overall_acc": summary.get("overall_acc"),
        "mcq_acc": summary.get("mcq_acc"),
        "free_form_acc": summary.get("free_form_acc"),
        "n_correct": summary.get("n_correct"),
        "schema_valid_rate": formatting.get("schema_valid_rate"),
        "extractable_rate": formatting.get("extractable_rate"),
        "formatting_failure_rate": formatting.get("formatting_failure_rate"),
        "retry_rate": formatting.get("retry_rate"),
        "retry_used_rate": formatting.get("retry_used_rate"),
        "harness_avg_pass_rate": harness.get("harness_avg_pass_rate"),
        "harness_full_pass_rate": harness.get("harness_full_pass_rate"),
        "output_dir": str(output_dir) if output_dir else None,
        "output_jsonl_path": report.get("output_jsonl_path"),
        "debug_jsonl_path": report.get("debug_jsonl_path"),
        "submission_csv_path": report.get("submission_csv_path"),
    }


def run_category_experiment(
    problem_set,
    category,
    model_bundle,
    generation_config,
    retry_generation_config=None,
    strategy_name="baseline3",
    experiment_name=None,
    batch_size=32,
    limit=None,
    score=True,
    output_dir=None,
    comparison_csv_path=None,
    harness=None,
    show_progress=True,
):
    experiment_name = experiment_name or f"{category}_{strategy_name}"
    subset = filter_problem_set_by_category(problem_set, category)

    if limit is not None:
        subset = subset.head(limit)

    output_dir = Path(output_dir) if output_dir else None

    if output_dir:
        category_dir = output_dir / _safe_name(experiment_name) / _safe_name(category) / _safe_name(strategy_name)
        category_dir.mkdir(parents=True, exist_ok=True)

        output_jsonl_path = category_dir / "results.jsonl"
        debug_jsonl_path = category_dir / "debug.jsonl"
        report_json_path = category_dir / "report.json"
    else:
        category_dir = None
        output_jsonl_path = None
        debug_jsonl_path = None
        report_json_path = None

    result = run_baseline2_problem_set(
        problem_set=subset,
        model_bundle=model_bundle,
        generation_config=generation_config,
        retry_generation_config=retry_generation_config,
        batch_size=batch_size,
        limit=None,
        score=score,
        strategy_name=strategy_name,
        report_label=experiment_name,
        output_jsonl_path=output_jsonl_path,
        debug_jsonl_path=debug_jsonl_path,
        report_json_path=None,
        show_progress=show_progress,
    )

    harness = harness or get_harness(category)
    apply_harness(subset.records, result.scored_rows, harness)
    harness_summary = summarize_harness(result.scored_rows)

    result.report["harness"] = harness_summary
    result.report["category"] = category
    result.report["strategy_name"] = strategy_name
    result.report["experiment_name"] = experiment_name

    if output_jsonl_path:
        save_jsonl(result.scored_rows, output_jsonl_path)

    if report_json_path:
        write_report(result.report, report_json_path)

    if comparison_csv_path:
        row = build_category_comparison_row(
            experiment_name=experiment_name,
            category=category,
            strategy_name=strategy_name,
            result=result,
            output_dir=category_dir,
        )
        append_comparison_row(comparison_csv_path, row)

    return result


def run_category_strategy_grid(
    problem_set,
    category,
    strategy_names,
    model_bundle,
    generation_config,
    retry_generation_config=None,
    experiment_name=None,
    batch_size=32,
    limit=None,
    score=True,
    output_dir=None,
    comparison_csv_path=None,
    harness=None,
    show_progress=True,
):
    results = {}

    for strategy_name in strategy_names:
        print("=" * 100)
        print("Category:", category)
        print("Strategy:", strategy_name)

        result = run_category_experiment(
            problem_set=problem_set,
            category=category,
            model_bundle=model_bundle,
            generation_config=generation_config,
            retry_generation_config=retry_generation_config,
            strategy_name=strategy_name,
            experiment_name=experiment_name or f"{category}_strategy_grid",
            batch_size=batch_size,
            limit=limit,
            score=score,
            output_dir=output_dir,
            comparison_csv_path=comparison_csv_path,
            harness=harness,
            show_progress=show_progress,
        )

        results[strategy_name] = result

    return results


def _weighted_formatting_summary(reports):
    totals = {
        "n_outputs": 0,
        "schema_valid_count": 0,
        "extractable_count": 0,
        "formatting_failure_count": 0,
        "retry_count": 0,
        "retry_used_count": 0,
    }

    for report in reports:
        formatting = report.get("formatting") or {}

        for key in totals:
            totals[key] += formatting.get(key) or 0

    n = totals["n_outputs"]

    if not n:
        return totals

    totals["schema_valid_rate"] = totals["schema_valid_count"] / n
    totals["extractable_rate"] = totals["extractable_count"] / n
    totals["formatting_failure_rate"] = totals["formatting_failure_count"] / n
    totals["retry_rate"] = totals["retry_count"] / n
    totals["retry_used_rate"] = totals["retry_used_count"] / n

    return totals


def run_mixed_category_experiment(
    problem_set,
    category_to_strategy,
    model_bundle,
    generation_config,
    retry_generation_config=None,
    default_strategy_name="baseline3",
    experiment_name="mixed_category_specialization",
    batch_size=32,
    limit_per_category=None,
    score=True,
    output_dir=None,
    comparison_csv_path=None,
    show_progress=True,
):
    output_dir = Path(output_dir) if output_dir else None

    all_scored_rows = []
    category_results = {}
    category_reports = []

    categories = list(BASELINE3_CATEGORIES)

    for category in categories:
        subset = filter_problem_set_by_category(problem_set, category)

        if len(subset) == 0:
            continue

        strategy_name = category_to_strategy.get(category, default_strategy_name)

        result = run_category_experiment(
            problem_set=problem_set,
            category=category,
            model_bundle=model_bundle,
            generation_config=generation_config,
            retry_generation_config=retry_generation_config,
            strategy_name=strategy_name,
            experiment_name=experiment_name,
            batch_size=batch_size,
            limit=limit_per_category,
            score=score,
            output_dir=output_dir,
            comparison_csv_path=comparison_csv_path,
            harness=get_harness(category),
            show_progress=show_progress,
        )

        category_results[category] = result
        category_reports.append(result.report)
        all_scored_rows.extend(result.scored_rows)

    summary = summarize_results(all_scored_rows)
    formatting = _weighted_formatting_summary(category_reports)

    report = {
        "experiment_name": experiment_name,
        "problem_set": {
            "name": problem_set.name,
            "n": len(problem_set),
            "category_counts": category_counts(problem_set),
        },
        "category_to_strategy": dict(category_to_strategy),
        "default_strategy_name": default_strategy_name,
        "summary": summary,
        "formatting": formatting,
        "category_reports": {
            category: result.report
            for category, result in category_results.items()
        },
    }

    if output_dir:
        mixed_dir = output_dir / _safe_name(experiment_name) / "_mixed"
        mixed_dir.mkdir(parents=True, exist_ok=True)

        output_jsonl_path = mixed_dir / "results.jsonl"
        report_json_path = mixed_dir / "report.json"
        submission_csv_path = mixed_dir / "submission.csv"

        save_jsonl(all_scored_rows, output_jsonl_path)
        write_report(report, report_json_path)

        if not score:
            save_submission_csv(all_scored_rows, submission_csv_path)
            report["submission_csv_path"] = str(submission_csv_path)

        report["output_jsonl_path"] = str(output_jsonl_path)
        report["report_json_path"] = str(report_json_path)

    if comparison_csv_path:
        fake_result = type("ResultView", (), {})()
        fake_result.report = report
        fake_result.summary = summary

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
            "category": "_mixed",
            "strategy_name": "category_to_strategy",
            "problem_set_name": problem_set.name,
            "n": len(all_scored_rows),
            "n_scored": summary.get("n_scored"),
            "overall_acc": summary.get("overall_acc"),
            "mcq_acc": summary.get("mcq_acc"),
            "free_form_acc": summary.get("free_form_acc"),
            "n_correct": summary.get("n_correct"),
            "schema_valid_rate": formatting.get("schema_valid_rate"),
            "extractable_rate": formatting.get("extractable_rate"),
            "formatting_failure_rate": formatting.get("formatting_failure_rate"),
            "retry_rate": formatting.get("retry_rate"),
            "retry_used_rate": formatting.get("retry_used_rate"),
            "harness_avg_pass_rate": None,
            "harness_full_pass_rate": None,
            "output_dir": str(output_dir) if output_dir else None,
            "output_jsonl_path": report.get("output_jsonl_path"),
            "debug_jsonl_path": None,
            "submission_csv_path": report.get("submission_csv_path"),
        }

        append_comparison_row(comparison_csv_path, row)

    return {
        "scored_rows": all_scored_rows,
        "summary": summary,
        "formatting": formatting,
        "category_results": category_results,
        "report": report,
    }