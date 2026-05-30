import csv
import json
import time
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

from prompting.strategies import BASELINE3_CATEGORIES

from .baseline2_runner import run_baseline2_problem_set
from .category_harnesses import get_harness, apply_harness, summarize_harness
from .datasets import ProblemSet, save_jsonl
from .runner import save_submission_csv, write_report
from .scoring import summarize_results


def _safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_").replace(":", "_")


def _require_pandas():
    if pd is None:
        raise ImportError("pandas is required for this ablation summary helper")
    return pd


def _read_jsonl(path):
    rows = []
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_tagged_problem_set(data_jsonl_path, tags_jsonl_path, name="tagged_problem_set"):
    data_rows = _read_jsonl(data_jsonl_path)
    tag_rows = _read_jsonl(tags_jsonl_path)

    tag_by_id = {
        str(row.get("id")): row
        for row in tag_rows
    }

    merged = []

    for record in data_rows:
        tag = tag_by_id.get(str(record.get("id")), {})
        primary_category = tag.get("primary_category") or record.get("primary_category") or "general_math"
        qwen_categories = tag.get("qwen_categories") or [primary_category]

        merged_record = dict(record)
        merged_record["primary_category"] = primary_category
        merged_record["qwen_categories"] = [primary_category]
        merged_record["category_tag_raw_output"] = tag.get("category_tag_raw_output", "")
        merged_record["category_tag_parse_ok"] = tag.get("category_tag_parse_ok", False)
        merged_record["category_tag_source"] = tag.get("category_tag_source", "unknown")

        merged.append(merged_record)

    return ProblemSet(name, merged)


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


def category_summary_frame_rows(problem_set):
    counts = category_counts(problem_set)
    total = len(problem_set)

    rows = []

    for category, count in counts.items():
        rows.append({
            "category": category,
            "n": count,
            "share_of_dataset": count / total if total else None,
        })

    return rows


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
        "n_errors": _n_errors_from_summary(summary),
        "error_rate": _error_rate_from_summary(summary),
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
        "wrong_rows_jsonl_path": report.get("wrong_rows_jsonl_path"),
        "wrong_rows_txt_path": report.get("wrong_rows_txt_path"),
    }


def _n_errors_from_summary(summary):
    n_scored = summary.get("n_scored")
    n_correct = summary.get("n_correct")

    if n_scored is None or n_correct is None:
        return None

    return n_scored - n_correct


def _error_rate_from_summary(summary):
    n_scored = summary.get("n_scored")
    n_errors = _n_errors_from_summary(summary)

    if not n_scored:
        return None

    return n_errors / n_scored


def _row_text(value, max_chars):
    text = str(value or "")

    if max_chars is None:
        return text

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[TRUNCATED]..."


def _tail_text(value, max_chars):
    text = str(value or "")

    if max_chars is None:
        return text

    if len(text) <= max_chars:
        return text

    return text[-max_chars:]


def wrong_rows_for_analysis(
    scored_rows,
    max_rows=None,
    max_question_chars=1200,
    max_response_chars=2500,
    include_correct=False,
):
    rows = []

    for row in scored_rows:
        if not include_correct and row.get("correct") is True:
            continue

        rows.append({
            "id": row.get("id"),
            "category": row.get("category"),
            "route_name": row.get("route_name"),
            "template_name": row.get("template_name"),
            "is_mcq": row.get("is_mcq"),
            "options": row.get("options"),
            "gold": row.get("gold"),
            "boxed_answer": row.get("boxed_answer"),
            "extracted_final_answer": row.get("extracted_final_answer"),
            "correct": row.get("correct"),
            "schema_valid": row.get("schema_valid"),
            "schema_errors": row.get("schema_errors"),
            "harness_pass_rate": row.get("harness_pass_rate"),
            "harness_check_results": row.get("harness_check_results"),
            "question": _row_text(row.get("question"), max_question_chars),
            "response_tail": _tail_text(row.get("response"), max_response_chars),
            "response_full": _row_text(row.get("response"), max_response_chars),
        })

        if max_rows is not None and len(rows) >= max_rows:
            break

    return rows


def save_wrong_rows_artifacts(
    scored_rows,
    output_dir,
    max_rows_for_txt=25,
    max_question_chars=1200,
    max_response_chars=2500,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrong_rows = wrong_rows_for_analysis(
        scored_rows=scored_rows,
        max_rows=None,
        max_question_chars=None,
        max_response_chars=None,
        include_correct=False,
    )

    wrong_rows_jsonl_path = output_dir / "wrong_rows.jsonl"
    save_jsonl(wrong_rows, wrong_rows_jsonl_path)

    wrong_rows_txt_path = output_dir / "wrong_rows_for_copy_paste.txt"

    display_rows = wrong_rows_for_analysis(
        scored_rows=scored_rows,
        max_rows=max_rows_for_txt,
        max_question_chars=max_question_chars,
        max_response_chars=max_response_chars,
        include_correct=False,
    )

    with open(wrong_rows_txt_path, "w", encoding="utf-8") as f:
        f.write(f"WRONG ROWS FOR ANALYSIS\n")
        f.write(f"n_wrong_total: {len(wrong_rows)}\n")
        f.write("=" * 100 + "\n\n")

        for idx, row in enumerate(display_rows, start=1):
            f.write("=" * 100 + "\n")
            f.write(f"WRONG ROW {idx}\n")
            f.write("=" * 100 + "\n")
            f.write(f"id: {row.get('id')}\n")
            f.write(f"category: {row.get('category')}\n")
            f.write(f"route_name: {row.get('route_name')}\n")
            f.write(f"template_name: {row.get('template_name')}\n")
            f.write(f"is_mcq: {row.get('is_mcq')}\n")
            f.write(f"gold: {row.get('gold')}\n")
            f.write(f"boxed_answer: {row.get('boxed_answer')}\n")
            f.write(f"extracted_final_answer: {row.get('extracted_final_answer')}\n")
            f.write(f"correct: {row.get('correct')}\n")
            f.write(f"schema_valid: {row.get('schema_valid')}\n")
            f.write(f"schema_errors: {row.get('schema_errors')}\n")
            f.write(f"harness_pass_rate: {row.get('harness_pass_rate')}\n")
            f.write(f"harness_check_results: {json.dumps(row.get('harness_check_results'), ensure_ascii=False)}\n")
            f.write("\nQUESTION:\n")
            f.write(str(row.get("question") or ""))
            f.write("\n\nRESPONSE TAIL:\n")
            f.write(str(row.get("response_tail") or ""))
            f.write("\n\n")

    return {
        "wrong_rows_count": len(wrong_rows),
        "wrong_rows_jsonl_path": str(wrong_rows_jsonl_path),
        "wrong_rows_txt_path": str(wrong_rows_txt_path),
    }


def inspect_wrong_rows(result, max_rows=10, max_question_chars=800, max_response_chars=1800):
    return wrong_rows_for_analysis(
        scored_rows=result.scored_rows,
        max_rows=max_rows,
        max_question_chars=max_question_chars,
        max_response_chars=max_response_chars,
        include_correct=False,
    )


def print_wrong_rows_for_copy_paste(result, max_rows=8, max_question_chars=1000, max_response_chars=2200):
    rows = inspect_wrong_rows(
        result,
        max_rows=max_rows,
        max_question_chars=max_question_chars,
        max_response_chars=max_response_chars,
    )

    for idx, row in enumerate(rows, start=1):
        print("=" * 100)
        print(f"WRONG ROW {idx}")
        print("=" * 100)
        print("id:", row.get("id"))
        print("category:", row.get("category"))
        print("route_name:", row.get("route_name"))
        print("template_name:", row.get("template_name"))
        print("is_mcq:", row.get("is_mcq"))
        print("options:", row.get("options"))
        print("gold:", row.get("gold"))
        print("boxed_answer:", row.get("boxed_answer"))
        print("extracted_final_answer:", row.get("extracted_final_answer"))
        print("correct:", row.get("correct"))
        print("schema_valid:", row.get("schema_valid"))
        print("schema_errors:", row.get("schema_errors"))
        print("harness_pass_rate:", row.get("harness_pass_rate"))
        print("harness_check_results:", row.get("harness_check_results"))
        print("\nQUESTION:")
        print(row.get("question"))
        print("\nRESPONSE TAIL:")
        print(row.get("response_tail"))
        print()


def _category_output_dir(output_dir, experiment_name, category, strategy_name):
    return (
        Path(output_dir)
        / _safe_name(category)
        / _safe_name(experiment_name)
        / _safe_name(strategy_name)
    )


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
    export_wrong_rows=True,
):
    experiment_name = experiment_name or f"{category}_{strategy_name}"
    subset = filter_problem_set_by_category(problem_set, category)

    if limit is not None:
        subset = subset.head(limit)

    output_dir = Path(output_dir) if output_dir else None

    if output_dir:
        category_dir = _category_output_dir(output_dir, experiment_name, category, strategy_name)
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

    for record, row in zip(subset.records, result.scored_rows):
        row.setdefault("question", record.get("question"))
        row.setdefault("options", record.get("options"))
        row.setdefault("answer", record.get("answer"))
        row.setdefault("primary_category", record.get("primary_category"))
        row.setdefault("qwen_categories", record.get("qwen_categories"))
        row.setdefault("category_tag_source", record.get("category_tag_source"))
        row.setdefault("category_tag_parse_ok", record.get("category_tag_parse_ok"))
        row.setdefault("derived_rules", record.get("derived_rules"))

    harness = harness or get_harness(category)
    apply_harness(subset.records, result.scored_rows, harness)
    harness_summary = summarize_harness(result.scored_rows)

    result.report["harness"] = harness_summary
    result.report["category"] = category
    result.report["strategy_name"] = strategy_name
    result.report["experiment_name"] = experiment_name

    if category_dir and export_wrong_rows:
        wrong_artifacts = save_wrong_rows_artifacts(result.scored_rows, category_dir)
        result.report.update(wrong_artifacts)

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


def run_multi_category_experiments(
    problem_set,
    categories,
    model_bundle,
    generation_config,
    retry_generation_config=None,
    strategy_name="baseline3",
    experiment_name=None,
    batch_size=32,
    limit_per_category=None,
    score=True,
    output_dir=None,
    comparison_csv_path=None,
    show_progress=True,
):
    experiment_name = experiment_name or f"multi_category_{strategy_name}"
    results = {}

    for category in categories:
        print("=" * 100)
        print("Running category:", category)
        print("Strategy:", strategy_name)

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

        results[category] = result

    return results


def summarize_category_error_contribution(category_results, problem_set=None):
    rows = []

    total_errors = 0
    total_scored = 0
    total_correct = 0

    for category, result in category_results.items():
        summary = result.report.get("summary") or result.summary or {}
        n_scored = summary.get("n_scored") or 0
        n_correct = summary.get("n_correct") or 0
        n_errors = n_scored - n_correct

        total_errors += n_errors
        total_scored += n_scored
        total_correct += n_correct

    dataset_counts = category_counts(problem_set) if problem_set is not None else {}
    dataset_total = len(problem_set) if problem_set is not None else None

    for category, result in category_results.items():
        summary = result.report.get("summary") or result.summary or {}
        formatting = result.report.get("formatting") or {}
        harness = result.report.get("harness") or {}

        n_scored = summary.get("n_scored") or 0
        n_correct = summary.get("n_correct") or 0
        n_errors = n_scored - n_correct

        dataset_n = dataset_counts.get(category)

        rows.append({
            "category": category,
            "dataset_n": dataset_n,
            "dataset_share": dataset_n / dataset_total if dataset_total else None,
            "n_scored": n_scored,
            "n_correct": n_correct,
            "n_errors": n_errors,
            "accuracy": n_correct / n_scored if n_scored else None,
            "error_rate_within_category": n_errors / n_scored if n_scored else None,
            "share_of_total_errors": n_errors / total_errors if total_errors else None,
            "schema_valid_rate": formatting.get("schema_valid_rate"),
            "extractable_rate": formatting.get("extractable_rate"),
            "harness_avg_pass_rate": harness.get("harness_avg_pass_rate"),
            "wrong_rows_txt_path": result.report.get("wrong_rows_txt_path"),
            "wrong_rows_jsonl_path": result.report.get("wrong_rows_jsonl_path"),
        })

    rows.sort(key=lambda row: row["share_of_total_errors"] or 0, reverse=True)

    return {
        "total_scored": total_scored,
        "total_correct": total_correct,
        "total_errors": total_errors,
        "overall_accuracy": total_correct / total_scored if total_scored else None,
        "rows": rows,
    }


def save_error_contribution_report(report, output_dir, filename="error_contribution_report.json"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / filename.replace(".json", ".csv")

    rows = report.get("rows") or []

    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return {
        "json_path": str(path),
        "csv_path": str(csv_path),
    }


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

    error_contribution = summarize_category_error_contribution(
        category_results,
        problem_set=problem_set,
    )

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
        "error_contribution": error_contribution,
        "category_reports": {
            category: result.report
            for category, result in category_results.items()
        },
    }

    if output_dir:
        mixed_dir = output_dir / _safe_name("_mixed") / _safe_name(experiment_name)
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
        "error_contribution": error_contribution,
        "report": report,
    }

GLOBAL_STRATEGIES = {
    "baseline",
    "baseline2",
    "baseline3",
    "baseline3_adaptive_rules",
}


def _as_list(value):
    if value is None:
        return None

    if value == "all":
        return "all"

    if isinstance(value, str):
        return [value]

    return list(value)


def normalize_strategy_spec(strategy_spec):
    if isinstance(strategy_spec, str):
        return {
            "name": strategy_spec,
            "label": strategy_spec,
            "categories": None,
            "enabled": True,
            "notes": "",
        }

    spec = dict(strategy_spec)
    name = spec["name"]

    return {
        "name": name,
        "label": spec.get("label") or name,
        "categories": _as_list(spec.get("categories")),
        "enabled": spec.get("enabled", True),
        "notes": spec.get("notes", ""),
    }


def strategy_applies_to_category(strategy_spec, category):
    spec = normalize_strategy_spec(strategy_spec)

    if not spec["enabled"]:
        return False, "disabled"

    categories = spec.get("categories")
    name = spec["name"]

    if categories == "all":
        return True, "explicit_all"

    if isinstance(categories, list):
        if category in categories:
            return True, "explicit_category"
        return False, "explicit_category_mismatch"

    if name in GLOBAL_STRATEGIES:
        return True, "global_strategy"

    if name.startswith(category):
        return True, "inferred_from_name_prefix"

    if f"{category}_" in name:
        return True, "inferred_from_name_contains_category"

    return False, "not_applicable_without_explicit_category"


def expand_category_strategy_plan(categories, strategies):
    plan_rows = []
    skipped_rows = []

    for category in categories:
        for strategy_spec in strategies:
            spec = normalize_strategy_spec(strategy_spec)
            applies, reason = strategy_applies_to_category(spec, category)

            row = {
                "category": category,
                "strategy_name": spec["name"],
                "strategy_label": spec["label"],
                "applicability_reason": reason,
                "strategy_notes": spec.get("notes", ""),
            }

            if applies:
                plan_rows.append(row)
            else:
                skipped_rows.append(row)

    return {
        "plan_rows": plan_rows,
        "skipped_rows": skipped_rows,
    }


def _result_summary_row(result, category, strategy_name, experiment_name, applicability_reason=None):
    report = result.report or {}
    summary = report.get("summary") or result.summary or {}
    formatting = report.get("formatting") or {}
    harness = report.get("harness") or {}

    n_scored = summary.get("n_scored") or 0
    n_correct = summary.get("n_correct") or 0
    n_errors = n_scored - n_correct

    return {
        "experiment_name": experiment_name,
        "category": category,
        "strategy_name": strategy_name,
        "applicability_reason": applicability_reason,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_errors": n_errors,
        "overall_acc": summary.get("overall_acc"),
        "mcq_acc": summary.get("mcq_acc"),
        "free_form_acc": summary.get("free_form_acc"),
        "schema_valid_rate": formatting.get("schema_valid_rate"),
        "extractable_rate": formatting.get("extractable_rate"),
        "formatting_failure_rate": formatting.get("formatting_failure_rate"),
        "retry_rate": formatting.get("retry_rate"),
        "retry_used_rate": formatting.get("retry_used_rate"),
        "harness_avg_pass_rate": harness.get("harness_avg_pass_rate"),
        "harness_full_pass_rate": harness.get("harness_full_pass_rate"),
        "output_jsonl_path": report.get("output_jsonl_path"),
        "debug_jsonl_path": report.get("debug_jsonl_path"),
        "report_json_path": report.get("report_json_path"),
        "wrong_rows_jsonl_path": report.get("wrong_rows_jsonl_path"),
        "wrong_rows_txt_path": report.get("wrong_rows_txt_path"),
    }


def _add_error_contribution(summary_rows):
    total_errors = sum(row.get("n_errors") or 0 for row in summary_rows)

    for row in summary_rows:
        n_errors = row.get("n_errors") or 0
        row["share_of_ablation_errors"] = n_errors / total_errors if total_errors else None

    return summary_rows


def _save_ablation_summary(summary_rows, skipped_rows, output_dir, experiment_name):
    output_dir = Path(output_dir)
    ablation_dir = output_dir / "_ablation" / _safe_name(experiment_name)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = ablation_dir / "ablation_summary.csv"
    skipped_csv_path = ablation_dir / "ablation_skipped.csv"
    summary_json_path = ablation_dir / "ablation_summary.json"

    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    if skipped_rows:
        fieldnames = sorted({key for row in skipped_rows for key in row.keys()})
        with open(skipped_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in skipped_rows:
                writer.writerow(row)

    payload = {
        "experiment_name": experiment_name,
        "summary_rows": summary_rows,
        "skipped_rows": skipped_rows,
        "summary_csv_path": str(summary_csv_path),
        "skipped_csv_path": str(skipped_csv_path),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "ablation_dir": str(ablation_dir),
        "summary_csv_path": str(summary_csv_path),
        "skipped_csv_path": str(skipped_csv_path),
        "summary_json_path": str(summary_json_path),
    }


def run_category_strategy_ablation(
    problem_set,
    categories,
    strategies,
    model_bundle,
    generation_config,
    retry_generation_config=None,
    experiment_name="category_strategy_ablation",
    batch_size=32,
    limit_per_combo=None,
    score=True,
    output_dir=None,
    comparison_csv_path=None,
    show_progress=True,
    dry_plan=False,
):
    plan = expand_category_strategy_plan(categories, strategies)
    plan_rows = plan["plan_rows"]
    skipped_rows = plan["skipped_rows"]

    if dry_plan:
        return {
            "plan_rows": plan_rows,
            "skipped_rows": skipped_rows,
            "results": {},
            "summary_rows": [],
            "artifacts": {},
        }

    results = {}
    summary_rows = []

    for plan_row in plan_rows:
        category = plan_row["category"]
        strategy_name = plan_row["strategy_name"]
        applicability_reason = plan_row["applicability_reason"]

        print("=" * 100)
        print("Ablation experiment:", experiment_name)
        print("Category:", category)
        print("Strategy:", strategy_name)
        print("Applicability:", applicability_reason)

        result = run_category_experiment(
            problem_set=problem_set,
            category=category,
            model_bundle=model_bundle,
            generation_config=generation_config,
            retry_generation_config=retry_generation_config,
            strategy_name=strategy_name,
            experiment_name=experiment_name,
            batch_size=batch_size,
            limit=limit_per_combo,
            score=score,
            output_dir=output_dir,
            comparison_csv_path=comparison_csv_path,
            harness=get_harness(category),
            show_progress=show_progress,
            export_wrong_rows=True,
        )

        results.setdefault(category, {})[strategy_name] = result

        summary_rows.append(
            _result_summary_row(
                result=result,
                category=category,
                strategy_name=strategy_name,
                experiment_name=experiment_name,
                applicability_reason=applicability_reason,
            )
        )

    summary_rows = _add_error_contribution(summary_rows)

    artifacts = {}
    if output_dir:
        artifacts = _save_ablation_summary(
            summary_rows=summary_rows,
            skipped_rows=skipped_rows,
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    return {
        "plan_rows": plan_rows,
        "skipped_rows": skipped_rows,
        "results": results,
        "summary_rows": summary_rows,
        "artifacts": artifacts,
    }


def ablation_summary_frame(ablation_result):
    try:
        pandas = _require_pandas()
        return pandas.DataFrame(ablation_result.get("summary_rows") or [])
    except Exception:
        return ablation_result.get("summary_rows") or []


def ablation_plan_frame(categories, strategies):
    plan = expand_category_strategy_plan(categories, strategies)

    try:
        pandas = _require_pandas()
        return (
            pandas.DataFrame(plan["plan_rows"]),
            pandas.DataFrame(plan["skipped_rows"]),
        )
    except Exception:
        return plan["plan_rows"], plan["skipped_rows"]


def get_ablation_result(ablation_result, category, strategy_name):
    return (
        ablation_result
        .get("results", {})
        .get(category, {})
        .get(strategy_name)
    )


def print_ablation_wrong_rows(
    ablation_result,
    category,
    strategy_name,
    max_rows=8,
    max_question_chars=1200,
    max_response_chars=2500,
):
    result = get_ablation_result(ablation_result, category, strategy_name)

    if result is None:
        print("No result found for", category, strategy_name)
        return

    print_wrong_rows_for_copy_paste(
        result,
        max_rows=max_rows,
        max_question_chars=max_question_chars,
        max_response_chars=max_response_chars,
    )


def ablation_wrong_rows_frame(
    ablation_result,
    category,
    strategy_name,
    max_rows=20,
    max_question_chars=1000,
    max_response_chars=2500,
):
    result = get_ablation_result(ablation_result, category, strategy_name)

    if result is None:
        return []

    rows = inspect_wrong_rows(
        result,
        max_rows=max_rows,
        max_question_chars=max_question_chars,
        max_response_chars=max_response_chars,
    )

    try:
        pandas = _require_pandas()
        return pandas.DataFrame(rows)
    except Exception:
        return rows
    
def filter_problem_set_by_rule(problem_set, rule_name, category=None, name=None):
    records = []

    for record in problem_set.records:
        if category is not None and (record.get("primary_category") or "general_math") != category:
            continue

        if rule_name in (record.get("derived_rules") or []):
            records.append(record)

    return ProblemSet(
        name or f"{problem_set.name}_{rule_name}",
        records,
    )


def run_rule_strategy_ablation(
    problem_set,
    category,
    rule_names,
    strategies,
    model_bundle,
    generation_config,
    retry_generation_config=None,
    experiment_name="rule_strategy_ablation",
    batch_size=32,
    limit_per_combo=None,
    score=True,
    output_dir=None,
    comparison_csv_path=None,
    show_progress=True,
):
    results = {}
    summary_rows = []

    for rule_name in rule_names:
        rule_set = filter_problem_set_by_rule(
            problem_set=problem_set,
            rule_name=rule_name,
            category=category,
            name=f"{problem_set.name}_{category}_{rule_name}",
        )

        if len(rule_set) == 0:
            print("Skipping empty rule set:", category, rule_name)
            continue

        results.setdefault(rule_name, {})

        for strategy_spec in strategies:
            spec = normalize_strategy_spec(strategy_spec)
            strategy_name = spec["name"]

            applies, reason = strategy_applies_to_category(spec, category)
            if not applies:
                continue

            print("=" * 100)
            print("Rule ablation:", experiment_name)
            print("Category:", category)
            print("Rule:", rule_name)
            print("Strategy:", strategy_name)
            print("n:", len(rule_set))

            result = run_category_experiment(
                problem_set=rule_set,
                category=category,
                model_bundle=model_bundle,
                generation_config=generation_config,
                retry_generation_config=retry_generation_config,
                strategy_name=strategy_name,
                experiment_name=f"{experiment_name}_{rule_name}",
                batch_size=batch_size,
                limit=limit_per_combo,
                score=score,
                output_dir=output_dir,
                comparison_csv_path=comparison_csv_path,
                harness=get_harness(category),
                show_progress=show_progress,
                export_wrong_rows=True,
            )

            results[rule_name][strategy_name] = result

            row = _result_summary_row(
                result=result,
                category=category,
                strategy_name=strategy_name,
                experiment_name=experiment_name,
                applicability_reason=f"rule_filtered:{rule_name}",
            )
            row["rule_name"] = rule_name
            summary_rows.append(row)

    summary_rows = _add_error_contribution(summary_rows)

    artifacts = {}
    if output_dir:
        artifacts = _save_ablation_summary(
            summary_rows=summary_rows,
            skipped_rows=[],
            output_dir=output_dir,
            experiment_name=experiment_name,
        )

    return {
        "results": results,
        "summary_rows": summary_rows,
        "artifacts": artifacts,
    }
