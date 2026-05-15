import csv
import json
from datetime import datetime, timezone
from pathlib import Path


COMPARISON_FIELDS = [
    "timestamp_utc",
    "experiment_name",
    "baseline",
    "strategy_name",
    "split",
    "n",
    "backend",
    "model_id",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "accuracy",
    "mcq_acc",
    "free_form_acc",
    "n_correct",
    "n_scored",
    "formatting_failure_rate",
    "schema_valid_rate",
    "extractable_rate",
    "retry_count",
    "retry_schema_success_count",
    "category_counts_json",
    "category_tags_path",
    "category_one_hot_csv_path",
    "report_json_path",
    "output_jsonl_path",
    "debug_jsonl_path",
    "submission_csv_path",
]


def _summary_value(report, section, key):
    value = report.get(section) or {}
    return value.get(key)


def build_comparison_row(
    report,
    experiment_name=None,
    strategy_name=None,
    split=None,
    report_json_path=None,
):
    config = report.get("generation_config") or {}
    problem_set = report.get("problem_set") or {}

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name or report.get("baseline"),
        "baseline": report.get("baseline"),
        "strategy_name": strategy_name,
        "split": split or problem_set.get("name"),
        "n": problem_set.get("n"),
        "backend": report.get("backend"),
        "model_id": report.get("model_id") or config.get("model_id"),
        "max_new_tokens": config.get("max_new_tokens"),
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k"),
        "accuracy": _summary_value(report, "summary", "overall_acc"),
        "mcq_acc": _summary_value(report, "summary", "mcq_acc"),
        "free_form_acc": _summary_value(report, "summary", "free_form_acc"),
        "n_correct": _summary_value(report, "summary", "n_correct"),
        "n_scored": _summary_value(report, "summary", "n_scored"),
        "formatting_failure_rate": _summary_value(report, "formatting", "formatting_failure_rate"),
        "schema_valid_rate": _summary_value(report, "formatting", "schema_valid_rate"),
        "extractable_rate": _summary_value(report, "formatting", "extractable_rate"),
        "retry_count": _summary_value(report, "formatting", "retry_count"),
        "retry_schema_success_count": _summary_value(report, "formatting", "retry_schema_success_count"),
        "category_counts_json": json.dumps(report.get("category_counts") or {}, sort_keys=True),
        "category_tags_path": report.get("category_tags_path"),
        "category_one_hot_csv_path": report.get("category_one_hot_csv_path"),
        "report_json_path": str(report_json_path) if report_json_path else report.get("report_json_path"),
        "output_jsonl_path": report.get("output_jsonl_path"),
        "debug_jsonl_path": report.get("debug_jsonl_path"),
        "submission_csv_path": report.get("submission_csv_path"),
    }


def append_comparison_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in COMPARISON_FIELDS})

    return path
