import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


VED_CATEGORIES = ("calculus", "general_math")


def read_jsonl(path):
    rows = []
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return path


def _record_map(data_jsonl_path):
    if not data_jsonl_path:
        return {}

    path = Path(data_jsonl_path)
    if not path.exists():
        return {}

    return {str(row.get("id")): row for row in read_jsonl(path)}


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _lower_question(row):
    return str(row.get("question") or "").lower()


def _rules(row):
    values = row.get("derived_rules") or []
    if isinstance(values, str):
        return [values]
    return list(values)


def _harness_failed(row):
    failed = []
    for item in row.get("harness_check_results") or []:
        if item.get("passed") is False:
            failed.append(item.get("name"))
    return failed


def classify_root_cause(row):
    question = _lower_question(row)
    rules = set(_rules(row))
    failed_checks = set(_harness_failed(row))

    if not row.get("extractable"):
        return "answer_extraction_failure"
    if not row.get("schema_valid") or not row.get("strict_well_formed"):
        return "formatting_failure"
    if row.get("is_mcq") and "no_unmapped_numeric_for_mcq" in failed_checks:
        return "mcq_option_mapping_failure"
    if "answer_count_valid" in failed_checks or "multi_answer_placeholder_discipline" in failed_checks:
        return "multi_answer_count_mismatch"

    if row.get("category") == "calculus":
        if "calculus_integral" in rules or any(term in question for term in ("integral", "integrate", "\\int")):
            return "calculus_integral_error"
        if "calculus_limit_asymptotic" in rules or "limit" in question or "\\lim" in question:
            return "calculus_limit_error"
        if (
            "calculus_derivative_extrema" in rules
            or "calculus_endpoint_extrema_check" in rules
            or any(term in question for term in ("derivative", "differentiate", "tangent", "maximum", "minimum"))
        ):
            return "calculus_derivative_extrema_error"
        if "calculus_differential_equation" in rules or "differential equation" in question:
            return "calculus_differential_equation_error"
        if "calculus_series_approximation" in rules or any(term in question for term in ("series", "taylor", "maclaurin")):
            return "calculus_series_approximation_error"
        if "calculus_complex_residue" in rules or any(term in question for term in ("residue", "contour")):
            return "calculus_complex_residue_error"

    if row.get("category") == "general_math":
        if "general_math_mcq_option_mapping" in rules and row.get("is_mcq"):
            return "mcq_option_mapping_failure"
        if "general_math_text_answer" in rules:
            return "general_math_text_answer_error"
        if "general_math_unit_or_conversion" in rules or "general_math_percentage_or_rate" in rules:
            return "general_math_unit_rate_error"
        if "general_math_direct_formula" in rules:
            return "general_math_direct_formula_error"
        if "general_math_arithmetic_simplification" in rules:
            return "arithmetic_or_simplification_error"
        if "general_math_ordered_answer" in rules:
            return "ordered_answer_error"

    if any(term in question for term in ("simplify", "evaluate", "calculate", "fraction", "round")):
        return "arithmetic_or_simplification_error"

    return "multi_step_reasoning_error"


def explain_wrong(row, root_cause):
    gold = row.get("gold") or row.get("answer")
    model_answer = row.get("boxed_answer") or row.get("extracted_final_answer")

    if root_cause == "answer_extraction_failure":
        return "The final answer could not be extracted from the model output."
    if root_cause == "formatting_failure":
        return "The output did not satisfy the required reasoning/final-answer schema."
    if root_cause == "mcq_option_mapping_failure":
        return "The problem is multiple-choice, but the final answer was not mapped to a single option letter."
    if root_cause == "multi_answer_count_mismatch":
        return "The final answer count did not match the number of requested blanks or parts."

    return f"The extracted answer {model_answer!r} does not match the ground truth {gold!r}."


def build_failure_analysis_rows(scored_rows, records_by_id=None, categories=VED_CATEGORIES):
    records_by_id = records_by_id or {}
    wanted = set(categories)
    rows = []

    for row in scored_rows:
        category = row.get("category") or row.get("primary_category")
        if category not in wanted:
            continue
        if row.get("correct") is True:
            continue

        record = records_by_id.get(str(row.get("id")), {})
        merged = dict(record)
        merged.update(row)

        root_cause = classify_root_cause(merged)
        prompt_metadata = merged.get("prompt_metadata") or {}

        rows.append({
            "id": merged.get("id"),
            "question": merged.get("question"),
            "ground_truth_answer": merged.get("gold") or merged.get("answer"),
            "model_answer": merged.get("boxed_answer") or merged.get("extracted_final_answer"),
            "category": category,
            "prompt_template_used": merged.get("template_name"),
            "strategy_used": prompt_metadata.get("strategy_label") or merged.get("strategy_name"),
            "route_name": merged.get("route_name"),
            "derived_rules": _as_text(merged.get("derived_rules") or []),
            "is_mcq": merged.get("is_mcq"),
            "schema_valid": merged.get("schema_valid"),
            "strict_well_formed": merged.get("strict_well_formed"),
            "extractable": merged.get("extractable"),
            "retry_needed": merged.get("retry_needed"),
            "retry_used": merged.get("retry_used"),
            "harness_failed_checks": _as_text(_harness_failed(merged)),
            "why_answer_is_wrong": explain_wrong(merged, root_cause),
            "root_cause_classification": root_cause,
            "raw_model_output": merged.get("response") or merged.get("raw_output"),
        })

    return rows


def summarize_failure_patterns(rows):
    by_category = defaultdict(Counter)
    by_category_strategy = defaultdict(Counter)
    by_category_template = defaultdict(Counter)

    for row in rows:
        category = row.get("category")
        by_category[category][row.get("root_cause_classification")] += 1
        by_category_strategy[category][row.get("strategy_used")] += 1
        by_category_template[category][row.get("prompt_template_used")] += 1

    return {
        "n_failures": len(rows),
        "root_causes_by_category": {
            category: dict(counter.most_common())
            for category, counter in sorted(by_category.items())
        },
        "strategies_by_category": {
            category: dict(counter.most_common())
            for category, counter in sorted(by_category_strategy.items())
        },
        "templates_by_category": {
            category: dict(counter.most_common())
            for category, counter in sorted(by_category_template.items())
        },
    }


def save_failure_analysis(rows, output_dir, prefix="ved"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = write_jsonl(rows, output_dir / f"{prefix}_failure_analysis.jsonl")
    csv_path = output_dir / f"{prefix}_failure_analysis.csv"
    summary_path = output_dir / f"{prefix}_failure_summary.json"

    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["no_failures"])

    summary = summarize_failure_patterns(rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "failure_analysis_jsonl_path": str(jsonl_path),
        "failure_analysis_csv_path": str(csv_path),
        "failure_summary_json_path": str(summary_path),
        "failure_summary": summary,
    }


def analyze_results(results_jsonl_path, output_dir, data_jsonl_path=None, prefix="ved"):
    scored_rows = read_jsonl(results_jsonl_path)
    records_by_id = _record_map(data_jsonl_path)
    rows = build_failure_analysis_rows(scored_rows, records_by_id=records_by_id)
    return save_failure_analysis(rows, output_dir=output_dir, prefix=prefix)


def main():
    parser = argparse.ArgumentParser(description="Analyze calculus and general_math validation failures.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--data-jsonl", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="ved")
    args = parser.parse_args()

    artifacts = analyze_results(
        results_jsonl_path=args.results_jsonl,
        data_jsonl_path=args.data_jsonl,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print(json.dumps(artifacts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
