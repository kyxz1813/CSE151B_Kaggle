import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .datasets import ProblemSet


@dataclass
class DerivedRule:
    name: str
    category: str
    detector: object
    validator: object = None
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def applies(self, record):
        try:
            return bool(self.detector(record))
        except Exception:
            return False

    def evaluate(self, record, scored_row):
        if self.validator is None:
            return None

        try:
            value = self.validator(record, scored_row)
            if isinstance(value, dict):
                return value
            return {"passed": bool(value), "details": {}}
        except Exception as exc:
            return {
                "passed": False,
                "details": {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            }


class RuleRegistry:
    def __init__(self):
        self._rules = []

    def register(self, rule):
        self._rules.append(rule)

    def rules_for_category(self, category):
        return [rule for rule in self._rules if rule.category == category]

    def all_rules(self):
        return list(self._rules)


RULE_REGISTRY = RuleRegistry()


def register_rule(rule):
    RULE_REGISTRY.register(rule)


def detect_rules_for_record(record, category=None, registry=None):
    registry = registry or RULE_REGISTRY
    category = category or record.get("primary_category") or "general_math"

    matched = []

    for rule in registry.rules_for_category(category):
        if rule.applies(record):
            matched.append(rule.name)

    return matched


def annotate_records_with_rules(records, registry=None):
    annotated = []

    for record in records:
        category = record.get("primary_category") or "general_math"
        rule_names = detect_rules_for_record(record, category=category, registry=registry)

        new_record = dict(record)
        new_record["derived_rules"] = rule_names
        annotated.append(new_record)

    return annotated


def annotate_problem_set_with_rules(problem_set, registry=None, name=None):
    records = annotate_records_with_rules(problem_set.records, registry=registry)
    return ProblemSet(name or f"{problem_set.name}_with_rules", records)


def rule_distribution(records):
    counts = {}

    for record in records:
        for rule in record.get("derived_rules") or []:
            counts[rule] = counts.get(rule, 0) + 1

    return dict(sorted(counts.items()))


def save_rule_one_hot_csv(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_rules = sorted({
        rule
        for record in records
        for rule in (record.get("derived_rules") or [])
    })

    fieldnames = [
        "id",
        "primary_category",
        *all_rules,
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            active = set(record.get("derived_rules") or [])

            row = {
                "id": record.get("id"),
                "primary_category": record.get("primary_category") or "general_math",
            }

            for rule in all_rules:
                row[rule] = 1 if rule in active else 0

            writer.writerow(row)

    return path


def save_rule_annotations_jsonl(problem_set, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for record in problem_set.records:
        rows.append({
            "id": record.get("id"),
            "primary_category": record.get("primary_category"),
            "derived_rules": record.get("derived_rules") or [],
            "rule_annotation_source": record.get("rule_annotation_source", "deterministic_rules"),
        })

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return path


def load_rule_annotation_map(path):
    path = Path(path)
    rows = {}

    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            rows[str(row.get("id"))] = row

    return rows


def apply_rule_annotations(problem_set, annotation_jsonl_path, name=None):
    annotation_map = load_rule_annotation_map(annotation_jsonl_path)
    records = []

    for record in problem_set.records:
        row = annotation_map.get(str(record.get("id")), {})
        new_record = dict(record)
        new_record["derived_rules"] = row.get("derived_rules") or []
        new_record["rule_annotation_source"] = row.get("rule_annotation_source", "loaded_rule_annotations")
        records.append(new_record)

    return ProblemSet(name or f"{problem_set.name}_with_loaded_rules", records)


def annotate_problem_set_with_rules_for_categories(problem_set, categories=None, registry=None, name=None):
    categories = set(categories or [])
    records = []

    for record in problem_set.records:
        category = record.get("primary_category") or "general_math"

        if categories and category not in categories:
            new_record = dict(record)
            new_record["derived_rules"] = record.get("derived_rules") or []
            new_record["rule_annotation_source"] = record.get("rule_annotation_source", "not_selected_for_rule_annotation")
            records.append(new_record)
            continue

        rule_names = detect_rules_for_record(record, category=category, registry=registry)

        if record.get("options"):
            if category == "linear_algebra":
                rule_names.append("linear_algebra_mcq_option_mapping")
            if category == "discrete_algorithm":
                rule_names.append("discrete_mcq_option_mapping")

        if not record.get("options") and str(record.get("question", "")).count("[ANS]") > 1:
            if category == "linear_algebra":
                rule_names.append("linear_algebra_freeform_multi_answer")

        rule_names = sorted(set(rule_names))

        new_record = dict(record)
        new_record["derived_rules"] = rule_names
        new_record["rule_annotation_source"] = "deterministic_rules"
        records.append(new_record)

    return ProblemSet(name or f"{problem_set.name}_with_rules", records)


def prepare_and_save_rule_annotations(
    problem_set,
    categories,
    output_jsonl_path,
    output_one_hot_csv_path=None,
    registry=None,
    name=None,
):
    annotated = annotate_problem_set_with_rules_for_categories(
        problem_set=problem_set,
        categories=categories,
        registry=registry,
        name=name,
    )

    save_rule_annotations_jsonl(annotated, output_jsonl_path)

    if output_one_hot_csv_path:
        selected_records = [
            record for record in annotated.records
            if not categories or record.get("primary_category") in set(categories)
        ]
        save_rule_one_hot_csv(selected_records, output_one_hot_csv_path)

    return annotated