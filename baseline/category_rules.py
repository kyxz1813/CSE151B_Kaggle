import csv
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