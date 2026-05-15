from dataclasses import dataclass, field


@dataclass
class HarnessCheck:
    name: str
    fn: object
    weight: float = 1.0
    description: str = ""


@dataclass
class CategoryHarness:
    category: str
    checks: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def evaluate_row(self, record, scored_row):
        check_results = []
        total_weight = 0.0
        passed_weight = 0.0

        for check in self.checks:
            result = _run_check(check, record, scored_row)
            check_results.append(result)

            if result["passed"] is not None:
                total_weight += check.weight
                if result["passed"]:
                    passed_weight += check.weight

        pass_rate = passed_weight / total_weight if total_weight else None

        return {
            "harness_category": self.category,
            "harness_check_results": check_results,
            "harness_pass_rate": pass_rate,
            "harness_passed": pass_rate == 1.0 if pass_rate is not None else None,
        }


class HarnessRegistry:
    def __init__(self):
        self._harnesses = {}

    def register(self, harness):
        self._harnesses[harness.category] = harness

    def get(self, category):
        return self._harnesses.get(category) or build_default_harness(category)

    def names(self):
        return sorted(self._harnesses)


def _run_check(check, record, scored_row):
    try:
        value = check.fn(record, scored_row)

        if isinstance(value, dict):
            passed = value.get("passed")
            details = value.get("details", {})
        elif isinstance(value, tuple):
            passed = value[0]
            details = value[1] if len(value) > 1 else {}
        else:
            passed = value
            details = {}

        if passed is not None:
            passed = bool(passed)

        return {
            "name": check.name,
            "passed": passed,
            "weight": check.weight,
            "details": details,
        }

    except Exception as exc:
        return {
            "name": check.name,
            "passed": False,
            "weight": check.weight,
            "details": {
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        }


def check_schema_valid(record, scored_row):
    return scored_row.get("schema_valid")


def check_extractable(record, scored_row):
    return scored_row.get("extractable")


def check_answer_count(record, scored_row):
    expected = scored_row.get("expected_answer_count")
    actual = scored_row.get("actual_answer_count")

    if expected is None or actual is None:
        return None

    return {
        "passed": expected == actual,
        "details": {
            "expected_answer_count": expected,
            "actual_answer_count": actual,
        },
    }


def check_mcq_letter(record, scored_row):
    if not record.get("options"):
        return None

    boxed = str(scored_row.get("boxed_answer") or "").strip().upper()
    valid_letters = [chr(ord("A") + idx) for idx in range(len(record.get("options") or []))]

    return {
        "passed": boxed in valid_letters,
        "details": {
            "boxed_answer": boxed,
            "valid_letters": valid_letters,
        },
    }


def check_official_correct(record, scored_row):
    if scored_row.get("correct") is None:
        return None

    return scored_row.get("correct")


def build_default_harness(category):
    return CategoryHarness(
        category=category,
        checks=[
            HarnessCheck("schema_valid", check_schema_valid, weight=1.0),
            HarnessCheck("extractable", check_extractable, weight=1.0),
            HarnessCheck("answer_count", check_answer_count, weight=1.0),
            HarnessCheck("mcq_letter", check_mcq_letter, weight=1.0),
            HarnessCheck("official_correct", check_official_correct, weight=2.0),
        ],
        metadata={
            "kind": "default",
        },
    )


HARNESS_REGISTRY = HarnessRegistry()


def register_harness(harness):
    HARNESS_REGISTRY.register(harness)


def get_harness(category, registry=None):
    registry = registry or HARNESS_REGISTRY
    return registry.get(category)


def apply_harness(records, scored_rows, harness):
    for record, row in zip(records, scored_rows):
        row.update(harness.evaluate_row(record, row))

    return scored_rows


def summarize_harness(scored_rows):
    rows = [row for row in scored_rows if row.get("harness_pass_rate") is not None]

    if not rows:
        return {
            "harness_n": 0,
            "harness_avg_pass_rate": None,
            "harness_full_pass_rate": None,
        }

    full_pass = [row for row in rows if row.get("harness_passed") is True]

    check_counts = {}
    check_passes = {}

    for row in rows:
        for check in row.get("harness_check_results") or []:
            name = check["name"]

            if check["passed"] is None:
                continue

            check_counts[name] = check_counts.get(name, 0) + 1
            check_passes[name] = check_passes.get(name, 0) + int(bool(check["passed"]))

    check_summary = {}

    for name in sorted(check_counts):
        check_summary[name] = {
            "n": check_counts[name],
            "pass_count": check_passes[name],
            "pass_rate": check_passes[name] / check_counts[name] if check_counts[name] else None,
        }

    return {
        "harness_n": len(rows),
        "harness_avg_pass_rate": sum(row["harness_pass_rate"] for row in rows) / len(rows),
        "harness_full_pass_count": len(full_pass),
        "harness_full_pass_rate": len(full_pass) / len(rows),
        "harness_check_summary": check_summary,
    }