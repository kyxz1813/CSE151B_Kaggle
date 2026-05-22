import re

from baseline.category_harnesses import (
    CategoryHarness,
    HarnessCheck,
    register_harness,
)

from baseline.category_rules import (
    DerivedRule,
    register_rule,
)


CATEGORY = "applied_word_problem"

DEFAULT_STRATEGY = "applied_word_problem_v2_model_building"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "applied_word_problem_v2_model_building",
    "applied_word_problem_multi_step",
    "applied_word_problem_piecewise",
    "applied_word_problem_rate_distance",
    "applied_word_problem_financial",
]


PROMPT_TEMPLATE_NOTES = """
Prompt/template work for this category lives in:
- baseline/baseline3_prompts.py
- prompting/templates.py
- prompting/strategies.py

This file owns:
- applied word problem harness checks
- applied word problem derived rules
- default strategy lists
"""


# ============================================================
# UTILITIES
# ============================================================

def _text(record):
    pieces = [str(record.get("question", ""))]

    for option in record.get("options") or []:
        pieces.append(str(option))

    return "\n".join(pieces).lower()


def _response(row):
    return str(
        row.get("response")
        or row.get("raw_output")
        or ""
    )


def _boxed(row):
    return str(
        row.get("boxed_answer")
        or row.get("extracted_final_answer")
        or ""
    ).strip()


def _ans_count(record):
    return str(record.get("question", "")).count("[ANS]")


def _has_any(text, terms):
    return any(term in text for term in terms)


# ============================================================
# GENERALIZED DETECTORS
# ============================================================

def is_rate_or_ratio_model_problem(record):
    text = _text(record)

    rate_terms = [
        "per",
        "rate",
        "tax",
        "interest",
        "cost",
        "price",
        "speed",
        "mph",
        "mi/h",
        "percent",
        "%",
        "sales tax",
    ]

    context_terms = [
        "dollars",
        "hours",
        "miles",
        "people",
        "investment",
        "balance",
        "beam",
        "packages",
    ]

    return (
        _has_any(text, rate_terms)
        and _has_any(text, context_terms)
    )


def is_linear_modeling_problem(record):
    text = _text(record)

    return _has_any(text, [
        "linear model",
        "linear function",
        "break-even",
        "profit",
        "cost function",
        "revenue",
        "manufacture",
        "selling price",
        "service charge",
        "piecewise",
        "y=mx+b",
    ])


def is_exponential_modeling_problem(record):
    text = _text(record)

    return _has_any(text, [
        "exponential",
        "decreases by",
        "increases by",
        "cools",
        "growth",
        "decay",
        "investment",
        "compound",
        "population",
    ])


def is_geometry_application_problem(record):
    text = _text(record)

    geometry_terms = [
        "rectangle",
        "window",
        "semicircle",
        "area",
        "perimeter",
        "width",
        "square room",
        "tiles",
        "wheel",
        "circumference",
    ]

    return _has_any(text, geometry_terms)


def is_function_interpretation_problem(record):
    text = _text(record)

    return (
        _has_any(text, [
            "f(x)",
            "inverse",
            "modeled by",
            "formula",
            "function",
        ])
        and _has_any(text, [
            "meaning",
            "represents",
            "interpret",
            "can be manufactured",
            "population",
        ])
    )


def is_piecewise_or_case_problem(record):
    text = _text(record)

    return _has_any(text, [
        "piecewise",
        "if",
        "otherwise",
        "discount",
        "at least",
        "or more",
    ])


def is_table_or_schedule_reasoning_problem(record):
    text = _text(record)

    return _has_any(text, [
        "table",
        "billing cycle",
        "days",
        "balance",
        "schedule",
        "transactions",
    ])


def is_multi_step_quantity_tracking_problem(record):
    text = _text(record)

    tracking_terms = [
        "remaining balance",
        "payment",
        "purchase",
        "left",
        "started with",
        "after",
        "returned",
        "trip back",
        "billing cycle",
        "transactions",
    ]

    return _has_any(text, tracking_terms)


def is_inequality_or_constraint_problem(record):
    text = _text(record)

    return _has_any(text, [
        "maximum",
        "minimum",
        "between",
        "at least",
        "at most",
        "support",
        "inclusive",
        "range",
        "whole number",
    ])


def is_unit_conversion_application_problem(record):
    text = _text(record)

    return (
        _has_any(text, [
            "convert",
            "celsius",
            "fahrenheit",
            "miles",
            "inches",
            "km",
            "hours",
        ])
        and _has_any(text, [
            "wheel",
            "plane",
            "temperature",
            "speed",
        ])
    )


def is_multi_step_applied_problem(record):
    text = _text(record)

    indicators = 0

    indicators += int(_ans_count(record) >= 2)
    indicators += int("part (a)" in text)
    indicators += int("part (b)" in text)
    indicators += int("(a)" in text)
    indicators += int("(b)" in text)
    indicators += int("find a formula" in text)
    indicators += int("using this model" in text)

    return indicators >= 2


def is_multi_answer_problem(record):
    return _ans_count(record) >= 2


def is_mcq_problem(record):
    return bool(record.get("options"))


# ============================================================
# HARNESS CHECKS
# ============================================================

def schema_valid(record, row):
    return row.get("schema_valid")


def extractable(record, row):
    return row.get("extractable")


def answer_count_valid(record, row):
    expected = row.get("expected_answer_count")

    if expected is None:
        expected = _ans_count(record)

    actual = row.get("actual_answer_count")

    if expected is None or actual is None:
        return None

    return {
        "passed": expected == actual,
        "details": {
            "expected": expected,
            "actual": actual,
        },
    }


def mcq_letter_valid(record, row):
    options = record.get("options") or []

    if not options:
        return None

    boxed = _boxed(row).upper()

    valid = [
        chr(ord("A") + i)
        for i in range(len(options))
    ]

    return {
        "passed": boxed in valid,
        "details": {
            "boxed": boxed,
            "valid_letters": valid,
        },
    }


def units_or_context_preserved(record, row):
    text = _text(record)

    if not _has_any(text, [
        "hours",
        "miles",
        "dollars",
        "degrees",
        "feet",
        "kg",
        "lb",
        "percent",
        "celsius",
        "fahrenheit",
        "kilograms",
        "pounds",
        "people",
        "oz",
        "ounces",
    ]):
        return None

    response = _response(row).lower()

    context_markers = [
        "hour",
        "mile",
        "dollar",
        "degree",
        "ft",
        "kg",
        "lb",
        "%",
        "oz",
    ]

    present = any(
        marker in response
        for marker in context_markers
    )

    return {
        "passed": present,
        "details": {
            "unit_context_present": present,
        },
    }


def exact_form_preferred_unless_requested(record, row):
    text = _text(record)

    approximation_requested = _has_any(text, [
        "round",
        "nearest",
        "approximate",
        "approximation",
        "nearest tenth",
        "nearest hundredth",
    ])

    if approximation_requested:
        return None

    response = _response(row).lower()

    suspicious_terms = [
        "≈",
        "approximately",
        "approx",
    ]

    found = [
        t for t in suspicious_terms
        if t in response
    ]

    return {
        "passed": len(found) == 0,
        "details": {
            "approximation_requested": approximation_requested,
            "found": found,
        },
    }


def function_or_equation_present_when_expected(record, row):
    text = _text(record)

    expects_model = _has_any(text, [
        "formula",
        "function",
        "model",
        "piecewise",
        "y=",
        "p(x)",
        "t(n)",
    ])

    if not expects_model:
        return None

    response = _boxed(row)

    equation_markers = [
        "=",
        "x",
        "y",
        "f(",
        "p(",
        "{",
    ]

    present = any(
        marker in response
        for marker in equation_markers
    )

    return {
        "passed": present,
        "details": {
            "equation_like_structure_present": present,
        },
    }


def inequality_or_interval_present(record, row):
    if not is_inequality_or_constraint_problem(record):
        return None

    response = _response(row)

    markers = [
        "<",
        ">",
        "≤",
        "≥",
        "(",
        ")",
        "[",
        "]",
        ",",
        "\\infty",
        "∞",
    ]

    present = any(
        marker in response
        for marker in markers
    )

    return {
        "passed": present,
        "details": {
            "inequality_or_interval_present": present,
        },
    }


def multi_answer_placeholder_discipline(record, row):
    expected = _ans_count(record)

    if expected <= 1:
        return None

    actual = row.get("actual_answer_count")

    return {
        "passed": expected == actual,
        "details": {
            "expected": expected,
            "actual": actual,
        },
    }


def avoids_unjustified_integer_rounding(record, row):
    text = _text(record)

    integer_required = _has_any(text, [
        "integer",
        "whole number",
        "packages",
        "beams",
        "people",
    ])

    if integer_required:
        return None

    response = _response(row).lower()

    suspicious_terms = [
        "must be an integer",
        "round up",
        "round down",
        "nearest whole",
    ]

    found = [
        term for term in suspicious_terms
        if term in response
    ]

    return {
        "passed": len(found) == 0,
        "details": {
            "integer_required": integer_required,
            "suspicious_terms_found": found,
        },
    }


def official_correct(record, row):
    if row.get("correct") is None:
        return None

    return row.get("correct")


def build_applied_word_problem_harness():
    return CategoryHarness(
        category=CATEGORY,
        checks=[
            HarnessCheck(
                "schema_valid",
                schema_valid,
                weight=1.0,
            ),
            HarnessCheck(
                "extractable",
                extractable,
                weight=1.0,
            ),
            HarnessCheck(
                "answer_count_valid",
                answer_count_valid,
                weight=1.0,
            ),
            HarnessCheck(
                "mcq_letter_valid",
                mcq_letter_valid,
                weight=1.0,
            ),
            HarnessCheck(
                "units_or_context_preserved",
                units_or_context_preserved,
                weight=0.5,
            ),
            HarnessCheck(
                "exact_form_preferred_unless_requested",
                exact_form_preferred_unless_requested,
                weight=0.75,
            ),
            HarnessCheck(
                "function_or_equation_present_when_expected",
                function_or_equation_present_when_expected,
                weight=0.75,
            ),
            HarnessCheck(
                "inequality_or_interval_present",
                inequality_or_interval_present,
                weight=0.5,
            ),
            HarnessCheck(
                "multi_answer_placeholder_discipline",
                multi_answer_placeholder_discipline,
                weight=0.75,
            ),
            HarnessCheck(
                "avoids_unjustified_integer_rounding",
                avoids_unjustified_integer_rounding,
                weight=0.75,
            ),
            HarnessCheck(
                "official_correct",
                official_correct,
                weight=2.0,
            ),
        ],
        metadata={
            "category": CATEGORY,
            "kind": "applied_word_problem_general",
        },
    )


def register_category_harness():
    harness = build_applied_word_problem_harness()
    register_harness(harness)
    return harness


def register_category_rules():
    rules = [
        DerivedRule(
            name="applied_word_problem_rate_ratio_model",
            category=CATEGORY,
            detector=is_rate_or_ratio_model_problem,
            description="Rate, ratio, pricing, tax, or proportional modeling problem.",
        ),
        DerivedRule(
            name="applied_word_problem_linear_modeling",
            category=CATEGORY,
            detector=is_linear_modeling_problem,
            description="Linear cost/profit/revenue/break-even modeling problem.",
        ),
        DerivedRule(
            name="applied_word_problem_exponential_modeling",
            category=CATEGORY,
            detector=is_exponential_modeling_problem,
            description="Exponential growth/decay/cooling/application problem.",
        ),
        DerivedRule(
            name="applied_word_problem_geometry_application",
            category=CATEGORY,
            detector=is_geometry_application_problem,
            description="Applied geometry or measurement problem.",
        ),
        DerivedRule(
            name="applied_word_problem_function_interpretation",
            category=CATEGORY,
            detector=is_function_interpretation_problem,
            description="Function interpretation or inverse-function meaning problem.",
        ),
        DerivedRule(
            name="applied_word_problem_piecewise_case",
            category=CATEGORY,
            detector=is_piecewise_or_case_problem,
            description="Piecewise-defined or conditional-case modeling problem.",
        ),
        DerivedRule(
            name="applied_word_problem_table_schedule_reasoning",
            category=CATEGORY,
            detector=is_table_or_schedule_reasoning_problem,
            description="Table-, schedule-, or transaction-based reasoning problem.",
        ),
        DerivedRule(
            name="applied_word_problem_quantity_tracking",
            category=CATEGORY,
            detector=is_multi_step_quantity_tracking_problem,
            description="Multi-step quantity/accounting/tracking problem.",
        ),
        DerivedRule(
            name="applied_word_problem_unit_conversion_application",
            category=CATEGORY,
            detector=is_unit_conversion_application_problem,
            description="Applied unit conversion or measurement interpretation problem.",
        ),
        DerivedRule(
            name="applied_word_problem_inequality_constraint",
            category=CATEGORY,
            detector=is_inequality_or_constraint_problem,
            description="Constraint, range, or inequality-based application problem.",
        ),
        DerivedRule(
            name="applied_word_problem_multi_step",
            category=CATEGORY,
            detector=is_multi_step_applied_problem,
            description="Multi-part or multi-step applied problem.",
        ),
        DerivedRule(
            name="applied_word_problem_multi_answer",
            category=CATEGORY,
            detector=is_multi_answer_problem,
            description="Problem requiring multiple extracted answers.",
        ),
        DerivedRule(
            name="applied_word_problem_mcq",
            category=CATEGORY,
            detector=is_mcq_problem,
            description="Multiple-choice applied word problem.",
        ),
    ]

    for rule in rules:
        register_rule(rule)

    return rules


def register_all():
    harness = register_category_harness()
    rules = register_category_rules()

    return {
        "category": CATEGORY,
        "default_strategy": DEFAULT_STRATEGY,
        "candidate_strategies": CANDIDATE_STRATEGIES,
        "harness": harness,
        "rules": rules,
    }