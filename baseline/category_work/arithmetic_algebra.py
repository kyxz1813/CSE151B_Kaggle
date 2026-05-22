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


CATEGORY = "arithmetic_algebra"

DEFAULT_STRATEGY = "arithmetic_algebra_v2_general"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "arithmetic_algebra_v2_general",
    "arithmetic_algebra_symbolic",
    "arithmetic_algebra_numeric",
    "arithmetic_algebra_multi_answer",
    "arithmetic_algebra_mcq",
]



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


def is_numeric_evaluation_problem(record):
    text = _text(record)

    return _has_any(text, [
        "evaluate",
        "compute",
        "find the value",
        "what is",
        "correct to",
        "find"
    ])


def is_symbolic_manipulation_problem(record):
    text = _text(record)

    symbolic_terms = [
        "factor",
        "expand",
        "simplify",
        "rewrite",
        "equivalent",
        "express",
        "composition",
        "inverse",
        "reduce"
    ]

    algebraic_markers = [
        "^",
        "(",
        ")",
        "x",
        "y",
        "t",
        "/"
    ]

    return (
        _has_any(text, symbolic_terms)
        and _has_any(text, algebraic_markers)
    )


def is_equation_solving_problem(record):
    text = _text(record)

    return _has_any(text, [
        "solve",
        "solutions",
        "roots",
        "zeros",
        "real number solutions",
        "complex numbers",
    ])


def is_function_or_relation_problem(record):
    text = _text(record)

    return _has_any(text, [
        "function",
        "f(x)",
        "h(x)",
        "r(x)",
        "inverse",
        "table of values",
        "sequence",
        "formula"
    ])


def is_discrete_integer_problem(record):
    text = _text(record)

    return (
        _has_any(text, [
            "integer",
            "divisible",
            "divisors",
            "multiple",
            "remainder",
            "positive integer",
            "smallest",
            "greatest",
        ])
    )


def is_conversion_or_representation_problem(record):
    text = _text(record)

    return _has_any(text, [
        "percent",
        "convert",
        "unit conversion",
        "hours",
        "minutes",
        "seconds",
        "log",
        "\\ln",
        "change of base",
    ])


def is_interval_or_set_problem(record):
    text = _text(record)

    return _has_any(text, [
        "union",
        "intersection",
        "interval",
        "subset",
        "\\cup",
        "\\cap",
    ])


def is_mcq_problem(record):
    return bool(record.get("options"))


def is_multi_answer_problem(record):
    return _ans_count(record) >= 2


def is_table_or_sequence_problem(record):
    text = _text(record)

    return (
        _has_any(text, [
            "table",
            "sequence",
            "classification",
        ])
    )



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
            "valid": valid,
        },
    }


def exact_form_preferred_unless_requested(record, row):
    text = _text(record)
    response = _response(row).lower()

    approximation_requested = _has_any(text, [
        "decimal",
        "nearest",
        "approximate",
        "correct to",
    ])

    if approximation_requested:
        return None

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
            "suspicious_terms": found,
        },
    }


def symbolic_answer_not_over_decimalized(record, row):
    if not is_symbolic_manipulation_problem(record):
        return None

    response = _boxed(row)

    decimal_count = len(
        re.findall(r"\d+\.\d+", response)
    )

    symbolic_markers = [
        "sqrt",
        "^",
        "/",
        "(",
        ")",
        "x",
    ]

    symbolic_present = any(
        marker in response
        for marker in symbolic_markers
    )

    return {
        "passed": (
            symbolic_present
            or decimal_count == 0
        ),
        "details": {
            "decimal_count": decimal_count,
            "symbolic_present": symbolic_present,
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


def interval_notation_present(record, row):
    if not is_interval_or_set_problem(record):
        return None

    response = _response(row)

    interval_markers = [
        "(",
        ")",
        "[",
        "]",
        "\\infty",
        "∞",
    ]

    present = any(
        marker in response
        for marker in interval_markers
    )

    return {
        "passed": present,
        "details": {
            "interval_notation_present": present,
        },
    }


def official_correct(record, row):
    if row.get("correct") is None:
        return None

    return row.get("correct")


def build_arithmetic_algebra_harness():
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
                "exact_form_preferred_unless_requested",
                exact_form_preferred_unless_requested,
                weight=0.75,
            ),
            HarnessCheck(
                "symbolic_answer_not_over_decimalized",
                symbolic_answer_not_over_decimalized,
                weight=0.75,
            ),
            HarnessCheck(
                "multi_answer_placeholder_discipline",
                multi_answer_placeholder_discipline,
                weight=0.75,
            ),
            HarnessCheck(
                "interval_notation_present",
                interval_notation_present,
                weight=0.5,
            ),
            HarnessCheck(
                "official_correct",
                official_correct,
                weight=2.0,
            ),
        ],
        metadata={
            "category": CATEGORY,
            "kind": "general_arithmetic_algebra",
        },
    )



def register_category_harness():
    harness = build_arithmetic_algebra_harness()
    register_harness(harness)
    return harness


def register_category_rules():
    rules = [
        DerivedRule(
            name="arithmetic_algebra_numeric_evaluation",
            category=CATEGORY,
            detector=is_numeric_evaluation_problem,
            description="Primarily numeric computation/evaluation problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_symbolic_manipulation",
            category=CATEGORY,
            detector=is_symbolic_manipulation_problem,
            description="Algebraic symbolic manipulation problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_equation_solving",
            category=CATEGORY,
            detector=is_equation_solving_problem,
            description="Equation solving/root-finding problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_function_or_relation",
            category=CATEGORY,
            detector=is_function_or_relation_problem,
            description="Function/sequence/table/inverse-function problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_discrete_integer",
            category=CATEGORY,
            detector=is_discrete_integer_problem,
            description="Discrete math/integer/divisibility problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_conversion_representation",
            category=CATEGORY,
            detector=is_conversion_or_representation_problem,
            description="Unit/representation/logarithm/percent conversion problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_interval_or_set",
            category=CATEGORY,
            detector=is_interval_or_set_problem,
            description="Interval/set notation problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_mcq",
            category=CATEGORY,
            detector=is_mcq_problem,
            description="Multiple-choice arithmetic/algebra problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_multi_answer",
            category=CATEGORY,
            detector=is_multi_answer_problem,
            description="Problem requiring multiple extracted answers.",
        ),
        DerivedRule(
            name="arithmetic_algebra_table_or_sequence",
            category=CATEGORY,
            detector=is_table_or_sequence_problem,
            description="Table-completion or sequence-classification problem.",
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