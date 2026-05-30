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


CATEGORY = "general_math"

DEFAULT_STRATEGY = "general_math_v1_structured"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "baseline3_adaptive_rules",
    "general_math_v1_structured",
    "general_math_mcq_verifier",
    "general_math_multi_answer",
    "general_math_text_or_unit",
]


def _text(record):
    pieces = [str(record.get("question", ""))]
    for option in record.get("options") or []:
        pieces.append(str(option))
    return "\n".join(pieces).lower()


def _response(row):
    return str(row.get("response") or row.get("raw_output") or "")


def _boxed(row):
    return str(row.get("boxed_answer") or row.get("extracted_final_answer") or "").strip()


def _ans_count(record):
    return str(record.get("question", "")).count("[ANS]")


def _has_any(text, terms):
    return any(term in text for term in terms)


def is_mcq_option_mapping(record):
    return bool(record.get("options"))


def is_multi_answer(record):
    return _ans_count(record) > 1


def is_text_answer(record):
    text = _text(record)
    return _has_any(text, [
        "enter your answer as",
        "enter:",
        "answer as",
        "do not put a period",
        "phrase",
        "weekday",
        "month",
        "increased or decreased",
        "true or false",
    ])


def is_unit_or_conversion(record):
    text = _text(record)
    return _has_any(text, [
        "convert",
        "conversion",
        "units",
        "liters",
        "milliliters",
        "meters",
        "miles",
        "feet",
        "hours",
        "minutes",
        "degrees",
        "percent",
        "%",
    ])


def is_percentage_or_rate(record):
    text = _text(record)
    return _has_any(text, [
        "percent",
        "%",
        "percentage",
        "rate",
        "ratio",
        "increased by",
        "decreased by",
        "discount",
        "markup",
        "interest",
    ])


def is_direct_formula(record):
    text = _text(record)
    return _has_any(text, [
        "formula",
        "modeled by",
        "function",
        "plug in",
        "substitute",
        "where x is",
        "where ",
    ])


def is_ordered_answer(record):
    text = _text(record)
    return _has_any(text, [
        "in order",
        "ordered pair",
        "from least to greatest",
        "from greatest to least",
        "list",
        "comma-separated",
    ]) or is_multi_answer(record)


def is_arithmetic_simplification(record):
    text = _text(record)
    return _has_any(text, [
        "simplify",
        "evaluate",
        "calculate",
        "fraction",
        "decimal",
        "nearest",
        "round",
        "absolute value",
    ])


def schema_valid(record, row):
    return row.get("schema_valid")


def extractable(record, row):
    return row.get("extractable")


def answer_count_valid(record, row):
    expected = row.get("expected_answer_count")
    actual = row.get("actual_answer_count")

    if expected is None:
        expected = _ans_count(record)

    if expected is None or actual is None:
        return None

    return {
        "passed": expected == actual,
        "details": {
            "expected_answer_count": expected,
            "actual_answer_count": actual,
        },
    }


def mcq_letter_valid(record, row):
    options = record.get("options") or []
    if not options:
        return None

    boxed = _boxed(row).upper()
    valid = [chr(ord("A") + idx) for idx in range(len(options))]

    return {
        "passed": boxed in valid,
        "details": {
            "boxed_answer": boxed,
            "valid_letters": valid,
        },
    }


def no_unmapped_numeric_for_mcq(record, row):
    if not record.get("options"):
        return None

    boxed = _boxed(row).strip()
    return {
        "passed": bool(re.fullmatch(r"[A-Z]", boxed.upper())),
        "details": {
            "boxed_answer": boxed,
        },
    }


def multi_answer_placeholder_discipline(record, row):
    if not is_multi_answer(record):
        return None

    actual = row.get("actual_answer_count")
    expected = _ans_count(record)
    return {
        "passed": actual == expected,
        "details": {
            "question_ans_count": expected,
            "actual_answer_count": actual,
            "boxed_answer": _boxed(row),
        },
    }


def ordered_answer_discipline(record, row):
    if not is_ordered_answer(record):
        return None

    boxed = _boxed(row)
    return {
        "passed": bool(boxed),
        "details": {
            "boxed_answer": boxed,
            "ordered_prompt": True,
        },
    }


def final_answer_not_explanatory(record, row):
    boxed = _boxed(row)
    if not boxed:
        return None

    bad_markers = [
        "because",
        "therefore",
        "so ",
        "since",
        "final answer",
    ]
    lower = boxed.lower()
    found = [marker for marker in bad_markers if marker in lower]

    return {
        "passed": len(found) == 0,
        "details": {
            "bad_markers_found": found,
            "boxed_answer": boxed[:200],
        },
    }


def official_correct(record, row):
    if row.get("correct") is None:
        return None
    return row.get("correct")


def build_general_math_harness():
    return CategoryHarness(
        category=CATEGORY,
        checks=[
            HarnessCheck("schema_valid", schema_valid, weight=1.0),
            HarnessCheck("extractable", extractable, weight=1.0),
            HarnessCheck("answer_count_valid", answer_count_valid, weight=1.0),
            HarnessCheck("mcq_letter_valid", mcq_letter_valid, weight=1.0),
            HarnessCheck("no_unmapped_numeric_for_mcq", no_unmapped_numeric_for_mcq, weight=0.75),
            HarnessCheck("multi_answer_placeholder_discipline", multi_answer_placeholder_discipline, weight=0.75),
            HarnessCheck("ordered_answer_discipline", ordered_answer_discipline, weight=0.5),
            HarnessCheck("final_answer_not_explanatory", final_answer_not_explanatory, weight=0.5),
            HarnessCheck("official_correct", official_correct, weight=2.0),
        ],
        metadata={
            "category": CATEGORY,
            "kind": "general_math_specialized",
        },
    )


def register_category_harness():
    harness = build_general_math_harness()
    register_harness(harness)
    return harness


def register_category_rules():
    rules = [
        DerivedRule(
            name="general_math_mcq_option_mapping",
            category=CATEGORY,
            detector=is_mcq_option_mapping,
            description="Multiple-choice fallback problem requiring final option-letter mapping.",
        ),
        DerivedRule(
            name="general_math_multi_answer",
            category=CATEGORY,
            detector=is_multi_answer,
            description="Fallback problem with multiple [ANS] blanks.",
        ),
        DerivedRule(
            name="general_math_text_answer",
            category=CATEGORY,
            detector=is_text_answer,
            description="Problem requiring a phrase, word, true/false, direction, or exact text answer.",
        ),
        DerivedRule(
            name="general_math_unit_or_conversion",
            category=CATEGORY,
            detector=is_unit_or_conversion,
            description="Fallback problem involving units, percentages, or conversion checks.",
        ),
        DerivedRule(
            name="general_math_percentage_or_rate",
            category=CATEGORY,
            detector=is_percentage_or_rate,
            description="Fallback problem involving percentage, ratio, rate, discount, markup, or interest.",
        ),
        DerivedRule(
            name="general_math_direct_formula",
            category=CATEGORY,
            detector=is_direct_formula,
            description="Fallback problem solved by direct substitution into a formula or model.",
        ),
        DerivedRule(
            name="general_math_ordered_answer",
            category=CATEGORY,
            detector=is_ordered_answer,
            description="Fallback problem requiring ordered or comma-separated final answers.",
        ),
        DerivedRule(
            name="general_math_arithmetic_simplification",
            category=CATEGORY,
            detector=is_arithmetic_simplification,
            description="Fallback problem requiring careful arithmetic, simplification, or rounding.",
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
