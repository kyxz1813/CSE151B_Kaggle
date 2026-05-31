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


CATEGORY = "calculus"

DEFAULT_STRATEGY = "calculus_v1_structured"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "baseline3_adaptive_rules",
    "calculus_v1_structured",
    "calculus_limit_asymptotic",
    "calculus_integral",
    "calculus_derivative_extrema",
    "calculus_differential_equation",
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


def is_limit_asymptotic(record):
    text = _text(record)
    return _has_any(text, [
        "limit",
        "lim_",
        "\\lim",
        "as n approaches",
        "as x approaches",
        "approaches infinity",
        "asymptotic",
    ])


def is_integral(record):
    text = _text(record)
    return _has_any(text, [
        "integral",
        "integrate",
        "antiderivative",
        "\\int",
        "contour",
        "residue",
        "residues",
    ])


def is_derivative_extrema(record):
    text = _text(record)
    return _has_any(text, [
        "derivative",
        "differentiate",
        "tangent",
        "linear approximation",
        "best first-degree approximation",
        "maximum",
        "minimum",
        "extrema",
        "critical point",
        "rate of change",
    ])


def is_endpoint_extrema(record):
    text = _text(record)
    return is_derivative_extrema(record) and _has_any(text, [
        "closed interval",
        "on [",
        "on the interval",
        "absolute maximum",
        "absolute minimum",
        "global maximum",
        "global minimum",
    ])


def is_series_approximation(record):
    text = _text(record)
    return _has_any(text, [
        "series",
        "taylor",
        "maclaurin",
        "power series",
        "approximation polynomial",
        "first-degree approximation",
    ])


def is_differential_equation(record):
    text = _text(record)
    return _has_any(text, [
        "differential equation",
        "dy/dx",
        "separable",
        "initial condition",
        "newton's law of cooling",
        "exponential decay",
    ])


def is_initial_condition(record):
    text = _text(record)
    return _has_any(text, [
        "initial condition",
        " y(",
        " f(",
        "when x =",
        "when t =",
        "at time",
    ]) and is_differential_equation(record)


def is_complex_residue(record):
    text = _text(record)
    return _has_any(text, [
        "residue",
        "residues",
        "contour",
        "complex integral",
        "analytic function",
        "tan(\\pi z)",
        "tan(πz)",
    ])


def is_mcq_option_mapping(record):
    return bool(record.get("options"))


def is_improper_parameter_integral(record):
    text = _text(record)
    return is_integral(record) and _has_any(text, [
        "-infty",
        "+infty",
        "infinity",
        "parameter",
        " a ",
        " a^",
        "improper",
    ])


def is_definite_integral_substitution(record):
    text = _text(record)
    return is_integral(record) and _has_any(text, [
        "int_{0}",
        "int_0",
        "substitution",
        "sqrt",
        "u=",
        "definite",
        "compute the integral",
    ])


def is_equation_root_dichotomy(record):
    text = _text(record)
    return _has_any(text, [
        "find all solutions",
        "solve",
        "dichotomy",
        "bisection",
        "graphing",
        "root",
    ]) and _has_any(text, ["equation", "= 0", "=0"])


def is_mcq_absent_option_risk(record):
    text = _text(record)
    return bool(record.get("options")) and _has_any(text, [
        "integral",
        "compute",
        "evaluate",
        "what is",
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


def calculus_method_evidence(record, row):
    text = _text(record)
    response = _response(row).lower()

    expected_terms = []
    if is_limit_asymptotic(record):
        expected_terms.extend(["limit", "dominant", "l'hopital", "taylor", "expand", "rational"])
    if is_integral(record):
        expected_terms.extend(["integral", "substitution", "parts", "antiderivative", "residue", "bounds"])
    if is_derivative_extrema(record):
        expected_terms.extend(["derivative", "differentiate", "critical", "tangent", "slope"])
    if is_differential_equation(record):
        expected_terms.extend(["separate", "differential", "initial", "constant", "exponential"])

    if not expected_terms:
        return None

    found = [term for term in expected_terms if term in response]

    return {
        "passed": bool(found),
        "details": {
            "matched_terms": found,
            "question_triggered": text[:160],
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


def final_answer_not_explanatory(record, row):
    boxed = _boxed(row)
    if not boxed:
        return None

    lower = boxed.lower()
    bad_markers = [
        "because",
        "therefore",
        "so ",
        "since",
        "final answer",
        "we get",
    ]
    found = [marker for marker in bad_markers if marker in lower]

    return {
        "passed": not found,
        "details": {
            "bad_markers_found": found,
            "boxed_answer": boxed[:200],
        },
    }


def mcq_must_not_be_blank(record, row):
    if not record.get("options"):
        return None

    boxed = _boxed(row)
    return {
        "passed": bool(boxed),
        "details": {
            "boxed": boxed,
        },
    }


def official_correct(record, row):
    if row.get("correct") is None:
        return None
    return row.get("correct")


def build_calculus_harness():
    return CategoryHarness(
        category=CATEGORY,
        checks=[
            HarnessCheck("schema_valid", schema_valid, weight=1.0),
            HarnessCheck("extractable", extractable, weight=1.0),
            HarnessCheck("answer_count_valid", answer_count_valid, weight=1.0),
            HarnessCheck("mcq_letter_valid", mcq_letter_valid, weight=1.0),
            HarnessCheck("calculus_method_evidence", calculus_method_evidence, weight=0.75),
            HarnessCheck("no_unmapped_numeric_for_mcq", no_unmapped_numeric_for_mcq, weight=0.75),
            HarnessCheck("final_answer_not_explanatory", final_answer_not_explanatory, weight=0.5),
            HarnessCheck("official_correct", official_correct, weight=2.0),
        ],
        metadata={
            "category": CATEGORY,
            "kind": "calculus_specialized",
        },
    )


def register_category_harness():
    harness = build_calculus_harness()
    register_harness(harness)
    return harness


def register_category_rules():
    rules = [
        DerivedRule(
            name="calculus_limit_asymptotic",
            category=CATEGORY,
            detector=is_limit_asymptotic,
            description="Limit, asymptotic, or sequence-limit calculus problem.",
        ),
        DerivedRule(
            name="calculus_integral",
            category=CATEGORY,
            detector=is_integral,
            description="Integral, antiderivative, contour, or residue problem.",
        ),
        DerivedRule(
            name="calculus_derivative_extrema",
            category=CATEGORY,
            detector=is_derivative_extrema,
            description="Derivative, tangent, rate, approximation, or extrema problem.",
        ),
        DerivedRule(
            name="calculus_endpoint_extrema_check",
            category=CATEGORY,
            detector=is_endpoint_extrema,
            description="Extrema problem requiring critical point and endpoint comparison.",
        ),
        DerivedRule(
            name="calculus_series_approximation",
            category=CATEGORY,
            detector=is_series_approximation,
            description="Taylor/Maclaurin/series or approximation polynomial problem.",
        ),
        DerivedRule(
            name="calculus_differential_equation",
            category=CATEGORY,
            detector=is_differential_equation,
            description="Differential equation or calculus model problem.",
        ),
        DerivedRule(
            name="calculus_initial_condition_check",
            category=CATEGORY,
            detector=is_initial_condition,
            description="Differential equation or model requiring constants from conditions.",
        ),
        DerivedRule(
            name="calculus_complex_residue",
            category=CATEGORY,
            detector=is_complex_residue,
            description="Complex contour or residue theorem problem.",
        ),
        DerivedRule(
            name="calculus_mcq_option_mapping",
            category=CATEGORY,
            detector=is_mcq_option_mapping,
            description="Calculus MCQ requiring final option-letter mapping.",
        ),
        DerivedRule(
            name="calculus_improper_parameter_integral",
            category=CATEGORY,
            detector=is_improper_parameter_integral,
            description="Improper integral with a parameter or infinite bounds.",
        ),
        DerivedRule(
            name="calculus_definite_integral_substitution",
            category=CATEGORY,
            detector=is_definite_integral_substitution,
            description="Definite integral likely requiring substitution and endpoint tracking.",
        ),
        DerivedRule(
            name="calculus_equation_root_dichotomy",
            category=CATEGORY,
            detector=is_equation_root_dichotomy,
            description="Equation/root-solving numerical calculus problem.",
        ),
        DerivedRule(
            name="calculus_mcq_absent_option_fallback",
            category=CATEGORY,
            detector=is_mcq_absent_option_risk,
            description="MCQ where computed expression must still be mapped to one option letter.",
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
