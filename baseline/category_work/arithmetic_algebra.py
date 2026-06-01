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



def is_temperature_conversion_problem(record):
    text = _text(record)
    return _has_any(text, [
        "degrees celsius",
        "degrees kelvin",
        "degrees rankine",
        "fahrenheit",
        "celsius",
        "kelvin",
        "rankine",
        "^\circ",
    ]) and _has_any(text, ["fahrenheit", "f"])


def is_bernstein_polynomial_problem(record):
    text = _text(record)
    return "bernstein" in text and "polynomial" in text


def is_base_arithmetic_problem(record):
    text = _text(record)
    return _has_any(text, [
        "binary",
        "base",
        "base-",
        "add the following binary",
    ])


def is_numeric_precision_problem(record):
    text = _text(record)
    return _has_any(text, [
        "at least",
        "accurate to",
        "correct to",
        "decimal",
        "graphically",
        "approx",
        "approximately",
    ])


def is_floor_log_sum_problem(record):
    text = _text(record)
    return (
        ("floor" in text or "\\lfloor" in text)
        and _has_any(text, ["log_2", "log2", "integer part", "remainder"])
    )


def is_letter_set_selection_problem(record):
    text = _text(record)
    return _has_any(text, [
        "list the letter",
        "which of the following",
        "answer(s)",
        "symmetry",
        "equivalent",
    ]) and not record.get("options")



def is_formula_then_evaluate_problem(record):
    text = _text(record)
    return (
        _ans_count(record) >= 2
        and _has_any(text, [
            "can be simplified to the form",
            "simplified to the form",
            "where a and b",
            "a=",
            "b=",
            "suppose that",
            "then",
            "contains no fractions",
            "contain no fractions",
        ])
        and _has_any(text, [
            "formula",
            "expression",
            "simplified",
            "suppose",
            "then",
        ])
    )


def is_exponential_log_solve_problem(record):
    text = _text(record)
    return (
        _has_any(text, ["graphically", "solve", "log", "ln"])
        and re.search(r"\b[a-z]\s*=\s*[-+]?\d", text) is not None
        and re.search(r"\([^)]*\)\s*\^\s*[a-z]", text) is not None
    ) or (
        "^q" in text and _has_any(text, ["graphically", "solve"])
    )


def is_ordered_pair_answer_problem(record):
    text = _text(record)
    return _has_any(text, [
        "ordered pair",
        "point",
        "coordinate",
        "zero of the polynomial",
        "zeros of the polynomial",
        "pair",
    ]) and _ans_count(record) >= 1


def is_canonical_symbolic_syntax_problem(record):
    text = _text(record)
    return (
        _has_any(text, [
            "formula",
            "express your answer",
            "simplified to the form",
            "write the formula",
            "find a simplified formula",
        ])
        and _has_any(text, ["t", "x", "y", "e^", "exp", "^", "*"])
    )


def is_scientific_expression_style_problem(record):
    text = _text(record)
    return _has_any(text, [
        "e^",
        "exp",
        "exponential",
        "simplified formula",
        "let u(x)",
        "let v(x)",
    ]) and _ans_count(record) >= 1


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


def high_precision_when_requested(record, row):
    if not is_numeric_precision_problem(record):
        return None

    boxed = _boxed(row)
    decimals = re.findall(r"\d+\.(\d+)", boxed)

    if not decimals:
        return None

    max_digits = max(len(item) for item in decimals)

    return {
        "passed": max_digits >= 4,
        "details": {
            "max_decimal_digits": max_digits,
            "boxed": boxed,
        },
    }


def binary_answer_digits_valid(record, row):
    if not is_base_arithmetic_problem(record):
        return None

    boxed = _boxed(row).replace(" ", "")
    pieces = [item for item in boxed.split(",") if item]

    if not pieces:
        return None

    bad = [item for item in pieces if not re.fullmatch(r"[01]+", item)]

    return {
        "passed": len(bad) == 0,
        "details": {
            "bad_binary_items": bad,
            "boxed": boxed,
        },
    }


def temperature_answer_count_valid(record, row):
    if not is_temperature_conversion_problem(record):
        return None

    expected = _ans_count(record)
    actual = row.get("actual_answer_count")

    return {
        "passed": actual == expected,
        "details": {
            "expected_temperature_answers": expected,
            "actual_answer_count": actual,
        },
    }


def temperature_high_precision_valid(record, row):
    if not is_temperature_conversion_problem(record):
        return None

    boxed = _boxed(row)
    parts = [part.strip() for part in boxed.split(",") if part.strip()]

    if len(parts) < 2:
        return {
            "passed": False,
            "details": {"reason": "fewer_than_two_temperature_answers", "boxed": boxed},
        }

    decimal_digits = []
    for part in parts[:2]:
        match = re.search(r"\.([0-9]+)", part)
        decimal_digits.append(len(match.group(1)) if match else 0)

    return {
        "passed": min(decimal_digits) >= 10,
        "details": {
            "first_two_decimal_digits": decimal_digits,
            "boxed": boxed,
        },
    }


def bernstein_canonical_syntax_valid(record, row):
    if not is_bernstein_polynomial_problem(record):
        return None

    boxed = _boxed(row)
    expected = _ans_count(record)
    actual = row.get("actual_answer_count")

    has_explicit_multiplication = "*" in boxed
    has_power_one = "^1" in boxed
    has_implicit_coeff_var = re.search(r"\b\d+[a-zA-Z]", boxed) is not None

    return {
        "passed": (
            actual == expected
            and has_explicit_multiplication
            and has_power_one
            and not has_implicit_coeff_var
        ),
        "details": {
            "actual_answer_count": actual,
            "expected_answer_count": expected,
            "has_explicit_multiplication": has_explicit_multiplication,
            "has_power_one": has_power_one,
            "has_implicit_coeff_var": bool(has_implicit_coeff_var),
            "boxed": boxed,
        },
    }


def formula_blank_not_numeric_substitution(record, row):
    if not is_formula_then_evaluate_problem(record):
        return None

    boxed = _boxed(row)
    parts = [part.strip() for part in boxed.split(",") if part.strip()]

    if len(parts) < 2:
        return {
            "passed": False,
            "details": {"reason": "not_enough_parts", "boxed": boxed},
        }

    first_two_have_symbols = all(
        re.search(r"[A-Za-z]", part) is not None
        for part in parts[:2]
    )

    first_two_are_plain_numbers = any(
        re.fullmatch(r"[-+]?\d+(?:\.\d+)?", part) is not None
        for part in parts[:2]
    )

    return {
        "passed": first_two_have_symbols and not first_two_are_plain_numbers,
        "details": {
            "first_two_have_symbols": first_two_have_symbols,
            "first_two_are_plain_numbers": first_two_are_plain_numbers,
            "first_two_parts": parts[:2],
        },
    }


def letter_set_no_commas_valid(record, row):
    if not is_letter_set_selection_problem(record):
        return None

    boxed = _boxed(row).strip()
    compact = boxed.replace(" ", "")

    if not re.fullmatch(r"[A-Z,]+", compact):
        return None

    return {
        "passed": "," not in compact,
        "details": {
            "boxed": boxed,
            "compact": compact,
        },
    }


def ordered_pair_format_valid(record, row):
    if not is_ordered_pair_answer_problem(record):
        return None

    boxed = _boxed(row).strip()
    looks_like_pair = bool(re.search(r"\([^()]+,[^()]+\)", boxed))

    return {
        "passed": looks_like_pair,
        "details": {
            "boxed": boxed,
            "looks_like_pair": looks_like_pair,
        },
    }


def exponential_solution_precision_valid(record, row):
    if not is_exponential_log_solve_problem(record):
        return None

    boxed = _boxed(row)
    decimals = re.findall(r"\d+\.([0-9]+)", boxed)

    if not decimals:
        return {
            "passed": False,
            "details": {"reason": "no_decimal_answer", "boxed": boxed},
        }

    max_digits = max(len(item) for item in decimals)

    return {
        "passed": max_digits >= 4,
        "details": {
            "max_decimal_digits": max_digits,
            "boxed": boxed,
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
                "high_precision_when_requested",
                high_precision_when_requested,
                weight=0.5,
            ),
            HarnessCheck(
                "binary_answer_digits_valid",
                binary_answer_digits_valid,
                weight=0.5,
            ),
            HarnessCheck(
                "temperature_answer_count_valid",
                temperature_answer_count_valid,
                weight=0.5,
            ),
            HarnessCheck(
                "temperature_high_precision_valid",
                temperature_high_precision_valid,
                weight=0.5,
            ),
            HarnessCheck(
                "bernstein_canonical_syntax_valid",
                bernstein_canonical_syntax_valid,
                weight=0.5,
            ),
            HarnessCheck(
                "formula_blank_not_numeric_substitution",
                formula_blank_not_numeric_substitution,
                weight=0.5,
            ),
            HarnessCheck(
                "letter_set_no_commas_valid",
                letter_set_no_commas_valid,
                weight=0.5,
            ),
            HarnessCheck(
                "ordered_pair_format_valid",
                ordered_pair_format_valid,
                weight=0.5,
            ),
            HarnessCheck(
                "exponential_solution_precision_valid",
                exponential_solution_precision_valid,
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
        DerivedRule(
            name="arithmetic_algebra_temperature_conversion",
            category=CATEGORY,
            detector=is_temperature_conversion_problem,
            description="Temperature conversion among Fahrenheit, Celsius, Kelvin, and Rankine.",
        ),
        DerivedRule(
            name="arithmetic_algebra_bernstein_polynomial",
            category=CATEGORY,
            detector=is_bernstein_polynomial_problem,
            description="Bernstein polynomial formula problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_base_arithmetic",
            category=CATEGORY,
            detector=is_base_arithmetic_problem,
            description="Binary/base arithmetic problem requiring carry in the stated base.",
        ),
        DerivedRule(
            name="arithmetic_algebra_numeric_precision",
            category=CATEGORY,
            detector=is_numeric_precision_problem,
            description="Numeric approximation problem where extra precision is safer.",
        ),
        DerivedRule(
            name="arithmetic_algebra_floor_log_sum",
            category=CATEGORY,
            detector=is_floor_log_sum_problem,
            description="Floor-logarithm summation/counting problem.",
        ),
        DerivedRule(
            name="arithmetic_algebra_letter_set_selection",
            category=CATEGORY,
            detector=is_letter_set_selection_problem,
            description="Free-form answer consisting of selected letters such as BCEG.",
        ),
        DerivedRule(
            name="arithmetic_algebra_formula_then_evaluate",
            category=CATEGORY,
            detector=is_formula_then_evaluate_problem,
            description="Problem asks for symbolic formula blanks first and numerical evaluation later.",
        ),
        DerivedRule(
            name="arithmetic_algebra_exponential_log_solve",
            category=CATEGORY,
            detector=is_exponential_log_solve_problem,
            description="Exponential equation solved using logarithms with high precision.",
        ),
        DerivedRule(
            name="arithmetic_algebra_ordered_pair_answer",
            category=CATEGORY,
            detector=is_ordered_pair_answer_problem,
            description="Problem expects an ordered pair or coordinate-style final answer.",
        ),
        DerivedRule(
            name="arithmetic_algebra_canonical_symbolic_syntax",
            category=CATEGORY,
            detector=is_canonical_symbolic_syntax_problem,
            description="Symbolic answer should use explicit multiplication and canonical syntax.",
        ),
        DerivedRule(
            name="arithmetic_algebra_scientific_expression_style",
            category=CATEGORY,
            detector=is_scientific_expression_style_problem,
            description="Expression answer involving e, exp, powers, or simplified symbolic formula style.",
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