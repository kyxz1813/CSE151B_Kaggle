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

DEFAULT_STRATEGY = "calculus_v2_subtype_adaptive"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "baseline3_adaptive_rules",
    "calculus_v1_structured",
    "calculus_v2_subtype_adaptive",
    "calculus_integral_verifier",
    "calculus_precision_freeform",
    "calculus_differential_equation",
]


def _text(record):
    pieces = [str(record.get("question", ""))]
    for option in record.get("options") or []:
        pieces.append(str(option))
    return "\n".join(pieces).lower()


def _question_text(record):
    return str(record.get("question", "")).lower()


def _raw_response(row):
    return str(
        row.get("raw_output")
        or row.get("initial_raw_output")
        or row.get("retry_raw_output")
        or row.get("response")
        or ""
    )


def _response(row):
    return str(row.get("response") or row.get("raw_output") or "")


def _boxed(row):
    return str(row.get("boxed_answer") or row.get("extracted_final_answer") or "").strip()


def _ans_count(record):
    return str(record.get("question", "")).count("[ANS]")


def _has_any(text, terms):
    return any(term in text for term in terms)


def _has_all(text, terms):
    return all(term in text for term in terms)


def _options_text(record):
    return "\n".join(str(option) for option in record.get("options") or []).lower()


def _is_mcq(record):
    return bool(record.get("options"))


def _freeform(record):
    return not _is_mcq(record)


# ============================================================
# Broad existing detectors
# ============================================================

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
        "int_",
        "int_{",
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
        "slope",
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
        "sum of the series",
        "estimate error",
    ])


def is_differential_equation(record):
    text = _text(record)
    return _has_any(text, [
        "differential equation",
        "dy/dx",
        "y'",
        "y^{\\prime}",
        "y^{\prime}",
        "separable",
        "initial condition",
        "newton's law of cooling",
        "exponential decay",
        "ivp",
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
        "where y",
        "y(\\pi)",
        "y(\pi)",
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
    return _is_mcq(record)


# ============================================================
# Calculus v2 fine-grained detectors
# ============================================================

def is_calculus_numeric_precision_freeform(record):
    text = _text(record)
    return _freeform(record) and (
        _has_any(text, [
            "round",
            "nearest",
            "approximately",
            "approximate",
            "calculator",
            "graphically",
            "estimate",
            "at least",
            "decimal",
            "possible error",
            "maximum error",
            "minimum",
            "maximum",
        ])
        or bool(re.search(r"\[ans\].*(hours|fahrenheit|cm|million|tons|percent|%)", text))
    )


def is_calculus_exponential_model(record):
    text = _text(record)
    return _has_any(text, [
        "newton's law of cooling",
        "cools",
        "cool to",
        "cooling",
        "temperature",
        "room temperature",
        "continuous rate",
        "growth rate",
        "exponential",
        "increasing at a continuous rate",
        "decreasing at a continuous rate",
        "production",
    ]) and _freeform(record)


def is_calculus_newton_cooling(record):
    text = _text(record)
    return _has_any(text, [
        "cool",
        "cools",
        "cooling",
        "turkey",
        "oven",
        "room temperature",
        "newton",
    ])


def is_calculus_actual_max_error(record):
    text = _text(record)
    return _has_all(text, ["maximum error", "measured"]) or _has_any(text, [
        "possible error in measurement",
        "using this value",
        "maximum error must be less than",
    ])


def is_calculus_antiderivative_mcq(record):
    text = _question_text(record)
    options = _options_text(record)
    return _is_mcq(record) and is_integral(record) and (
        "+c" in options
        or "+ c" in options
        or "antiderivative" in text
        or "compute the integral" in text
        or bool(re.search(r"\\int(?!_)", text))
    )


def is_calculus_definite_integral_mcq(record):
    text = _question_text(record)
    return _is_mcq(record) and is_integral(record) and (
        "int_" in text
        or "\\int_" in text
        or "int_{" in text
        or "from" in text and "to" in text
        or "estimate" in text
    )


def is_calculus_trig_integral(record):
    text = _text(record)
    return is_integral(record) and _has_any(text, [
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
    ])


def is_calculus_improper_parameter_integral(record):
    text = _text(record)
    return is_integral(record) and _has_any(text, [
        "infty",
        "infinity",
        "-infty",
        "+infty",
        "parameter",
    ]) and _has_any(text, [" a", "a^", "a^{", "frac{a", "sqrt{a"])


def is_calculus_trig_derivative(record):
    text = _text(record)
    return is_derivative_extrema(record) and _has_any(text, [
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
    ])


def is_calculus_differentiation_under_integral(record):
    text = _text(record)
    return is_integral(record) and _has_any(text, [
        "d} {\\mathrm{d} y",
        "d}} {\\mathrm{d} y",
        "frac{\\mathrm{d}}{\\mathrm{d} y}",
        "frac{\\mathrm{d}} {\\mathrm{d} y}",
        "d/dy",
        "with respect to y",
    ])


def is_calculus_implicit_differentiation(record):
    text = _text(record)
    return _has_any(text, [
        "determined by the equation",
        "implicit",
        "e}^{x+y}",
        "e^{x+y}",
        "xy+1",
    ]) and _has_any(text, ["d}x", "d}y", "dx", "dy", "then ( )"])


def is_calculus_piecewise_differential_equation(record):
    text = _text(record)
    return is_differential_equation(record) and _has_any(text, [
        "x \\leq\\pi",
        "x \\leq\pi",
        "x <= pi",
        "x>\\pi",
        "x > \\pi",
        "continuous at",
        "piecewise",
        "\\begin{matrix}",
    ])


def is_calculus_first_order_ivp(record):
    text = _text(record)
    return is_differential_equation(record) and _has_any(text, [
        "ivp",
        "y(\\pi)",
        "y(\pi)",
        "when t=pi/2",
        "when t=pi",
        "where y",
        "initial",
    ])


def is_calculus_volume_revolution(record):
    text = _text(record)
    return _has_any(text, [
        "volume of",
        "solid of revolution",
        "volume of the solid",
        "volume of a vase",
        "rotating the curve",
        "rotated about",
        "around the x-axis",
        "about the x-axis",
    ])


def is_calculus_surface_area_revolution(record):
    text = _text(record)
    return _has_any(text, [
        "surface formed by rotating",
        "surface area",
        "surface of revolution",
        "astroid",
    ])


def is_calculus_area_between_curves(record):
    text = _text(record)
    return _has_any(text, [
        "area of the figure enclosed",
        "area enclosed",
        "area between",
        "between the curves",
        "enclosed between",
    ])


def is_calculus_optimization_geometry(record):
    text = _text(record)
    return is_derivative_extrema(record) and _has_any(text, [
        "wire",
        "square",
        "circle",
        "cube",
        "volume",
        "area",
        "maximum error",
        "minimum",
        "maximize",
        "minimize",
    ])


def is_calculus_domain_range_radical_rational(record):
    text = _text(record)
    return _has_any(text, ["domain", "range"]) and _has_any(text, [
        "sqrt",
        "\\sqrt",
        "radical",
        "denominator",
    ])


def is_calculus_fourier_sobolev_boundary(record):
    text = _text(record)
    return _has_any(text, [
        "sobolev",
        "h^{1}",
        "fourier",
        "e^{-2 \\pi i n x}",
        "lim}_{n",
        "n \\int",
        "n int",
    ])


def is_calculus_series_special_function(record):
    text = _text(record)
    return is_series_approximation(record) and _has_any(text, [
        "infty",
        "infinite",
        "factorial",
        "2\\cdot 4",
        "2^2",
        "estimate error",
        "sum of the series",
    ])


def is_calculus_discrete_series_convergence(record):
    text = _text(record)
    return _has_any(text, [
        "base 3 representation",
        "number of zeroes",
        "x^{a(n)}",
        "n^3",
        "series",
        "converge",
    ])


def is_calculus_numeric_mcq_close_options(record):
    if not _is_mcq(record):
        return False
    options = record.get("options") or []
    numeric_count = 0
    for option in options:
        if re.search(r"-?\d+(?:\.\d+)?", str(option)):
            numeric_count += 1
    return numeric_count >= max(3, len(options) // 2)


# ============================================================
# Harness checks
# ============================================================

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
    response = _raw_response(row).lower()

    expected_terms = []
    if is_limit_asymptotic(record):
        expected_terms.extend(["limit", "dominant", "l'hopital", "taylor", "expand", "rational"])
    if is_integral(record):
        expected_terms.extend(["integral", "substitution", "parts", "antiderivative", "residue", "bounds", "differentiate"])
    if is_derivative_extrema(record):
        expected_terms.extend(["derivative", "differentiate", "critical", "tangent", "slope"])
    if is_differential_equation(record):
        expected_terms.extend(["separate", "differential", "initial", "constant", "exponential", "integrating factor"])
    if is_calculus_volume_revolution(record) or is_calculus_surface_area_revolution(record):
        expected_terms.extend(["pi", "integral", "surface", "volume", "rotate", "axis"])

    if not expected_terms:
        return None

    found = [term for term in expected_terms if term in response]

    return {
        "passed": bool(found),
        "details": {
            "matched_terms": found,
            "question_triggered": text[:220],
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


def high_precision_numeric_freeform(record, row):
    if not is_calculus_numeric_precision_freeform(record):
        return None

    boxed = _boxed(row)
    numbers = re.findall(r"-?\d+\.\d+", boxed)
    if not numbers:
        return None

    low_precision = []
    for number in numbers:
        decimals = number.split(".", 1)[1]
        if len(decimals) < 4:
            low_precision.append(number)

    return {
        "passed": len(low_precision) == 0,
        "details": {
            "low_precision_numbers": low_precision,
            "boxed_answer": boxed[:200],
        },
    }


def mcq_option_verification_evidence(record, row):
    if not _is_mcq(record):
        return None

    response = _raw_response(row).lower()
    evidence_terms = [
        "option",
        "choice",
        "compare",
        "matches",
        "differentiate the option",
        "closest",
        "evaluate each",
        "numerically",
    ]
    found = [term for term in evidence_terms if term in response]

    return {
        "passed": bool(found),
        "details": {
            "evidence_terms_found": found,
        },
    }


def actual_error_not_differential_only(record, row):
    if not is_calculus_actual_max_error(record):
        return None

    response = _raw_response(row).lower()
    boxed = _boxed(row)
    used_exact_change = any(term in response for term in [
        "(28.006)",
        "actual",
        "exact error",
        "upper value",
        "new volume",
        "difference",
    ])
    used_only_differential = "dv" in response and "3l" in response and not used_exact_change

    return {
        "passed": used_exact_change and not used_only_differential,
        "details": {
            "used_exact_change": used_exact_change,
            "used_only_differential": used_only_differential,
            "boxed_answer": boxed,
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
            HarnessCheck("mcq_option_verification_evidence", mcq_option_verification_evidence, weight=0.75),
            HarnessCheck("high_precision_numeric_freeform", high_precision_numeric_freeform, weight=0.75),
            HarnessCheck("actual_error_not_differential_only", actual_error_not_differential_only, weight=0.75),
            HarnessCheck("no_unmapped_numeric_for_mcq", no_unmapped_numeric_for_mcq, weight=0.75),
            HarnessCheck("final_answer_not_explanatory", final_answer_not_explanatory, weight=0.5),
            HarnessCheck("official_correct", official_correct, weight=2.0),
        ],
        metadata={
            "category": CATEGORY,
            "kind": "calculus_specialized_v2",
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
            name="calculus_numeric_precision_freeform",
            category=CATEGORY,
            detector=is_calculus_numeric_precision_freeform,
            description="Free-form calculus answer where extra numeric precision is safer.",
        ),
        DerivedRule(
            name="calculus_exponential_model",
            category=CATEGORY,
            detector=is_calculus_exponential_model,
            description="Newton cooling, exponential growth/decay, or continuous-rate model.",
        ),
        DerivedRule(
            name="calculus_newton_cooling",
            category=CATEGORY,
            detector=is_calculus_newton_cooling,
            description="Newton cooling temperature problem.",
        ),
        DerivedRule(
            name="calculus_actual_max_error",
            category=CATEGORY,
            detector=is_calculus_actual_max_error,
            description="Maximum error problem requiring actual endpoint difference, not only differential estimate.",
        ),
        DerivedRule(
            name="calculus_antiderivative_mcq_verify",
            category=CATEGORY,
            detector=is_calculus_antiderivative_mcq,
            description="Indefinite integral MCQ where options should be verified by differentiating.",
        ),
        DerivedRule(
            name="calculus_definite_integral_numeric_mcq",
            category=CATEGORY,
            detector=is_calculus_definite_integral_mcq,
            description="Definite integral MCQ where numeric evaluation and option comparison are useful.",
        ),
        DerivedRule(
            name="calculus_trig_integral",
            category=CATEGORY,
            detector=is_calculus_trig_integral,
            description="Trigonometric integral requiring identity/substitution/sign care.",
        ),
        DerivedRule(
            name="calculus_improper_parameter_integral",
            category=CATEGORY,
            detector=is_calculus_improper_parameter_integral,
            description="Improper parameter integral, often reducible to a standard integral formula.",
        ),
        DerivedRule(
            name="calculus_trig_derivative_simplification",
            category=CATEGORY,
            detector=is_calculus_trig_derivative,
            description="Trig derivative requiring identity simplification before MCQ matching.",
        ),
        DerivedRule(
            name="calculus_differentiation_under_integral",
            category=CATEGORY,
            detector=is_calculus_differentiation_under_integral,
            description="Derivative with respect to a parameter under an integral sign.",
        ),
        DerivedRule(
            name="calculus_implicit_differentiation",
            category=CATEGORY,
            detector=is_calculus_implicit_differentiation,
            description="Implicit differentiation problem, possibly asking for dx in terms of dy.",
        ),
        DerivedRule(
            name="calculus_piecewise_differential_equation",
            category=CATEGORY,
            detector=is_calculus_piecewise_differential_equation,
            description="Piecewise differential equation requiring continuity matching at a join point.",
        ),
        DerivedRule(
            name="calculus_first_order_ivp",
            category=CATEGORY,
            detector=is_calculus_first_order_ivp,
            description="First-order IVP requiring solution constant before evaluation.",
        ),
        DerivedRule(
            name="calculus_volume_revolution",
            category=CATEGORY,
            detector=is_calculus_volume_revolution,
            description="Volume of revolution problem.",
        ),
        DerivedRule(
            name="calculus_surface_area_revolution",
            category=CATEGORY,
            detector=is_calculus_surface_area_revolution,
            description="Surface area of revolution problem.",
        ),
        DerivedRule(
            name="calculus_area_between_curves",
            category=CATEGORY,
            detector=is_calculus_area_between_curves,
            description="Area between/enclosed by curves problem.",
        ),
        DerivedRule(
            name="calculus_optimization_geometry",
            category=CATEGORY,
            detector=is_calculus_optimization_geometry,
            description="Optimization problem involving geometric quantities.",
        ),
        DerivedRule(
            name="calculus_domain_range_radical_rational",
            category=CATEGORY,
            detector=is_calculus_domain_range_radical_rational,
            description="Domain/range problem involving radical or rational expression.",
        ),
        DerivedRule(
            name="calculus_fourier_sobolev_boundary",
            category=CATEGORY,
            detector=is_calculus_fourier_sobolev_boundary,
            description="Fourier coefficient/Sobolev boundary-term asymptotic problem.",
        ),
        DerivedRule(
            name="calculus_series_special_function",
            category=CATEGORY,
            detector=is_calculus_series_special_function,
            description="Special-function or numeric series problem.",
        ),
        DerivedRule(
            name="calculus_discrete_series_convergence",
            category=CATEGORY,
            detector=is_calculus_discrete_series_convergence,
            description="Convergence problem mixing series with discrete digit/counting structure.",
        ),
        DerivedRule(
            name="calculus_numeric_mcq_close_options",
            category=CATEGORY,
            detector=is_calculus_numeric_mcq_close_options,
            description="MCQ with many numeric options requiring high precision and closest-option comparison.",
        ),
        DerivedRule(
            name="calculus_mcq_option_mapping",
            category=CATEGORY,
            detector=is_mcq_option_mapping,
            description="Calculus MCQ requiring final option-letter mapping.",
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
