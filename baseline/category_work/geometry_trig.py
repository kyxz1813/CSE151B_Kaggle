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


CATEGORY = "geometry_trig"

DEFAULT_STRATEGY = "geometry_trig_frq"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "baseline3_adaptive_rules",
    "geometry_trig_frq",
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


def is_angle_conversion(record):
    text = _text(record)

    return _has_any(text, [
        "radian measure",
        "degree measure",
        "degrees is",
        "radians is",
        "convert to radians",
        "convert to degrees",
        "degree",
        "radian",
    ])


def is_arc_length_sector(record):
    text = _text(record)

    return _has_any(text, [
        "arc length",
        "central angle",
        "sector area",
        "sector of a circle",
        "subtends",
        "radius of the circle",
        "circumference",
    ])


def is_trig_equation(record):
    text = _text(record)

    trig_terms = [
        "sin(",
        "cos(",
        "tan(",
        "sin ",
        "cos ",
        "tan ",
        "sec(",
        "csc(",
        "cot(",
        "\\sin",
        "\\cos",
        "\\tan",
        "solve for theta",
        "solve for θ",
        "all solutions",
    ]

    return _has_any(text, trig_terms)


def is_inverse_trig(record):
    text = _text(record)

    return _has_any(text, [
        "arcsin",
        "arccos",
        "arctan",
        "asin",
        "acos",
        "atan",
        "inverse sine",
        "inverse cosine",
        "inverse tangent",
    ])


def is_quadrant_sign(record):
    text = _text(record)

    return _has_any(text, [
        "first quadrant",
        "second quadrant",
        "third quadrant",
        "fourth quadrant",
        "quadrant i",
        "quadrant ii",
        "quadrant iii",
        "quadrant iv",
        "reference angle",
        "terminal side",
        "unit circle",
    ])


def is_coordinate_point_trig(record):
    text = _text(record)

    return _has_any(text, [
        "point in the first quadrant",
        "point in the second quadrant",
        "point in the third quadrant",
        "point in the fourth quadrant",
        "valid point",
        "one valid point",
        "coordinate",
        "(x, y)",
        "(x,y)",
        "terminal side passes through",
    ])


def is_right_triangle(record):
    text = _text(record)

    return _has_any(text, [
        "right triangle",
        "hypotenuse",
        "opposite side",
        "adjacent side",
        "pythagorean",
        "angle of elevation",
        "angle of depression",
        "height of",
        "distance from",
    ])


def is_law_of_sines_cosines(record):
    text = _text(record)

    return _has_any(text, [
        "law of sines",
        "law of cosines",
        "sas",
        "sss",
        "asa",
        "aas",
        "ambiguous case",
        "triangle abc",
    ])


def is_circle_geometry(record):
    text = _text(record)

    return _has_any(text, [
        "circle",
        "radius",
        "diameter",
        "circumference",
        "chord",
        "tangent line",
        "inscribed",
        "central angle",
        "arc",
        "sector",
    ])


def is_multi_answer_problem(record):
    return _ans_count(record) > 1


def is_mcq_option_mapping(record):
    return bool(record.get("options"))


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
    valid = [
        chr(ord("A") + idx)
        for idx in range(len(options))
    ]

    return {
        "passed": boxed in valid,
        "details": {
            "boxed_answer": boxed,
            "valid_letters": valid,
        },
    }


def geometry_trig_method_evidence(record, row):
    response = _response(row).lower()

    expected_terms = []

    if is_angle_conversion(record):
        expected_terms.extend([
            "pi",
            "π",
            "180",
            "degree",
            "radian",
        ])

    if is_arc_length_sector(record):
        expected_terms.extend([
            "arc",
            "radius",
            "sector",
            "theta",
            "θ",
            "radian",
        ])

    if is_trig_equation(record):
        expected_terms.extend([
            "sin",
            "cos",
            "tan",
            "period",
            "quadrant",
            "reference angle",
        ])

    if is_quadrant_sign(record):
        expected_terms.extend([
            "quadrant",
            "positive",
            "negative",
            "reference angle",
            "unit circle",
        ])

    if is_coordinate_point_trig(record):
        expected_terms.extend([
            "point",
            "coordinate",
            "quadrant",
            "x",
            "y",
            "sqrt",
        ])

    if is_right_triangle(record):
        expected_terms.extend([
            "hypotenuse",
            "opposite",
            "adjacent",
            "pythagorean",
            "sin",
            "cos",
            "tan",
        ])

    if is_law_of_sines_cosines(record):
        expected_terms.extend([
            "law of sines",
            "law of cosines",
            "sin",
            "cos",
            "triangle",
        ])

    if not expected_terms:
        return None

    found = [
        term
        for term in expected_terms
        if term in response
    ]

    return {
        "passed": bool(found),
        "details": {
            "matched_terms": found,
        },
    }


def angle_unit_check(record, row):
    if not is_angle_conversion(record):
        return None

    question = _text(record)
    response = _response(row).lower()
    boxed = _boxed(row).lower()

    asks_for_radians = "radian" in question
    asks_for_degrees = "degree" in question and "radian" not in question

    if asks_for_radians:
        passed = (
            "pi" in response
            or "π" in response
            or bool(re.search(r"\d+\.\d+", boxed))
        )

        expected_unit = "radians"

    elif asks_for_degrees:
        passed = bool(
            re.search(r"-?\d+(\.\d+)?", boxed)
        )

        expected_unit = "degrees"

    else:
        return None

    return {
        "passed": passed,
        "details": {
            "expected_unit": expected_unit,
            "boxed_answer": boxed,
        },
    }


def periodic_solution_evidence(record, row):
    text = _text(record)

    if not (
        is_trig_equation(record)
        and _has_any(text, [
            "all solutions",
            "n is any integer",
            "any integer",
            "general solution",
        ])
    ):
        return None

    response = _response(row).lower()
    boxed = _boxed(row).lower()

    periodic_markers = [
        "pi",
        "π",
        "2pi",
        "2π",
        "n",
        "integer",
        "period",
    ]

    found = [
        marker
        for marker in periodic_markers
        if marker in response or marker in boxed
    ]

    return {
        "passed": bool(found),
        "details": {
            "matched_terms": found,
            "boxed_answer": boxed,
        },
    }


def quadrant_sign_evidence(record, row):
    if not is_quadrant_sign(record):
        return None

    response = _response(row).lower()

    found = [
        term
        for term in [
            "quadrant",
            "positive",
            "negative",
            "sign",
            "reference angle",
        ]
        if term in response
    ]

    return {
        "passed": bool(found),
        "details": {
            "matched_terms": found,
        },
    }


def no_unmapped_numeric_for_mcq(record, row):
    if not record.get("options"):
        return None

    boxed = _boxed(row).strip()

    return {
        "passed": bool(
            re.fullmatch(r"[A-Z]", boxed.upper())
        ),
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

    found = [
        marker
        for marker in bad_markers
        if marker in lower
    ]

    return {
        "passed": not found,
        "details": {
            "bad_markers_found": found,
            "boxed_answer": boxed[:200],
        },
    }


def official_correct(record, row):
    if row.get("correct") is None:
        return None

    return row.get("correct")


def build_geometry_trig_harness():
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
                "geometry_trig_method_evidence",
                geometry_trig_method_evidence,
                weight=0.75,
            ),
            HarnessCheck(
                "angle_unit_check",
                angle_unit_check,
                weight=0.75,
            ),
            HarnessCheck(
                "periodic_solution_evidence",
                periodic_solution_evidence,
                weight=0.75,
            ),
            HarnessCheck(
                "quadrant_sign_evidence",
                quadrant_sign_evidence,
                weight=0.75,
            ),
            HarnessCheck(
                "no_unmapped_numeric_for_mcq",
                no_unmapped_numeric_for_mcq,
                weight=0.75,
            ),
            HarnessCheck(
                "final_answer_not_explanatory",
                final_answer_not_explanatory,
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
            "kind": "geometry_trig_specialized",
        },
    )


def register_category_harness():
    harness = build_geometry_trig_harness()
    register_harness(harness)

    return harness


def register_category_rules():
    rules = [
        DerivedRule(
            name="geometry_trig_angle_conversion",
            category=CATEGORY,
            detector=is_angle_conversion,
            description=(
                "Degree-radian conversion requiring correct unit handling."
            ),
        ),
        DerivedRule(
            name="geometry_trig_arc_length_sector",
            category=CATEGORY,
            detector=is_arc_length_sector,
            description=(
                "Arc length, sector area, radius, or central-angle problem."
            ),
        ),
        DerivedRule(
            name="geometry_trig_trig_equation",
            category=CATEGORY,
            detector=is_trig_equation,
            description=(
                "Trigonometric equation requiring correct branches and periodicity."
            ),
        ),
        DerivedRule(
            name="geometry_trig_inverse_trig",
            category=CATEGORY,
            detector=is_inverse_trig,
            description=(
                "Inverse-trigonometric evaluation requiring principal-value checks."
            ),
        ),
        DerivedRule(
            name="geometry_trig_quadrant_sign",
            category=CATEGORY,
            detector=is_quadrant_sign,
            description=(
                "Quadrant-sensitive trigonometric sign and reference-angle problem."
            ),
        ),
        DerivedRule(
            name="geometry_trig_coordinate_point",
            category=CATEGORY,
            detector=is_coordinate_point_trig,
            description=(
                "Coordinate-point trigonometry problem requiring a valid point "
                "and correct sign choices."
            ),
        ),
        DerivedRule(
            name="geometry_trig_right_triangle",
            category=CATEGORY,
            detector=is_right_triangle,
            description=(
                "Right-triangle geometry or applied trigonometry problem."
            ),
        ),
        DerivedRule(
            name="geometry_trig_law_of_sines_cosines",
            category=CATEGORY,
            detector=is_law_of_sines_cosines,
            description=(
                "Non-right-triangle problem requiring the law of sines or cosines."
            ),
        ),
        DerivedRule(
            name="geometry_trig_circle_geometry",
            category=CATEGORY,
            detector=is_circle_geometry,
            description=(
                "Circle-geometry problem requiring radius, arc, chord, "
                "sector, or central-angle reasoning."
            ),
        ),
        DerivedRule(
            name="geometry_trig_multi_answer_order",
            category=CATEGORY,
            detector=is_multi_answer_problem,
            description=(
                "Multiple-answer problem requiring the exact number of answers "
                "in the same order as the blanks."
            ),
        ),
        DerivedRule(
            name="geometry_trig_mcq_option_mapping",
            category=CATEGORY,
            detector=is_mcq_option_mapping,
            description=(
                "Geometry or trigonometry MCQ requiring final option-letter mapping."
            ),
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