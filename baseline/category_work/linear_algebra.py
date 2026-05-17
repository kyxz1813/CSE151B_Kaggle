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


CATEGORY = "linear_algebra"

DEFAULT_STRATEGY = "linear_algebra_v1_structured_verify"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "linear_algebra_v1_structured_verify",
    "linear_algebra_mcq_option_verifier",
    "linear_algebra_freeform_multi_answer",
    "linear_algebra_lp_systems",
]


PROMPT_TEMPLATE_NOTES = """
Prompt/template work for this category lives in:
- baseline/baseline3_prompts.py
- prompting/templates.py
- prompting/strategies.py

This file owns:
- linear algebra harness checks
- linear algebra derived rules
- default strategy lists
"""


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


def is_rank_parameter_condition(record):
    text = _text(record)
    return (
        "rank" in text
        and (
            "lambda" in text
            or "\\lambda" in text
            or "λ" in text
            or "parameter" in text
        )
    )


def is_determinant_or_matrix_property(record):
    text = _text(record)
    return _has_any(text, [
        "determinant",
        "\\det",
        "matrix",
        "matrices",
        "invertible",
        "singular",
        "eigen",
        "rank",
    ])


def is_vector_projection(record):
    text = _text(record)
    return _has_any(text, [
        "projection",
        "perpendicular vectors",
        "orthogonal",
        "\\textbf{u}",
        "\\textbf{v}",
        "dot product",
    ])


def is_linear_programming_word_problem(record):
    text = _text(record)
    return (
        _has_any(text, [
            "maximize",
            "minimize",
            "improve",
            "most",
            "constraints",
            "subject to",
            "linear programming",
            "allocate",
        ])
        and _has_any(text, [
            "spend",
            "budget",
            "charges",
            "hours",
            "cost",
            "sleep",
            "points",
            "resources",
        ])
    )


def is_system_word_multi_answer(record):
    text = _text(record)
    return (
        _ans_count(record) >= 2
        and _has_any(text, [
            "write a mathematical expression",
            "equation",
            "revenue",
            "sold",
            "burgers",
            "hot dogs",
            "system",
            "x + y",
        ])
    )


def is_combinatorial_matrix_determinant(record):
    text = _text(record)
    return (
        "nonempty subsets" in text
        and "matrix" in text
        and "determinant" in text
    )


def is_matrix_transformation_or_mapping(record):
    text = _text(record)
    return _has_any(text, [
        "mapping",
        "linear transformation",
        "transformation",
        "jacobian",
        "matrix a",
        "matrix $a",
    ])


def is_column_dependence_multi_part(record):
    text = _text(record)
    return _has_any(text, [
        "linear dependent columns",
        "linearly dependent columns",
        "dependent columns",
        "column vectors",
        "columns are dependent",
    ])


def is_mistag_xlist_ylist_sequence(record):
    text = _text(record)
    return "x_list" in text and "y_list" in text


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


def rank_candidate_verification_present(record, row):
    if not is_rank_parameter_condition(record):
        return None

    response = _response(row).lower()

    mentions_multiple_candidates = (
        ("0 or" in response)
        or ("or -3" in response)
        or ("lambda = 0" in response and "-3" in response)
        or ("\\lambda = 0" in response and "-3" in response)
    )

    verification_terms = [
        "check each candidate",
        "verify each candidate",
        "rank 1",
        "rank one",
        "not rank 2",
        "not rank two",
        "reject",
        "discard",
        "only",
    ]

    verified = any(term in response for term in verification_terms)

    if mentions_multiple_candidates and not verified:
        return {
            "passed": False,
            "details": {
                "reason": "determinant-zero candidates appear, but exact-rank candidate verification is not evident",
            },
        }

    return {
        "passed": True,
        "details": {
            "mentions_multiple_candidates": mentions_multiple_candidates,
            "verification_evidence": verified,
        },
    }


def no_integer_assumption_unless_stated(record, row):
    if not is_linear_programming_word_problem(record):
        return None

    question = _text(record)
    response = _response(row).lower()

    integer_required = _has_any(question, [
        "integer",
        "whole number",
        "whole-number",
        "nonnegative integer",
        "positive integer",
    ])

    if integer_required:
        return None

    bad_terms = [
        "integer solution",
        "integer point",
        "integer points",
        "feasible integer",
        "hours are probably integers",
        "m, c are integers",
        "m and c are integers",
        "try integer",
    ]

    found = [term for term in bad_terms if term in response]

    return {
        "passed": len(found) == 0,
        "details": {
            "integer_required_by_question": integer_required,
            "bad_terms_found": found,
        },
    }


def multi_answer_placeholder_discipline(record, row):
    count = _ans_count(record)
    if count <= 1:
        return None

    actual = row.get("actual_answer_count")
    boxed = _boxed(row)

    return {
        "passed": actual == count,
        "details": {
            "question_ans_count": count,
            "actual_answer_count": actual,
            "boxed_answer": boxed,
        },
    }


def official_correct(record, row):
    if row.get("correct") is None:
        return None
    return row.get("correct")


def build_linear_algebra_harness():
    return CategoryHarness(
        category=CATEGORY,
        checks=[
            HarnessCheck("schema_valid", schema_valid, weight=1.0),
            HarnessCheck("extractable", extractable, weight=1.0),
            HarnessCheck("answer_count_valid", answer_count_valid, weight=1.0),
            HarnessCheck("mcq_letter_valid", mcq_letter_valid, weight=1.0),
            HarnessCheck("rank_candidate_verification_present", rank_candidate_verification_present, weight=0.75),
            HarnessCheck("no_integer_assumption_unless_stated", no_integer_assumption_unless_stated, weight=0.75),
            HarnessCheck("multi_answer_placeholder_discipline", multi_answer_placeholder_discipline, weight=0.75),
            HarnessCheck("official_correct", official_correct, weight=2.0),
        ],
        metadata={
            "category": CATEGORY,
            "kind": "linear_algebra_specialized",
        },
    )


def register_category_harness():
    harness = build_linear_algebra_harness()
    register_harness(harness)
    return harness


def register_category_rules():
    rules = [
        DerivedRule(
            name="linear_algebra_rank_parameter_condition",
            category=CATEGORY,
            detector=is_rank_parameter_condition,
            description="Rank/nullity/linear-dependence problems with a parameter such as lambda.",
        ),
        DerivedRule(
            name="linear_algebra_determinant_or_matrix_property",
            category=CATEGORY,
            detector=is_determinant_or_matrix_property,
            description="Matrix determinant, rank, eigenvalue, invertibility, or matrix property problem.",
        ),
        DerivedRule(
            name="linear_algebra_vector_projection",
            category=CATEGORY,
            detector=is_vector_projection,
            description="Vector projection, orthogonality, dot-product, or perpendicular vector problem.",
        ),
        DerivedRule(
            name="linear_algebra_linear_programming_word_problem",
            category=CATEGORY,
            detector=is_linear_programming_word_problem,
            description="Linear constraints/objective word problem, usually continuous unless integers are stated.",
        ),
        DerivedRule(
            name="linear_algebra_system_word_multi_answer",
            category=CATEGORY,
            detector=is_system_word_multi_answer,
            description="System-of-equations word problem with multiple requested answers.",
        ),
        DerivedRule(
            name="linear_algebra_combinatorial_matrix_determinant",
            category=CATEGORY,
            detector=is_combinatorial_matrix_determinant,
            description="Subset/intersection matrix determinant problem.",
        ),
        DerivedRule(
            name="linear_algebra_matrix_transformation_or_mapping",
            category=CATEGORY,
            detector=is_matrix_transformation_or_mapping,
            description="Matrix mapping/transformation/Jacobian-style problem.",
        ),
        DerivedRule(
            name="linear_algebra_column_dependence_multi_part",
            category=CATEGORY,
            detector=is_column_dependence_multi_part,
            description="Column dependence or multi-part linear dependence problem.",
        ),
        DerivedRule(
            name="linear_algebra_mistag_xlist_ylist_sequence",
            category=CATEGORY,
            detector=is_mistag_xlist_ylist_sequence,
            description="Likely discrete sequence x_list/y_list problem currently routed as linear algebra.",
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