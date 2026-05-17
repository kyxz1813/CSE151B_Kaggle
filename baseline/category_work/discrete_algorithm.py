import ast
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


CATEGORY = "discrete_algorithm"

DEFAULT_STRATEGY = "discrete_algorithm_v1_subtype_router"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "discrete_algorithm_v1_subtype_router",
    "discrete_sequence_option_verifier",
    "discrete_boolean_logic_v1",
    "discrete_counting_dp_v1",
    "discrete_number_theory_v1",
]


PROMPT_TEMPLATE_NOTES = """
Prompt/template work for this category lives in:
- baseline/baseline3_prompts.py
- prompting/templates.py
- prompting/strategies.py

This file owns:
- discrete algorithm harness checks
- discrete algorithm derived rules
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


def _has_any(text, terms):
    return any(term in text for term in terms)


def _letter_to_option(record, letter):
    options = record.get("options") or []
    if not letter:
        return None

    letter = str(letter).strip().upper()

    if len(letter) != 1:
        return None

    idx = ord(letter) - ord("A")
    if idx < 0 or idx >= len(options):
        return None

    return options[idx]


def _parse_list_like(value):
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return None
    return None


def _extract_x_list(record):
    question = str(record.get("question", ""))
    match = re.search(r"x_list.*?\[([^\]]+)\]", question, flags=re.IGNORECASE | re.DOTALL)

    if not match:
        return None

    raw = "[" + match.group(1) + "]"
    return _parse_list_like(raw)


def is_xlist_ylist_sequence(record):
    text = _text(record)
    return "x_list" in text and "y_list" in text


def is_named_sequence(record):
    text = _text(record)
    return is_xlist_ylist_sequence(record) and _has_any(text, [
        "definition of a(n)",
        "coordination sequence",
        "crystal ball sequence",
        "molien series",
        "doudna sequence",
        "gaussian primes",
        "catalan",
        "posets",
        "tiling",
        "dodecahedron",
        "mongean shuffle",
    ])


def is_computable_definition_sequence(record):
    text = _text(record)
    return is_xlist_ylist_sequence(record) and _has_any(text, [
        "write n-1 in binary",
        "power of the k-th prime",
        "number of",
        "least",
        "coefficients",
        "expansion",
        "recurrence",
        "defined by",
    ])


def is_boolean_logic_calculus(record):
    text = _text(record)
    return _has_any(text, [
        "logical calculus",
        "boolean",
        "truth value",
        "truth values",
        "logic",
        "proposition",
        "idempotent",
    ])


def is_counting_dp(record):
    text = _text(record)
    return _has_any(text, [
        "number of",
        "how many",
        "ways",
        "using only the digits",
        "adjacent digits",
        "no more than",
        "rectangle",
        "tilings",
        "grid",
        "walk",
        "path",
        "choose",
        "arrangements",
    ])


def is_graph_grid_walk(record):
    text = _text(record)
    return _has_any(text, [
        "ant starts",
        "cartesian plane",
        "grid",
        "walk",
        "path",
        "rectangle",
        "vertices",
        "edges",
        "graph",
        "lattice",
    ])


def is_number_theory(record):
    text = _text(record)
    return _has_any(text, [
        "prime",
        "gaussian prime",
        "norm",
        "modulo",
        "modular",
        "divides",
        "divisible",
        "gcd",
        "lcm",
        "least positive",
        "congruence",
        "between 0 and",
    ])


def is_numeric_root_mistag(record):
    text = _text(record)
    return _has_any(text, [
        "graphing calculator",
        "find all solutions to the equation",
        "solutions to the equation",
        "correct to two decimal places",
        "if there are no solutions",
    ])


def is_set_family_extremal(record):
    text = _text(record)
    return _has_any(text, [
        "finite set",
        "subsets",
        "family",
        "positive integers",
        "such that no",
        "there exists",
    ])


def is_mcq_option_mapping(record):
    return bool(record.get("options"))


def schema_valid(record, row):
    return row.get("schema_valid")


def extractable(record, row):
    return row.get("extractable")


def answer_count_valid(record, row):
    expected = row.get("expected_answer_count")
    actual = row.get("actual_answer_count")

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


def xlist_selected_option_length_valid(record, row):
    if not is_xlist_ylist_sequence(record):
        return None

    if not record.get("options"):
        return None

    x_list = _extract_x_list(record)
    selected = _letter_to_option(record, _boxed(row))
    selected_list = _parse_list_like(selected)

    if x_list is None or selected_list is None:
        return None

    return {
        "passed": len(x_list) == len(selected_list),
        "details": {
            "x_list_length": len(x_list),
            "selected_option_length": len(selected_list),
            "selected_option": selected,
        },
    }


def no_unsupported_named_sequence_guessing(record, row):
    if not is_named_sequence(record):
        return None

    response = _response(row).lower()

    bad_terms = [
        "i'll guess",
        "i will guess",
        "guess based",
        "after research",
        "i recall",
        "known results show",
        "most likely",
        "closest",
        "in the ballpark",
        "standard problems",
    ]

    found = [term for term in bad_terms if term in response]

    return {
        "passed": len(found) == 0,
        "details": {
            "bad_terms_found": found,
        },
    }


def boolean_not_standard_arithmetic(record, row):
    if not is_boolean_logic_calculus(record):
        return None

    response = _response(row).lower()

    bad_terms = [
        "standard arithmetic",
        "ordinary arithmetic",
        "in standard arithmetic",
    ]

    found = [term for term in bad_terms if term in response]

    return {
        "passed": len(found) == 0,
        "details": {
            "bad_terms_found": found,
        },
    }


def numeric_root_precision(record, row):
    if not is_numeric_root_mistag(record):
        return None

    answer = _boxed(row)
    decimal_match = re.search(r"\d+\.(\d+)", answer)

    if not decimal_match:
        return {
            "passed": False,
            "details": {
                "reason": "numeric root answer does not contain decimal precision",
                "boxed_answer": answer,
            },
        }

    digits = len(decimal_match.group(1))

    return {
        "passed": digits >= 6,
        "details": {
            "decimal_digits": digits,
            "boxed_answer": answer,
            "reason": "judge may prefer high precision even when prompt says two decimals",
        },
    }


def scorer_anomaly_gold_equals_boxed(record, row):
    gold = row.get("gold")
    boxed = _boxed(row)

    if gold is None or boxed == "":
        return None

    gold_text = str(gold).strip().upper()
    boxed_text = str(boxed).strip().upper()

    if row.get("correct") is False and gold_text == boxed_text:
        return {
            "passed": False,
            "details": {
                "reason": "gold and boxed answer look identical but official_correct is false",
                "gold": gold,
                "boxed": boxed,
            },
        }

    return None


def official_correct(record, row):
    if row.get("correct") is None:
        return None
    return row.get("correct")


def build_discrete_algorithm_harness():
    return CategoryHarness(
        category=CATEGORY,
        checks=[
            HarnessCheck("schema_valid", schema_valid, weight=1.0),
            HarnessCheck("extractable", extractable, weight=1.0),
            HarnessCheck("answer_count_valid", answer_count_valid, weight=1.0),
            HarnessCheck("mcq_letter_valid", mcq_letter_valid, weight=1.0),
            HarnessCheck("xlist_selected_option_length_valid", xlist_selected_option_length_valid, weight=0.5),
            HarnessCheck("no_unsupported_named_sequence_guessing", no_unsupported_named_sequence_guessing, weight=0.75),
            HarnessCheck("boolean_not_standard_arithmetic", boolean_not_standard_arithmetic, weight=0.75),
            HarnessCheck("numeric_root_precision", numeric_root_precision, weight=0.75),
            HarnessCheck("scorer_anomaly_gold_equals_boxed", scorer_anomaly_gold_equals_boxed, weight=0.25),
            HarnessCheck("official_correct", official_correct, weight=2.0),
        ],
        metadata={
            "category": CATEGORY,
            "kind": "discrete_algorithm_specialized",
        },
    )


def register_category_harness():
    harness = build_discrete_algorithm_harness()
    register_harness(harness)
    return harness


def register_category_rules():
    rules = [
        DerivedRule(
            name="discrete_xlist_ylist_sequence",
            category=CATEGORY,
            detector=is_xlist_ylist_sequence,
            description="Problems giving x_list and asking for y_list.",
        ),
        DerivedRule(
            name="discrete_named_sequence",
            category=CATEGORY,
            detector=is_named_sequence,
            description="Named sequence or OEIS-like definition problems.",
        ),
        DerivedRule(
            name="discrete_computable_definition_sequence",
            category=CATEGORY,
            detector=is_computable_definition_sequence,
            description="Sequence problem with a definition that may be directly computable.",
        ),
        DerivedRule(
            name="discrete_boolean_logic_calculus",
            category=CATEGORY,
            detector=is_boolean_logic_calculus,
            description="Logical calculus or Boolean algebra problem.",
        ),
        DerivedRule(
            name="discrete_counting_dp",
            category=CATEGORY,
            detector=is_counting_dp,
            description="Counting, recurrence, dynamic programming, or finite state counting problem.",
        ),
        DerivedRule(
            name="discrete_graph_grid_walk",
            category=CATEGORY,
            detector=is_graph_grid_walk,
            description="Graph/grid/path/walk/lattice problem.",
        ),
        DerivedRule(
            name="discrete_number_theory",
            category=CATEGORY,
            detector=is_number_theory,
            description="Prime, modulo, divisibility, Gaussian integer, or congruence problem.",
        ),
        DerivedRule(
            name="discrete_numeric_root_mistag",
            category=CATEGORY,
            detector=is_numeric_root_mistag,
            description="Numerical equation/root solving problem currently routed as discrete.",
        ),
        DerivedRule(
            name="discrete_set_family_extremal",
            category=CATEGORY,
            detector=is_set_family_extremal,
            description="Finite set/family/extremal combinatorics problem.",
        ),
        DerivedRule(
            name="discrete_mcq_option_mapping",
            category=CATEGORY,
            detector=is_mcq_option_mapping,
            description="MCQ problem requiring final letter mapping.",
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