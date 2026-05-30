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


CATEGORY = "statistics_probability"

DEFAULT_STRATEGY = "statistics_probability_v1_structured"

CANDIDATE_STRATEGIES = [
    "baseline3",
    "baseline3_adaptive_rules",
    "statistics_probability_v1_structured",
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


def is_hypothesis_test(record):
    text = _text(record)
    return _has_any(text, [
        "hypothesis",
        "null hypothesis",
        "alternative hypothesis",
        "h_0",
        "h0",
        "h_a",
        "ha",
        "significance level",
        "p-value",
        "p value",
        "reject",
        "fail to reject",
        "type i error",
        "type ii error",
        "power",
    ])


def is_type_i_type_ii_power(record):
    text = _text(record)
    return _has_any(text, [
        "type i error",
        "type ii error",
        "power of the test",
        "probability of a type",
        "beta",
    ])


def is_confidence_interval(record):
    text = _text(record)
    return _has_any(text, [
        "confidence interval",
        "confidence level",
        "margin of error",
        "interval estimate",
        "lower bound",
        "upper bound",
    ])


def is_regression_correlation(record):
    text = _text(record)
    return _has_any(text, [
        "regression",
        "correlation",
        "least squares",
        "slope",
        "intercept",
        "residual",
        "r-squared",
        "r^2",
        "predict",
    ])


def is_probability_counting(record):
    text = _text(record)
    return _has_any(text, [
        "probability",
        "randomly selected",
        "without replacement",
        "with replacement",
        "at least",
        "at most",
        "exactly",
        "conditional probability",
        "independent",
        "mutually exclusive",
    ])


def is_distribution_problem(record):
    text = _text(record)
    return _has_any(text, [
        "normal distribution",
        "standard normal",
        "z-score",
        "z score",
        "binomial",
        "poisson",
        "exponential distribution",
        "uniform distribution",
        "mean",
        "standard deviation",
        "variance",
    ])


def is_expected_value_variance(record):
    text = _text(record)
    return _has_any(text, [
        "expected value",
        "expectation",
        "variance",
        "standard deviation",
        "mean of",
        "random variable",
    ])


def is_sampling_distribution(record):
    text = _text(record)
    return _has_any(text, [
        "sample mean",
        "sampling distribution",
        "central limit theorem",
        "standard error",
        "sample size",
    ])


def is_mcq_option_mapping(record):
    return bool(record.get("options"))


def is_multi_answer(record):
    return _ans_count(record) > 1


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


def hypothesis_test_method_evidence(record, row):
    if not is_hypothesis_test(record):
        return None

    response = _response(row).lower()

    evidence_terms = [
        "null",
        "alternative",
        "critical",
        "p-value",
        "p value",
        "reject",
        "fail to reject",
        "standard error",
        "z",
        "t",
    ]

    found = [term for term in evidence_terms if term in response]

    return {
        "passed": len(found) >= 2,
        "details": {
            "evidence_terms_found": found,
        },
    }


def probability_range_valid(record, row):
    if not (
        is_probability_counting(record)
        or is_type_i_type_ii_power(record)
        or is_distribution_problem(record)
    ):
        return None

    answer = _boxed(row)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer)

    if not nums:
        return None

    suspicious = []
    for raw in nums:
        try:
            value = float(raw)
        except Exception:
            continue

        if value < 0 or value > 1:
            suspicious.append(value)

    if "%" in answer:
        return None

    return {
        "passed": len(suspicious) == 0,
        "details": {
            "suspicious_values": suspicious,
            "boxed_answer": answer,
        },
    }


def final_answer_not_explanatory(record, row):
    answer = _boxed(row)
    bad_terms = ["because", "therefore", "reject", "fail to reject", "p-value is"]
    found = [term for term in bad_terms if term in answer.lower()]

    return {
        "passed": len(found) == 0,
        "details": {
            "bad_terms_found": found,
        },
    }


def official_correct(record, row):
    if row.get("correct") is None:
        return None
    return row.get("correct")


def build_statistics_probability_harness():
    return CategoryHarness(
        category=CATEGORY,
        checks=[
            HarnessCheck("schema_valid", schema_valid, weight=1.0),
            HarnessCheck("extractable", extractable, weight=1.0),
            HarnessCheck("answer_count_valid", answer_count_valid, weight=1.0),
            HarnessCheck("mcq_letter_valid", mcq_letter_valid, weight=1.0),
            HarnessCheck("hypothesis_test_method_evidence", hypothesis_test_method_evidence, weight=0.75),
            HarnessCheck("probability_range_valid", probability_range_valid, weight=0.75),
            HarnessCheck("final_answer_not_explanatory", final_answer_not_explanatory, weight=0.5),
            HarnessCheck("official_correct", official_correct, weight=2.0),
        ],
        metadata={
            "category": CATEGORY,
            "kind": "statistics_probability_specialized",
        },
    )


def register_category_harness():
    harness = build_statistics_probability_harness()
    register_harness(harness)
    return harness


def register_category_rules():
    rules = [
        DerivedRule(
            name="statistics_probability_hypothesis_test",
            category=CATEGORY,
            detector=is_hypothesis_test,
            description="Hypothesis test, p-value, rejection, significance-level problem.",
        ),
        DerivedRule(
            name="statistics_probability_type_i_type_ii_power",
            category=CATEGORY,
            detector=is_type_i_type_ii_power,
            description="Type I/II error or power problem.",
        ),
        DerivedRule(
            name="statistics_probability_confidence_interval",
            category=CATEGORY,
            detector=is_confidence_interval,
            description="Confidence interval or margin-of-error problem.",
        ),
        DerivedRule(
            name="statistics_probability_regression_correlation",
            category=CATEGORY,
            detector=is_regression_correlation,
            description="Regression/correlation/residual/prediction problem.",
        ),
        DerivedRule(
            name="statistics_probability_probability_counting",
            category=CATEGORY,
            detector=is_probability_counting,
            description="Probability using counting, replacement, independence, conditional probability.",
        ),
        DerivedRule(
            name="statistics_probability_distribution",
            category=CATEGORY,
            detector=is_distribution_problem,
            description="Distribution problem involving normal/binomial/Poisson/uniform/etc.",
        ),
        DerivedRule(
            name="statistics_probability_expected_value_variance",
            category=CATEGORY,
            detector=is_expected_value_variance,
            description="Expected value, variance, standard deviation of random variables.",
        ),
        DerivedRule(
            name="statistics_probability_sampling_distribution",
            category=CATEGORY,
            detector=is_sampling_distribution,
            description="Sample mean, standard error, CLT, or sampling distribution problem.",
        ),
        DerivedRule(
            name="statistics_probability_mcq_option_mapping",
            category=CATEGORY,
            detector=is_mcq_option_mapping,
            description="MCQ statistics/probability problem requiring final option-letter mapping.",
        ),
        DerivedRule(
            name="statistics_probability_multi_answer",
            category=CATEGORY,
            detector=is_multi_answer,
            description="Multi-answer statistics/probability problem.",
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