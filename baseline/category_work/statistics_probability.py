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
        "hypothesis", "null hypothesis", "alternative hypothesis", "h_0", "h0",
        "h_a", "ha", "significance level", "p-value", "p value",
        "reject", "fail to reject", "type i error", "type ii error", "power",
    ])


def is_type_i_type_ii_power(record):
    text = _text(record)
    return _has_any(text, [
        "type i error", "type ii error", "power of the test", "probability of a type", "beta",
    ])


def is_confidence_interval(record):
    text = _text(record)
    return _has_any(text, [
        "confidence interval", "confidence level", "margin of error", "interval estimate",
        "lower bound", "upper bound",
    ])


def is_regression_correlation(record):
    text = _text(record)
    return _has_any(text, [
        "regression", "correlation", "least squares", "slope", "intercept", "residual",
        "r-squared", "r^2", "predict",
    ])


def is_probability_counting(record):
    text = _text(record)
    return _has_any(text, [
        "probability", "randomly selected", "without replacement", "with replacement",
        "at least", "at most", "exactly", "conditional probability", "independent",
        "mutually exclusive",
    ])


def is_distribution_problem(record):
    text = _text(record)
    return _has_any(text, [
        "normal distribution", "standard normal", "z-score", "z score", "binomial",
        "poisson", "exponential distribution", "uniform distribution", "mean", "standard deviation", "variance",
    ])


def is_expected_value_variance(record):
    text = _text(record)
    return _has_any(text, [
        "expected value", "expectation", "variance", "standard deviation", "mean of", "random variable",
    ])


def is_sampling_distribution(record):
    text = _text(record)
    return _has_any(text, [
        "sample mean", "sampling distribution", "central limit theorem", "standard error", "sample size",
    ])


def is_descriptive_table(record):
    text = _text(record)
    return _ans_count(record) >= 4 and _has_any(text, [
        "standard deviation", "variance", "deviation", "squared", "table", "data set", "data values",
    ])


def is_chi_square_goodness_fit(record):
    text = _text(record)
    return "chi" in text and _has_any(text, ["goodness", "expected", "observed", "fit"])


def is_chi_square_independence(record):
    text = _text(record)
    return "chi" in text and _has_any(text, ["independence", "contingency", "row", "column", "expected frequencies"])


def is_sample_size_margin_error(record):
    text = _text(record)
    return _has_any(text, ["sample size", "margin of error", "minimum sample", "smallest sample", "how many"])


def is_multi_blank_table(record):
    text = _text(record)
    return _ans_count(record) >= 4 and _has_any(text, ["table", "row", "column", "blank", "expected", "observed"])


def is_rounding_direction(record):
    text = _text(record)
    return _has_any(text, ["smallest integer", "minimum", "at least", "round up", "ceil", "nearest", "round"])


def is_mcq_option_mapping(record):
    return bool(record.get("options"))


def is_multi_answer(record):
    return _ans_count(record) > 1

def is_statistics_table_high_precision(record):
    text = _text(record)
    return _ans_count(record) >= 4 and _has_any(text, [
        "standard deviation", "variance", "squared", "sum", "mean",
        "use at least three decimal", "table below",
    ])


def is_chi_square_decision_contract(record):
    text = _text(record)
    return "chi" in text and _has_any(text, [
        "is there sufficient data", "support the claim", "test the claim",
        "significance level", "critical value",
    ])


def is_chi_square_assumption_letters(record):
    text = _text(record)
    return _has_any(text, [
        "assumption 1", "assumption 2", "expected frequencies are 1 or greater",
        "20 percent of the expected frequencies are less than 5",
    ])


def is_sample_size_raw_value(record):
    text = _text(record)
    return is_sample_size_margin_error(record) and _has_any(text, [
        "bound of error", "margin of error", "estimate the mean", "estimate the difference",
        "confidence interval", "how large should", "how many",
    ])


def is_type_ii_error_beta(record):
    text = _text(record)
    return _has_any(text, [
        "type ii error", "p(type ii", "probability of making a type ii", "given that mu",
    ])


def is_probability_threshold_log(record):
    text = _text(record)
    return _has_any(text, [
        "rollover", "all losers", "does not win", "greater than", "less than",
        "fewer than", "independently sold tickets",
    ])


def is_embedded_letter_choices(record):
    text = str(record.get("question", ""))
    lowered = text.lower()
    if record.get("options"):
        return False
    if "[ANS]" not in text:
        return False
    return (
        re.search(r"\bA\.", text) is not None
        and re.search(r"\bB\.", text) is not None
        and _has_any(lowered, ["c.", "d.", "option", "assumption", "f-curve", "relative frequencies"])
    )


def is_f_critical_embedded_choices(record):
    text = _text(record)
    return _has_any(text, [
        "f-curve", "f value", "f-value", "degrees of freedom", "df=", "area",
        "to its right",
    ]) and is_embedded_letter_choices(record)


def is_long_regression_vector(record):
    text = _text(record)
    return (
        "x=c(" in text
        and "y=c(" in text
        and _has_any(text, ["regression", "correlation", "least squares", "predict", "residual"])
    )


def is_uppercase_categorical_problem(record):
    text = _text(record)
    return _has_any(text, [
        "yes", "no", "increasing", "decreasing", "support the claim",
        "sufficient data", "rollover increasing or decreasing",
    ])


def is_high_precision_stats_numeric(record):
    text = _text(record)
    return _has_any(text, [
        "standard deviation", "variance", "test statistic", "critical value",
        "type ii error", "sample size", "bound of error", "margin of error",
        "probability", "regression", "correlation", "confidence interval",
    ])

def is_representative_choice_letters(record):
    text = _text(record)
    return (
        "representative" in text
        and "non-representative" in text
        and "[ans]" in text
    )


def is_measurement_scale_abbreviation(record):
    text = _text(record)
    return (
        "[ans]" in text
        and _has_any(text, ["nominal", "ordinal", "interval", "ratio"])
        and _has_any(text, ["questionaire", "questionnaire", "possible responses", "indicate whether"])
    )


def is_true_false_abbreviation(record):
    text = _text(record)
    return (
        "[ans]" in text
        and _has_any(text, ["select true or false", "true or false", "corresponding statement is true or false"])
    )


def is_coded_categorical_stats_answer(record):
    return (
        is_representative_choice_letters(record)
        or is_measurement_scale_abbreviation(record)
        or is_true_false_abbreviation(record)
    )


def is_two_mean_equal_sample_size_problem(record):
    text = _text(record)
    return (
        is_sample_size_margin_error(record)
        and _has_any(text, ["difference between two population means", "n_1", "n1", "n_2", "n2", "equal size"])
    )

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
    return {"passed": expected == actual, "details": {"expected_answer_count": expected, "actual_answer_count": actual}}


def mcq_letter_valid(record, row):
    options = record.get("options") or []
    if not options:
        return None
    boxed = _boxed(row).upper()
    valid = [chr(ord("A") + idx) for idx in range(len(options))]
    return {"passed": boxed in valid, "details": {"boxed_answer": boxed, "valid_letters": valid}}


def hypothesis_test_method_evidence(record, row):
    if not is_hypothesis_test(record):
        return None
    response = _response(row).lower()
    evidence_terms = ["null", "alternative", "critical", "p-value", "p value", "reject", "fail to reject", "standard error", "z", "t"]
    found = [term for term in evidence_terms if term in response]
    return {"passed": len(found) >= 2, "details": {"evidence_terms_found": found}}


def probability_range_valid(record, row):
    if not (is_probability_counting(record) or is_type_i_type_ii_power(record)):
        return None
    text = _text(record)
    if any(term in text for term in ["standard deviation", "sample size", "chi", "test statistic", "critical value"]):
        return None
    answer = _boxed(row)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if not nums or "%" in answer:
        return None
    suspicious = []
    for raw in nums:
        try:
            value = float(raw)
        except Exception:
            continue
        if value < 0 or value > 1:
            suspicious.append(value)
    return {"passed": len(suspicious) == 0, "details": {"suspicious_values": suspicious, "boxed_answer": answer}}


def table_multi_blank_not_collapsed(record, row):
    if not (is_descriptive_table(record) or is_multi_blank_table(record)):
        return None
    expected = _ans_count(record)
    actual = row.get("actual_answer_count")
    return {"passed": actual == expected, "details": {"expected_table_entries": expected, "actual_answer_count": actual}}


def final_answer_not_explanatory(record, row):
    answer = _boxed(row)
    bad_terms = ["because", "therefore", "reject", "fail to reject", "p-value is"]
    found = [term for term in bad_terms if term in answer.lower()]
    return {"passed": len(found) == 0, "details": {"bad_terms_found": found}}

def stats_high_precision_numeric(record, row):
    if not is_high_precision_stats_numeric(record):
        return None

    answer = _boxed(row)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer)

    if not nums:
        return None

    decimal_lengths = []
    for raw in nums:
        if "." in raw:
            decimal_lengths.append(len(raw.split(".")[-1]))

    if not decimal_lengths:
        return {"passed": False, "details": {"reason": "no_decimal_precision", "boxed_answer": answer}}

    return {
        "passed": max(decimal_lengths) >= 4,
        "details": {"max_decimal_digits": max(decimal_lengths), "boxed_answer": answer},
    }


def decision_box_no_reject_phrase(record, row):
    if not is_chi_square_decision_contract(record) and not is_hypothesis_test(record):
        return None

    answer = _boxed(row).lower()
    bad_terms = ["reject", "fail to reject", "h0", "h_0", "null hypothesis", "p-value"]
    found = [term for term in bad_terms if term in answer]

    return {"passed": len(found) == 0, "details": {"bad_terms_found": found, "boxed_answer": _boxed(row)}}


def categorical_uppercase_valid(record, row):
    if not is_uppercase_categorical_problem(record):
        return None

    answer = _boxed(row)
    targets = ["yes", "no", "increasing", "decreasing"]
    found_lower = []

    for target in targets:
        if re.search(rf"\b{target}\b", answer):
            found_lower.append(target)

    return {
        "passed": len(found_lower) == 0,
        "details": {"lowercase_terms_found": found_lower, "boxed_answer": answer},
    }


def sample_size_not_ceiled(record, row):
    if not is_sample_size_raw_value(record):
        return None

    text = _text(record)
    if _has_any(text, ["smallest integer", "minimum integer", "whole number", "round up"]):
        return None

    answer = _boxed(row).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer)

    if len(nums) != 1:
        return None

    raw = nums[0]
    is_integer_like = "." not in raw

    return {
        "passed": not is_integer_like,
        "details": {"integer_like": is_integer_like, "boxed_answer": answer},
    }


def embedded_letter_answer_format(record, row):
    if not is_embedded_letter_choices(record):
        return None

    answer = _boxed(row).strip()
    parts = [part.strip() for part in answer.split(",") if part.strip()]
    valid = all(re.fullmatch(r"[A-D]", part.upper()) is not None for part in parts)

    return {
        "passed": valid and len(parts) > 0,
        "details": {"parts": parts, "boxed_answer": answer},
    }


def chi_square_contract_answer_count(record, row):
    if not is_chi_square_decision_contract(record):
        return None

    actual = row.get("actual_answer_count")
    answer = _boxed(row)
    contains_reject_phrase = any(term in answer.lower() for term in ["reject", "fail to reject", "h0", "h_0"])

    return {
        "passed": not contains_reject_phrase,
        "details": {"actual_answer_count": actual, "contains_reject_phrase": contains_reject_phrase},
    }


def long_regression_vector_warning(record, row):
    if not is_long_regression_vector(record):
        return None

    return {
        "passed": bool(row.get("schema_valid")) and bool(row.get("extractable")),
        "details": {"note": "long vector regression task; likely needs high token budget and may remain numerically hard"},
    }

def official_correct(record, row):
    if row.get("correct") is None:
        return None
    return row.get("correct")

def coded_categorical_answer_format(record, row):
    if not is_coded_categorical_stats_answer(record):
        return None

    answer = _boxed(row).strip()
    parts = [part.strip().upper() for part in answer.split(",") if part.strip()]

    if not parts:
        return {"passed": False, "details": {"reason": "empty_answer", "boxed_answer": answer}}

    if is_representative_choice_letters(record):
        valid_letters = {"A", "B"}
        passed = all(part in valid_letters for part in parts)
        return {
            "passed": passed,
            "details": {
                "expected_format": "A/B letters only",
                "parts": parts,
                "boxed_answer": answer,
            },
        }

    if is_measurement_scale_abbreviation(record):
        valid_letters = {"N", "O", "I", "R"}
        passed = all(part in valid_letters for part in parts)
        return {
            "passed": passed,
            "details": {
                "expected_format": "N/O/I/R abbreviations only",
                "parts": parts,
                "boxed_answer": answer,
            },
        }

    if is_true_false_abbreviation(record):
        valid_letters = {"T", "F"}
        passed = all(part in valid_letters for part in parts)
        return {
            "passed": passed,
            "details": {
                "expected_format": "T/F letters only",
                "parts": parts,
                "boxed_answer": answer,
            },
        }

    return None

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
            HarnessCheck("table_multi_blank_not_collapsed", table_multi_blank_not_collapsed, weight=0.75),
            HarnessCheck("stats_high_precision_numeric", stats_high_precision_numeric, weight=0.75),
            HarnessCheck("decision_box_no_reject_phrase", decision_box_no_reject_phrase, weight=0.75),
            HarnessCheck("categorical_uppercase_valid", categorical_uppercase_valid, weight=0.5),
            HarnessCheck("sample_size_not_ceiled", sample_size_not_ceiled, weight=0.75),
            HarnessCheck("embedded_letter_answer_format", embedded_letter_answer_format, weight=0.75),
            HarnessCheck("coded_categorical_answer_format", coded_categorical_answer_format, weight=0.75),
            HarnessCheck("chi_square_contract_answer_count", chi_square_contract_answer_count, weight=0.75),
            HarnessCheck("long_regression_vector_warning", long_regression_vector_warning, weight=0.25),
            HarnessCheck("final_answer_not_explanatory", final_answer_not_explanatory, weight=0.5),
            HarnessCheck("official_correct", official_correct, weight=2.0),
        ],
        metadata={"category": CATEGORY, "kind": "statistics_probability_specialized_v2"},
    )


def register_category_harness():
    harness = build_statistics_probability_harness()
    register_harness(harness)
    return harness


def register_category_rules():
    rules = [
        DerivedRule("statistics_probability_hypothesis_test", CATEGORY, is_hypothesis_test, description="Hypothesis test, p-value, rejection, significance-level problem."),
        DerivedRule("statistics_probability_type_i_type_ii_power", CATEGORY, is_type_i_type_ii_power, description="Type I/II error or power problem."),
        DerivedRule("statistics_probability_confidence_interval", CATEGORY, is_confidence_interval, description="Confidence interval or margin-of-error problem."),
        DerivedRule("statistics_probability_regression_correlation", CATEGORY, is_regression_correlation, description="Regression/correlation/residual/prediction problem."),
        DerivedRule("statistics_probability_probability_counting", CATEGORY, is_probability_counting, description="Probability using counting, replacement, independence, conditional probability."),
        DerivedRule("statistics_probability_distribution", CATEGORY, is_distribution_problem, description="Distribution problem involving normal/binomial/Poisson/uniform/etc."),
        DerivedRule("statistics_probability_expected_value_variance", CATEGORY, is_expected_value_variance, description="Expected value, variance, standard deviation of random variables."),
        DerivedRule("statistics_probability_sampling_distribution", CATEGORY, is_sampling_distribution, description="Sample mean, standard error, CLT, or sampling distribution problem."),
        DerivedRule("statistics_probability_descriptive_table", CATEGORY, is_descriptive_table, description="Descriptive-statistics table with many blanks."),
        DerivedRule("statistics_probability_chi_square_goodness_fit", CATEGORY, is_chi_square_goodness_fit, description="Chi-square goodness-of-fit problem."),
        DerivedRule("statistics_probability_chi_square_independence", CATEGORY, is_chi_square_independence, description="Chi-square independence/contingency-table problem."),
        DerivedRule("statistics_probability_sample_size_margin_error", CATEGORY, is_sample_size_margin_error, description="Sample-size or margin-of-error problem."),
        DerivedRule("statistics_probability_multi_blank_table", CATEGORY, is_multi_blank_table, description="Statistics table/multi-blank problem."),
        DerivedRule("statistics_probability_rounding_direction", CATEGORY, is_rounding_direction, description="Problem where rounding direction must be interpreted carefully."),
        DerivedRule("statistics_probability_mcq_option_mapping", CATEGORY, is_mcq_option_mapping, description="MCQ statistics/probability problem requiring final option-letter mapping."),
        DerivedRule("statistics_probability_multi_answer", CATEGORY, is_multi_answer, description="Multi-answer statistics/probability problem."),

        DerivedRule("statistics_probability_stat_table_high_precision", CATEGORY, is_statistics_table_high_precision, description="Descriptive statistics table requiring high precision and all table entries."),
        DerivedRule("statistics_probability_chi_square_decision_contract", CATEGORY, is_chi_square_decision_contract, description="Chi-square test with final YES/NO decision contract."),
        DerivedRule("statistics_probability_chi_square_assumption_letters", CATEGORY, is_chi_square_assumption_letters, description="Chi-square assumption checking with embedded letter choices."),
        DerivedRule("statistics_probability_sample_size_raw_value", CATEGORY, is_sample_size_raw_value, description="Sample size calculation where raw computed value should be preserved unless integer rounding is explicit."),
        DerivedRule("statistics_probability_type_ii_error_beta", CATEGORY, is_type_ii_error_beta, description="Type II error beta calculation under alternative mean."),
        DerivedRule("statistics_probability_probability_threshold_log", CATEGORY, is_probability_threshold_log, description="Probability threshold solved with logarithms."),
        DerivedRule("statistics_probability_embedded_letter_choices", CATEGORY, is_embedded_letter_choices, description="Free-form problem with embedded A/B/C/D choices for each blank."),
        DerivedRule("statistics_probability_f_critical_embedded_choices", CATEGORY, is_f_critical_embedded_choices, description="F critical value problem with embedded answer choices."),
        DerivedRule("statistics_probability_long_regression_vector", CATEGORY, is_long_regression_vector, description="Long x=c(...), y=c(...) regression/correlation vector task."),
        DerivedRule("statistics_probability_uppercase_categorical", CATEGORY, is_uppercase_categorical_problem, description="Categorical final answers should use uppercase canonical words."),
        DerivedRule("statistics_probability_high_precision_numeric", CATEGORY, is_high_precision_stats_numeric, description="Statistics numeric answer likely requires high precision."),
    
        DerivedRule("statistics_probability_representative_choice_letters", CATEGORY, is_representative_choice_letters, description="Representative/non-representative prompts where final answers should be A/B letters."),
        DerivedRule("statistics_probability_measurement_scale_abbreviation", CATEGORY, is_measurement_scale_abbreviation, description="Nominal/ordinal/interval/ratio prompts where final answers should use N/O/I/R abbreviations."),
        DerivedRule("statistics_probability_true_false_abbreviation", CATEGORY, is_true_false_abbreviation, description="True/False statistics prompts where final answers should use T/F letters."),
        DerivedRule("statistics_probability_coded_categorical_answer", CATEGORY, is_coded_categorical_stats_answer, description="Categorical statistics answer that should use compact code letters rather than full words."),
        DerivedRule("statistics_probability_two_mean_equal_sample_size", CATEGORY, is_two_mean_equal_sample_size_problem, description="Two-population mean-difference sample-size problem with equal sample sizes."),
    ]
    for rule in rules:
        register_rule(rule)
    return rules


def register_all():
    harness = register_category_harness()
    rules = register_category_rules()
    return {"category": CATEGORY, "default_strategy": DEFAULT_STRATEGY, "candidate_strategies": CANDIDATE_STRATEGIES, "harness": harness, "rules": rules}
