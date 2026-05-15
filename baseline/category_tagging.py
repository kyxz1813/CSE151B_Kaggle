import csv
import json
import re
from pathlib import Path

from prompting.strategies import BASELINE3_CATEGORIES

from .datasets import ProblemSet, save_jsonl
from .generation import GenerationConfig, generate_prompt_texts


CATEGORY_TAG_SYSTEM_PROMPT = """\
/no_think
You categorize math problems for prompt routing.

Choose exactly one category from this fixed list:
statistics_probability, calculus, geometry_trig, linear_algebra, discrete_algorithm, arithmetic_algebra, applied_word_problem, general_math

Return only one JSON object in this exact schema:
{"primary_category":"<category>","categories":["<same category>"]}

Rules:
- Choose exactly one primary_category.
- categories must contain exactly the same one category.
- Use general_math only when no more specific category fits.
- Do not solve the problem.
- Do not explain.
- Do not include markdown.
- Do not include code fences.
- Do not include text before or after the JSON.
"""


CATEGORY_ONE_HOT_FIELDS = (
    "id",
    "answer_format",
    "is_mcq",
    "primary_category",
    "category_tag_parse_ok",
    "category_tag_source",
    *BASELINE3_CATEGORIES,
)


def _format_options(options):
    lines = []
    for idx, option in enumerate(options or []):
        lines.append(f"{chr(ord('A') + idx)}. {option}")
    return "\n".join(lines)


def build_category_tag_messages(record):
    question = str(record.get("question", "")).strip()
    options = record.get("options") or []
    option_block = _format_options(options)
    answer_format = "mcq" if options else "free_form"

    user_prompt = (
        "/no_think\n"
        f"Answer format: {answer_format}\n\n"
        f"Question:\n{question}"
    )

    if option_block:
        user_prompt += f"\n\nAnswer choices:\n{option_block}"

    user_prompt += "\n\nReturn only the JSON object."

    return [
        {"role": "system", "content": CATEGORY_TAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _coerce_valid_category(value):
    category = str(value or "").strip()

    if category in BASELINE3_CATEGORIES:
        return category

    normalized = (
        category.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "")
        .replace(",", "")
        .replace("`", "")
        .replace('"', "")
        .replace("'", "")
    )

    if normalized in BASELINE3_CATEGORIES:
        return normalized

    aliases = {
        "probability": "statistics_probability",
        "statistics": "statistics_probability",
        "stats": "statistics_probability",
        "statistical_probability": "statistics_probability",
        "hypothesis_testing": "statistics_probability",
        "type_i_error": "statistics_probability",
        "type_ii_error": "statistics_probability",

        "calc": "calculus",
        "integral": "calculus",
        "integrals": "calculus",
        "derivative": "calculus",
        "derivatives": "calculus",
        "limit": "calculus",
        "limits": "calculus",
        "complex_analysis": "calculus",
        "residue": "calculus",
        "residues": "calculus",
        "contour_integral": "calculus",
        "differential_equation": "calculus",
        "differential_equations": "calculus",

        "geometry": "geometry_trig",
        "trigonometry": "geometry_trig",
        "trig": "geometry_trig",
        "triangle": "geometry_trig",

        "linearalgebra": "linear_algebra",
        "linear_algebra": "linear_algebra",
        "matrix": "linear_algebra",
        "matrices": "linear_algebra",
        "vector": "linear_algebra",
        "vectors": "linear_algebra",

        "discrete": "discrete_algorithm",
        "algorithm": "discrete_algorithm",
        "algorithms": "discrete_algorithm",
        "combinatorics": "discrete_algorithm",
        "number_theory": "discrete_algorithm",
        "modular_arithmetic": "discrete_algorithm",

        "arithmetic": "arithmetic_algebra",
        "algebra": "arithmetic_algebra",
        "prealgebra": "arithmetic_algebra",
        "pre_algebra": "arithmetic_algebra",

        "word_problem": "applied_word_problem",
        "applied": "applied_word_problem",
        "application": "applied_word_problem",
        "unit_conversion": "applied_word_problem",
        "conversion": "applied_word_problem",

        "general": "general_math",
        "math": "general_math",
        "other": "general_math",
    }

    return aliases.get(normalized)


def _parse_json_candidates(text):
    if not text:
        return []

    decoder = json.JSONDecoder()
    candidates = []

    for idx, char in enumerate(text):
        if char != "{":
            continue

        try:
            obj, end = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                candidates.append({
                    "start": idx,
                    "end": idx + end,
                    "obj": obj,
                })
        except json.JSONDecodeError:
            continue

    return candidates


def _category_from_json_object(obj):
    primary = _coerce_valid_category(obj.get("primary_category"))

    if primary:
        return primary

    raw_categories = obj.get("categories") or []
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]

    for category in raw_categories:
        primary = _coerce_valid_category(category)
        if primary:
            return primary

    return None


def _category_from_json(text):
    candidates = _parse_json_candidates(text)

    valid = []
    for candidate in candidates:
        category = _category_from_json_object(candidate["obj"])
        if category:
            valid.append((candidate["start"], category))

    if not valid:
        return None

    valid.sort(key=lambda item: item[0])
    return valid[-1][1]


def _category_from_explicit_label(text):
    text = text or ""
    hits = []

    patterns = [
        r"CATEGORY\s*:\s*([A-Za-z_ -]+)",
        r"primary_category\s*[:=]\s*([A-Za-z_ -]+)",
        r"category\s*[:=]\s*([A-Za-z_ -]+)",
        r"best category is\s+([A-Za-z_ -]+)",
        r"the category is\s+([A-Za-z_ -]+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            category = _coerce_valid_category(match.group(1))
            if category:
                hits.append((match.start(), category))

    lower = text.lower()
    for category in BASELINE3_CATEGORIES:
        start = 0
        while True:
            idx = lower.find(category.lower(), start)
            if idx == -1:
                break
            hits.append((idx, category))
            start = idx + 1

    if not hits:
        return None

    hits.sort(key=lambda item: item[0])
    return hits[-1][1]


def _question_text(record):
    pieces = [str(record.get("question", ""))]
    for option in record.get("options") or []:
        pieces.append(str(option))
    return "\n".join(pieces).lower()


def _rule_based_category(record):
    text = _question_text(record)

    statistics_probability_terms = [
        "probability",
        "random",
        "expected value",
        "expectation",
        "variance",
        "standard deviation",
        "normal distribution",
        "hypothesis",
        "null hypothesis",
        "alternative hypothesis",
        "p-value",
        "p value",
        "type i",
        "type ii",
        "confidence interval",
        "sample mean",
        "population mean",
        "sigma",
        "alpha",
        "regression",
        "binomial",
        "poisson",
        "z-score",
        "z score",
    ]

    calculus_terms = [
        "integral",
        "integrate",
        "differentiate",
        "derivative",
        "limit",
        "lim_",
        "antiderivative",
        "taylor",
        "maclaurin",
        "series expansion",
        "approximation polynomial",
        "best first-degree approximation",
        "residue",
        "residues",
        "contour",
        "complex integral",
        "tan(πz)",
        "tan(\\pi z)",
        "differential equation",
        "exponential decay",
        "newton's law of cooling",
    ]

    geometry_trig_terms = [
        "triangle",
        "circle",
        "angle",
        "cos",
        "sin",
        "tan",
        "trig",
        "radian",
        "degree",
        "law of cosines",
        "law of sines",
        "polygon",
        "area",
        "perimeter",
        "volume",
        "surface area",
    ]

    linear_algebra_terms = [
        "matrix",
        "matrices",
        "determinant",
        "eigen",
        "vector",
        "rank",
        "basis",
        "linear transformation",
        "null space",
        "column space",
        "analytic function",
    ]

    discrete_algorithm_terms = [
        "algorithm",
        "recurrence",
        "graph",
        "tree",
        "modulo",
        "modular",
        "integer sequence",
        "combinatorics",
        "permutation",
        "combination",
        "divisibility",
        "prime",
        "gcd",
        "lcm",
    ]

    applied_word_problem_terms = [
        "miles",
        "feet",
        "fahrenheit",
        "celsius",
        "dollars",
        "turkey",
        "brick",
        "kiln",
        "room temperature",
        "convert",
        "conversion",
        "units",
        "mph",
        "rate",
        "mixture",
        "work together",
    ]

    arithmetic_algebra_terms = [
        "simplify",
        "reduce the fraction",
        "solve for",
        "factor",
        "equation",
        "expression",
        "polynomial",
        "fraction",
        "exponent",
        "logarithm",
        "absolute value",
        "equivalent",
        "[ans]",
    ]

    def has_any(terms):
        return any(term in text for term in terms)

    if has_any(statistics_probability_terms):
        return "statistics_probability"

    if has_any(calculus_terms):
        return "calculus"

    if has_any(geometry_trig_terms):
        return "geometry_trig"

    if has_any(linear_algebra_terms):
        return "linear_algebra"

    if has_any(discrete_algorithm_terms):
        return "discrete_algorithm"

    if has_any(applied_word_problem_terms):
        return "applied_word_problem"

    if has_any(arithmetic_algebra_terms):
        return "arithmetic_algebra"

    return "general_math"


def normalize_category_tags(raw_output, record=None):
    primary = _category_from_json(raw_output)
    source = "qwen_json"
    qwen_parse_ok = primary is not None

    if primary is None:
        primary = _category_from_explicit_label(raw_output)
        source = "qwen_text"
        qwen_parse_ok = primary is not None

    if record is not None:
        rule_category = _rule_based_category(record)

        if primary is None:
            primary = rule_category
            source = "rule_fallback"
        elif primary == "general_math" and rule_category != "general_math":
            primary = rule_category
            source = "rule_override_general_math"

    if primary is None:
        primary = "general_math"
        source = "default_fallback"

    return {
        "categories": [primary],
        "primary_category": primary,
        "tag_parse_ok": qwen_parse_ok,
        "category_tag_source": source,
    }


def apply_category_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load_category_tag_map(path):
    rows = {}
    path = Path(path)

    if not path.exists():
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            normalized = normalize_category_tags(
                row.get("category_tag_raw_output", ""),
                record=row,
            )

            primary = _coerce_valid_category(row.get("primary_category"))
            if primary is None:
                primary = normalized["primary_category"]

            row["primary_category"] = primary
            row["qwen_categories"] = [primary]
            row["category_tag_parse_ok"] = row.get(
                "category_tag_parse_ok",
                normalized["tag_parse_ok"],
            )
            row["category_tag_source"] = row.get(
                "category_tag_source",
                normalized["category_tag_source"],
            )

            rows[str(row["id"])] = row

    return rows


def _one_hot_row(row, record=None):
    primary = row.get("primary_category") or "general_math"
    options = record.get("options") if record else None
    answer_format = "mcq" if options else "free_form"

    out = {
        "id": row.get("id"),
        "answer_format": answer_format,
        "is_mcq": bool(options),
        "primary_category": primary,
        "category_tag_parse_ok": bool(row.get("category_tag_parse_ok")),
        "category_tag_source": row.get("category_tag_source") or "unknown",
    }

    for category in BASELINE3_CATEGORIES:
        out[category] = 1 if category == primary else 0

    return out


def save_category_one_hot_csv(tag_rows, records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record_by_id = {str(record.get("id")): record for record in records}

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATEGORY_ONE_HOT_FIELDS)
        writer.writeheader()

        for row in tag_rows:
            record = record_by_id.get(str(row.get("id")), {})
            writer.writerow(_one_hot_row(row, record=record))

    return path


def category_distribution(tag_rows):
    counts = {}

    for row in tag_rows:
        category = row.get("primary_category") or "general_math"
        counts[category] = counts.get(category, 0) + 1

    return dict(sorted(counts.items()))


def tag_problem_set_with_qwen(
    problem_set,
    model_bundle,
    generation_config=None,
    batch_size=32,
    output_jsonl_path=None,
    output_one_hot_csv_path=None,
    existing_tags_path=None,
    limit=None,
    show_progress=True,
):
    records = problem_set.records[:limit] if limit is not None else list(problem_set.records)
    existing_tags = load_category_tag_map(existing_tags_path) if existing_tags_path else {}

    tag_rows = []
    prompt_texts = []
    prompt_records = []

    for record in records:
        existing = existing_tags.get(str(record.get("id")))

        if existing:
            primary = _coerce_valid_category(existing.get("primary_category"))
            if primary is None:
                primary = "general_math"

            existing = dict(existing)
            existing["primary_category"] = primary
            existing["qwen_categories"] = [primary]
            existing.setdefault("category_tag_source", "existing_tags")
            tag_rows.append(existing)
            continue

        messages = build_category_tag_messages(record)
        prompt_texts.append(apply_category_chat_template(model_bundle.tokenizer, messages))
        prompt_records.append((record, messages))

    gen = generation_config or GenerationConfig(
        max_new_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        do_sample=False,
    )

    if prompt_texts:
        generations = generate_prompt_texts(
            model_bundle=model_bundle,
            prompt_texts=prompt_texts,
            generation_config=gen,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        for (record, messages), raw_output in zip(prompt_records, generations["responses"]):
            normalized = normalize_category_tags(raw_output, record=record)
            primary = normalized["primary_category"]

            tag_rows.append({
                "id": record.get("id"),
                "question": record.get("question"),
                "qwen_categories": [primary],
                "primary_category": primary,
                "category_tag_raw_output": raw_output,
                "category_tag_parse_ok": normalized["tag_parse_ok"],
                "category_tag_source": normalized["category_tag_source"],
                "category_tag_prompt": messages,
            })

    by_id = {str(row["id"]): row for row in tag_rows}
    tagged_records = []
    ordered_tag_rows = []

    for record in records:
        row = by_id.get(str(record.get("id")))

        if row is None:
            normalized = normalize_category_tags("", record=record)
            row = {
                "id": record.get("id"),
                "question": record.get("question"),
                "qwen_categories": normalized["categories"],
                "primary_category": normalized["primary_category"],
                "category_tag_raw_output": "",
                "category_tag_parse_ok": normalized["tag_parse_ok"],
                "category_tag_source": normalized["category_tag_source"],
                "category_tag_prompt": build_category_tag_messages(record),
            }

        primary = _coerce_valid_category(row.get("primary_category"))
        if primary is None:
            primary = "general_math"

        row = dict(row)
        row["primary_category"] = primary
        row["qwen_categories"] = [primary]
        row.setdefault("category_tag_source", "unknown")

        tagged = dict(record)
        tagged["qwen_categories"] = [primary]
        tagged["primary_category"] = primary
        tagged["category_tag_raw_output"] = row.get("category_tag_raw_output", "")
        tagged["category_tag_parse_ok"] = row.get("category_tag_parse_ok", False)
        tagged["category_tag_source"] = row.get("category_tag_source", "unknown")

        tagged_records.append(tagged)
        ordered_tag_rows.append(row)

    if output_jsonl_path:
        save_jsonl(ordered_tag_rows, output_jsonl_path)

    if output_one_hot_csv_path:
        save_category_one_hot_csv(ordered_tag_rows, records, output_one_hot_csv_path)

    return ProblemSet(problem_set.name, tagged_records), ordered_tag_rows