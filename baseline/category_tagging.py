import json
import re
from pathlib import Path

from prompting.strategies import BASELINE3_CATEGORIES

from .datasets import ProblemSet, save_jsonl
from .generation import GenerationConfig, generate_prompt_texts


CATEGORY_TAG_SYSTEM_PROMPT = """You categorize math problems for prompt routing.

Choose one or more categories from this fixed list:
statistics_probability, calculus, geometry_trig, linear_algebra, discrete_algorithm, arithmetic_algebra, applied_word_problem, general_math

Return only JSON in this schema:
{"categories":["<category>"],"primary_category":"<category>"}

Rules:
- Use general_math only when no more specific category fits.
- primary_category must be one of categories.
- Do not solve the problem.
"""


def _format_options(options):
    lines = []
    for idx, option in enumerate(options or []):
        lines.append(f"{chr(ord('A') + idx)}. {option}")
    return "\n".join(lines)


def build_category_tag_messages(record):
    question = str(record.get("question", "")).strip()
    options = record.get("options") or []
    option_block = _format_options(options)

    user_prompt = f"Question:\n{question}"
    if option_block:
        user_prompt += f"\n\nAnswer choices:\n{option_block}"

    return [
        {"role": "system", "content": CATEGORY_TAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_object(text):
    if not text:
        return None

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_category_tags(raw_output):
    parsed = _extract_json_object(raw_output)
    categories = []
    primary = None

    if isinstance(parsed, dict):
        raw_categories = parsed.get("categories") or []
        if isinstance(raw_categories, str):
            raw_categories = [raw_categories]

        for category in raw_categories:
            category = str(category).strip()
            if category in BASELINE3_CATEGORIES and category not in categories:
                categories.append(category)

        raw_primary = str(parsed.get("primary_category") or "").strip()
        if raw_primary in BASELINE3_CATEGORIES:
            primary = raw_primary

    if not categories:
        categories = ["general_math"]

    if primary not in categories:
        primary = categories[0]

    if "general_math" in categories and len(categories) > 1:
        categories = [category for category in categories if category != "general_math"]
        if primary == "general_math":
            primary = categories[0]

    return {
        "categories": categories,
        "primary_category": primary,
        "tag_parse_ok": parsed is not None,
    }


def _apply_chat_template(tokenizer, messages):
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
            rows[str(row["id"])] = row

    return rows


def tag_problem_set_with_qwen(
    problem_set,
    model_bundle,
    generation_config=None,
    batch_size=32,
    output_jsonl_path=None,
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
            tag_rows.append(existing)
            continue

        messages = build_category_tag_messages(record)
        prompt_texts.append(_apply_chat_template(model_bundle.tokenizer, messages))
        prompt_records.append((record, messages))

    gen = generation_config or GenerationConfig(
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
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
            normalized = normalize_category_tags(raw_output)
            tag_rows.append({
                "id": record.get("id"),
                "question": record.get("question"),
                "qwen_categories": normalized["categories"],
                "primary_category": normalized["primary_category"],
                "category_tag_raw_output": raw_output,
                "category_tag_parse_ok": normalized["tag_parse_ok"],
                "category_tag_prompt": messages,
            })

    by_id = {str(row["id"]): row for row in tag_rows}
    tagged_records = []

    for record in records:
        row = by_id.get(str(record.get("id")))
        if row is None:
            row = {
                "id": record.get("id"),
                "question": record.get("question"),
                "qwen_categories": ["general_math"],
                "primary_category": "general_math",
                "category_tag_raw_output": "",
                "category_tag_parse_ok": False,
                "category_tag_prompt": build_category_tag_messages(record),
            }

        tagged = dict(record)
        tagged["qwen_categories"] = row["qwen_categories"]
        tagged["primary_category"] = row["primary_category"]
        tagged["category_tag_raw_output"] = row["category_tag_raw_output"]
        tagged["category_tag_parse_ok"] = row["category_tag_parse_ok"]
        tagged_records.append(tagged)

    ordered_tag_rows = [by_id[str(record.get("id"))] for record in records if str(record.get("id")) in by_id]
    if output_jsonl_path:
        save_jsonl(ordered_tag_rows, output_jsonl_path)

    return ProblemSet(problem_set.name, tagged_records), ordered_tag_rows
