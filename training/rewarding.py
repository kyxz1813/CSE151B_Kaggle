import json
import re
from baseline.scoring import score_one, load_judger

try:
    from baseline.output_parsing import parse_model_output
except Exception:
    parse_model_output = None

FINAL_ANSWER_RE = re.compile(r"Final\s+Answer\s*:", re.IGNORECASE)


def extract_completion_text(completion):
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "")
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def expected_answer_count(record):
    if not record:
        return None
    if record.get("options"):
        return 1
    return max(1, str(record.get("question", "")).count("[ANS]"))


def split_boxed_answer(answer):
    answer = str(answer or "").strip()
    if not answer:
        return []
    if "," not in answer:
        return [answer]
    return [part.strip() for part in answer.split(",")]


def actual_answer_count(answer):
    return len(split_boxed_answer(answer))


def safe_parse_output(text, record=None):
    if parse_model_output is not None:
        try:
            return parse_model_output(text, record=record, sanitize=True)
        except TypeError:
            try:
                return parse_model_output(text)
            except Exception:
                pass
        except Exception:
            pass
    boxed = ""
    if "\\boxed{" in text:
        idx = text.rfind("\\boxed{")
        start = idx + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    boxed = text[start:i].strip()
                    break
            i += 1
    well_formed = bool(FINAL_ANSWER_RE.search(text) and boxed)
    return {
        "raw_output": text or "",
        "response_for_submission": text or "",
        "extracted_answer": boxed,
        "boxed_answer": boxed,
        "extractable": bool(boxed),
        "schema_valid": well_formed,
        "strict_well_formed": well_formed,
        "well_formed": well_formed,
        "schema_errors": [] if well_formed else ["not_well_formed"],
        "expected_answer_count": expected_answer_count(record) if record else None,
        "actual_answer_count": actual_answer_count(boxed),
    }


def valid_mcq_letter(record, parsed):
    if not record or not record.get("options"):
        return True
    boxed = str(parsed.get("boxed_answer") or parsed.get("extracted_answer") or "").strip().upper()
    if len(boxed) != 1:
        return False
    valid = [chr(ord("A") + i) for i in range(len(record.get("options") or []))]
    return boxed in valid


def score_correctness(record, completion, judger=None):
    if record is None or record.get("answer") is None:
        return 0.0
    text = extract_completion_text(completion)
    try:
        correct = score_one(record, text, judger=judger)
        return 1.0 if correct else 0.0
    except Exception:
        return 0.0


def reward_breakdown(record, completion, judger=None):
    text = extract_completion_text(completion)
    parsed = safe_parse_output(text, record=record)
    exp = expected_answer_count(record)
    actual = parsed.get("actual_answer_count")
    count_ok = exp is not None and actual is not None and exp == actual
    correct = score_correctness(record, text, judger=judger)
    schema_valid = 1.0 if parsed.get("schema_valid") else 0.0
    extractable = 1.0 if parsed.get("extractable") else 0.0
    answer_count = 1.0 if count_ok else 0.0
    mcq_valid = 1.0 if valid_mcq_letter(record, parsed) else 0.0
    missing_final = 0.0 if FINAL_ANSWER_RE.search(text) else -0.10
    multiple_final = -0.05 * max(0, len(FINAL_ANSWER_RE.findall(text)) - 1)
    combined = 1.0 * correct + 0.15 * schema_valid + 0.10 * extractable + 0.10 * answer_count + 0.05 * mcq_valid + missing_final + multiple_final
    return {
        "combined": float(combined),
        "correct": float(correct),
        "schema_valid": schema_valid,
        "extractable": extractable,
        "answer_count": answer_count,
        "mcq_valid": mcq_valid,
        "missing_final_penalty": missing_final,
        "multiple_final_penalty": multiple_final,
        "boxed_answer": parsed.get("boxed_answer"),
        "expected_answer_count": exp,
        "actual_answer_count": actual,
    }


def record_from_json_value(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def _records_from_kwargs(n, kwargs):
    records = kwargs.get("records")
    if records is None:
        records = kwargs.get("record")
    if records is None:
        raw_records = []
        for i in range(n):
            row = {}
            for key in ["id", "question", "options", "answer", "primary_category", "derived_rules"]:
                val = kwargs.get(key)
                if isinstance(val, list) and len(val) == n:
                    row[key] = val[i]
                elif val is not None and not isinstance(val, list):
                    row[key] = val
            raw_records.append(row)
        return raw_records
    if isinstance(records, list):
        return [record_from_json_value(x) or {} for x in records]
    parsed = record_from_json_value(records)
    if isinstance(parsed, list):
        return parsed
    return [parsed or {} for _ in range(n)]


_JUDGER = None


def get_cached_judger(judger_dir="."):
    global _JUDGER
    if _JUDGER is None:
        try:
            _JUDGER = load_judger(judger_dir)
        except Exception:
            _JUDGER = False
    return None if _JUDGER is False else _JUDGER


def correctness_reward_func(prompts=None, completions=None, **kwargs):
    completions = completions or []
    records = _records_from_kwargs(len(completions), kwargs)
    judger = get_cached_judger(kwargs.get("judger_dir", "."))
    return [score_correctness(record, completion, judger=judger) for record, completion in zip(records, completions)]


def format_reward_func(prompts=None, completions=None, **kwargs):
    completions = completions or []
    records = _records_from_kwargs(len(completions), kwargs)
    rewards = []
    for record, completion in zip(records, completions):
        text = extract_completion_text(completion)
        parsed = safe_parse_output(text, record=record)
        reward = 0.0
        reward += 0.40 if parsed.get("schema_valid") else 0.0
        reward += 0.25 if parsed.get("extractable") else 0.0
        reward += 0.20 if parsed.get("actual_answer_count") == parsed.get("expected_answer_count") else 0.0
        reward += 0.15 if valid_mcq_letter(record, parsed) else 0.0
        rewards.append(float(reward))
    return rewards


def combined_reward_func(prompts=None, completions=None, **kwargs):
    completions = completions or []
    records = _records_from_kwargs(len(completions), kwargs)
    judger = get_cached_judger(kwargs.get("judger_dir", "."))
    return [reward_breakdown(record, completion, judger=judger)["combined"] for record, completion in zip(records, completions)]
