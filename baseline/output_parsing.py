import re


FINAL_ANSWER_RE = re.compile(r"Final\s+Answer\s*:", re.IGNORECASE)
ANS_RE = re.compile(r"\[ANS\]", re.IGNORECASE)


def _find_boxed_spans(text):
    marker = "\\boxed{"
    results = []
    start = 0

    while True:
        marker_pos = text.find(marker, start)
        if marker_pos == -1:
            break

        content_start = marker_pos + len(marker)
        depth = 1
        i = content_start

        while i < len(text):
            char = text[i]
            prev = text[i - 1] if i > 0 else ""

            if char == "{" and prev != "\\":
                depth += 1
            elif char == "}" and prev != "\\":
                depth -= 1
                if depth == 0:
                    results.append((marker_pos, i + 1, text[content_start:i].strip()))
                    start = i + 1
                    break

            i += 1
        else:
            break

    return results


def _find_boxed_contents(text):
    return [content for _, _, content in _find_boxed_spans(text)]


def _last_final_section(text):
    matches = list(FINAL_ANSWER_RE.finditer(text or ""))

    if not matches:
        return None

    return (text or "")[matches[-1].end():].strip()


def _top_level_comma_split(text):
    parts = []
    current = []
    depth = 0

    pairs = {
        "{": "}",
        "(": ")",
        "[": "]",
    }

    closing = set(pairs.values())

    for char in text or "":
        if char in pairs:
            depth += 1
            current.append(char)
        elif char in closing:
            depth = max(0, depth - 1)
            current.append(char)
        elif char == "," and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
        else:
            current.append(char)

    piece = "".join(current).strip()
    if piece:
        parts.append(piece)

    return parts


def expected_answer_count(record):
    if not record:
        return 1

    if record.get("options"):
        return 1

    question = str(record.get("question", ""))
    count = len(ANS_RE.findall(question))

    return max(1, count)


def valid_mcq_letters(record):
    if not record or not record.get("options"):
        return []

    return [chr(ord("A") + idx) for idx in range(len(record.get("options") or []))]


def extract_final_answer(raw_output):
    text = raw_output or ""
    final_section = _last_final_section(text)

    if final_section is not None:
        boxed = _find_boxed_contents(final_section)
        if boxed:
            return boxed[-1].strip()

        return final_section.strip().strip("`").strip()

    boxed = _find_boxed_contents(text)
    if boxed:
        return boxed[-1].strip()

    return ""


def validate_output_schema(raw_output, record=None):
    text = raw_output or ""
    errors = []

    has_reasoning = "Reasoning:" in text
    if not has_reasoning:
        errors.append("missing_reasoning_section")

    final_matches = list(FINAL_ANSWER_RE.finditer(text))
    has_final_marker = bool(final_matches)

    if not has_final_marker:
        errors.append("missing_final_answer_marker")
        final_section = text.strip()
    else:
        final_section = text[final_matches[-1].end():].strip()

    boxed_spans = _find_boxed_spans(final_section)

    if "\\boxed{" in final_section and not boxed_spans:
        errors.append("unclosed_boxed_answer")

    if not boxed_spans:
        errors.append("missing_boxed_answer_after_final")
        extracted_answer = extract_final_answer(text)
        return {
            "schema_errors": errors,
            "extracted_answer": extracted_answer,
            "boxed_answer": "",
            "boxed_answers_after_final": [],
            "expected_answer_count": expected_answer_count(record),
            "actual_answer_count": 0,
            "extractable": bool(extracted_answer),
            "well_formed": False,
            "strict_well_formed": False,
            "schema_valid": False,
        }

    if len(boxed_spans) > 1:
        errors.append("multiple_boxed_answers_after_final")

    start, end, content = boxed_spans[-1]
    boxed_answer = content.strip()
    boxed_answers = [span[2].strip() for span in boxed_spans]

    if not boxed_answer:
        errors.append("empty_boxed_answer")

    if len(boxed_spans) == 1:
        only_start, only_end, _ = boxed_spans[0]

        if only_start != 0:
            errors.append("text_before_final_box")

        trailing = final_section[only_end:].strip()
        if trailing:
            errors.append("trailing_text_after_final_box")

    expected_count = expected_answer_count(record)
    actual_count = 1

    if record and record.get("options"):
        letters = valid_mcq_letters(record)
        candidate = boxed_answer.strip().upper()

        if not re.fullmatch(r"[A-Z]", candidate or ""):
            errors.append("mcq_answer_not_single_letter")
        elif letters and candidate not in letters:
            errors.append("mcq_answer_letter_out_of_range")

    elif expected_count > 1:
        parts = _top_level_comma_split(boxed_answer)
        actual_count = len(parts)

        if actual_count != expected_count:
            errors.append("multi_answer_count_mismatch")

    structural_errors = {
        "missing_reasoning_section",
        "missing_final_answer_marker",
        "missing_boxed_answer_after_final",
        "unclosed_boxed_answer",
        "multiple_boxed_answers_after_final",
        "empty_boxed_answer",
        "text_before_final_box",
        "trailing_text_after_final_box",
    }

    well_formed = not any(error in structural_errors for error in errors)
    schema_valid = not errors

    return {
        "schema_errors": errors,
        "extracted_answer": boxed_answer,
        "boxed_answer": boxed_answer,
        "boxed_answers_after_final": boxed_answers,
        "expected_answer_count": expected_count,
        "actual_answer_count": actual_count,
        "extractable": bool(boxed_answer),
        "well_formed": well_formed,
        "strict_well_formed": well_formed,
        "schema_valid": schema_valid,
    }


def is_well_formed_output(raw_output):
    return validate_output_schema(raw_output)["well_formed"]


def parse_model_output(raw_output, record=None):
    text = raw_output or ""
    validation = validate_output_schema(text, record=record)

    return {
        "raw_output": text,
        "response_for_submission": text,
        "response_for_scoring": text,
        "extracted_answer": validation["extracted_answer"],
        "boxed_answer": validation["boxed_answer"],
        "boxed_answers_after_final": validation["boxed_answers_after_final"],
        "expected_answer_count": validation["expected_answer_count"],
        "actual_answer_count": validation["actual_answer_count"],
        "schema_errors": validation["schema_errors"],
        "extractable": validation["extractable"],
        "well_formed": validation["well_formed"],
        "strict_well_formed": validation["strict_well_formed"],
        "schema_valid": validation["schema_valid"],
        "repaired_response": text,
    }