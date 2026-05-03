import re


FINAL_ANSWER_RE = re.compile(r"Final\s+Answer\s*:", re.IGNORECASE)


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


def extract_final_answer(raw_output):
    text = raw_output or ""
    boxed = _find_boxed_contents(text)

    if boxed:
        return boxed[-1].strip()

    matches = list(FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return ""

    answer = text[matches[-1].end():].strip()
    return answer.strip().strip("`").strip()


def is_well_formed_output(raw_output):
    text = raw_output or ""
    if "Reasoning:" not in text:
        return False

    matches = list(FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return False

    final_section = text[matches[-1].end():].strip()
    boxed_spans = _find_boxed_spans(final_section)
    if len(boxed_spans) != 1:
        return False

    start, end, content = boxed_spans[0]
    if start != 0 or not content:
        return False

    return final_section[end:].strip() == ""


def parse_model_output(raw_output):
    return {
        "raw_output": raw_output or "",
        "extracted_answer": extract_final_answer(raw_output),
        "well_formed": is_well_formed_output(raw_output),
    }
