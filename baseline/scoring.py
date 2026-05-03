import re
import sys
from pathlib import Path


def extract_letter(text):
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()

    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response, gold_letter):
    return extract_letter(response) == str(gold_letter).strip().upper()


def load_judger(judger_dir="."):
    judger_dir = str(Path(judger_dir).resolve())
    if judger_dir not in sys.path:
        sys.path.insert(0, judger_dir)

    from .judger import Judger

    return Judger(strict_extract=False)


def score_free_form(response, gold, judger):
    gold_list = gold if isinstance(gold, list) else [gold]

    try:
        return bool(judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        ))
    except Exception:
        return False


def score_one(record, response, judger=None):
    if "answer" not in record or record.get("answer") is None:
        return None

    if record.get("options"):
        return score_mcq(response, record.get("answer"))

    if judger is None:
        judger = load_judger()

    return score_free_form(response, record.get("answer"), judger)


def score_records(records, responses, judger=None):
    rows = []

    for record, response in zip(records, responses):
        is_mcq = bool(record.get("options"))
        correct = score_one(record, response, judger=judger)

        rows.append({
            "id": record.get("id"),
            "is_mcq": is_mcq,
            "gold": record.get("answer"),
            "response": response,
            "correct": correct,
        })

    return rows


def summarize_results(results):
    answered = [r for r in results if r["correct"] is not None]
    mcq = [r for r in answered if r["is_mcq"]]
    free = [r for r in answered if not r["is_mcq"]]

    def acc(rows):
        if not rows:
            return None
        return sum(bool(r["correct"]) for r in rows) / len(rows)

    return {
        "n_scored": len(answered),
        "overall_acc": acc(answered),
        "mcq_acc": acc(mcq),
        "free_form_acc": acc(free),
        "n_mcq": len(mcq),
        "n_free_form": len(free),
        "n_correct": sum(bool(r["correct"]) for r in answered),
    }
