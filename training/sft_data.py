import argparse
import json
import random
from pathlib import Path

from prompting.models import problem_from_record
from prompting.prompt_chain import build_prompt_chain


def read_jsonl(path):
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def records_by_id(records):
    return {str(record.get("id")): record for record in records}


def normalize_gold_answer(record):
    answer = record.get("answer")
    if record.get("options"):
        return str(answer).strip().upper()
    if isinstance(answer, list):
        return ", ".join(str(x).strip() for x in answer)
    return str(answer).strip()


def count_ans_placeholders(record):
    return str(record.get("question", "")).count("[ANS]")


def answer_format(record):
    return "mcq" if record.get("options") else "free_form"


def build_gold_target_response(record, reasoning_style="brief"):
    gold = normalize_gold_answer(record)
    is_mcq = bool(record.get("options"))
    n_blanks = count_ans_placeholders(record)
    if reasoning_style == "none":
        reasoning = "Reasoning:\n"
    elif is_mcq:
        reasoning = "Reasoning:\nSolve the problem, compare the result to the answer choices, and return only the matching letter.\n"
    elif n_blanks > 1:
        reasoning = f"Reasoning:\nCount the {n_blanks} answer blank(s), solve each requested part, and preserve the requested order.\n"
    else:
        reasoning = "Reasoning:\nSolve the problem carefully and preserve exact form or enough precision for the requested answer.\n"
    return f"{reasoning}\nFinal Answer: \\boxed{{{gold}}}"


def response_from_result_row(row):
    for key in ["response_for_submission", "response", "raw_output", "initial_raw_output"]:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def is_valid_training_response(text):
    text = str(text or "")
    return "Final Answer:" in text and "\\boxed{" in text and len(text.strip()) > 0


def render_prompt_messages(record, strategy_name="baseline3_adaptive_rules"):
    chain = build_prompt_chain(strategy_name=strategy_name)
    problem = problem_from_record(record)
    spec = chain.build_spec(problem)
    return spec.to_messages(), spec.metadata


def make_sft_example(record, target_response, strategy_name="baseline3_adaptive_rules", source="gold", extra_metadata=None):
    messages, prompt_metadata = render_prompt_messages(record, strategy_name=strategy_name)
    messages = list(messages)
    messages.append({"role": "assistant", "content": target_response})
    metadata = {
        "id": record.get("id"),
        "source": source,
        "answer_format": answer_format(record),
        "primary_category": record.get("primary_category"),
        "derived_rules": record.get("derived_rules") or [],
        "prompt_strategy": strategy_name,
        "prompt_metadata": prompt_metadata,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {"messages": messages, "metadata": metadata}


def build_gold_sft_examples(records, strategy_name="baseline3_adaptive_rules", reasoning_style="brief", skip_unanswered=True):
    examples = []
    for record in records:
        if skip_unanswered and record.get("answer") is None:
            continue
        target = build_gold_target_response(record, reasoning_style=reasoning_style)
        examples.append(make_sft_example(record, target, strategy_name=strategy_name, source="gold_concise"))
    return examples


def build_result_trace_examples(records, result_rows, strategy_name="baseline3_adaptive_rules", include_correct=True, include_wrong_corrections=True, include_malformed_corrections=True, reasoning_style="brief"):
    record_map = records_by_id(records)
    examples = []
    for row in result_rows:
        record = record_map.get(str(row.get("id")))
        if not record:
            continue
        correct = row.get("correct")
        schema_valid = row.get("schema_valid")
        trace = response_from_result_row(row)
        if include_correct and correct is True and is_valid_training_response(trace):
            examples.append(make_sft_example(record, trace, strategy_name=strategy_name, source="correct_model_trace", extra_metadata={"source_result_correct": correct, "source_schema_valid": schema_valid}))
            continue
        needs_correction = include_wrong_corrections and record.get("answer") is not None and correct is False
        needs_format_correction = include_malformed_corrections and record.get("answer") is not None and not schema_valid
        if needs_correction or needs_format_correction:
            target = build_gold_target_response(record, reasoning_style=reasoning_style)
            examples.append(make_sft_example(record, target, strategy_name=strategy_name, source="gold_correction_from_result", extra_metadata={"source_result_correct": correct, "source_schema_valid": schema_valid, "source_boxed_answer": row.get("boxed_answer"), "source_schema_errors": row.get("schema_errors") or []}))
    return examples


def balanced_sample_by_category(examples, max_per_category=None, seed=414):
    if max_per_category is None:
        return list(examples)
    rng = random.Random(seed)
    groups = {}
    for ex in examples:
        cat = ex.get("metadata", {}).get("primary_category") or "unknown"
        groups.setdefault(cat, []).append(ex)
    sampled = []
    for cat, rows in sorted(groups.items()):
        rows = list(rows)
        rng.shuffle(rows)
        sampled.extend(rows[:max_per_category])
    rng.shuffle(sampled)
    return sampled


def split_examples(examples, val_frac=0.10, seed=414):
    rng = random.Random(seed)
    rows = list(examples)
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]


def write_summary(examples, path):
    counts = {}
    sources = {}
    for ex in examples:
        meta = ex.get("metadata", {})
        cat = meta.get("primary_category") or "unknown"
        src = meta.get("source") or "unknown"
        counts[cat] = counts.get(cat, 0) + 1
        sources[src] = sources.get(src, 0) + 1
    summary = {"n_examples": len(examples), "category_counts": dict(sorted(counts.items())), "source_counts": dict(sorted(sources.items()))}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def build_sft_dataset(records_path, output_dir, results_paths=None, strategy_name="baseline3_adaptive_rules", max_per_category=None, val_frac=0.10, seed=414, reasoning_style="brief"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(records_path)
    examples = build_gold_sft_examples(records, strategy_name=strategy_name, reasoning_style=reasoning_style)
    for result_path in results_paths or []:
        rows = read_jsonl(result_path)
        examples.extend(build_result_trace_examples(records, rows, strategy_name=strategy_name, reasoning_style=reasoning_style))
    dedup = {}
    for ex in examples:
        meta = ex.get("metadata", {})
        key = (str(meta.get("id")), meta.get("source"), ex["messages"][-1]["content"])
        dedup[key] = ex
    examples = balanced_sample_by_category(list(dedup.values()), max_per_category=max_per_category, seed=seed)
    train, val = split_examples(examples, val_frac=val_frac, seed=seed)
    train_path = output_dir / "train_sft.jsonl"
    val_path = output_dir / "val_sft.jsonl"
    all_path = output_dir / "all_sft.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)
    write_jsonl(examples, all_path)
    summary = write_summary(examples, summary_path)
    return {"train_path": str(train_path), "val_path": str(val_path), "all_path": str(all_path), "summary_path": str(summary_path), "summary": summary}


def parse_args():
    parser = argparse.ArgumentParser(description="Build SFT JSONL datasets from competition records and baseline result traces.")
    parser.add_argument("--records-path", required=True)
    parser.add_argument("--output-dir", default="results/sft_data")
    parser.add_argument("--results-paths", nargs="*", default=[])
    parser.add_argument("--strategy-name", default="baseline3_adaptive_rules")
    parser.add_argument("--max-per-category", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=414)
    parser.add_argument("--reasoning-style", choices=["brief", "none"], default="brief")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_sft_dataset(args.records_path, args.output_dir, results_paths=args.results_paths, strategy_name=args.strategy_name, max_per_category=args.max_per_category, val_frac=args.val_frac, seed=args.seed, reasoning_style=args.reasoning_style)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
