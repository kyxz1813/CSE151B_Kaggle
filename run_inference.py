import argparse
import csv
import importlib
import json
import os
from pathlib import Path

from tqdm.auto import tqdm


FINAL_CATEGORIES = [
    "statistics_probability",
    "calculus",
    "geometry_trig",
    "linear_algebra",
    "discrete_algorithm",
    "arithmetic_algebra",
    "applied_word_problem",
    "general_math",
]


def register_final_components():
    try:
        import baseline.rule_guidance as rule_guidance
        rule_guidance.register_all_default_guidance()
    except Exception as exc:
        print("Warning: could not register rule guidance:", repr(exc))

    module_names = [
        "baseline.category_work.arithmetic_algebra",
        "baseline.category_work.statistics_probability",
        "baseline.category_work.applied_word_problem",
        "baseline.category_work.geometry_trig",
        "baseline.category_work.calculus",
        "baseline.category_work.linear_algebra",
        "baseline.category_work.discrete_algorithm",
        "baseline.category_work.general_math",
    ]

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "register_all"):
                module.register_all()
        except ModuleNotFoundError:
            continue
        except Exception as exc:
            print(f"Warning: could not register {module_name}: {repr(exc)}")


def load_jsonl_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl_records(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


def save_submission_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "id": row["id"],
                "response": row["response"],
            })

    return path


def build_problem_set(name, records):
    from baseline.datasets import ProblemSet
    return ProblemSet(name=name, records=records)


def load_private_problem_set(private_data_path):
    try:
        from baseline.datasets import load_private_set
        return load_private_set(private_data_path)
    except Exception:
        records = load_jsonl_records(private_data_path)
        return build_problem_set("private", records)


def build_model_bundle(
    model_id,
    backend,
    cache_dir,
    gpu_id,
    max_input_tokens,
    max_model_len,
    gpu_memory_utilization,
    max_num_seqs,
    max_num_batched_tokens,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from baseline.modeling import ModelConfig, load_model

    model_config = ModelConfig(
        model_id=model_id,
        backend=backend,
        cache_dir=cache_dir,
        gpu_id=str(gpu_id),
        max_input_tokens=max_input_tokens,
        max_model_len=max_model_len,
        dtype="bfloat16",
        torch_dtype="bfloat16",
        load_in_4bit=False,
        device_map="auto",
        low_cpu_mem_usage=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        reuse_loaded=True,
    )

    return load_model(model_config)


def generation_config(
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    min_p,
    repetition_penalty,
    presence_penalty,
    do_sample,
):
    from baseline.generation import GenerationConfig

    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        do_sample=do_sample,
    )


def tag_private_problem_set(
    problem_set,
    model_bundle,
    tag_cache_path,
    one_hot_csv_path,
    batch_size,
    show_progress,
):
    from baseline.category_tagging import tag_problem_set_with_qwen

    tag_gen = generation_config(
        max_new_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        do_sample=False,
    )

    existing = tag_cache_path if Path(tag_cache_path).exists() else None

    tagged_problem_set, tag_rows = tag_problem_set_with_qwen(
        problem_set=problem_set,
        model_bundle=model_bundle,
        generation_config=tag_gen,
        batch_size=batch_size,
        output_jsonl_path=tag_cache_path,
        output_one_hot_csv_path=one_hot_csv_path,
        existing_tags_path=existing,
        limit=None,
        show_progress=show_progress,
    )

    return tagged_problem_set, tag_rows


def annotate_rules(problem_set, output_jsonl_path=None):
    try:
        from baseline.category_rules import annotate_problem_set_with_rules

        try:
            annotated = annotate_problem_set_with_rules(
                problem_set=problem_set,
                categories=FINAL_CATEGORIES,
            )
        except TypeError:
            annotated = annotate_problem_set_with_rules(problem_set)

        if output_jsonl_path is not None:
            save_jsonl_records(annotated.records, output_jsonl_path)

        return annotated

    except Exception as exc:
        print("Warning: rule annotation failed; continuing with category-only records:", repr(exc))
        return problem_set


def make_problem(record):
    from prompting.models import Problem

    metadata = dict(record)

    tags = set()
    for value in record.get("qwen_categories") or []:
        if value:
            tags.add(str(value))

    if record.get("primary_category"):
        tags.add(str(record.get("primary_category")))

    for value in record.get("derived_rules") or []:
        if value:
            tags.add(str(value))

    return Problem(
        id=int(record["id"]),
        question=str(record["question"]),
        options=record.get("options"),
        answer=record.get("answer"),
        tags=tags,
        metadata=metadata,
    )


def final_strategy_for_record(record):
    category = record.get("primary_category") or "general_math"
    is_mcq = bool(record.get("options"))

    if is_mcq and category in {"applied_word_problem", "geometry_trig"}:
        return "baseline3"

    return "baseline3_adaptive_rules"


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        return "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        ) + "\nassistant:"


def build_prompt_rows(records, tokenizer):
    from prompting.prompt_chain import build_prompt_chain

    chains = {}
    rows = []

    for idx, record in enumerate(tqdm(records, desc="Building prompts")):
        strategy_name = final_strategy_for_record(record)

        if strategy_name not in chains:
            chains[strategy_name] = build_prompt_chain(strategy_name=strategy_name)

        problem = make_problem(record)
        spec = chains[strategy_name].build_spec(problem)
        messages = spec.to_messages()
        prompt_text = apply_chat_template(tokenizer, messages)

        rows.append({
            "idx": idx,
            "id": record.get("id"),
            "record": record,
            "strategy_name": strategy_name,
            "template_name": spec.name,
            "messages": messages,
            "prompt_text": prompt_text,
        })

    return rows


def generate_prompt_texts_batched(
    model_bundle,
    prompt_texts,
    gen_config,
    batch_size,
    show_progress,
):
    from baseline.generation import generate_prompt_texts

    generated = generate_prompt_texts(
        model_bundle=model_bundle,
        prompt_texts=prompt_texts,
        generation_config=gen_config,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    return generated["responses"]


def fallback_parse_output(text, record):
    import re

    boxed_matches = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    boxed = boxed_matches[-1].strip() if boxed_matches else ""

    is_mcq = bool(record.get("options"))
    expected = 1 if is_mcq else str(record.get("question", "")).count("[ANS]")

    if boxed:
        actual = 1 if is_mcq else len([part for part in boxed.split(",") if part.strip()])
    else:
        actual = 0

    valid = bool(boxed)
    if is_mcq:
        valid = bool(re.fullmatch(r"[A-Z]", boxed.upper()))

    if boxed.strip() in {"", ".", "...", "?", "unknown", "None"}:
        valid = False

    return {
        "schema_valid": valid,
        "extractable": bool(boxed),
        "boxed_answer": boxed,
        "extracted_final_answer": boxed,
        "expected_answer_count": expected,
        "actual_answer_count": actual,
        "schema_errors": [] if valid else ["fallback_schema_invalid"],
    }


def parse_output(text, record):
    try:
        from baseline.output_parsing import parse_model_output

        try:
            return parse_model_output(text, record=record, sanitize=True)
        except TypeError:
            return parse_model_output(text, record)
    except Exception:
        return fallback_parse_output(text, record)


def build_retry_prompt_text(tokenizer, record, previous_output, parsed):
    options = record.get("options") or []
    option_lines = []

    for idx, opt in enumerate(options):
        option_lines.append(f"{chr(ord('A') + idx)}. {opt}")

    option_block = ""
    if option_lines:
        option_block = "\n\nAnswer choices:\n" + "\n".join(option_lines)

    error_text = ", ".join(parsed.get("schema_errors") or [])

    if options:
        format_text = "The final answer must be exactly one answer-choice letter inside \\boxed{}, e.g. \\boxed{C}."
    else:
        expected = str(record.get("question", "")).count("[ANS]")
        format_text = (
            f"The final answer must contain exactly {expected} answer(s), "
            "comma-separated inside one \\boxed{}."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You repair math-model outputs for the Kaggle submission. "
                "Solve concisely if needed, but prioritize producing a valid final boxed answer. "
                "Never output placeholders such as ..., ?, unknown, None, or an empty box."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{record.get('question')}"
                f"{option_block}\n\n"
                f"Previous model output:\n{previous_output}\n\n"
                f"Detected formatting/schema issue: {error_text}\n\n"
                f"{format_text}\n\n"
                "Return a concise response ending with:\n"
                "Final Answer: \\boxed{...}"
            ),
        },
    ]

    return apply_chat_template(tokenizer, messages)


def choose_response(record, raw_response, retry_response=None):
    raw_parsed = parse_output(raw_response, record)

    if raw_parsed.get("schema_valid"):
        return raw_response, raw_parsed, False

    if retry_response is not None:
        retry_parsed = parse_output(retry_response, record)
        if retry_parsed.get("schema_valid"):
            return retry_response, retry_parsed, True

    boxed = raw_parsed.get("boxed_answer") or raw_parsed.get("extracted_final_answer") or ""
    if boxed and boxed.strip() not in {"", ".", "...", "?", "unknown", "None"}:
        synthesized = f"Reasoning:\nAnswer extracted from completed final answer.\n\nFinal Answer: \\boxed{{{boxed}}}"
        synth_parsed = parse_output(synthesized, record)
        if synth_parsed.get("schema_valid"):
            return synthesized, synth_parsed, False

    return raw_response, raw_parsed, False


def run_inference(
    private_data_path="data/private.jsonl",
    output_csv_path="results/final_submission/submission.csv",
    output_jsonl_path="results/final_submission/private_results.jsonl",
    tag_cache_path="results/final_submission/private_category_tags.jsonl",
    tag_one_hot_csv_path="results/final_submission/private_category_tags_one_hot.csv",
    rule_jsonl_path="results/final_submission/private_with_rules.jsonl",
    model_id="Qwen/Qwen3-4B-Thinking-2507",
    backend="vllm",
    cache_dir=None,
    gpu_id="0",
    batch_size=32,
    retry_batch_size=32,
    max_input_tokens=32768,
    max_model_len=32768,
    max_new_tokens=32768,
    retry_max_new_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    gpu_memory_utilization=0.85,
    max_num_seqs=256,
    max_num_batched_tokens=32768,
    show_progress=True,
    limit=None,
):
    register_final_components()

    output_csv_path = Path(output_csv_path)
    output_jsonl_path = Path(output_jsonl_path)
    tag_cache_path = Path(tag_cache_path)
    tag_one_hot_csv_path = Path(tag_one_hot_csv_path)
    rule_jsonl_path = Path(rule_jsonl_path)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading private dataset...")
    private_set = load_private_problem_set(private_data_path)

    if limit is not None:
        private_set = build_problem_set(
            f"{private_set.name}_head{limit}",
            list(private_set.records[:limit]),
        )

    print(private_set.summary())

    print("Loading model...")
    model_bundle = build_model_bundle(
        model_id=model_id,
        backend=backend,
        cache_dir=cache_dir,
        gpu_id=gpu_id,
        max_input_tokens=max_input_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    print("Tagging private set or loading cached tags...")
    tagged_private_set, tag_rows = tag_private_problem_set(
        problem_set=private_set,
        model_bundle=model_bundle,
        tag_cache_path=tag_cache_path,
        one_hot_csv_path=tag_one_hot_csv_path,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    print("Annotating category rules...")
    final_problem_set = annotate_rules(
        tagged_private_set,
        output_jsonl_path=rule_jsonl_path,
    )

    records = list(final_problem_set.records)

    main_gen = generation_config(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        do_sample=True,
    )

    retry_gen = generation_config(
        max_new_tokens=retry_max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        do_sample=False,
    )

    print("Building final prompts...")
    prompt_rows = build_prompt_rows(records, model_bundle.tokenizer)
    prompt_texts = [row["prompt_text"] for row in prompt_rows]

    print("Running main inference...")
    raw_responses = generate_prompt_texts_batched(
        model_bundle=model_bundle,
        prompt_texts=prompt_texts,
        gen_config=main_gen,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    preliminary = []
    retry_prompt_texts = []
    retry_indices = []

    for idx, (row, raw_response) in enumerate(zip(prompt_rows, raw_responses)):
        parsed = parse_output(raw_response, row["record"])
        preliminary.append({
            "row": row,
            "raw_response": raw_response,
            "raw_parsed": parsed,
        })

        if not parsed.get("schema_valid"):
            retry_prompt_texts.append(
                build_retry_prompt_text(
                    tokenizer=model_bundle.tokenizer,
                    record=row["record"],
                    previous_output=raw_response,
                    parsed=parsed,
                )
            )
            retry_indices.append(idx)

    retry_responses_by_idx = {}

    if retry_prompt_texts:
        print(f"Running retry/repair pass for {len(retry_prompt_texts)} rows...")
        retry_responses = generate_prompt_texts_batched(
            model_bundle=model_bundle,
            prompt_texts=retry_prompt_texts,
            gen_config=retry_gen,
            batch_size=retry_batch_size,
            show_progress=show_progress,
        )

        for idx, retry_response in zip(retry_indices, retry_responses):
            retry_responses_by_idx[idx] = retry_response

    print("Post-processing and writing outputs...")
    result_rows = []

    for idx, item in enumerate(preliminary):
        row = item["row"]
        record = row["record"]
        raw_response = item["raw_response"]
        retry_response = retry_responses_by_idx.get(idx)

        final_response, parsed, retry_used = choose_response(
            record=record,
            raw_response=raw_response,
            retry_response=retry_response,
        )

        result_rows.append({
            "id": record.get("id"),
            "primary_category": record.get("primary_category"),
            "strategy_name": row.get("strategy_name"),
            "template_name": row.get("template_name"),
            "response": final_response,
            "raw_response": raw_response,
            "retry_response": retry_response,
            "retry_used": retry_used,
            "schema_valid": parsed.get("schema_valid"),
            "extractable": parsed.get("extractable"),
            "boxed_answer": parsed.get("boxed_answer"),
            "schema_errors": parsed.get("schema_errors"),
        })

    save_jsonl_records(result_rows, output_jsonl_path)
    save_submission_csv(result_rows, output_csv_path)

    n_schema_valid = sum(1 for row in result_rows if row.get("schema_valid"))
    n_retry_used = sum(1 for row in result_rows if row.get("retry_used"))

    report = {
        "n": len(result_rows),
        "schema_valid": n_schema_valid,
        "schema_valid_rate": n_schema_valid / len(result_rows) if result_rows else None,
        "retry_used": n_retry_used,
        "retry_used_rate": n_retry_used / len(result_rows) if result_rows else None,
        "output_csv_path": str(output_csv_path),
        "output_jsonl_path": str(output_jsonl_path),
        "tag_cache_path": str(tag_cache_path),
        "rule_jsonl_path": str(rule_jsonl_path),
    }

    report_path = output_csv_path.parent / "run_inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run final CSE 151B Kaggle private inference.")
    parser.add_argument("--private-data-path", default="data/private.jsonl")
    parser.add_argument("--output-csv", default="results/final_submission/submission.csv")
    parser.add_argument("--output-jsonl", default="results/final_submission/private_results.jsonl")
    parser.add_argument("--tag-cache-path", default="results/final_submission/private_category_tags.jsonl")
    parser.add_argument("--tag-one-hot-csv-path", default="results/final_submission/private_category_tags_one_hot.csv")
    parser.add_argument("--rule-jsonl-path", default="results/final_submission/private_with_rules.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retry-batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--retry-max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    run_inference(
        private_data_path=args.private_data_path,
        output_csv_path=args.output_csv,
        output_jsonl_path=args.output_jsonl,
        tag_cache_path=args.tag_cache_path,
        tag_one_hot_csv_path=args.tag_one_hot_csv_path,
        rule_jsonl_path=args.rule_jsonl_path,
        model_id=args.model_id,
        backend=args.backend,
        cache_dir=args.cache_dir,
        gpu_id=args.gpu_id,
        batch_size=args.batch_size,
        retry_batch_size=args.retry_batch_size,
        max_input_tokens=args.max_input_tokens,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        retry_max_new_tokens=args.retry_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        show_progress=not args.no_progress,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()