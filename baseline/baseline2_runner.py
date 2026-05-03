import time

from prompting.prompt_chain import build_prompt_chain

from .baseline2_prompts import BASELINE2_ASSISTANT_PREFILL, build_baseline2_retry_messages
from .datasets import save_jsonl
from .generation import GenerationConfig, generate_prompt_texts
from .output_parsing import parse_model_output
from .prompt_sets import build_prompt_texts
from .runner import RunResult, save_submission_csv, write_report, maybe_limit_problem_set
from .scoring import load_judger, score_records, summarize_results


def _build_retry_prompt_text(tokenizer, record):
    prompt_text = tokenizer.apply_chat_template(
        build_baseline2_retry_messages(record),
        tokenize=False,
        add_generation_prompt=True,
    )

    role_start = "<|im_start|>" + "assist" + "ant\n"
    thinking_prefill = f"{role_start}<think>\n"
    answer_prefill = f"{role_start}{BASELINE2_ASSISTANT_PREFILL}"
    if prompt_text.endswith(thinking_prefill):
        prompt_text = prompt_text[:-len(thinking_prefill)] + answer_prefill

    return prompt_text


def _with_assistant_prefill(output):
    text = output or ""
    if text.startswith(BASELINE2_ASSISTANT_PREFILL):
        return text
    return BASELINE2_ASSISTANT_PREFILL + text.lstrip()


def _formatting_summary(debug_rows):
    total = len(debug_rows)
    retry_rows = [row for row in debug_rows if row["retry_needed"]]
    malformed = [row for row in debug_rows if not row["well_formed"]]
    retry_success = [
        row for row in retry_rows
        if row["well_formed"] and not row["initial_well_formed"]
    ]

    return {
        "n_outputs": total,
        "formatting_failure_count": len(malformed),
        "formatting_failure_rate": len(malformed) / total if total else None,
        "retry_count": len(retry_rows),
        "retry_success_count": len(retry_success),
    }


def run_baseline2_problem_set(
    problem_set,
    model_bundle,
    generation_config=None,
    batch_size=1,
    limit=None,
    score=True,
    judger_dir=".",
    output_jsonl_path=None,
    debug_jsonl_path=None,
    submission_csv_path=None,
    report_json_path=None,
    show_progress=True,
):
    problem_set = maybe_limit_problem_set(problem_set, limit)
    timings = {}

    t0 = time.perf_counter()
    prompt_chain = build_prompt_chain(strategy_name="baseline2")
    prompt_rows = build_prompt_texts(problem_set, model_bundle.tokenizer, prompt_chain=prompt_chain)
    prompt_texts = [row["prompt_text"] for row in prompt_rows]
    timings["prompt_build_sec"] = time.perf_counter() - t0

    gen = generation_config or GenerationConfig()

    t0 = time.perf_counter()
    generations = generate_prompt_texts(
        model_bundle=model_bundle,
        prompt_texts=prompt_texts,
        generation_config=gen,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    timings["generation_sec"] = time.perf_counter() - t0

    raw_outputs = [_with_assistant_prefill(output) for output in generations["responses"]]
    parsed = [parse_model_output(output) for output in raw_outputs]
    malformed_indices = [idx for idx, row in enumerate(parsed) if not row["well_formed"]]
    retry_outputs = {}

    if malformed_indices:
        retry_prompt_texts = [
            _build_retry_prompt_text(model_bundle.tokenizer, problem_set.records[idx])
            for idx in malformed_indices
        ]

        t0 = time.perf_counter()
        retry_generations = generate_prompt_texts(
            model_bundle=model_bundle,
            prompt_texts=retry_prompt_texts,
            generation_config=gen,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        timings["retry_generation_sec"] = time.perf_counter() - t0

        for idx, retry_output in zip(malformed_indices, retry_generations["responses"]):
            retry_output = _with_assistant_prefill(retry_output)
            retry_outputs[idx] = retry_output
            raw_outputs[idx] = retry_output
            parsed[idx] = parse_model_output(retry_output)
    else:
        timings["retry_generation_sec"] = 0.0

    responses = [row["extracted_answer"] for row in parsed]

    debug_rows = []
    for idx, (record, raw_output, parsed_row) in enumerate(zip(problem_set.records, raw_outputs, parsed)):
        initial_parsed = parse_model_output(_with_assistant_prefill(generations["responses"][idx]))
        debug_rows.append({
            "id": record.get("id"),
            "question": record.get("question"),
            "raw_output": raw_output,
            "extracted_final_answer": parsed_row["extracted_answer"],
            "retry_needed": idx in retry_outputs,
            "initial_well_formed": initial_parsed["well_formed"],
            "well_formed": parsed_row["well_formed"],
        })

    t0 = time.perf_counter()
    score_available = score and any("answer" in r and r.get("answer") is not None for r in problem_set.records)
    judger = load_judger(judger_dir) if score_available else None

    if score_available:
        scored_rows = score_records(problem_set.records, responses, judger=judger)
    else:
        scored_rows = []
        for record, response in zip(problem_set.records, responses):
            scored_rows.append({
                "id": record.get("id"),
                "is_mcq": bool(record.get("options")),
                "gold": record.get("answer"),
                "response": response,
                "correct": None,
            })

    for scored_row, debug_row in zip(scored_rows, debug_rows):
        scored_row.update({
            "raw_output": debug_row["raw_output"],
            "extracted_final_answer": debug_row["extracted_final_answer"],
            "retry_needed": debug_row["retry_needed"],
            "well_formed": debug_row["well_formed"],
        })

    summary = summarize_results(scored_rows)
    formatting_summary = _formatting_summary(debug_rows)
    timings["scoring_sec"] = time.perf_counter() - t0

    if output_jsonl_path:
        save_jsonl(scored_rows, output_jsonl_path)

    if debug_jsonl_path:
        save_jsonl(debug_rows, debug_jsonl_path)

    if submission_csv_path:
        save_submission_csv(scored_rows, submission_csv_path)

    report = {
        "baseline": "baseline2_prompt_format",
        "problem_set": problem_set.summary(),
        "backend": model_bundle.backend,
        "batch_size": batch_size,
        "generation_config": vars(gen),
        "score_available": score_available,
        "summary": summary,
        "formatting": formatting_summary,
        "generation": generations,
        "timings": timings,
        "output_jsonl_path": str(output_jsonl_path) if output_jsonl_path else None,
        "debug_jsonl_path": str(debug_jsonl_path) if debug_jsonl_path else None,
        "submission_csv_path": str(submission_csv_path) if submission_csv_path else None,
    }

    if report_json_path:
        write_report(report, report_json_path)

    return RunResult(
        problem_set=problem_set,
        prompt_rows=prompt_rows,
        generations=generations,
        scored_rows=scored_rows,
        summary=summary,
        timings=timings,
        report=report,
    )
