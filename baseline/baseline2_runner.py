import time

from prompting.prompt_chain import build_prompt_chain

from .baseline2_prompts import BASELINE2_RESPONSE_PREFILL, build_baseline2_retry_messages
from .datasets import save_jsonl
from .experiments import append_comparison_row, build_comparison_row
from .generation import GenerationConfig, generate_prompt_texts
from .output_parsing import parse_model_output
from .prompt_sets import build_prompt_texts
from .runner import RunResult, save_submission_csv, write_report, maybe_limit_problem_set
from .scoring import load_judger, score_records, summarize_results


def _build_retry_prompt_text(tokenizer, record, previous_output=None, extracted_answer=None, retry_mode="short_resolve"):
    prompt_text = tokenizer.apply_chat_template(
        build_baseline2_retry_messages(
            record=record,
            previous_output=previous_output,
            extracted_answer=extracted_answer,
            retry_mode=retry_mode,
        ),
        tokenize=False,
        add_generation_prompt=True,
    )

    role_start = "<|im_start|>" + "assist" + "ant\n"
    thinking_prefill = f"{role_start}<think>\n"
    answer_prefill = f"{role_start}{BASELINE2_RESPONSE_PREFILL}"

    if prompt_text.endswith(thinking_prefill):
        prompt_text = prompt_text[:-len(thinking_prefill)] + answer_prefill

    return prompt_text


def _with_response_prefill(output):
    text = output or ""

    if text.startswith(BASELINE2_RESPONSE_PREFILL):
        return text

    return BASELINE2_RESPONSE_PREFILL + text.lstrip()


def _default_retry_generation_config(gen):
    return GenerationConfig(
        max_new_tokens=min(256, getattr(gen, "max_new_tokens", 256)),
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        do_sample=False,
    )


def _schema_score(parsed_row):
    score = 0

    if parsed_row.get("extractable"):
        score += 1

    if parsed_row.get("well_formed"):
        score += 1

    if parsed_row.get("strict_well_formed"):
        score += 1

    if parsed_row.get("schema_valid"):
        score += 3

    if parsed_row.get("sanitized"):
        score += 0.25

    score -= 0.10 * len(parsed_row.get("schema_errors") or [])

    return score


def _should_retry(parsed_row):
    return not parsed_row.get("schema_valid")


def _retry_mode_for(parsed_row):
    if parsed_row.get("extractable"):
        return "format_repair"

    return "short_resolve"


def _choose_better(current_parsed, candidate_parsed):
    return _schema_score(candidate_parsed) > _schema_score(current_parsed)


def _formatting_summary(debug_rows):
    total = len(debug_rows)

    retry_rows = [row for row in debug_rows if row["retry_needed"]]
    retry_used_rows = [row for row in debug_rows if row["retry_used"]]
    sanitized_rows = [row for row in debug_rows if row["sanitized"]]

    initial_schema_valid = [row for row in debug_rows if row["initial_schema_valid"]]
    final_schema_valid = [row for row in debug_rows if row["schema_valid"]]

    initial_extractable = [row for row in debug_rows if row["initial_extractable"]]
    final_extractable = [row for row in debug_rows if row["extractable"]]

    initial_malformed = [row for row in debug_rows if not row["initial_strict_well_formed"]]
    final_malformed = [row for row in debug_rows if not row["strict_well_formed"]]

    retry_schema_success = [
        row for row in retry_rows
        if row["schema_valid"] and not row["post_sanitize_schema_valid"]
    ]

    retry_extract_success = [
        row for row in retry_rows
        if row["extractable"] and not row["post_sanitize_extractable"]
    ]

    error_counts = {}
    for row in debug_rows:
        for error in row.get("schema_errors") or []:
            error_counts[error] = error_counts.get(error, 0) + 1

    initial_error_counts = {}
    for row in debug_rows:
        for error in row.get("initial_schema_errors") or []:
            initial_error_counts[error] = initial_error_counts.get(error, 0) + 1

    post_sanitize_error_counts = {}
    for row in debug_rows:
        for error in row.get("post_sanitize_schema_errors") or []:
            post_sanitize_error_counts[error] = post_sanitize_error_counts.get(error, 0) + 1

    return {
        "n_outputs": total,

        "initial_schema_valid_count": len(initial_schema_valid),
        "initial_schema_valid_rate": len(initial_schema_valid) / total if total else None,

        "schema_valid_count": len(final_schema_valid),
        "schema_valid_rate": len(final_schema_valid) / total if total else None,

        "initial_formatting_failure_count": len(initial_malformed),
        "initial_formatting_failure_rate": len(initial_malformed) / total if total else None,

        "formatting_failure_count": len(final_malformed),
        "formatting_failure_rate": len(final_malformed) / total if total else None,

        "initial_extractable_count": len(initial_extractable),
        "initial_extractable_rate": len(initial_extractable) / total if total else None,

        "extractable_count": len(final_extractable),
        "extractable_rate": len(final_extractable) / total if total else None,

        "unextractable_count": total - len(final_extractable),
        "unextractable_rate": (total - len(final_extractable)) / total if total else None,

        "sanitized_count": len(sanitized_rows),
        "sanitized_rate": len(sanitized_rows) / total if total else None,

        "retry_count": len(retry_rows),
        "retry_rate": len(retry_rows) / total if total else None,

        "retry_used_count": len(retry_used_rows),
        "retry_used_rate": len(retry_used_rows) / total if total else None,

        "retry_schema_success_count": len(retry_schema_success),
        "retry_extract_success_count": len(retry_extract_success),

        "initial_schema_error_counts": dict(sorted(initial_error_counts.items())),
        "post_sanitize_schema_error_counts": dict(sorted(post_sanitize_error_counts.items())),
        "schema_error_counts": dict(sorted(error_counts.items())),
    }


def _category_summary(scored_rows):
    rows_by_category = {}

    for row in scored_rows:
        category = row.get("category") or "uncategorized"
        rows_by_category.setdefault(category, []).append(row)

    summary = {}

    for category, rows in sorted(rows_by_category.items()):
        scored = [row for row in rows if row.get("correct") is not None]
        extractable = [row for row in rows if row.get("extractable")]
        schema_valid = [row for row in rows if row.get("schema_valid")]
        malformed = [row for row in rows if not row.get("strict_well_formed")]

        summary[category] = {
            "n": len(rows),
            "n_scored": len(scored),
            "accuracy": (
                sum(bool(row["correct"]) for row in scored) / len(scored)
                if scored else None
            ),
            "extractable_rate": len(extractable) / len(rows) if rows else None,
            "schema_valid_rate": len(schema_valid) / len(rows) if rows else None,
            "formatting_failure_rate": len(malformed) / len(rows) if rows else None,
        }

    return summary


def _category_counts(debug_rows):
    counts = {}

    for row in debug_rows:
        category = row.get("category") or "uncategorized"
        counts[category] = counts.get(category, 0) + 1

    return dict(sorted(counts.items()))


def _schema_accuracy_summary(scored_rows):
    scored = [row for row in scored_rows if row.get("correct") is not None]

    valid_rows = [row for row in scored if row.get("schema_valid")]
    invalid_rows = [row for row in scored if not row.get("schema_valid")]
    retry_rows = [row for row in scored if row.get("retry_used")]
    sanitized_rows = [row for row in scored if row.get("sanitized")]

    def acc(rows):
        if not rows:
            return None
        return sum(bool(row["correct"]) for row in rows) / len(rows)

    return {
        "schema_valid_n": len(valid_rows),
        "schema_valid_accuracy": acc(valid_rows),

        "schema_invalid_n": len(invalid_rows),
        "schema_invalid_accuracy": acc(invalid_rows),

        "retry_used_n": len(retry_rows),
        "retry_used_accuracy": acc(retry_rows),

        "sanitized_n": len(sanitized_rows),
        "sanitized_accuracy": acc(sanitized_rows),
    }


def _build_debug_row(
    record,
    prompt_row,
    initial_output,
    initial_raw_parsed,
    post_sanitize_parsed,
    final_parsed,
    retry_output,
    retry_parsed,
    retry_needed,
    retry_used,
    retry_mode,
):
    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "is_mcq": bool(record.get("options")),
        "gold": record.get("answer"),

        "category": prompt_row["metadata"].get("category"),
        "qwen_categories": prompt_row["metadata"].get("qwen_categories"),
        "route_name": prompt_row["metadata"].get("route_name"),
        "template_name": prompt_row["spec"].name,
        "prompt_metadata": prompt_row["metadata"],

        "initial_raw_output": initial_output,
        "raw_output": final_parsed["response_for_submission"],
        "response": final_parsed["response_for_submission"],
        "response_for_submission": final_parsed["response_for_submission"],

        "extracted_final_answer": final_parsed["extracted_answer"],
        "boxed_answer": final_parsed["boxed_answer"],
        "expected_answer_count": final_parsed["expected_answer_count"],
        "actual_answer_count": final_parsed["actual_answer_count"],

        "retry_needed": retry_needed,
        "retry_used": retry_used,
        "retry_mode": retry_mode,
        "retry_raw_output": retry_output,
        "retry_schema_errors": retry_parsed["schema_errors"] if retry_parsed else None,

        "sanitized": final_parsed["sanitized"],

        "initial_extractable": initial_raw_parsed["extractable"],
        "post_sanitize_extractable": post_sanitize_parsed["extractable"],
        "extractable": final_parsed["extractable"],

        "initial_strict_well_formed": initial_raw_parsed["strict_well_formed"],
        "post_sanitize_strict_well_formed": post_sanitize_parsed["strict_well_formed"],
        "strict_well_formed": final_parsed["strict_well_formed"],

        "initial_well_formed": initial_raw_parsed["well_formed"],
        "post_sanitize_well_formed": post_sanitize_parsed["well_formed"],
        "well_formed": final_parsed["well_formed"],

        "initial_schema_valid": initial_raw_parsed["schema_valid"],
        "post_sanitize_schema_valid": post_sanitize_parsed["schema_valid"],
        "schema_valid": final_parsed["schema_valid"],

        "initial_schema_errors": initial_raw_parsed["schema_errors"],
        "post_sanitize_schema_errors": post_sanitize_parsed["schema_errors"],
        "schema_errors": final_parsed["schema_errors"],
    }


def run_baseline2_problem_set(
    problem_set,
    model_bundle,
    generation_config=None,
    retry_generation_config=None,
    batch_size=1,
    limit=None,
    score=True,
    strategy_name="baseline2",
    report_label="baseline2_prompt_format",
    judger_dir=".",
    output_jsonl_path=None,
    debug_jsonl_path=None,
    submission_csv_path=None,
    report_json_path=None,
    comparison_csv_path=None,
    experiment_name=None,
    split_name=None,
    show_progress=True,
):
    problem_set = maybe_limit_problem_set(problem_set, limit)
    timings = {}

    t0 = time.perf_counter()
    prompt_chain = build_prompt_chain(strategy_name=strategy_name)
    prompt_rows = build_prompt_texts(problem_set, model_bundle.tokenizer, prompt_chain=prompt_chain)
    prompt_texts = [row["prompt_text"] for row in prompt_rows]
    timings["prompt_build_sec"] = time.perf_counter() - t0

    gen = generation_config or GenerationConfig()
    retry_gen = retry_generation_config or _default_retry_generation_config(gen)

    t0 = time.perf_counter()
    generations = generate_prompt_texts(
        model_bundle=model_bundle,
        prompt_texts=prompt_texts,
        generation_config=gen,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    timings["generation_sec"] = time.perf_counter() - t0

    initial_outputs = [_with_response_prefill(output) for output in generations["responses"]]

    initial_raw_parsed = [
        parse_model_output(output, record=record, sanitize=False)
        for output, record in zip(initial_outputs, problem_set.records)
    ]

    post_sanitize_parsed = [
        parse_model_output(output, record=record, sanitize=True)
        for output, record in zip(initial_outputs, problem_set.records)
    ]

    retry_needed_indices = [
        idx for idx, parsed_row in enumerate(post_sanitize_parsed)
        if _should_retry(parsed_row)
    ]

    retry_outputs = {}
    retry_parsed = {}
    retry_modes = {}

    if retry_needed_indices:
        retry_prompt_texts = []

        for idx in retry_needed_indices:
            record = problem_set.records[idx]
            parsed_row = post_sanitize_parsed[idx]
            retry_mode = _retry_mode_for(parsed_row)
            retry_modes[idx] = retry_mode

            retry_prompt_texts.append(
                _build_retry_prompt_text(
                    tokenizer=model_bundle.tokenizer,
                    record=record,
                    previous_output=post_sanitize_parsed[idx]["response_for_submission"],
                    extracted_answer=post_sanitize_parsed[idx]["extracted_answer"],
                    retry_mode=retry_mode,
                )
            )

        t0 = time.perf_counter()
        retry_generations = generate_prompt_texts(
            model_bundle=model_bundle,
            prompt_texts=retry_prompt_texts,
            generation_config=retry_gen,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        timings["retry_generation_sec"] = time.perf_counter() - t0

        for idx, retry_output in zip(retry_needed_indices, retry_generations["responses"]):
            retry_output = _with_response_prefill(retry_output)
            retry_outputs[idx] = retry_output
            retry_parsed[idx] = parse_model_output(
                retry_output,
                record=problem_set.records[idx],
                sanitize=True,
            )
    else:
        timings["retry_generation_sec"] = 0.0

    final_parsed = []
    retry_used_indices = set()

    for idx, parsed_row in enumerate(post_sanitize_parsed):
        chosen_parsed = parsed_row

        if idx in retry_parsed and _choose_better(chosen_parsed, retry_parsed[idx]):
            chosen_parsed = retry_parsed[idx]
            retry_used_indices.add(idx)

        final_parsed.append(chosen_parsed)

    final_outputs = [row["response_for_submission"] for row in final_parsed]

    debug_rows = []

    for idx, record in enumerate(problem_set.records):
        debug_rows.append(
            _build_debug_row(
                record=record,
                prompt_row=prompt_rows[idx],
                initial_output=initial_outputs[idx],
                initial_raw_parsed=initial_raw_parsed[idx],
                post_sanitize_parsed=post_sanitize_parsed[idx],
                final_parsed=final_parsed[idx],
                retry_output=retry_outputs.get(idx),
                retry_parsed=retry_parsed.get(idx),
                retry_needed=idx in retry_needed_indices,
                retry_used=idx in retry_used_indices,
                retry_mode=retry_modes.get(idx),
            )
        )

    t0 = time.perf_counter()
    score_available = score and any(
        "answer" in record and record.get("answer") is not None
        for record in problem_set.records
    )
    judger = load_judger(judger_dir) if score_available else None

    if score_available:
        scored_rows = score_records(problem_set.records, final_outputs, judger=judger)
    else:
        scored_rows = []

        for record, response in zip(problem_set.records, final_outputs):
            scored_rows.append({
                "id": record.get("id"),
                "is_mcq": bool(record.get("options")),
                "gold": record.get("answer"),
                "response": response,
                "correct": None,
            })

    for scored_row, debug_row in zip(scored_rows, debug_rows):
        scored_row.update({
            "category": debug_row["category"],
            "qwen_categories": debug_row["qwen_categories"],
            "route_name": debug_row["route_name"],
            "template_name": debug_row["template_name"],
            "prompt_metadata": debug_row["prompt_metadata"],

            "raw_output": debug_row["raw_output"],
            "initial_raw_output": debug_row["initial_raw_output"],
            "response": debug_row["response"],
            "response_for_submission": debug_row["response_for_submission"],

            "extracted_final_answer": debug_row["extracted_final_answer"],
            "boxed_answer": debug_row["boxed_answer"],
            "expected_answer_count": debug_row["expected_answer_count"],
            "actual_answer_count": debug_row["actual_answer_count"],

            "retry_needed": debug_row["retry_needed"],
            "retry_used": debug_row["retry_used"],
            "retry_mode": debug_row["retry_mode"],
            "retry_raw_output": debug_row["retry_raw_output"],
            "retry_schema_errors": debug_row["retry_schema_errors"],

            "sanitized": debug_row["sanitized"],

            "initial_extractable": debug_row["initial_extractable"],
            "post_sanitize_extractable": debug_row["post_sanitize_extractable"],
            "extractable": debug_row["extractable"],

            "initial_strict_well_formed": debug_row["initial_strict_well_formed"],
            "post_sanitize_strict_well_formed": debug_row["post_sanitize_strict_well_formed"],
            "strict_well_formed": debug_row["strict_well_formed"],

            "initial_well_formed": debug_row["initial_well_formed"],
            "post_sanitize_well_formed": debug_row["post_sanitize_well_formed"],
            "well_formed": debug_row["well_formed"],

            "initial_schema_valid": debug_row["initial_schema_valid"],
            "post_sanitize_schema_valid": debug_row["post_sanitize_schema_valid"],
            "schema_valid": debug_row["schema_valid"],

            "initial_schema_errors": debug_row["initial_schema_errors"],
            "post_sanitize_schema_errors": debug_row["post_sanitize_schema_errors"],
            "schema_errors": debug_row["schema_errors"],
        })

    summary = summarize_results(scored_rows)
    formatting_summary = _formatting_summary(debug_rows)
    schema_accuracy_summary = _schema_accuracy_summary(scored_rows)
    category_counts = _category_counts(debug_rows)
    category_summary = _category_summary(scored_rows)
    timings["scoring_sec"] = time.perf_counter() - t0

    if output_jsonl_path:
        save_jsonl(scored_rows, output_jsonl_path)

    if debug_jsonl_path:
        save_jsonl(debug_rows, debug_jsonl_path)

    if submission_csv_path:
        save_submission_csv(scored_rows, submission_csv_path)

    report = {
        "baseline": report_label,
        "problem_set": problem_set.summary(),
        "backend": model_bundle.backend,
        "model_id": getattr(getattr(model_bundle, "config", None), "model_id", None),
        "batch_size": batch_size,
        "generation_config": vars(gen),
        "retry_generation_config": vars(retry_gen),
        "score_available": score_available,
        "summary": summary,
        "formatting": formatting_summary,
        "schema_accuracy": schema_accuracy_summary,
        "category_counts": category_counts,
        "category_summary": category_summary,
        "generation": generations,
        "timings": timings,
        "output_jsonl_path": str(output_jsonl_path) if output_jsonl_path else None,
        "debug_jsonl_path": str(debug_jsonl_path) if debug_jsonl_path else None,
        "submission_csv_path": str(submission_csv_path) if submission_csv_path else None,
        "report_json_path": str(report_json_path) if report_json_path else None,
    }

    if report_json_path:
        write_report(report, report_json_path)

    if comparison_csv_path:
        row = build_comparison_row(
            report=report,
            experiment_name=experiment_name,
            strategy_name=strategy_name,
            split=split_name,
            report_json_path=report_json_path,
        )
        append_comparison_row(comparison_csv_path, row)

    return RunResult(
        problem_set=problem_set,
        prompt_rows=prompt_rows,
        generations=generations,
        scored_rows=scored_rows,
        summary=summary,
        timings=timings,
        report=report,
    )


def run_baseline3_problem_set(*args, **kwargs):
    kwargs.setdefault("strategy_name", "baseline3")
    kwargs.setdefault("report_label", "baseline3_qwen_category_prompts")
    return run_baseline2_problem_set(*args, **kwargs)
