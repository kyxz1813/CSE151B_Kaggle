import csv
import json
import time
from pathlib import Path

from prompting.prompt_chain import build_prompt_chain

from .datasets import save_jsonl
from .prompt_sets import build_prompt_texts
from .generation import generate_prompt_texts, GenerationConfig
from .scoring import load_judger, score_records, summarize_results


class RunResult:
    def __init__(self, problem_set, prompt_rows, generations, scored_rows, summary, timings, report):
        self.problem_set = problem_set
        self.prompt_rows = prompt_rows
        self.generations = generations
        self.scored_rows = scored_rows
        self.results = scored_rows
        self.rows = scored_rows
        self.summary = summary
        self.timings = timings
        self.report = report

    def to_dict(self):
        return self.report


def save_submission_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()

        for row in results:
            writer.writerow({
                "id": row["id"],
                "response": row["response"],
            })


def write_report(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def maybe_limit_problem_set(problem_set, limit):
    if limit is None:
        return problem_set
    return problem_set.head(limit)


def run_problem_set(
    problem_set,
    model_bundle,
    prompt_chain=None,
    generation_config=None,
    batch_size=1,
    limit=None,
    score=True,
    judger_dir=".",
    output_jsonl_path=None,
    submission_csv_path=None,
    report_json_path=None,
    show_progress=True,
):
    problem_set = maybe_limit_problem_set(problem_set, limit)
    timings = {}

    t0 = time.perf_counter()
    chain = prompt_chain or build_prompt_chain()
    prompt_rows = build_prompt_texts(problem_set, model_bundle.tokenizer, prompt_chain=chain)
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

    responses = generations["responses"]

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

    summary = summarize_results(scored_rows)
    timings["scoring_sec"] = time.perf_counter() - t0

    if output_jsonl_path:
        save_jsonl(scored_rows, output_jsonl_path)

    if submission_csv_path:
        save_submission_csv(scored_rows, submission_csv_path)

    report = {
        "problem_set": problem_set.summary(),
        "backend": model_bundle.backend,
        "batch_size": batch_size,
        "generation_config": vars(gen),
        "score_available": score_available,
        "summary": summary,
        "generation": generations,
        "timings": timings,
        "output_jsonl_path": str(output_jsonl_path) if output_jsonl_path else None,
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


def benchmark_batch_sizes(
    problem_set,
    model_bundle,
    batch_sizes,
    prompt_chain=None,
    generation_config=None,
    sample_size=16,
    score=False,
    judger_dir=".",
    show_progress=False,
):
    rows = []
    sample_set = problem_set.head(sample_size)

    for batch_size in batch_sizes:
        t0 = time.perf_counter()
        try:
            result = run_problem_set(
                problem_set=sample_set,
                model_bundle=model_bundle,
                prompt_chain=prompt_chain,
                generation_config=generation_config,
                batch_size=batch_size,
                score=score,
                judger_dir=judger_dir,
                show_progress=show_progress,
            )
            elapsed = time.perf_counter() - t0
            ok = True
            error = None
            sec_per_problem = elapsed / len(sample_set) if len(sample_set) else None
            generation_sec_per_problem = result.generations.get("sec_per_problem")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            ok = False
            error = repr(exc)
            sec_per_problem = None
            generation_sec_per_problem = None

        rows.append({
            "batch_size": batch_size,
            "ok": ok,
            "elapsed_sec": elapsed,
            "sec_per_problem": sec_per_problem,
            "generation_sec_per_problem": generation_sec_per_problem,
            "error": error,
        })

    valid = [r for r in rows if r["ok"] and r["sec_per_problem"] is not None]
    best = min(valid, key=lambda r: r["sec_per_problem"]) if valid else None

    return {
        "sample_size": len(sample_set),
        "rows": rows,
        "best": best,
        "best_batch_size": best["batch_size"] if best else None,
    }
