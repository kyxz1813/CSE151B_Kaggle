import csv
import time
from pathlib import Path

from prompting.prompt_chain import build_prompt_chain

from .datasets import save_jsonl
from .prompt_sets import build_prompt_texts
from .generation import generate_prompt_texts, GenerationConfig
from .scoring import load_judger, score_records, summarize_results


class RunResult:
    def __init__(self, problem_set, prompt_rows, generations, scored_rows, summary, timings):
        self.problem_set = problem_set
        self.prompt_rows = prompt_rows
        self.generations = generations
        self.scored_rows = scored_rows
        self.summary = summary
        self.timings = timings


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


def run_problem_set(
    problem_set,
    model_bundle,
    prompt_chain=None,
    generation_config=None,
    batch_size=1,
    score=True,
    judger_dir=".",
    output_jsonl_path=None,
    submission_csv_path=None,
):
    timings = {}

    t0 = time.perf_counter()
    chain = prompt_chain or build_prompt_chain()
    prompt_rows = build_prompt_texts(problem_set, model_bundle.tokenizer, prompt_chain=chain)
    timings["prompt_build_sec"] = time.perf_counter() - t0

    prompt_texts = [row["prompt_text"] for row in prompt_rows]

    gen = generation_config or GenerationConfig()

    t0 = time.perf_counter()
    generations = generate_prompt_texts(
        model_bundle=model_bundle,
        prompt_texts=prompt_texts,
        generation_config=gen,
        batch_size=batch_size,
        show_progress=True,
    )
    timings["generation_sec"] = time.perf_counter() - t0

    responses = generations["responses"]

    t0 = time.perf_counter()
    judger = load_judger(judger_dir) if score else None
    scored_rows = score_records(problem_set.records, responses, judger=judger) if score else []

    if not score:
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

    return RunResult(
        problem_set=problem_set,
        prompt_rows=prompt_rows,
        generations=generations,
        scored_rows=scored_rows,
        summary=summary,
        timings=timings,
    )