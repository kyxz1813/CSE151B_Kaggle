import argparse
import json
from pathlib import Path
from pprint import pprint

from baseline.baseline2_runner import run_baseline2_problem_set
from baseline.datasets import load_private_set, load_public_splits
from baseline.generation import GenerationConfig


class DryRunTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class DryRunModelBundle:
    backend = "dry_run"
    tokenizer = DryRunTokenizer()


def install_dry_run_generator():
    import baseline.baseline2_runner as runner
    import baseline.category_tagging as category_tagging

    call_state = {"n": 0}

    def dry_run_generate(model_bundle, prompt_texts, generation_config=None, batch_size=1, show_progress=True):
        call_state["n"] += 1
        responses = []

        for idx, _ in enumerate(prompt_texts):
            if call_state["n"] == 1 and idx == 0:
                responses.append(
                    "Reasoning:\nThis is a formatting smoke-test response.\n\n"
                    "Final Answer: \\boxed{0}\n\nExtra trailing text."
                )
            elif call_state["n"] == 1:
                responses.append("Malformed smoke-test response")
            else:
                responses.append("Reasoning:\nThis retry follows the required format.\n\nFinal Answer: \\boxed{A}")

        return {
            "responses": responses,
            "elapsed_sec": 0.0,
            "n": len(responses),
            "sec_per_problem": 0.0,
            "backend": model_bundle.backend,
            "batch_size": batch_size,
            "max_new_tokens": generation_config.max_new_tokens if generation_config else None,
        }

    runner.generate_prompt_texts = dry_run_generate

    def dry_run_category_generate(model_bundle, prompt_texts, generation_config=None, batch_size=1, show_progress=True):
        categories = [
            "statistics_probability",
            "calculus",
            "geometry_trig",
            "linear_algebra",
            "discrete_algorithm",
            "arithmetic_algebra",
            "applied_word_problem",
            "general_math",
        ]
        responses = []
        for idx, _ in enumerate(prompt_texts):
            category = categories[idx % len(categories)]
            responses.append(
                f'{{"categories":["{category}"],"primary_category":"{category}"}}'
            )
        return {
            "responses": responses,
            "elapsed_sec": 0.0,
            "n": len(responses),
            "sec_per_problem": 0.0,
            "backend": model_bundle.backend,
            "batch_size": batch_size,
            "max_new_tokens": generation_config.max_new_tokens if generation_config else None,
        }

    category_tagging.generate_prompt_texts = dry_run_category_generate


def parse_args(
    description="Run Baseline 2 prompt-formatting inference.",
    default_output_dir="results/baseline2_prompt_format",
    extra_args_fn=None,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--split", choices=["train", "val", "public", "private"], default="val")
    parser.add_argument("--public-data-path", default="data/public.jsonl")
    parser.add_argument("--private-data-path", default="data/private.jsonl")
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=414)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--comparison-csv", default="results/experiment_comparison.csv")
    if extra_args_fn is not None:
        extra_args_fn(parser)
    return parser.parse_args()


def load_problem_set(args):
    if args.split == "private":
        return load_private_set(args.private_data_path)

    splits = load_public_splits(
        args.public_data_path,
        val_frac=args.val_frac,
        seed=args.split_seed,
    )
    return splits[args.split]


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_set = load_problem_set(args)
    score = args.score if args.score is not None else args.split != "private" and not args.dry_run

    if args.dry_run:
        install_dry_run_generator()
        model_bundle = DryRunModelBundle()
    else:
        from baseline.modeling import ModelConfig, load_model

        model_config = ModelConfig(
            model_id=args.model_id,
            backend=args.backend,
            cache_dir=args.cache_dir,
            gpu_id=args.gpu_id,
            max_input_tokens=args.max_input_tokens,
            max_model_len=args.max_model_len,
            dtype="bfloat16",
            torch_dtype="bfloat16",
            load_in_4bit=False,
            device_map="auto",
            low_cpu_mem_usage=True,
            gpu_memory_utilization=0.85,
            max_num_seqs=256,
            max_num_batched_tokens=32768,
            reuse_loaded=True,
        )
        model_bundle = load_model(model_config)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        do_sample=not args.no_sample,
    )

    result = run_baseline2_problem_set(
        problem_set=problem_set,
        model_bundle=model_bundle,
        generation_config=generation_config,
        batch_size=args.batch_size,
        limit=args.limit,
        score=score,
        output_jsonl_path=output_dir / f"{args.split}_results.jsonl",
        debug_jsonl_path=output_dir / f"{args.split}_debug.jsonl",
        submission_csv_path=output_dir / "submission.csv" if args.split == "private" else None,
        report_json_path=output_dir / f"{args.split}_report.json",
        comparison_csv_path=args.comparison_csv,
        experiment_name="baseline2_prompt_format",
        split_name=args.split,
    )

    printed_report = {
        "summary": result.report["summary"],
        "formatting": result.report["formatting"],
        "output_jsonl_path": result.report["output_jsonl_path"],
        "debug_jsonl_path": result.report["debug_jsonl_path"],
        "submission_csv_path": result.report["submission_csv_path"],
    }

    baseline1_report_path = Path("results") / "baseline1_weakest" / f"{args.split}_report.json"
    if baseline1_report_path.exists():
        with open(baseline1_report_path, "r", encoding="utf-8") as f:
            baseline1_report = json.load(f)
        printed_report["baseline1_summary"] = baseline1_report.get("summary")

    pprint(printed_report)


if __name__ == "__main__":
    main()
