from pathlib import Path

from baseline.category_experiments import filter_problem_set_by_category, run_category_experiment
from baseline.category_harnesses import get_harness
from baseline.category_rules import (
    annotate_problem_set_with_rules,
    save_rule_one_hot_csv,
    rule_distribution,
)


def category_output_dir(base_dir, category, experiment_name, strategy_name):
    path = (
        Path(base_dir)
        / str(category)
        / str(experiment_name)
        / str(strategy_name)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def inspect_category_records(problem_set, category, n=10):
    subset = filter_problem_set_by_category(problem_set, category)

    rows = []
    for record in subset.records[:n]:
        rows.append({
            "id": record.get("id"),
            "primary_category": record.get("primary_category"),
            "is_mcq": bool(record.get("options")),
            "answer": record.get("answer"),
            "question": str(record.get("question", ""))[:500],
        })

    return rows


def prepare_category_rules(problem_set, category, output_dir):
    subset = filter_problem_set_by_category(problem_set, category)
    annotated = annotate_problem_set_with_rules(subset)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rule_csv_path = output_dir / "rule_one_hot.csv"
    save_rule_one_hot_csv(annotated.records, rule_csv_path)

    return {
        "problem_set": annotated,
        "rule_distribution": rule_distribution(annotated.records),
        "rule_one_hot_csv_path": str(rule_csv_path),
    }


def run_category_workspace(
    problem_set,
    category,
    strategy_name,
    model_bundle,
    generation_config,
    retry_generation_config,
    batch_size,
    output_root,
    experiment_name=None,
    limit=None,
    score=True,
    show_progress=True,
):
    experiment_name = experiment_name or f"{category}_{strategy_name}"

    output_dir = category_output_dir(
        base_dir=output_root,
        category=category,
        experiment_name=experiment_name,
        strategy_name=strategy_name,
    )

    rules_info = prepare_category_rules(
        problem_set=problem_set,
        category=category,
        output_dir=output_dir,
    )

    result = run_category_experiment(
        problem_set=rules_info["problem_set"],
        category=category,
        model_bundle=model_bundle,
        generation_config=generation_config,
        retry_generation_config=retry_generation_config,
        strategy_name=strategy_name,
        experiment_name=experiment_name,
        batch_size=batch_size,
        limit=limit,
        score=score,
        output_dir=output_root,
        comparison_csv_path=None,
        harness=get_harness(category),
        show_progress=show_progress,
    )

    result.report["category_workspace"] = {
        "category": category,
        "strategy_name": strategy_name,
        "experiment_name": experiment_name,
        "rule_distribution": rules_info["rule_distribution"],
        "rule_one_hot_csv_path": rules_info["rule_one_hot_csv_path"],
        "workspace_output_dir": str(output_dir),
    }

    return result