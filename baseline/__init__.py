_EXPORTS = {
    "ProblemSet": ("baseline.datasets", "ProblemSet"),
    "load_problem_set": ("baseline.datasets", "load_problem_set"),
    "load_public_splits": ("baseline.datasets", "load_public_splits"),
    "load_private_set": ("baseline.datasets", "load_private_set"),
    "stratified_split": ("baseline.datasets", "stratified_split"),
    "ModelConfig": ("baseline.modeling", "ModelConfig"),
    "ModelBundle": ("baseline.modeling", "ModelBundle"),
    "predownload_model": ("baseline.modeling", "predownload_model"),
    "load_transformers_model": ("baseline.modeling", "load_transformers_model"),
    "GenerationConfig": ("baseline.generation", "GenerationConfig"),
    "generate_prompt_texts": ("baseline.generation", "generate_prompt_texts"),
    "score_records": ("baseline.scoring", "score_records"),
    "summarize_results": ("baseline.scoring", "summarize_results"),
    "run_problem_set": ("baseline.runner", "run_problem_set"),
    "save_submission_csv": ("baseline.runner", "save_submission_csv"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'baseline' has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
