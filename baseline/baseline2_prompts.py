BASELINE2_SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem, but keep the response short and follow the required output format exactly.

Required output structure:

Reasoning:
- Use at most 8 concise lines.
- Do not repeat checks unless necessary.
- Do not continue once the final answer is written.

Final Answer: \\boxed{<final answer>}

Rules:
- The final answer must contain only the answer, not explanation.
- For multiple-choice questions, the final answer must be one capital letter when applicable.
- For numeric or symbolic answers, simplify when possible.
- For multiple [ANS] blanks, put all answers inside one box separated by commas, in the same order as the blanks.
- Use exactly one boxed expression after Final Answer.
- Do not write anything after the boxed final answer.
"""


BASELINE2_FORMAT_REPAIR_SYSTEM_PROMPT = """You are a formatting repair assistant.

Your job is not to solve a new problem. Your job is to rewrite the previous answer into the required final-answer schema.

Required output structure:

Reasoning:
Briefly state that the answer is being reformatted.

Final Answer: \\boxed{<final answer>}

Rules:
- Use exactly one boxed expression after Final Answer.
- Do not write anything after the boxed final answer.
- If an extracted answer is provided, preserve it exactly unless it clearly violates the requested answer-count format.
- For multiple-choice questions, the boxed answer must be a capital letter when applicable.
- For multiple [ANS] blanks, put all answers inside one box separated by commas, in the same order as the blanks.
"""


BASELINE2_SHORT_RESOLVE_SYSTEM_PROMPT = """You are an expert mathematician. Solve the original problem briefly and finish with the required final-answer format.

Required output structure:

Reasoning:
- Use at most 6 concise lines.
- Focus only on the shortest path to the answer.
- Do not re-check repeatedly.
- Do not continue once the final answer is written.

Final Answer: \\boxed{<final answer>}

Rules:
- The final answer must contain only the answer, not explanation.
- For multiple-choice questions, the final answer must be one capital letter when applicable.
- For numeric or symbolic answers, simplify when possible.
- For multiple [ANS] blanks, put all answers inside one box separated by commas, in the same order as the blanks.
- Use exactly one boxed expression after Final Answer.
- Do not write anything after the boxed final answer.
- Do not use external APIs, calculators, tools, code execution, or any other model.
"""


BASELINE2_ASSISTANT_PREFILL = "Reasoning:\n"


def format_options(options):
    lines = []
    for idx, option in enumerate(options or []):
        letter = chr(ord("A") + idx)
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def build_baseline2_user_prompt(context):
    problem = context.problem
    question = problem.question.strip()

    if not problem.options:
        return question

    return (
        f"{question}\n\n"
        f"Answer choices:\n{format_options(problem.options)}"
    )


def _format_original_problem_block(record):
    question = str(record.get("question", "")).strip()
    options = record.get("options") or []

    if options:
        return (
            f"Original question:\n{question}\n\n"
            f"Answer choices:\n{format_options(options)}"
        )

    return f"Original question:\n{question}"


def build_baseline2_retry_messages(
    record,
    previous_output=None,
    extracted_answer=None,
    retry_mode="short_resolve",
):
    original_block = _format_original_problem_block(record)

    if retry_mode == "format_repair":
        user_prompt = (
            f"{original_block}\n\n"
            f"Extracted answer candidate:\n{extracted_answer or ''}\n\n"
            f"Previous model output:\n{previous_output or ''}\n\n"
            "Rewrite the answer using the required schema. Do not solve from scratch unless the extracted answer is unusable."
        )

        return [
            {"role": "system", "content": BASELINE2_FORMAT_REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    user_prompt = (
        f"{original_block}\n\n"
        "The previous response did not contain a usable final boxed answer. "
        "Solve briefly and return the answer using the required schema."
    )

    return [
        {"role": "system", "content": BASELINE2_SHORT_RESOLVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]