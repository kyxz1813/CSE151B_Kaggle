BASELINE2_SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem and follow the required output format exactly.

Required output structure:

Reasoning:
<concise step-by-step reasoning>

Final Answer: \\boxed{<final answer>}

Rules:
- The final answer must contain only the answer, not explanation.
- For multiple-choice questions, the final answer must be a single capital letter when applicable.
- For numeric answers, simplify when possible.
- For multiple sub-answers, put all answers inside one box separated by commas.
- Never omit the final answer.
"""


BASELINE2_RETRY_SYSTEM_PROMPT = """You are an expert mathematician. Solve the original problem again and follow the required output format exactly.

Your output must include:

Final Answer: \\boxed{...}

Use this exact structure:

Reasoning:
<concise step-by-step reasoning>

Final Answer: \\boxed{<final answer>}

Rules:
- The final answer must contain only the answer, not explanation.
- For multiple-choice questions, the final answer must be a single capital letter when applicable.
- For numeric answers, simplify when possible.
- For multiple sub-answers, put all answers inside one box separated by commas.
- Never omit the final answer.
- Use exactly one boxed expression after Final Answer.
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


def build_baseline2_retry_messages(record):
    question = str(record.get("question", "")).strip()
    options = record.get("options") or []

    if options:
        user_prompt = (
            f"Original question:\n{question}\n\n"
            f"Answer choices:\n{format_options(options)}\n\n"
            "Return the answer using the required structure."
        )
    else:
        user_prompt = (
            f"Original question:\n{question}\n\n"
            "Return the answer using the required structure."
        )

    return [
        {"role": "system", "content": BASELINE2_RETRY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
