BASELINE2_SYSTEM_PROMPT = """You are an expert mathematician. Solve the problem and follow the required output format exactly.

Required output structure:

Reasoning:
<concise step-by-step reasoning>

Final Answer: \\boxed{<final answer>}

Global rules:
- You must end with exactly one line of the form: Final Answer: \\boxed{...}
- Stop immediately after the boxed final answer.
- Do not write anything after the final boxed answer.
- The final answer must contain only the answer, not explanation.
- If you are running out of space, stop reasoning and immediately write Final Answer: \boxed{...}.
- Do not continue with a second reasoning pass after the final answer.

Multiple-choice rules:
- For multiple-choice questions, compute the result, compare it to the answer choices, and put only the matching capital letter in the final box.
- Never put a raw number, formula, or expression in the final box for a multiple-choice question.
- If the exact computed result seems absent from the choices, choose the closest intended option and still box exactly one letter.

Multiple-answer rules:
- Before solving, count the number of [ANS] placeholders.
- If there are multiple [ANS] placeholders, make sure the final box contains exactly one comma-separated entry per blank, in the same order.
- Some blanks may ask for formulas, equations, words, directions, yes/no answers, or table entries, not just final numeric values.
- For table problems, fill every [ANS] cell, row by row, not only the final statistic.

Precision and exactness rules:
- Do not round prematurely.
- If a decimal approximation is requested, give more precision than the minimum when possible.
- If the problem says "at least N decimal places" or "correct to N decimal places", give 4-8 accurate decimal places when possible, unless it explicitly asks to round to exactly N places.
- Only round to an integer if the problem explicitly says to round to an integer, whole number, or smallest integer that works.
- Preserve exact forms such as fractions, radicals, logarithms, powers, and symbolic expressions when the prompt allows an exact answer.
- If the prompt allows decimal or fraction/expression, exact form is usually safer.
- If a value is computed numerically, keep 4-8 significant decimal digits unless the problem explicitly asks for exact rounding.
- For temperature conversions, use C=(F-32)*5/9, K=C+273.15, and R=F+459.67.
- For binary or base arithmetic, carry in the stated base and cross-check by decimal conversion when possible.
"""


BASELINE2_RETRY_SYSTEM_PROMPT = """You repair answer formatting.

You are given the original problem and a previous model response that was malformed or incomplete.

Your job is NOT to solve from scratch unless absolutely necessary.
Your job is to extract, repair, or choose the final answer from the previous response.

Return only this exact structure:

Final Answer: \\boxed{<final answer>}

Rules:
- Do not include reasoning.
- Do not include explanation.
- Do not include markdown.
- Do not write anything after the boxed final answer.
- Use exactly one boxed expression.
- For multiple-choice questions, the boxed answer must be one capital letter.
- For multiple [ANS] blanks, the boxed answer must contain exactly one comma-separated entry per blank, in order.
- If the previous response computed the answer but failed to box it, box that answer.
- If the previous response computed a multiple-choice value, compare it to the choices and box the matching letter.
- If no reliable answer is present, make the best possible answer choice and still return the required boxed final answer.
"""


BASELINE2_ASSISTANT_PREFILL = "Reasoning:\n"
BASELINE2_RESPONSE_PREFILL = "Reasoning:\n"
BASELINE2_RETRY_ASSISTANT_PREFILL = "Final Answer: "


def format_options(options):
    lines = []
    for idx, option in enumerate(options or []):
        letter = chr(ord("A") + idx)
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def count_answer_blanks(question):
    return str(question or "").count("[ANS]")


def build_baseline2_user_prompt(context):
    problem = context.problem
    question = problem.question.strip()
    n_blanks = count_answer_blanks(question)

    answer_contract = (
        "\n\nAnswer contract:\n"
        f"- Number of [ANS] blanks: {n_blanks}\n"
    )

    if problem.options:
        answer_contract += (
            "- This is multiple-choice. Final boxed answer must be exactly one capital option letter.\n"
        )
    elif n_blanks > 1:
        answer_contract += (
            f"- Final boxed answer must contain exactly {n_blanks} comma-separated entries in the same order as the blanks.\n"
        )
    else:
        answer_contract += (
            "- Final boxed answer must contain the single requested answer.\n"
        )

    answer_contract += (
        "- End with exactly one line: Final Answer: \\boxed{...}\n"
        "- Stop immediately after that line.\n"
    )

    if not problem.options:
        return question + answer_contract

    return (
        f"{question}\n\n"
        f"Answer choices:\n{format_options(problem.options)}"
        f"{answer_contract}"
    )


def build_baseline2_retry_messages(record, previous_response=None, extracted_answer=None, retry_mode=None):
    question = str(record.get("question", "")).strip()
    options = record.get("options") or []
    n_blanks = count_answer_blanks(question)

    if options:
        user_prompt = (
            f"Original question:\n{question}\n\n"
            f"Answer choices:\n{format_options(options)}\n\n"
            f"Number of [ANS] blanks: {n_blanks}\n\n"
            f"Previous response:\n{previous_response or ''}\n\n"
            f"Previously extracted answer, if any: {extracted_answer or ''}\n"
            f"Retry mode: {retry_mode or 'format_repair'}\n\n"
            "Repair the previous response. Return only Final Answer: \\boxed{<one capital letter>}."
        )
    else:
        user_prompt = (
            f"Original question:\n{question}\n\n"
            f"Number of [ANS] blanks: {n_blanks}\n\n"
            f"Previous response:\n{previous_response or ''}\n\n"
            f"Previously extracted answer, if any: {extracted_answer or ''}\n"
            f"Retry mode: {retry_mode or 'format_repair'}\n\n"
            "Repair the previous response. Return only Final Answer: \\boxed{...}."
        )

    return [
        {"role": "system", "content": BASELINE2_RETRY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]