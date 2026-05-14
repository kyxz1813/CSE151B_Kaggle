from prompting.prompt_chain import build_prompt_chain
from .baseline2_prompts import BASELINE2_RESPONSE_PREFILL


def apply_chat_template(tokenizer, messages, metadata=None):
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if metadata and metadata.get("strategy_name") in {"baseline2", "baseline3"}:
        role_start = "<|im_start|>" + "assist" + "ant\n"
        thinking_prefill = f"{role_start}<think>\n"
        answer_prefill = f"{role_start}{BASELINE2_RESPONSE_PREFILL}"
        if prompt_text.endswith(thinking_prefill):
            prompt_text = prompt_text[:-len(thinking_prefill)] + answer_prefill

    return prompt_text


def build_prompt_specs(problem_set, prompt_chain=None):
    chain = prompt_chain or build_prompt_chain()

    rows = []
    for problem in problem_set.problems():
        spec = chain.build_spec(problem)
        rows.append({
            "id": problem.id,
            "problem": problem,
            "spec": spec,
            "messages": spec.to_messages(),
            "metadata": spec.metadata,
            "generation_hints": spec.generation_hints,
        })

    return rows


def build_prompt_texts(problem_set, tokenizer, prompt_chain=None):
    prompt_rows = build_prompt_specs(problem_set, prompt_chain=prompt_chain)

    for row in prompt_rows:
        row["response_prefill"] = (
            BASELINE2_RESPONSE_PREFILL
            if row["metadata"].get("strategy_name") in {"baseline2", "baseline3"}
            else ""
        )
        row["prompt_text"] = apply_chat_template(
            tokenizer,
            row["messages"],
            metadata=row["metadata"],
        )

    return prompt_rows
