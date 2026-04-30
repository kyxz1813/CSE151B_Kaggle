from prompting.prompt_chain import build_prompt_chain


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
        row["prompt_text"] = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt_rows