import time
import torch

from tqdm import tqdm


class GenerationConfig:
    def __init__(
        self,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        do_sample=True,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample


def generate_prompt_texts(model_bundle, prompt_texts, generation_config=None, batch_size=1, show_progress=True):

    gen = generation_config or GenerationConfig()
    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = model_bundle.device()

    responses = []
    iterator = range(0, len(prompt_texts), batch_size)

    if show_progress:
        iterator = tqdm(iterator, total=(len(prompt_texts) + batch_size - 1) // batch_size, desc="Generating")

    start = time.perf_counter()

    for start_idx in iterator:
        batch_prompts = prompt_texts[start_idx:start_idx + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_bundle.config.max_input_tokens,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                temperature=gen.temperature,
                top_p=gen.top_p,
                top_k=gen.top_k,
                repetition_penalty=gen.repetition_penalty,
                do_sample=gen.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]

        for out in output_ids:
            new_tokens = out[prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(text)

    elapsed = time.perf_counter() - start

    return {
        "responses": responses,
        "elapsed_sec": elapsed,
        "n": len(responses),
        "sec_per_problem": elapsed / len(responses) if responses else None,
    }