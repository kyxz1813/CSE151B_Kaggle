import time


class GenerationConfig:
    def __init__(
        self,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        do_sample=True,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.do_sample = do_sample


def chunk_list(items, batch_size):
    if batch_size is None or batch_size <= 0:
        batch_size = len(items)
    for i in range(0, len(items), batch_size):
        yield i, items[i:i + batch_size]


def progress_iterator(items, show_progress, desc):
    if not show_progress:
        return items

    try:
        from tqdm.auto import tqdm
        return tqdm(items, desc=desc, total=len(items))
    except ImportError:
        return items


def generate_vllm(model_bundle, prompt_texts, generation_config, batch_size=None, show_progress=True):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        min_p=generation_config.min_p,
        presence_penalty=generation_config.presence_penalty,
        repetition_penalty=generation_config.repetition_penalty,
    )

    responses = []
    batch_ranges = list(chunk_list(prompt_texts, batch_size))
    iterator = progress_iterator(batch_ranges, show_progress, "Generating")

    for _, batch_prompts in iterator:
        outputs = model_bundle.llm.generate(batch_prompts, sampling_params=sampling_params)
        for out in outputs:
            responses.append(out.outputs[0].text.strip())

    return responses


def generate_transformers(model_bundle, prompt_texts, generation_config, batch_size=1, show_progress=True):
    import torch

    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    device = model_bundle.device()
    responses = []

    batch_ranges = list(chunk_list(prompt_texts, batch_size))
    iterator = progress_iterator(batch_ranges, show_progress, "Generating")

    for _, batch_prompts in iterator:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_bundle.config.max_input_tokens,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repetition_penalty=generation_config.repetition_penalty,
                do_sample=generation_config.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for out in output_ids:
            new_tokens = out[prompt_len:]
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return responses


def generate_prompt_texts(model_bundle, prompt_texts, generation_config=None, batch_size=1, show_progress=True):
    gen = generation_config or GenerationConfig()
    t0 = time.perf_counter()

    if model_bundle.backend == "vllm":
        responses = generate_vllm(model_bundle, prompt_texts, gen, batch_size=batch_size, show_progress=show_progress)
    elif model_bundle.backend == "transformers":
        responses = generate_transformers(model_bundle, prompt_texts, gen, batch_size=batch_size, show_progress=show_progress)
    else:
        raise ValueError(f"Unknown backend: {model_bundle.backend}")

    elapsed = time.perf_counter() - t0

    return {
        "responses": responses,
        "elapsed_sec": elapsed,
        "n": len(responses),
        "sec_per_problem": elapsed / len(responses) if responses else None,
        "backend": model_bundle.backend,
        "batch_size": batch_size,
        "max_new_tokens": gen.max_new_tokens,
    }
