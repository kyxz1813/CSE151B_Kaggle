import os
from huggingface_hub import snapshot_download

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelConfig:
    def __init__(
        self,
        model_id="Qwen/Qwen3-4B-Thinking-2507",
        cache_dir=None,
        gpu_id=None,
        load_in_4bit=True,
        device_map="auto",
        max_input_tokens=16384,
        torch_dtype="bfloat16",
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.gpu_id = gpu_id
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.max_input_tokens = max_input_tokens
        self.torch_dtype = torch_dtype


class ModelBundle:
    def __init__(self, config, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def device(self):
        return next(self.model.parameters()).device


def predownload_model(config):

    return snapshot_download(
        repo_id=config.model_id,
        cache_dir=config.cache_dir,
    )


def load_transformers_model(config):
    if config.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = {
        "trust_remote_code": True,
        "device_map": config.device_map,
        "cache_dir": config.cache_dir,
    }

    if config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        if config.torch_dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif config.torch_dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        **kwargs,
    )

    model.eval()

    return ModelBundle(config=config, tokenizer=tokenizer, model=model)