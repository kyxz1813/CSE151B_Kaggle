import os
from huggingface_hub import snapshot_download

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


_MODEL_CACHE = {}


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
        reuse_loaded=True,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.gpu_id = gpu_id
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.max_input_tokens = max_input_tokens
        self.torch_dtype = torch_dtype
        self.reuse_loaded = reuse_loaded

    def cache_key(self):
        return (
            self.model_id,
            self.cache_dir,
            self.gpu_id,
            self.load_in_4bit,
            self.device_map,
            self.max_input_tokens,
            self.torch_dtype,
        )


class ModelBundle:
    def __init__(self, config, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return "cpu"


def predownload_model(config):
    return snapshot_download(
        repo_id=config.model_id,
        cache_dir=config.cache_dir,
    )


def load_transformers_model(config):
    key = config.cache_key()

    if config.reuse_loaded and key in _MODEL_CACHE:
        print("Reusing model already loaded in this Python session.")
        return _MODEL_CACHE[key]

    if config.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)


    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.get_device_name(0))
    else:
        print("CUDA unavailable. Model will run on CPU and will be very slow.")

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

    if config.load_in_4bit and torch.cuda.is_available():
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

    bundle = ModelBundle(config=config, tokenizer=tokenizer, model=model)

    if config.reuse_loaded:
        _MODEL_CACHE[key] = bundle

    return bundle


def clear_loaded_model_cache():
    _MODEL_CACHE.clear()