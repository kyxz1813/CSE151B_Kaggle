import os

from huggingface_hub import snapshot_download
from vllm import LLM
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

_MODEL_CACHE = {}


class ModelConfig:
    def __init__(
        self,
        model_id="Qwen/Qwen3-4B-Thinking-2507",
        backend="vllm",
        cache_dir=None,
        gpu_id="0",
        trust_remote_code=True,
        reuse_loaded=True,
        max_input_tokens=4096,
        max_model_len=4096,
        dtype="bfloat16",
        torch_dtype="bfloat16",
        load_in_4bit=False,
        device_map="auto",
        low_cpu_mem_usage=True,
        gpu_memory_utilization=0.85,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        enforce_eager=False,
        enable_prefix_caching=False,
    ):
        self.model_id = model_id
        self.backend = backend
        self.cache_dir = cache_dir
        self.gpu_id = gpu_id
        self.trust_remote_code = trust_remote_code
        self.reuse_loaded = reuse_loaded
        self.max_input_tokens = max_input_tokens
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching

    def cache_key(self):
        return (
            self.model_id,
            self.backend,
            self.cache_dir,
            self.gpu_id,
            self.trust_remote_code,
            self.max_input_tokens,
            self.max_model_len,
            self.dtype,
            self.torch_dtype,
            self.load_in_4bit,
            self.device_map,
            self.low_cpu_mem_usage,
            self.gpu_memory_utilization,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.tensor_parallel_size,
            self.enforce_eager,
            self.enable_prefix_caching,
        )


class ModelBundle:
    def __init__(self, config, tokenizer, model=None, llm=None):
        self.config = config
        self.backend = config.backend
        self.tokenizer = tokenizer
        self.model = model
        self.llm = llm

    def device(self):
        if self.backend == "vllm":
            return "vllm"
        try:
            return next(self.model.parameters()).device
        except Exception:
            return "cpu"


class GPUInfo:
    def __init__(self, cuda_available, device_count, device_name=None, capability=None):
        self.cuda_available = cuda_available
        self.device_count = device_count
        self.device_name = device_name
        self.capability = capability

    def to_dict(self):
        return {
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "capability": self.capability,
        }


def detect_gpu_info():
    try:

        if not torch.cuda.is_available():
            return GPUInfo(False, 0)

        idx = torch.cuda.current_device()
        return GPUInfo(
            cuda_available=True,
            device_count=torch.cuda.device_count(),
            device_name=torch.cuda.get_device_name(idx),
            capability=torch.cuda.get_device_capability(idx),
        )
    except Exception:
        return GPUInfo(False, 0)


def set_cuda_visible_devices(config):
    if config.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)


def predownload_model(config):
    return snapshot_download(
        repo_id=config.model_id,
        cache_dir=config.cache_dir,
    )


def load_tokenizer(config):

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
        cache_dir=config.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def torch_dtype_from_name(name):

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return "auto"


def load_transformers_model(config):
    set_cuda_visible_devices(config)



    key = config.cache_key()
    if config.reuse_loaded and key in _MODEL_CACHE:
        print("Reusing cached Transformers model.")
        return _MODEL_CACHE[key]

    tokenizer = load_tokenizer(config)
    cuda_available = torch.cuda.is_available()

    kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "cache_dir": config.cache_dir,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
    }

    if cuda_available:
        kwargs["device_map"] = config.device_map
    else:
        kwargs["device_map"] = None

    if config.load_in_4bit and cuda_available:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype_from_name(config.torch_dtype),
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch_dtype_from_name(config.torch_dtype) if cuda_available else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        **kwargs,
    )
    model.eval()

    bundle = ModelBundle(config=config, tokenizer=tokenizer, model=model)

    if config.reuse_loaded:
        _MODEL_CACHE[key] = bundle

    if cuda_available:
        print("Transformers CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA unavailable. Transformers model is on CPU and will be very slow.")

    return bundle


def load_vllm_model(config):
    set_cuda_visible_devices(config)

    key = config.cache_key()
    if config.reuse_loaded and key in _MODEL_CACHE:
        print("Reusing cached vLLM model.")
        return _MODEL_CACHE[key]

    tokenizer = load_tokenizer(config)


    kwargs = {
        "model": config.model_id,
        "trust_remote_code": config.trust_remote_code,
        "download_dir": config.cache_dir,
        "dtype": config.dtype,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_num_seqs": config.max_num_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "tensor_parallel_size": config.tensor_parallel_size,
        "enforce_eager": config.enforce_eager,
        "enable_prefix_caching": config.enable_prefix_caching,
    }

    if config.load_in_4bit:
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"

    llm = LLM(**kwargs)
    bundle = ModelBundle(config=config, tokenizer=tokenizer, llm=llm)

    if config.reuse_loaded:
        _MODEL_CACHE[key] = bundle

    return bundle


def load_model(config):
    backend = (config.backend or "vllm").lower()
    config.backend = backend

    if backend == "vllm":
        return load_vllm_model(config)
    if backend == "transformers":
        return load_transformers_model(config)

    raise ValueError(f"Unknown backend: {backend}")


def clear_loaded_model_cache():
    _MODEL_CACHE.clear()
