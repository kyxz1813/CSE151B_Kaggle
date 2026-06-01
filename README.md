# CSE 151B Spring 2026 Kaggle Competition

## Hardware Used

Final inference was run on:

```text
NVIDIA A100, with High-Ram on Colab Pro
Approximate full private-set inference time: about 2 hours at max_new_tokens=32768
```

## Model Weights

We use the base HuggingFace model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

No fine-tuned checkpoint is required for the final submitted pipeline.

The model will be downloaded automatically by HuggingFace/vLLM when `run_inference()` is called.

## Environment Setup

```bash
pip install -U vllm transformers torch tqdm pandas numpy sympy antlr4-python3-runtime==4.11.1 bitsandbytes
```

What worked in colab:

```bash
!pip install prettyprint sympy numpy pandas matplotlib datasets peft trl transformers accelerate tqdm bitsandbytes ipykernel jupyter nvidia-nvjitlink antlr4-python3-runtime==4.11.1

!pip install -U uv
!uv pip install --system --reinstall --torch-backend=cu128 "vllm==0.18"
```

```python
import os

lib_path = "/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib"

if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] += f":{lib_path}"
else:
    os.environ["LD_LIBRARY_PATH"] = lib_path
```

## Reproducing the Final Submission

```bash
python run_inference.py \
  --private-data-path data/private.jsonl \
  --output-csv results/final_submission/submission.csv \
  --gpu-id 0 \
  --backend vllm \
  --batch-size 32 \
  --max-model-len 32768 \
  --max-new-tokens 32768
```

This performs the full end-to-end pipeline:

1. loads the model,
2. loads the private dataset,
3. categorizes private questions if cached tags are not already available,
4. applies category/rule annotations,
5. chooses the final prompting strategy per problem,
6. runs inference,
7. applies retry-based schema repair and answer normalization,
8. writes the final Kaggle submission CSV.

The output file has the required format:

```text
id,response
```

## Final Inference Hyperparameters

Main generation:

```text
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0.0
repetition_penalty = 1.0
presence_penalty = 0.0
max_new_tokens = 32768
```

Retry/repair generation:

```text
temperature = 0.0
top_p = 1.0
top_k = -1
max_new_tokens = 4096
```
