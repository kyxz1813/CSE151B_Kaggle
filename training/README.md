# CSE151B SFT/RL Training Patch

Files:
- `training/sft_data.py` builds chat-style SFT JSONL from tagged/rule-annotated records and optional baseline result traces.
- `training/sft_runner.py` runs QLoRA SFT with TRL + PEFT.
- `training/eval_adapter.py` evaluates a PEFT adapter with the existing Baseline 3 runner.
- `training/rewarding.py` contains correctness/format reward functions.
- `training/grpo_runner.py` is a GRPO scaffold.

Basic SFT flow:

```bash
python -m training.sft_data \
  --records-path results/baseline3_rule_annotations/public_all_rules_32k_final.jsonl \
  --results-paths results/baseline3_32k_final/public_full/public_full_results.jsonl \
  --output-dir results/sft_data/baseline3_sft_v0

python -m training.sft_runner \
  --train-jsonl results/sft_data/baseline3_sft_v0/train_sft.jsonl \
  --eval-jsonl results/sft_data/baseline3_sft_v0/val_sft.jsonl \
  --output-dir results/sft/qwen3_4b_math_sft_v0 \
  --max-seq-length 4096 \
  --num-train-epochs 1
```
