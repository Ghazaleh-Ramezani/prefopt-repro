#!/usr/bin/env bash
# Tiny end-to-end run on a single GPU (~15 min on an A10). Proves the wiring
# before you spend money on the real grid. The numbers are meaningless.
set -euo pipefail

SMALL="Qwen/Qwen2.5-0.5B-Instruct"
COMMON="--set model.base_model=$SMALL --set data.max_train_samples=200 --set data.max_eval_samples=50 --set model.attn_implementation=eager"

python -m prefopt.train_sft --config configs/sft.yaml $COMMON --output-dir runs/smoke_sft
python -m prefopt.train_rm  --config configs/rm.yaml  $COMMON --output-dir runs/smoke_rm
python -m prefopt.train_dpo --config configs/dpo.yaml $COMMON \
    --set init_adapter=runs/smoke_sft --output-dir runs/smoke_dpo
python -m prefopt.train_grpo --config configs/grpo.yaml $COMMON \
    --set init_adapter=runs/smoke_sft --set method.reward_model_path=runs/smoke_rm \
    --set method.num_generations=4 --set train.per_device_train_batch_size=4 \
    --set train.gradient_accumulation_steps=2 --output-dir runs/smoke_grpo

python -m prefopt.evaluate --config configs/dpo.yaml --policy runs/smoke_dpo \
    --baseline runs/smoke_sft --judge rm --rm-path runs/smoke_rm --n 20 --seeds 1
python -m prefopt.report --runs runs --out RESULTS.md
