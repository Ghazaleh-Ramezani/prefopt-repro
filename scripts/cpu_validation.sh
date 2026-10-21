#!/usr/bin/env bash
# End-to-end run of every stage on a CPU, with no GPU and no Hub access.
#
# Builds the offline fixture (scripts/make_cpu_fixture.py), then runs
# SFT -> reward model -> DPO -> GRPO -> win-rate evaluation -> reward-hacking
# diagnostics -> Goodhart gap at every checkpoint -> report. Takes about ten
# minutes on two cores.
#
# What it checks is the wiring: each stage consumes the previous stage's
# checkpoint, and evaluation rebuilds the model each stage trained. A 0.67M
# randomly initialised model says nothing about the methods themselves.
#
#   bash scripts/cpu_validation.sh            # writes results/cpu_validation/
set -euo pipefail

FIX=${FIX:-fixtures/cpu}
RUNS=${RUNS:-runs/cpu}
OUT=${OUT:-results/cpu_validation}

python scripts/make_cpu_fixture.py --out "$FIX"

COMMON=(
  --set model.base_model="$FIX/tiny-model"
  --set data.dataset="$FIX/prefdata"
  --set data.train_split=train
  --set data.eval_split=test
  --set data.sft_train_split=validation
  --set model.attn_implementation=eager
  --set model.torch_dtype=float32
  --set train.bf16=false
  --set data.max_train_samples=240
  --set data.max_eval_samples=60
  --set data.max_length=320
  --set data.max_prompt_length=96
  --set model.lora_r=8
  --set model.lora_alpha=16
)

python -m prefopt.train_sft --config configs/sft.yaml "${COMMON[@]}" \
    --set name=sft --output-dir "$RUNS/sft"

# A randomly initialised model needs a far larger learning rate than the
# 1.5B default to learn anything in a few epochs.
python -m prefopt.train_rm --config configs/rm.yaml "${COMMON[@]}" \
    --set train.num_train_epochs=8 --set train.learning_rate=5.0e-3 \
    --set model.lora_r=16 --set name=rm --output-dir "$RUNS/rm"

# An independently seeded reward model stands in for the external judge
# offline; on GPU the external judge is the LLM judge in prefopt.evaluate.
python -m prefopt.train_rm --config configs/rm.yaml "${COMMON[@]}" \
    --set seed=1 --set train.num_train_epochs=8 --set train.learning_rate=5.0e-3 \
    --set model.lora_r=16 --set name=rm_gold --output-dir "$RUNS/rm_gold"

# Keep every checkpoint, so the Goodhart gap can be traced across training.
CKPT=(--set train.save_steps=4 --set train.save_total_limit=null)

python -m prefopt.train_dpo --config configs/dpo.yaml "${COMMON[@]}" \
    --set init_adapter="$RUNS/sft" --set train.num_train_epochs=2 \
    --set train.learning_rate=1.0e-4 "${CKPT[@]}" --set name=dpo_b0.1 --output-dir "$RUNS/dpo_b0.1"

python -m prefopt.train_grpo --config configs/grpo.yaml "${COMMON[@]}" \
    --set init_adapter="$RUNS/sft" --set method.reward_model_path="$RUNS/rm" \
    --set method.num_generations=4 --set train.per_device_train_batch_size=4 \
    --set train.gradient_accumulation_steps=2 --set method.max_completion_length=48 \
    --set data.max_train_samples=32 "${CKPT[@]}" --set name=grpo_g4 --output-dir "$RUNS/grpo_g4"

python -m prefopt.evaluate --config configs/dpo.yaml "${COMMON[@]}" \
    --policy "$RUNS/dpo_b0.1" --baseline "$RUNS/sft" \
    --judge rm --rm-path "$RUNS/rm" --n 12 --seeds 2

for policy in dpo_b0.1 grpo_g4; do
  python -m prefopt.diagnostics --config configs/dpo.yaml "${COMMON[@]}" \
      --policy "$RUNS/$policy" --reference "$RUNS/sft" --rm-path "$RUNS/rm" --n 12
done

for policy in dpo_b0.1 grpo_g4; do
  python -m prefopt.goodhart --config configs/dpo.yaml "${COMMON[@]}" \
      --run "$RUNS/$policy" --reference "$RUNS/sft" --proxy-rm "$RUNS/rm" \
      --judge rm --judge-rm "$RUNS/rm_gold" --n 12 --seeds 2
done

mkdir -p "$OUT"
python -m prefopt.report --runs "$RUNS" --out "$OUT/RESULTS.md"

# ZeRO-3 sharding check: two CPU ranks against a single-process run.
if python -c "import deepspeed" 2>/dev/null; then
  DS_ACCELERATOR=cpu FIX="$FIX" torchrun --nproc_per_node 2 \
      scripts/zero3_cpu_check.py --out "$OUT/zero3_cpu_check.json"
fi
cat > "$OUT/README.md" <<'EOF'
# CPU validation run

Produced by `bash scripts/cpu_validation.sh` on an offline fixture: a
0.67M-parameter Qwen2-architecture model with a byte-level BPE tokenizer and a
240-pair synthetic preference set, built by `scripts/make_cpu_fixture.py`.

The run checks that each stage trains, saves its checkpoint, and that the next
stage and the evaluation harness rebuild exactly the model that was trained.
`goodhart.json` in the DPO and GRPO folders traces the Goodhart gap at every
saved checkpoint, with an independently seeded reward model (`rm_gold`)
standing in for the external judge. `zero3_cpu_check.json` compares DeepSpeed
ZeRO-3 training on two CPU ranks against a single-process run of the same
steps.
On a model this small the win-rates and diagnostics exercise the code paths;
comparisons between methods come from the 1.5B ablation grid.

Each subdirectory holds that stage's `config.resolved.yaml` and
`environment.json` (git SHA, library versions, hardware), plus its evaluation
output where the stage has one.
EOF

# Keep the text artifacts; adapter weights stay in $RUNS.
for run in "$RUNS"/*/; do
  name=$(basename "$run")
  mkdir -p "$OUT/$name"
  for f in config.resolved.yaml environment.json rm_eval.json dpo_eval.json \
           grpo_eval.json results.json diagnostics.json goodhart.json; do
    [ -f "$run/$f" ] && cp "$run/$f" "$OUT/$name/"
  done
done
echo "artifacts -> $OUT"
