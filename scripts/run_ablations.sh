#!/usr/bin/env bash
# Full ablation grid. Assumes SFT and the full-data RM already exist:
#   python -m prefopt.train_sft --config configs/sft.yaml
#   python -m prefopt.train_rm  --config configs/rm.yaml
#
# Runs are named after the varied axis so `python -m prefopt.report` can group
# them without extra bookkeeping.
set -euo pipefail

JUDGE="${JUDGE:-anthropic}"
EVAL_N="${EVAL_N:-300}"
SEEDS="${SEEDS:-3}"
BASELINE="runs/sft"

evaluate () {  # $1 = run dir, $2 = config
  python -m prefopt.evaluate --config "$2" --policy "$1" --baseline "$BASELINE" \
      --judge "$JUDGE" --n "$EVAL_N" --seeds "$SEEDS"
  python -m prefopt.diagnostics --config "$2" --policy "$1" \
      --reference "$BASELINE" --rm-path runs/rm_full --n 200
}

# ---- Axis 1: DPO KL strength ------------------------------------------------
for BETA in 0.01 0.05 0.1 0.5; do
  OUT="runs/dpo_b${BETA}"
  python -m prefopt.train_dpo --config configs/dpo.yaml \
      --set method.beta="$BETA" --set name="dpo_b${BETA}" --output-dir "$OUT"
  evaluate "$OUT" configs/dpo.yaml
done

# ---- Axis 2: GRPO group size ------------------------------------------------
# Effective batch stays fixed so the comparison is generations-per-prompt, not
# generations-per-step.
for G in 4 8 16; do
  OUT="runs/grpo_g${G}"
  python -m prefopt.train_grpo --config configs/grpo.yaml \
      --set method.num_generations="$G" --set name="grpo_g${G}" --output-dir "$OUT"
  evaluate "$OUT" configs/grpo.yaml
done

# ---- Axis 3: reward-model quality -------------------------------------------
# Train weaker RMs on nested subsets, then re-run GRPO against each.
for FRAC in 0.1 0.5; do
  RM="runs/rm_f${FRAC}"
  python -m prefopt.train_rm --config configs/rm.yaml \
      --set data.rm_pair_fraction="$FRAC" --set name="rm_f${FRAC}" --output-dir "$RM"
  OUT="runs/grpo_rm${FRAC}"
  python -m prefopt.train_grpo --config configs/grpo.yaml \
      --set method.reward_model_path="$RM" --set name="grpo_rm${FRAC}" --output-dir "$OUT"
  evaluate "$OUT" configs/grpo.yaml
done

python -m prefopt.report --runs runs --out RESULTS.md
