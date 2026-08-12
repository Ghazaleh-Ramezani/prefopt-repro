# prefopt-repro — DPO and GRPO under controlled ablation

A reproduction of two preference-optimization methods on an open-weight model,
built so that the *comparison* is trustworthy rather than so that the numbers
are large. Four stages — SFT, reward model, DPO, GRPO — one config system, one
evaluation harness, and a set of reward-hacking diagnostics that run on every
checkpoint.

The question the repo answers: **how do KL strength, group size, and reward-model
quality trade off against win-rate, and where does optimising the proxy start to
cost real quality?**

## Quick start

```bash
pip install -e ".[train,judge]"

# 15-minute wiring check on a 0.5B model — numbers are meaningless, but a green
# run means the pipeline is sound before you rent an H100.
bash scripts/smoke_test.sh
```

Then the real pipeline:

```bash
python -m prefopt.train_sft  --config configs/sft.yaml    # baseline
python -m prefopt.train_rm   --config configs/rm.yaml     # reward model
python -m prefopt.train_dpo  --config configs/dpo.yaml    # or train_grpo
python -m prefopt.evaluate    --config configs/dpo.yaml \
    --policy runs/dpo_b0.1 --baseline runs/sft --n 300 --seeds 3
python -m prefopt.diagnostics --config configs/dpo.yaml \
    --policy runs/dpo_b0.1 --reference runs/sft --rm-path runs/rm_full
python -m prefopt.report --runs runs --out RESULTS.md
```

Full grid: `bash scripts/run_ablations.sh` (or `make ablations`).

## What each stage does

| Stage | Script | Key knob | Output |
|---|---|---|---|
| SFT | `train_sft.py` | — | Baseline adapter; every win-rate is measured against it |
| Reward model | `train_rm.py` | `data.rm_pair_fraction` | Scalar-head RM + held-out pairwise accuracy |
| DPO | `train_dpo.py` | `method.beta` | Policy trained directly on preference pairs |
| GRPO | `train_grpo.py` | `method.num_generations`, `method.beta` | Policy trained on group-relative advantages against the RM |

DPO's `beta` and GRPO's `beta` play the same role — the strength of the KL
anchor to the reference policy — which is what makes the two methods comparable
on one axis rather than two.

## Ablation grid

| Axis | Values | Question |
|---|---|---|
| DPO KL strength | beta ∈ {0.01, 0.05, 0.1, 0.5} | Where does the win-rate peak, and how early do the hacking diagnostics turn? |
| GRPO group size | G ∈ {4, 8, 16} | Does lower-variance advantage estimation buy win-rate at fixed effective batch? |
| RM quality | RM trained on 10% / 50% / 100% of pairs | How much of the downstream gain is attributable to reward-model accuracy? |

Results land in `RESULTS.md`, one row per run, each backed by a
`config.resolved.yaml` and an `environment.json` in its run directory.

## Evaluation, and why it is built this way

`prefopt/evaluate.py` reports win-rate against the SFT baseline with three
protections that most single-number comparisons skip:

- **Position swap.** Every pair is judged twice, A/B and B/A. A win counts only
  when the judge agrees with itself; flips are scored as ties and surfaced as
  `swap_consistency`. LLM judges have large position bias, and an unswapped
  win-rate partly measures it.
- **Bootstrap CIs.** A 300-prompt win-rate carries roughly a ±6-point 95%
  interval. Reporting 61.2% without one overstates what the run established.
- **Multi-seed generation.** Sampling variance across seeds is often larger than
  the method delta being claimed, so the harness reports the seed spread.

## Reward-hacking diagnostics

`prefopt/diagnostics.py` is the part that distinguishes "the policy improved"
from "the policy learned the reward model":

| Diagnostic | Reads as trouble when |
|---|---|
| `length_reward_corr` | Spearman(length, RM score) above ~0.5 — the win-rate is partly a verbosity win |
| `kl_from_reference` | Rising while judge win-rate plateaus — classic proxy over-optimisation |
| `goodhart_gap` | RM-judged win-rate pulling away from held-out judge win-rate |
| `distinct_4` | Falling versus the SFT reference — degenerate repetition scoring well |
| `stylistic_tells` | Header/bullet/enthusiasm rates climbing — the RM rewarding format over content |

The over-optimisation curve — reward up, quality flat or down, KL climbing — is
the intended headline finding of the GRPO sweep, not an incidental check.

## Configuration

Configs are YAML with one level of inheritance and dotted CLI overrides:

```bash
python -m prefopt.train_dpo --config configs/dpo.yaml \
    --set method.beta=0.05 --set model.lora_r=64 --output-dir runs/dpo_b0.05
```

Unknown keys raise at startup rather than falling back to a default, so a typo
in a sweep fails in the first second instead of producing a run that quietly
answers the wrong question. The fully resolved config is written to the run
directory alongside library versions, CUDA version, GPU model, and git SHA.

## Multi-GPU

```bash
accelerate launch --config_file accelerate_ds.yaml -m prefopt.train_dpo \
    --config configs/dpo.yaml --set train.deepspeed=ds_zero3.json
```

For GRPO, run generation on a dedicated vLLM server and set
`method.use_vllm=true` — rollout otherwise dominates step time:

```bash
trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct   # separate GPU
```

## Publishing to the Hub

Cards are rendered from run artifacts rather than written by hand, so a number
on the Hub always has a config and an environment dump behind it:

```bash
python -m prefopt.report --runs runs \
    --card cards/model_card_policy.md.j2 \
    --card-run runs/dpo_b0.1 --card-out runs/dpo_b0.1/README.md
huggingface-cli upload <user>/qwen2.5-1.5b-dpo-b0.1 runs/dpo_b0.1
```

Templates: `cards/model_card_policy.md.j2`, `cards/model_card_rm.md.j2`,
`cards/dataset_card.md.j2`. The `base_model:` field in the card front matter is
what makes an adapter appear on the base model's Hub page, so leave it in.

## Repository layout

```
configs/            base.yaml + one config per stage
src/prefopt/
  config.py         typed config, inheritance, CLI overrides
  data.py           dataset loaders for each stage
  modeling.py       model/tokenizer/LoRA construction
  reward.py         reward functions and shaping terms for GRPO
  train_{sft,rm,dpo,grpo}.py
  evaluate.py       generation + position-swapped judging + bootstrap CIs
  diagnostics.py    reward-hacking measurements
  report.py         run aggregation and card rendering
cards/              Jinja templates for Hub model/dataset cards
scripts/            smoke test and full ablation grid
tests/              CPU-only tests for config, rewards, stats, reporting
```

## Known limitations

Stated here rather than buried, because they bound what the results support:

- One base-model family at 1.5B; nothing here shows the ordering holds at 7B+.
- LoRA rather than full-parameter training, which limits policy movement.
- A single LLM judge with no measured human agreement — `swap_consistency` is a
  self-consistency check, not a validity check.
- One preference dataset with known verbosity bias, so `length_reward_corr` is
  partly inherited from the data rather than created by training.
- GRPO here optimises a learned RM, not a verifiable reward, so it is the
  RLHF-style setting rather than the reasoning-with-checkable-answers setting
  GRPO was introduced for.

## Version note

TRL's trainer signatures change between minor releases. The pins in
`pyproject.toml` (`trl>=0.17,<0.21`) mark the tested window; if you install
outside it, check `SFTConfig` / `RewardConfig` / `DPOConfig` / `GRPOConfig`
argument names before opening an issue.

## License

Apache-2.0.
