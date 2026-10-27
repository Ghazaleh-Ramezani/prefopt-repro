"""Stage 3a — Direct Preference Optimization.

`method.beta` is the KL-regularization strength: the DPO objective is the
closed-form optimum of a KL-constrained reward-maximisation problem, so beta
plays the role PPO's KL penalty plays. Small beta lets the policy drift far
from the reference and is where reward hacking shows up; large beta pins the
policy near SFT and win-rate flatlines. Sweeping it is the headline ablation.

With PEFT the reference policy is free: TRL disables the adapter to recover
the base model, so no second copy of the weights sits in memory.

    python -m prefopt.train_dpo --config configs/dpo.yaml --set method.beta=0.05

`--eval-only` evaluates the adapter already saved in the output directory
without training, for a run whose final evaluation did not complete (the
adapter is saved before evaluation starts):

    python -m prefopt.train_dpo --config configs/dpo.yaml --output-dir runs/dpo_b0.1 \
        --set train.per_device_eval_batch_size=2 --eval-only
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from trl import DPOConfig, DPOTrainer

from .cli import build_parser, prepare
from .config import RunCfg
from .data import load_preference_dataset
from .modeling import load_policy, load_tokenizer, peft_config
from .utils import environment_metadata, get_logger, save_json

logger = get_logger(__name__)


def dpo_config(cfg: RunCfg) -> DPOConfig:
    return DPOConfig(
        output_dir=cfg.train.output_dir,
        beta=cfg.method.beta,
        loss_type=cfg.method.loss_type,
        label_smoothing=cfg.method.label_smoothing,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        max_grad_norm=cfg.train.max_grad_norm,
        bf16=cfg.train.bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": cfg.train.gradient_checkpointing_use_reentrant
        },
        logging_steps=cfg.train.logging_steps,
        eval_strategy=cfg.train.eval_strategy,
        eval_steps=cfg.train.eval_steps,
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        report_to=cfg.train.report_to,
        deepspeed=cfg.train.deepspeed,
        seed=cfg.seed,
        max_length=cfg.data.max_length,
        max_prompt_length=cfg.data.max_prompt_length,
    )


def load_trained_policy(cfg: RunCfg) -> Any:
    """The policy a finished DPO run trained: its SFT starting point with the
    saved DPO adapter on top, still as a PEFT model so that disabling the
    adapter gives back the reference, exactly as during training.

    The adapter is a delta on the SFT-merged weights (see
    `modeling.adapter_chain`), so it is applied only on top of the same
    `init_adapter` the run recorded; a different one raises.
    """
    from peft import PeftModel

    run = Path(cfg.train.output_dir)
    if not (run / "adapter_config.json").exists():
        raise FileNotFoundError(f"--eval-only needs a trained adapter in {run}; none found.")
    recorded = run / "config.resolved.yaml"
    if recorded.exists():
        trained_from = (yaml.safe_load(recorded.read_text()) or {}).get("init_adapter")
        if trained_from != cfg.init_adapter:
            raise ValueError(
                f"{run} was trained from init_adapter={trained_from!r}, but this "
                f"evaluation would start from {cfg.init_adapter!r}."
            )
    model = load_policy(cfg, adapter_path=cfg.init_adapter)
    return PeftModel.from_pretrained(model, str(run))


def main() -> None:
    parser = build_parser("Direct Preference Optimization")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the adapter already saved in the output directory; no training.",
    )
    cli_args = parser.parse_args()
    cfg = prepare(cli_args, record=not cli_args.eval_only)

    tokenizer = load_tokenizer(cfg)
    data = load_preference_dataset(cfg)
    if cli_args.eval_only:
        model, lora = load_trained_policy(cfg), None
    else:
        model = load_policy(cfg, adapter_path=cfg.init_adapter)
        lora = peft_config(cfg, task_type="CAUSAL_LM")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT adapter-disabling gives us the reference for free
        args=dpo_config(cfg),
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    if not cli_args.eval_only:
        trainer.train()
        # Saved before evaluating, so a failed evaluation can be rerun with
        # --eval-only instead of retraining.
        trainer.save_model(cfg.train.output_dir)
        tokenizer.save_pretrained(cfg.train.output_dir)

    metrics = trainer.evaluate()
    metrics["beta"] = cfg.method.beta
    metrics["loss_type"] = cfg.method.loss_type
    metrics["eval_batch_size"] = cfg.train.per_device_eval_batch_size
    if cli_args.eval_only:
        metrics["evaluated_separately"] = True
        save_json(environment_metadata(), Path(cfg.train.output_dir) / "eval_environment.json")
    save_json(metrics, Path(cfg.train.output_dir) / "dpo_eval.json")
    # rewards/margins and rewards/accuracies are the implicit-reward diagnostics
    # TRL logs; a high margin with a flat win-rate is the classic warning sign.
    logger.info("DPO eval: %s", {k: v for k, v in metrics.items() if "reward" in k})


if __name__ == "__main__":
    main()
