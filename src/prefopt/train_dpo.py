"""Stage 3a — Direct Preference Optimization.

`method.beta` is the KL-regularization strength: the DPO objective is the
closed-form optimum of a KL-constrained reward-maximisation problem, so beta
plays the role PPO's KL penalty plays. Small beta lets the policy drift far
from the reference and is where reward hacking shows up; large beta pins the
policy near SFT and win-rate flatlines. Sweeping it is the headline ablation.

With PEFT the reference policy is free: TRL disables the adapter to recover
the base model, so no second copy of the weights sits in memory.

    python -m prefopt.train_dpo --config configs/dpo.yaml --set method.beta=0.05
"""

from __future__ import annotations

from pathlib import Path

from trl import DPOConfig, DPOTrainer

from .cli import parse_and_prepare
from .data import load_preference_dataset
from .modeling import load_policy, load_tokenizer, peft_config
from .utils import get_logger, save_json

logger = get_logger(__name__)


def main() -> None:
    cfg = parse_and_prepare("Direct Preference Optimization")
    tokenizer = load_tokenizer(cfg)
    model = load_policy(cfg, adapter_path=cfg.init_adapter)
    data = load_preference_dataset(cfg)

    args = DPOConfig(
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

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT adapter-disabling gives us the reference for free
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        processing_class=tokenizer,
        peft_config=peft_config(cfg, task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)

    metrics = trainer.evaluate()
    metrics["beta"] = cfg.method.beta
    metrics["loss_type"] = cfg.method.loss_type
    save_json(metrics, Path(cfg.train.output_dir) / "dpo_eval.json")
    # rewards/margins and rewards/accuracies are the implicit-reward diagnostics
    # TRL logs; a high margin with a flat win-rate is the classic warning sign.
    logger.info("DPO eval: %s", {k: v for k, v in metrics.items() if "reward" in k})


if __name__ == "__main__":
    main()
