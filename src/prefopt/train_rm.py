"""Stage 2 — reward model.

Trains a scalar-head model on chosen/rejected pairs with the Bradley-Terry
loss. The held-out pairwise accuracy this reports is the *x-axis* of the
reward-model-quality ablation: train RMs on 10% / 50% / 100% of pairs, record
their accuracies, then plot downstream GRPO win-rate against them.

    python -m prefopt.train_rm --config configs/rm.yaml
    python -m prefopt.train_rm --config configs/rm.yaml --set data.rm_pair_fraction=0.1
"""

from __future__ import annotations

from pathlib import Path

from trl import RewardConfig, RewardTrainer

from .cli import parse_and_prepare
from .data import load_preference_dataset
from .modeling import load_reward_model, peft_config
from .utils import get_logger, save_json

logger = get_logger(__name__)


def main() -> None:
    cfg = parse_and_prepare("Reward model training")
    model, tokenizer = load_reward_model(cfg)
    data = load_preference_dataset(cfg, fraction=cfg.data.rm_pair_fraction)

    args = RewardConfig(
        output_dir=cfg.train.output_dir,
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
        center_rewards_coefficient=0.01,  # keeps the reward scale from drifting
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        processing_class=tokenizer,
        peft_config=peft_config(cfg, task_type="SEQ_CLS"),
    )
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)

    metrics = trainer.evaluate()
    metrics["rm_pair_fraction"] = cfg.data.rm_pair_fraction
    metrics["n_train_pairs"] = len(data["train"])
    save_json(metrics, Path(cfg.train.output_dir) / "rm_eval.json")
    logger.info(
        "RM pairwise accuracy: %.4f on %d held-out pairs",
        metrics.get("eval_accuracy", float("nan")),
        len(data["eval"]),
    )


if __name__ == "__main__":
    main()
