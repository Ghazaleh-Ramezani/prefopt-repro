"""Stage 1 — supervised fine-tuning.

This is the baseline every later number is measured against, so it is worth
being boring here: one epoch, modest LR, no clever tricks. If the SFT model is
weak, DPO and GRPO will both look artificially good.

    python -m prefopt.train_sft --config configs/sft.yaml
"""

from __future__ import annotations

from trl import SFTConfig, SFTTrainer

from .cli import parse_and_prepare
from .data import load_sft_dataset
from .modeling import load_policy, load_tokenizer, peft_config
from .utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = parse_and_prepare("Supervised fine-tuning")
    tokenizer = load_tokenizer(cfg)
    model = load_policy(cfg)
    data = load_sft_dataset(cfg)

    args = SFTConfig(
        output_dir=cfg.train.output_dir,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        weight_decay=cfg.train.weight_decay,
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
        packing=cfg.method.packing,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        processing_class=tokenizer,
        peft_config=peft_config(cfg, task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)
    logger.info("SFT adapter written to %s", cfg.train.output_dir)


if __name__ == "__main__":
    main()
