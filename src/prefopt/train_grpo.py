"""Stage 3b — Group Relative Policy Optimization.

GRPO samples G completions per prompt and standardises rewards *within the
group* to form the advantage, which removes the need for a learned value
network. The two knobs that matter:

  method.num_generations (G)  — larger groups give a lower-variance advantage
                                estimate at linear generation cost.
  method.beta                 — KL coefficient against the reference policy.

Set `method.use_vllm=true` and run a vLLM server to make generation cheap;
without it, rollout dominates step time.

    trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct    # separate GPU
    python -m prefopt.train_grpo --config configs/grpo.yaml --set method.num_generations=16
"""

from __future__ import annotations

from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

from .cli import parse_and_prepare
from .data import load_prompt_dataset
from .modeling import load_policy, load_tokenizer, peft_config
from .reward import build_reward_funcs
from .utils import get_logger, save_json

logger = get_logger(__name__)


def main() -> None:
    cfg = parse_and_prepare("Group Relative Policy Optimization")
    tokenizer = load_tokenizer(cfg)
    model = load_policy(cfg, adapter_path=cfg.init_adapter)
    prompts = load_prompt_dataset(cfg)
    reward_funcs, reward_weights = build_reward_funcs(cfg)

    # The effective batch must be divisible by the group size, or TRL cannot
    # form complete groups. Failing here is cheaper than failing at step 1.
    effective = (
        cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps
    )
    if effective % cfg.method.num_generations != 0:
        raise ValueError(
            f"per_device_train_batch_size * gradient_accumulation_steps ({effective}) "
            f"must be divisible by num_generations ({cfg.method.num_generations})."
        )

    args = GRPOConfig(
        output_dir=cfg.train.output_dir,
        beta=cfg.method.beta,
        num_generations=cfg.method.num_generations,
        max_completion_length=cfg.method.max_completion_length,
        max_prompt_length=cfg.data.max_prompt_length,
        temperature=cfg.method.temperature,
        reward_weights=reward_weights,
        use_vllm=cfg.method.use_vllm,
        learning_rate=cfg.train.learning_rate,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        max_grad_norm=cfg.train.max_grad_norm,
        bf16=cfg.train.bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        logging_steps=cfg.train.logging_steps,
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        report_to=cfg.train.report_to,
        deepspeed=cfg.train.deepspeed,
        seed=cfg.seed,
        log_completions=True,  # eyeballing rollouts catches hacking early
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=prompts,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        peft_config=peft_config(cfg, task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)

    save_json(
        {
            "beta": cfg.method.beta,
            "num_generations": cfg.method.num_generations,
            "reward_model_path": cfg.method.reward_model_path,
            "reward_weights": reward_weights,
            "log_history": trainer.state.log_history[-50:],
        },
        Path(cfg.train.output_dir) / "grpo_eval.json",
    )
    logger.info("GRPO policy written to %s", cfg.train.output_dir)


if __name__ == "__main__":
    main()
