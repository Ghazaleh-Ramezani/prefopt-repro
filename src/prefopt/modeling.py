"""Model / tokenizer construction shared by all four training stages."""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import RunCfg
from .utils import get_logger

logger = get_logger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_tokenizer(cfg: RunCfg) -> Any:
    tok = AutoTokenizer.from_pretrained(cfg.model.base_model, use_fast=True)
    if tok.pad_token is None:
        # Reusing EOS as PAD is fine for causal LM but silently breaks the
        # reward model's sequence-classification head if the model config is
        # not updated too — see load_reward_model below.
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched generation
    return tok


def _quant_config(cfg: RunCfg) -> BitsAndBytesConfig | None:
    if not cfg.model.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=DTYPES[cfg.model.torch_dtype],
    )


def _common_kwargs(cfg: RunCfg) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": DTYPES[cfg.model.torch_dtype],
        "quantization_config": _quant_config(cfg),
    }
    if cfg.model.attn_implementation:
        kwargs["attn_implementation"] = cfg.model.attn_implementation
    return kwargs


def peft_config(cfg: RunCfg, task_type: str = "CAUSAL_LM"):
    """LoRA config, or None when full-parameter training is requested."""
    if not cfg.model.use_peft:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=cfg.model.lora_target_modules,
        bias="none",
        task_type=task_type,
    )


def load_policy(cfg: RunCfg, adapter_path: str | None = None) -> Any:
    """Causal LM policy. If `adapter_path` is given, the SFT adapter is loaded
    and merged so that later stages start from the SFT checkpoint rather than
    stacking adapters on adapters."""
    model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, **_common_kwargs(cfg))
    if adapter_path:
        from peft import PeftModel

        logger.info("Loading and merging SFT adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    if cfg.train.gradient_checkpointing:
        model.config.use_cache = False
    return model


def load_reward_model(cfg: RunCfg, path: str | None = None) -> tuple[Any, Any]:
    """Sequence-classification model with a scalar head, used as the reward
    model for GRPO and as a cheap proxy scorer in diagnostics."""
    source = path or cfg.model.base_model
    tok = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # classification head reads the last non-pad token

    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=cfg.method.rm_num_labels,
        **_common_kwargs(cfg),
    )
    # This line is the single most common source of silently-wrong reward
    # models: without it the head pools over padding.
    model.config.pad_token_id = tok.pad_token_id
    return model, tok
