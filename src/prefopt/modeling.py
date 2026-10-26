"""Model / tokenizer construction shared by all four training stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
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


def inference_dtype(preferred: str | None = None) -> Any:
    """bfloat16 where the hardware supports it, float32 otherwise.

    Evaluation and diagnostics used to hardcode bfloat16, which is fine on an
    A100 and either very slow or unsupported on CPU.
    """
    if preferred and preferred != "bfloat16":
        return DTYPES[preferred]
    if torch.cuda.is_available() and native_bf16():
        return torch.bfloat16
    return torch.float32


def native_bf16() -> bool:
    """True when the GPU computes in bf16 natively (compute capability 8.0+).

    `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`
    and so also returns True on older GPUs such as the T4, where bf16 is
    emulated. Evaluation there ran in emulated bf16 even when the run itself
    trained in float32.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _is_adapter_dir(path: str) -> bool:
    p = Path(path)
    return (p / "adapter_config.json").exists() and not (p / "config.json").exists()


def _resolved_config(run: Path) -> Path | None:
    """The config.resolved.yaml that describes `run`.

    Trainer checkpoints (`<run>/checkpoint-<step>/`) hold adapter weights and
    trainer state but no config.resolved.yaml; they share the run's starting
    adapter, so the run directory's file applies to them. Without this, an
    intermediate DPO or GRPO checkpoint loads onto the raw base and silently
    drops the SFT adapter, the same failure as an unchained final run.
    """
    own = run / "config.resolved.yaml"
    if own.exists():
        return own
    if run.name.startswith("checkpoint-") and (run.parent / "config.resolved.yaml").exists():
        return run.parent / "config.resolved.yaml"
    return None


def adapter_chain(path: str) -> list[str]:
    """The adapters to apply, oldest first, to rebuild the model a run trained.

    DPO and GRPO start from the SFT adapter merged into the base
    (`load_policy`), then train a fresh adapter on top. The saved DPO adapter
    is therefore a delta on *SFT-merged* weights, while its adapter config
    names the raw base. Applying it to the raw base drops the SFT delta and
    scores a model that was never trained. Each run records its starting
    adapter as `init_adapter` in config.resolved.yaml, so the chain can be
    walked back to the base.
    """
    chain: list[str] = []
    current: str | None = path
    while current is not None:
        if not _is_adapter_dir(current):
            raise FileNotFoundError(
                f"{current} is not an adapter directory, but a run lists it as "
                f"its init_adapter; refusing to rebuild a different model."
            )
        if current in chain:
            raise ValueError(f"init_adapter cycle at {current}")
        chain.append(current)

        parent = None
        resolved = _resolved_config(Path(current))
        if resolved is not None:
            parent = (yaml.safe_load(resolved.read_text()) or {}).get("init_adapter")
        if parent and not Path(parent).exists():
            # Recorded relative to the working directory at training time;
            # also look beside the run directory.
            sibling = Path(current).parent / Path(parent).name
            if sibling.exists():
                parent = str(sibling)
        current = parent
    return list(reversed(chain))


def load_for_inference(
    path: str,
    base_model: str,
    *,
    head: str = "causal",
    dtype: Any = None,
    num_labels: int = 1,
) -> Any:
    """Load a run directory for inference, resolving LoRA adapters.

    `trainer.save_model()` under PEFT writes adapter weights only, so every run
    directory this repo produces holds an `adapter_config.json` and no
    `config.json`. `AutoModel*.from_pretrained` cannot open that: the base model
    has to be loaded first and the adapter applied on top. Passing the run
    directory straight to `from_pretrained` raises before a single prompt is
    scored, which is why `base_model` is threaded through to here.
    """
    dtype = dtype if dtype is not None else inference_dtype()
    device_map = "auto" if torch.cuda.is_available() else None

    if head == "causal":
        loader, kwargs = AutoModelForCausalLM, {}
    elif head == "sequence":
        loader, kwargs = AutoModelForSequenceClassification, {"num_labels": num_labels}
    else:
        raise ValueError(f"Unknown head: {head}")

    if not _is_adapter_dir(path):
        return loader.from_pretrained(path, dtype=dtype, device_map=device_map, **kwargs).eval()

    from peft import PeftModel

    model = loader.from_pretrained(base_model, dtype=dtype, device_map=device_map, **kwargs)
    for adapter in adapter_chain(path):
        logger.info("Applying and merging adapter %s", adapter)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.eval()


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
