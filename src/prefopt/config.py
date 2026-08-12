"""Typed configuration: YAML files with single-level inheritance + CLI overrides.

Every run is fully described by one resolved config, which is written to
`<output_dir>/config.resolved.yaml` so a run can be replayed exactly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class ModelCfg:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    use_peft: bool = True
    load_in_4bit: bool = False  # QLoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class DataCfg:
    dataset: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    eval_split: str = "test_prefs"
    sft_train_split: str = "train_sft"
    max_train_samples: int | None = 20000
    max_eval_samples: int | None = 500
    max_prompt_length: int = 768
    max_length: int = 1536
    # Reward-model-quality ablation axis: train the RM on a fraction of pairs.
    rm_pair_fraction: float = 1.0


@dataclass
class TrainCfg:
    output_dir: str = "runs/unnamed"
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 2
    report_to: str = "none"  # "wandb" once you have a project set up
    deepspeed: str | None = None  # e.g. "ds_zero3.json"


@dataclass
class MethodCfg:
    """Method-specific knobs. Only the fields relevant to the stage are read."""

    # DPO / GRPO shared: strength of the KL anchor to the reference policy.
    beta: float = 0.1
    # DPO
    loss_type: str = "sigmoid"  # sigmoid | ipo | hinge | robust
    label_smoothing: float = 0.0
    # GRPO
    num_generations: int = 8  # group size G
    max_completion_length: int = 512
    temperature: float = 0.9
    use_vllm: bool = False
    reward_model_path: str | None = None
    length_penalty_weight: float = 0.0  # >0 adds an explicit anti-verbosity term
    # Reward model
    rm_num_labels: int = 1
    # SFT
    packing: bool = True


@dataclass
class RunCfg:
    name: str = "unnamed"
    stage: str = "sft"  # sft | rm | dpo | grpo
    seed: int = 17
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    method: MethodCfg = field(default_factory=MethodCfg)
    # Path to the SFT adapter that DPO/GRPO/RM start from.
    init_adapter: str | None = None
    notes: str = ""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(raw: str) -> Any:
    """Turn a CLI override string into a Python scalar."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_override(tree: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _from_dict(cls: type, data: dict) -> Any:
    """Build a *flat* dataclass from a dict, rejecting unknown keys.

    Rejecting unknown keys is deliberate: a typo in an ablation config should
    fail loudly at startup, not silently run the default and pollute results.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunCfg:
    """Load a YAML config, resolving an optional `_base_` parent, then apply
    `--set key.subkey=value` overrides."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base_raw = yaml.safe_load(base_path.read_text()) or {}
        base_raw.pop("_base_", None)
        raw = _deep_merge(base_raw, raw)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, _, value = item.partition("=")
        _apply_override(raw, key.strip(), _coerce(value.strip()))

    # Nested dataclasses need explicit construction.
    cfg = RunCfg(
        name=raw.get("name", "unnamed"),
        stage=raw.get("stage", "sft"),
        seed=raw.get("seed", 17),
        model=_from_dict(ModelCfg, raw.get("model", {})),
        data=_from_dict(DataCfg, raw.get("data", {})),
        train=_from_dict(TrainCfg, raw.get("train", {})),
        method=_from_dict(MethodCfg, raw.get("method", {})),
        init_adapter=raw.get("init_adapter"),
        notes=raw.get("notes", ""),
    )
    return cfg


def to_dict(cfg: Any) -> dict:
    if not is_dataclass(cfg):
        return cfg
    return {f.name: to_dict(getattr(cfg, f.name)) for f in fields(cfg)}


def save_config(cfg: RunCfg, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "config.resolved.yaml"
    target.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
    return target
