"""Dataset construction for each stage.

Defaults target `HuggingFaceH4/ultrafeedback_binarized`, whose preference
splits carry `prompt`, `chosen`, and `rejected`, with the latter two as
message lists. TRL consumes that conversational format directly.

Every loader is deterministic given `cfg.seed`, and every subsample is taken
with an explicit seed so the RM-quality ablation compares like with like.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from .config import RunCfg
from .utils import get_logger

logger = get_logger(__name__)

PREFERENCE_COLUMNS = ["prompt", "chosen", "rejected"]


def _subsample(ds: Dataset, n: int | None, seed: int) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _keep_columns(ds: Dataset, columns: list[str]) -> Dataset:
    drop = [c for c in ds.column_names if c not in columns]
    return ds.remove_columns(drop) if drop else ds


def load_sft_dataset(cfg: RunCfg) -> dict[str, Dataset]:
    """SFT on the instruction-following split. Returns {'train', 'eval'}."""
    train = load_dataset(cfg.data.dataset, split=cfg.data.sft_train_split)
    train = _subsample(train, cfg.data.max_train_samples, cfg.seed)
    # Hold out from train so the SFT eval loss is not contaminated by the
    # preference splits we later evaluate win-rate on.
    split = train.train_test_split(test_size=0.02, seed=cfg.seed)
    logger.info("SFT: %d train / %d eval", len(split["train"]), len(split["test"]))
    return {"train": split["train"], "eval": split["test"]}


def load_preference_dataset(cfg: RunCfg, fraction: float = 1.0) -> dict[str, Dataset]:
    """Chosen/rejected pairs for the reward model and for DPO.

    `fraction` is the RM-quality ablation knob: an RM trained on 10% of pairs
    is a measurably worse reward signal, which is the point.
    """
    train = load_dataset(cfg.data.dataset, split=cfg.data.train_split)
    eval_ = load_dataset(cfg.data.dataset, split=cfg.data.eval_split)

    train = _keep_columns(train, PREFERENCE_COLUMNS)
    eval_ = _keep_columns(eval_, PREFERENCE_COLUMNS)

    train = _subsample(train, cfg.data.max_train_samples, cfg.seed)
    if fraction < 1.0:
        n = max(1, int(len(train) * fraction))
        train = _subsample(train, n, cfg.seed)  # same seed -> nested subsets
        logger.info("RM-quality ablation: using %.0f%% of pairs (%d)", fraction * 100, n)
    eval_ = _subsample(eval_, cfg.data.max_eval_samples, cfg.seed)

    logger.info("Preference: %d train / %d eval pairs", len(train), len(eval_))
    return {"train": train, "eval": eval_}


def load_prompt_dataset(cfg: RunCfg, split: str | None = None) -> Dataset:
    """Prompts only — GRPO samples its own completions, and the evaluator needs
    a held-out prompt set that no stage trained on."""
    ds = load_dataset(cfg.data.dataset, split=split or cfg.data.train_split)
    ds = _keep_columns(ds, ["prompt"])
    ds = _subsample(ds, cfg.data.max_train_samples, cfg.seed)
    return ds


def load_eval_prompts(cfg: RunCfg, n: int | None = None) -> list[str]:
    """Held-out prompts for win-rate evaluation, as plain strings."""
    ds = load_dataset(cfg.data.dataset, split=cfg.data.eval_split)
    ds = _subsample(ds, n or cfg.data.max_eval_samples, cfg.seed + 1)
    return [row["prompt"] for row in ds]


def to_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Apply the model's chat template to a bare user prompt."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
