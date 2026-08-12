"""Reward functions for GRPO.

The learned reward model is passed to TRL by path — TRL loads it as a
sequence-classification model and scores prompt+completion pairs on GPU.
The extra callables here are *shaping* terms, kept separate and separately
weighted so that any behaviour change can be attributed to a specific term.

Keeping a length penalty available (weight 0 by default) is deliberate: the
first thing an under-regularised policy learns on most reward models is to
write longer, and you want the knob in the repo when that happens rather than
inventing it after the fact.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .config import RunCfg


def _text_of(completion: Any) -> str:
    """GRPO passes completions as strings or as message lists depending on
    whether the dataset is conversational."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
    return str(completion)


def length_penalty(target_chars: int = 1200) -> Callable:
    """Smooth penalty for drifting away from a target length in either
    direction. Returns values in [-1, 0]."""

    def _fn(completions: list[Any], **kwargs: Any) -> list[float]:
        out = []
        for c in completions:
            n = len(_text_of(c))
            out.append(-abs(n - target_chars) / max(target_chars, 1))
        return [max(-1.0, v) for v in out]

    _fn.__name__ = "length_penalty"
    return _fn


def repetition_penalty(ngram: int = 4) -> Callable:
    """Penalise degenerate loops via distinct-n. Returns values in [-1, 0]."""

    def _fn(completions: list[Any], **kwargs: Any) -> list[float]:
        out = []
        for c in completions:
            tokens = re.findall(r"\w+", _text_of(c).lower())
            if len(tokens) < ngram:
                out.append(0.0)
                continue
            grams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
            distinct = len(set(grams)) / len(grams)
            out.append(distinct - 1.0)
        return out

    _fn.__name__ = "repetition_penalty"
    return _fn


def build_reward_funcs(cfg: RunCfg) -> tuple[list[Any], list[float]]:
    """Returns (reward_funcs, reward_weights) for GRPOConfig."""
    if not cfg.method.reward_model_path:
        raise ValueError(
            "method.reward_model_path is required for GRPO — train one with "
            "`python -m prefopt.train_rm --config configs/rm.yaml` first."
        )
    funcs: list[Any] = [cfg.method.reward_model_path]
    weights: list[float] = [1.0]

    if cfg.method.length_penalty_weight > 0:
        funcs.append(length_penalty())
        weights.append(cfg.method.length_penalty_weight)

    return funcs, weights
