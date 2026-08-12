"""Win-rate evaluation against the SFT baseline.

Three choices here are what make the number defensible:

1. **Position swap.** Every pair is judged twice, A/B and B/A. A win counts
   only when the judge agrees with itself; disagreement is scored as a tie and
   tracked as `swap_consistency`. Judges have a strong position bias and an
   unswapped win-rate mostly measures that.
2. **Bootstrap CI.** A 300-prompt win-rate has roughly a +/-6 point 95% CI.
   Reporting 61.2% without an interval invites a reviewer to assume you do not
   know that.
3. **Multi-seed generation.** Sampling temperature moves win-rate by more than
   most method deltas, so the harness repeats generation across seeds and
   reports the spread.

    python -m prefopt.evaluate --policy runs/dpo_b0.1 --baseline runs/sft \
        --config configs/dpo.yaml --judge anthropic --n 300 --seeds 3
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import load_config
from .data import load_eval_prompts
from .utils import bootstrap_ci, environment_metadata, get_logger, save_json, set_seed

logger = get_logger(__name__)

JUDGE_PROMPT = """You are comparing two assistant responses to the same user request.

[User request]
{prompt}

[Response A]
{a}

[Response B]
{b}

Judge which response is more helpful, correct, and appropriately concise.
Length alone is not quality: penalise padding, restatement, and filler.
Reply with exactly one token: A, B, or TIE."""


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate(
    model_path: str,
    prompts: list[str],
    base_model: str,
    seed: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    batch_size: int = 16,
) -> list[str]:
    """Sample one completion per prompt. Uses vLLM when installed (much faster
    for a few hundred prompts), otherwise falls back to HF generate."""
    try:
        from vllm import LLM, SamplingParams  # type: ignore

        llm = LLM(model=model_path, seed=seed, enable_lora=False)
        params = SamplingParams(
            temperature=temperature, top_p=0.95, max_tokens=max_new_tokens, seed=seed
        )
        outputs = llm.chat(
            [[{"role": "user", "content": p}] for p in prompts], params
        )
        return [o.outputs[0].text.strip() for o in outputs]
    except ImportError:
        logger.info("vLLM not available; falling back to transformers.generate")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    set_seed(seed)
    completions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True).to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        for i in range(len(chunk)):
            new_tokens = out[i][enc["input_ids"].shape[1] :]
            completions.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
    return completions


# --------------------------------------------------------------------------- #
# Judges
# --------------------------------------------------------------------------- #


class Judge(Protocol):
    def compare(self, prompt: str, a: str, b: str) -> str:
        """Returns 'A', 'B', or 'TIE'."""


def _parse_verdict(text: str) -> str:
    token = re.sub(r"[^A-Za-z]", "", text.strip().upper())[:3]
    if token.startswith("A"):
        return "A"
    if token.startswith("B"):
        return "B"
    return "TIE"


@dataclass
class AnthropicJudge:
    model: str = "claude-sonnet-4-6"

    def compare(self, prompt: str, a: str, b: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=self.model,
            max_tokens=8,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(prompt=prompt, a=a, b=b),
                }
            ],
        )
        return _parse_verdict("".join(b_.text for b_ in msg.content if b_.type == "text"))


@dataclass
class RewardModelJudge:
    """Free, fast, and circular: scoring a GRPO policy with the same RM it was
    trained against measures optimisation, not quality. Useful as a smoke test
    and as the second axis of the Goodhart plot — never as the headline."""

    rm_path: str

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.rm_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.rm_path, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

    def score(self, prompt: str, response: str) -> float:
        text = f"{prompt}\n\n{response}"
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=2048).to(
            self.model.device
        )
        with self._torch.no_grad():
            return float(self.model(**enc).logits[0][0])

    def compare(self, prompt: str, a: str, b: str) -> str:
        sa, sb = self.score(prompt, a), self.score(prompt, b)
        if abs(sa - sb) < 1e-3:
            return "TIE"
        return "A" if sa > sb else "B"


def build_judge(kind: str, rm_path: str | None, model: str | None) -> Judge:
    if kind == "anthropic":
        return AnthropicJudge(model=model or "claude-sonnet-4-6")
    if kind == "rm":
        if not rm_path:
            raise ValueError("--rm-path is required for --judge rm")
        return RewardModelJudge(rm_path=rm_path)
    raise ValueError(f"Unknown judge: {kind}")


# --------------------------------------------------------------------------- #
# Win-rate
# --------------------------------------------------------------------------- #


def win_rate(
    prompts: list[str],
    policy: list[str],
    baseline: list[str],
    judge: Judge,
) -> dict[str, Any]:
    """Position-swapped pairwise win-rate. Score is 1.0 win / 0.5 tie / 0.0 loss."""
    scores: list[float] = []
    consistent = 0
    records = []

    for prompt, pol, base in zip(prompts, policy, baseline):
        first = judge.compare(prompt, pol, base)  # policy as A
        second = judge.compare(prompt, base, pol)  # policy as B
        policy_won_first = first == "A"
        policy_won_second = second == "B"

        if policy_won_first == policy_won_second and first != "TIE" and second != "TIE":
            consistent += 1
            score = 1.0 if policy_won_first else 0.0
        elif first == "TIE" and second == "TIE":
            consistent += 1
            score = 0.5
        else:
            score = 0.5  # judge flipped with position: treat as a tie
        scores.append(score)
        records.append(
            {
                "prompt": prompt[:200],
                "verdict_ab": first,
                "verdict_ba": second,
                "score": score,
                "len_policy": len(pol),
                "len_baseline": len(base),
            }
        )

    mean, lo, hi = bootstrap_ci(scores)
    return {
        "win_rate": mean,
        "ci95": [lo, hi],
        "n": len(scores),
        "wins": sum(1 for s in scores if s == 1.0),
        "ties": sum(1 for s in scores if s == 0.5),
        "losses": sum(1 for s in scores if s == 0.0),
        "swap_consistency": consistent / max(len(scores), 1),
        "mean_len_policy": sum(len(p) for p in policy) / max(len(policy), 1),
        "mean_len_baseline": sum(len(b) for b in baseline) / max(len(baseline), 1),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Win-rate evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", required=True, help="Path to the policy checkpoint.")
    parser.add_argument("--baseline", required=True, help="Path to the SFT baseline.")
    parser.add_argument("--judge", default="anthropic", choices=["anthropic", "rm"])
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--rm-path", default=None)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    prompts = load_eval_prompts(cfg, n=args.n)
    judge = build_judge(args.judge, args.rm_path, args.judge_model)

    per_seed = []
    for seed in range(args.seeds):
        logger.info("Generation seed %d/%d", seed + 1, args.seeds)
        pol = generate(
            args.policy, prompts, cfg.model.base_model, seed=seed, temperature=args.temperature
        )
        base = generate(
            args.baseline, prompts, cfg.model.base_model, seed=seed, temperature=args.temperature
        )
        result = win_rate(prompts, pol, base, judge)
        result["seed"] = seed
        per_seed.append(result)
        logger.info(
            "seed %d: win-rate %.3f [%.3f, %.3f] | swap-consistency %.3f",
            seed,
            result["win_rate"],
            result["ci95"][0],
            result["ci95"][1],
            result["swap_consistency"],
        )

    rates = [r["win_rate"] for r in per_seed]
    summary = {
        "policy": args.policy,
        "baseline": args.baseline,
        "judge": args.judge,
        "judge_model": args.judge_model,
        "n_prompts": len(prompts),
        "seeds": args.seeds,
        "win_rate_mean": sum(rates) / len(rates),
        "win_rate_min": min(rates),
        "win_rate_max": max(rates),
        "per_seed": per_seed,
        "environment": environment_metadata(),
    }
    out = Path(args.out or Path(args.policy) / "results.json")
    save_json(summary, out)
    logger.info(
        "Win-rate %.1f%% (seed range %.1f–%.1f) -> %s",
        100 * summary["win_rate_mean"],
        100 * summary["win_rate_min"],
        100 * summary["win_rate_max"],
        out,
    )


if __name__ == "__main__":
    main()
