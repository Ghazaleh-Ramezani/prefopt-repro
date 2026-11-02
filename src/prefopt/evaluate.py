"""Win-rate evaluation against the SFT baseline.

Four choices here are what make the number defensible:

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
4. **Judge diagnostics.** Swap consistency says how often the judge agrees
   with itself, not which way it leans when it does not. `judge_diagnostics`
   reports the tie rate in each order, the parse rate, and a slot coefficient
   estimated from both orders, and the summary records the judge's format and
   decoding settings verbatim. A win-rate means little without them: the same
   pairs scored under a different format or temperature give a different number.

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
    for a few hundred prompts), otherwise falls back to HF generate.

    vLLM is used only for a full model directory. A run directory holds a LoRA
    adapter, possibly on top of an SFT adapter, and only `load_for_inference`
    rebuilds that chain; handing the directory to vLLM would fail on the
    missing config.json, and not with an ImportError, so there would be no
    fallback.
    """
    from .modeling import _is_adapter_dir, load_for_inference

    if not _is_adapter_dir(model_path):
        try:
            from vllm import LLM, SamplingParams  # type: ignore

            llm = LLM(model=model_path, seed=seed, enable_lora=False)
            params = SamplingParams(
                temperature=temperature, top_p=0.95, max_tokens=max_new_tokens, seed=seed
            )
            outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
            return [o.outputs[0].text.strip() for o in outputs]
        except ImportError:
            logger.info("vLLM not available; falling back to transformers.generate")

    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # Run directories hold a LoRA adapter, not a full model; `base_model` is
    # what makes them loadable.
    model = load_for_inference(model_path, base_model, head="causal")

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
        # Some fast tokenizers emit token_type_ids; a causal LM's generate()
        # rejects unknown kwargs outright rather than ignoring them.
        enc.pop("token_type_ids", None)
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


VALID_VERDICTS = ("A", "B", "TIE")


def _parse_verdict(text: str) -> str:
    """Return "A", "B" or "TIE" only when the judge's first token is exactly
    one of them; anything else is "INVALID".

    Matching on a prefix turned "Both are fine" into a B win and "Answer A"
    into an A win. An unparseable verdict is scored like a position flip — a
    tie — by win_rate, and counted, so a judge that stops following the
    format shows up in the results instead of silently shifting them.
    """
    match = re.match(r"\s*([A-Za-z]+)", text or "")
    token = match.group(1).upper() if match else ""
    return token if token in VALID_VERDICTS else "INVALID"


@dataclass
class AnthropicJudge:
    """Pairwise LLM judge with greedy decoding.

    The temperature is passed explicitly. Left out, the Messages API samples at
    temperature 1.0, so each verdict was a draw rather than the judge's
    preference: repeated calls on the same pair could disagree, which lowers
    swap consistency and turns flips into ties for reasons unrelated to
    position. Nothing errors when that happens.
    """

    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 8

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "llm_pairwise",
            "model": self.model,
            "prompt": JUDGE_PROMPT,
            "verdicts": list(VALID_VERDICTS),
            "decoding": {"temperature": self.temperature, "max_tokens": self.max_tokens},
        }

    def compare(self, prompt: str, a: str, b: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
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
    base_model: str | None = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoTokenizer

        from .modeling import load_for_inference

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.rm_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"  # the scalar head reads the last non-pad token
        # The RM run directory is an adapter too, so it needs the same
        # base-model resolution the policy does.
        self.model = load_for_inference(
            self.rm_path, self.base_model or self.rm_path, head="sequence", num_labels=1
        )
        self.model.config.pad_token_id = self.tok.pad_token_id

    def format(self, prompt: str, response: str) -> str:
        """The text the reward model scores: the conversation rendered through
        the tokenizer's chat template, which is what it was trained on.

        TRL's RewardTrainer renders conversational chosen/rejected pairs with
        the chat template before tokenizing, so the scalar head has only ever
        seen chat-formatted conversations. Scoring a bare "prompt\\n\\nresponse"
        string instead asks it about text outside its training distribution.
        Nothing errors; the scores are just a noisier and differently biased
        signal, which moves the win-rate, the length-reward correlation and
        the Goodhart gap. Public reward models such as Skywork-Reward-V2
        document the same chat-template input. The plain join is kept only for
        tokenizers that have no chat template.
        """
        if not getattr(self.tok, "chat_template", None):
            return f"{prompt}\n\n{response}"
        text = self.tok.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        )
        return text

    def score(self, prompt: str, response: str) -> float:
        enc = self.tok(
            self.format(prompt, response),
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        with self._torch.no_grad():
            return float(self.model(**enc).logits[0][0])

    def describe(self) -> dict[str, Any]:
        # Scored one response at a time, so slot position never enters the
        # verdict: swap consistency and the slot coefficient are trivially 1.0
        # and 0 under this judge and say nothing about it.
        return {
            "kind": "reward_model_pointwise",
            "rm_path": self.rm_path,
            "input_format": (
                "chat_template" if getattr(self.tok, "chat_template", None) else "plain_join"
            ),
            "tie_threshold": RM_TIE_THRESHOLD,
        }

    def compare(self, prompt: str, a: str, b: str) -> str:
        sa, sb = self.score(prompt, a), self.score(prompt, b)
        if abs(sa - sb) < RM_TIE_THRESHOLD:
            return "TIE"
        return "A" if sa > sb else "B"


RM_TIE_THRESHOLD = 1e-3


def build_judge(
    kind: str,
    rm_path: str | None,
    model: str | None,
    base_model: str | None = None,
    temperature: float = 0.0,
) -> Judge:
    if kind == "anthropic":
        return AnthropicJudge(model=model or "claude-sonnet-4-6", temperature=temperature)
    if kind == "rm":
        if not rm_path:
            raise ValueError("--rm-path is required for --judge rm")
        return RewardModelJudge(rm_path=rm_path, base_model=base_model)
    raise ValueError(f"Unknown judge: {kind}")


# --------------------------------------------------------------------------- #
# Judge diagnostics
# --------------------------------------------------------------------------- #


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def judge_diagnostics(
    records: list[dict[str, Any]],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    smoothing: float = 0.5,
) -> dict[str, Any]:
    """Tie rate, parse rate and slot bias of a pairwise judge, from the
    per-prompt verdicts `win_rate` records in both orders.

    The slot coefficient comes from a Bradley-Terry model with an additive
    position term,

        P(judge names slot A) = sigmoid(theta_A - theta_B + b),

    fitted on decisive verdicts (A or B; ties and invalid replies excluded).
    With the policy in slot A the logit of P(A) is d + b, and with the policy
    in slot B it is -d + b, where d = theta_policy - theta_baseline. So

        b = (logit p_AB + logit p_BA) / 2,    d = (logit p_AB - logit p_BA) / 2,

    with p_AB the share of decisive verdicts naming A when the policy is shown
    first and p_BA the same share with the order swapped. b > 0 leans toward
    slot A, b < 0 toward slot B. d is the quality difference with the position
    effect taken out; it is pooled over prompts, so with heterogeneous prompts
    it is attenuated toward zero, but b keeps its sign and scale.

    Counts are smoothed by `smoothing` per outcome (0.5 by default), so a judge
    that never names one slot still gets a finite b. A judge that agrees with
    itself on every pair gets b = 0 exactly, and so does any pointwise judge,
    such as a reward model, for which the coefficient is uninformative.

    The intervals resample prompts, the independent unit here. `b` is None when
    either order has no decisive verdict, because then nothing separates the
    position effect from the quality difference.
    """
    import numpy as np

    n = len(records)
    ab = np.array([r["verdict_ab"] for r in records], dtype=object)
    ba = np.array([r["verdict_ba"] for r in records], dtype=object)

    def order_stats(verdicts: Any) -> dict[str, Any]:
        decisive = int(np.isin(verdicts, ["A", "B"]).sum())
        return {
            "n": int(verdicts.size),
            "tie_rate": _rate(int((verdicts == "TIE").sum()), int(verdicts.size)),
            "invalid_rate": _rate(int((verdicts == "INVALID").sum()), int(verdicts.size)),
            "n_decisive": decisive,
            "p_slot_a": _rate(int((verdicts == "A").sum()), decisive),
        }

    out: dict[str, Any] = {
        "n_pairs": n,
        "order_ab": order_stats(ab),
        "order_ba": order_stats(ba),
        "verdict_tie_rate": _rate(int((ab == "TIE").sum() + (ba == "TIE").sum()), 2 * n),
        "parse_rate": (
            1.0 - int((ab == "INVALID").sum() + (ba == "INVALID").sum()) / (2 * n) if n else None
        ),
        "slot_a_rate": None,
        "slot_bias": None,
        "slot_bias_ci95": None,
        "quality_logit": None,
        "quality_logit_ci95": None,
        "smoothing": smoothing,
    }

    a1 = (ab == "A").astype(float)
    d1 = np.isin(ab, ["A", "B"]).astype(float)
    a2 = (ba == "A").astype(float)
    d2 = np.isin(ba, ["A", "B"]).astype(float)
    n1, n2 = d1.sum(), d2.sum()
    if n1 + n2:
        out["slot_a_rate"] = float((a1.sum() + a2.sum()) / (n1 + n2))
    if not (n1 and n2):
        return out

    def logit(k: Any, m: Any) -> Any:
        p = (k + smoothing) / (m + 2 * smoothing)
        return np.log(p / (1.0 - p))

    def estimate(k1: Any, m1: Any, k2: Any, m2: Any) -> tuple[Any, Any]:
        l1, l2 = logit(k1, m1), logit(k2, m2)
        return (l1 + l2) / 2.0, (l1 - l2) / 2.0

    b, d = estimate(a1.sum(), n1, a2.sum(), n2)

    # Resampling prompts only changes how many prompts fall into each
    # (verdict_ab, verdict_ba) combination, so a multinomial draw over the
    # combinations is the same bootstrap without materialising n_boot x n
    # indices.
    keys = sorted({(x, y) for x, y in zip(ab, ba)})
    counts = np.array([sum(1 for x, y in zip(ab, ba) if (x, y) == k) for k in keys])
    flags = np.array(
        [[x == "A", x in ("A", "B"), y == "A", y in ("A", "B")] for x, y in keys], dtype=float
    )
    rng = np.random.default_rng(seed)
    totals = rng.multinomial(n, counts / n, size=n_boot) @ flags
    bb, dd = estimate(totals[:, 0], totals[:, 1], totals[:, 2], totals[:, 3])
    q = [alpha / 2, 1 - alpha / 2]
    out["slot_bias"] = float(b)
    out["slot_bias_ci95"] = [float(v) for v in np.quantile(bb, q)]
    out["quality_logit"] = float(d)
    out["quality_logit_ci95"] = [float(v) for v in np.quantile(dd, q)]
    return out


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
    if not (len(prompts) == len(policy) == len(baseline)):
        # zip() would silently truncate to the shortest list, so a generation
        # step that dropped outputs would yield a win-rate on fewer prompts
        # than reported.
        raise ValueError(
            f"prompts, policy and baseline must align: got {len(prompts)}, "
            f"{len(policy)}, {len(baseline)}"
        )
    scores: list[float] = []
    consistent = 0
    invalid = 0
    records = []

    for prompt, pol, base in zip(prompts, policy, baseline):
        first = judge.compare(prompt, pol, base)  # policy as A
        second = judge.compare(prompt, base, pol)  # policy as B
        policy_won_first = first == "A"
        policy_won_second = second == "B"

        if "INVALID" in (first, second):
            # Checked first: two unparseable verdicts would otherwise satisfy
            # the agreement test below and be scored as a consistent loss.
            invalid += 1
            score = 0.5
        elif policy_won_first == policy_won_second and first != "TIE" and second != "TIE":
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
        "invalid_verdicts": invalid,
        "judge_diagnostics": judge_diagnostics(records),
        "mean_len_policy": sum(len(p) for p in policy) / max(len(policy), 1),
        "mean_len_baseline": sum(len(b) for b in baseline) / max(len(baseline), 1),
        "records": records,
    }


def _fmt_opt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Win-rate evaluation")
    parser.add_argument("--config", required=True)
    # The training stages take `--set` through cli.parse_and_prepare. Without
    # the same flag here, a run trained with overrides cannot be evaluated with
    # the settings it was trained under: this parser would reject the flag, and
    # the harness would silently read base_model from the YAML instead.
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-path override, e.g. --set model.base_model=... (repeatable).",
    )
    parser.add_argument("--policy", required=True, help="Path to the policy checkpoint.")
    parser.add_argument("--baseline", required=True, help="Path to the SFT baseline.")
    parser.add_argument("--judge", default="anthropic", choices=["anthropic", "rm"])
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Decoding temperature of the LLM judge (default greedy).",
    )
    parser.add_argument("--rm-path", default=None)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    prompts = load_eval_prompts(cfg, n=args.n)
    judge = build_judge(
        args.judge,
        args.rm_path,
        args.judge_model,
        cfg.model.base_model,
        temperature=args.judge_temperature,
    )

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
        diag = result["judge_diagnostics"]
        logger.info(
            "seed %d: win-rate %.3f [%.3f, %.3f] | swap-consistency %.3f | "
            "slot bias %s | verdict tie rate %s | parse rate %s",
            seed,
            result["win_rate"],
            result["ci95"][0],
            result["ci95"][1],
            result["swap_consistency"],
            _fmt_opt(diag["slot_bias"]),
            _fmt_opt(diag["verdict_tie_rate"]),
            _fmt_opt(diag["parse_rate"]),
        )

    rates = [r["win_rate"] for r in per_seed]
    summary = {
        "policy": args.policy,
        "baseline": args.baseline,
        "judge": args.judge,
        "judge_model": args.judge_model,
        "judge_format": judge.describe() if hasattr(judge, "describe") else None,
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
