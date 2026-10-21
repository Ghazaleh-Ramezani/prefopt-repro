"""Goodhart gap at every checkpoint of a DPO or GRPO run.

For the reference policy (step 0) and each saved checkpoint, the same prompts
and generation seeds are judged twice against the reference, both times with
position swapping:

  proxy_win_rate      judged by the reward model the policy was optimised
                      against (for DPO, the reward model of the same
                      preference data)
  external_win_rate   judged by an independent judge: the LLM judge, or a
                      reward model trained separately
  goodhart_gap        proxy_win_rate - external_win_rate

Each row also carries KL from the reference, mean completion length and the
length-reward correlation under the proxy. A gap that widens as KL grows is
the over-optimisation signature: the proxy keeps improving while the
independent judge does not.

    python -m prefopt.goodhart --config configs/grpo.yaml --run runs/grpo_g8 \
        --reference runs/sft --proxy-rm runs/rm_full --judge anthropic --n 300 --seeds 3

Writes <run>/goodhart.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .config import load_config
from .data import load_eval_prompts
from .utils import environment_metadata, get_logger, save_json

logger = get_logger(__name__)

_CKPT = re.compile(r"^checkpoint-(\d+)$")


def list_checkpoints(run_dir: str | Path) -> list[tuple[int | None, Path]]:
    """Saved checkpoints of a run, oldest first, as (global step, path).

    The final run directory is appended when its weights are not already a
    checkpoint, i.e. when no checkpoint sits at the run's last step.
    """
    run = Path(run_dir)
    ckpts = sorted(
        ((int(m.group(1)), p) for p in run.iterdir() if p.is_dir() and (m := _CKPT.match(p.name))),
        key=lambda t: t[0],
    )
    max_steps = None
    if ckpts:
        state = ckpts[-1][1] / "trainer_state.json"
        if state.exists():
            max_steps = json.loads(state.read_text()).get("max_steps")
    out: list[tuple[int | None, Path]] = list(ckpts)
    if (run / "adapter_config.json").exists() and (not ckpts or ckpts[-1][0] != max_steps):
        out.append((max_steps, run))
    return out


def _spearman(x: list[float], y: list[float]) -> float | None:
    from .diagnostics import spearman

    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return spearman(x, y)


def judge_pair(prompts, policy_out, reference_out, proxy, external) -> dict[str, Any]:
    """Proxy and external position-swapped win-rates for one set of completions."""
    from .evaluate import win_rate

    p = win_rate(prompts, policy_out, reference_out, proxy)
    e = win_rate(prompts, policy_out, reference_out, external)
    return {
        "proxy_win_rate": p["win_rate"],
        "external_win_rate": e["win_rate"],
        "goodhart_gap": p["win_rate"] - e["win_rate"],
        "external_swap_consistency": e["swap_consistency"],
        "external_invalid_verdicts": e["invalid_verdicts"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [r["step"] for r in rows if r["step"] is not None]
    gaps = [r["goodhart_gap"] for r in rows if r["step"] is not None]
    kls = [r["kl_from_reference"] for r in rows if r["step"] is not None and r.get("kl_from_reference") is not None]
    worst = max(rows, key=lambda r: r["goodhart_gap"])
    return {
        "n_points": len(rows),
        "gap_first": rows[0]["goodhart_gap"],
        "gap_last": rows[-1]["goodhart_gap"],
        "gap_max": worst["goodhart_gap"],
        "gap_max_step": worst["step"],
        "spearman_step_gap": _spearman([float(s) for s in steps], gaps),
        "spearman_kl_gap": _spearman(kls, gaps[-len(kls):]) if kls else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Goodhart gap at every checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run", required=True, help="DPO or GRPO run directory.")
    parser.add_argument("--reference", required=True, help="Reference policy, usually the SFT run.")
    parser.add_argument("--proxy-rm", required=True, help="Reward model the policy was optimised against.")
    parser.add_argument("--judge", default="anthropic", choices=["anthropic", "rm"])
    parser.add_argument("--judge-rm", default=None, help="Independent reward model, for --judge rm.")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip-kl", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from .diagnostics import mean_token_kl, spearman
    from .evaluate import RewardModelJudge, build_judge, generate

    if args.judge == "rm" and Path(args.judge_rm or "").resolve() == Path(args.proxy_rm).resolve():
        raise ValueError("the external judge must differ from the proxy reward model")

    cfg = load_config(args.config, args.overrides)
    base = cfg.model.base_model
    prompts = load_eval_prompts(cfg, n=args.n)
    proxy = RewardModelJudge(rm_path=args.proxy_rm, base_model=base)
    external = build_judge(args.judge, args.judge_rm, args.judge_model, base)

    reference_out = {
        s: generate(args.reference, prompts, base, seed=s, temperature=args.temperature) for s in range(args.seeds)
    }
    points: list[tuple[int | None, Path]] = [(0, Path(args.reference))] + list_checkpoints(args.run)

    rows = []
    for step, path in points:
        logger.info("step %s: %s", step, path)
        per_seed, lengths, scores = [], [], []
        for s in range(args.seeds):
            out = reference_out[s] if step == 0 else generate(str(path), prompts, base, seed=s, temperature=args.temperature)
            per_seed.append(judge_pair(prompts, out, reference_out[s], proxy, external))
            if s == 0:
                seed0 = out
                lengths = [len(c) for c in out]
                scores = [proxy.score(p, c) for p, c in zip(prompts, out)]
        row: dict[str, Any] = {
            "step": step,
            "checkpoint": path.name,
            **{k: sum(r[k] for r in per_seed) / len(per_seed) for k in per_seed[0]},
            "mean_chars": sum(lengths) / max(len(lengths), 1),
            "length_reward_corr": spearman(lengths, scores) if len(set(lengths)) > 1 else None,
        }
        if not args.skip_kl:
            row["kl_from_reference"] = 0.0 if step == 0 else mean_token_kl(
                str(path), args.reference, prompts, seed0, cfg.data.max_length, base_model=base
            )
        rows.append(row)
        logger.info(
            "step %s  proxy %.3f  external %.3f  gap %+.3f",
            step, row["proxy_win_rate"], row["external_win_rate"], row["goodhart_gap"],
        )

    report = {
        "run": args.run,
        "reference": args.reference,
        "proxy_reward_model": args.proxy_rm,
        "external_judge": f"rm:{args.judge_rm}" if args.judge == "rm" else f"{args.judge}:{args.judge_model or 'default'}",
        "n_prompts": len(prompts),
        "seeds": args.seeds,
        "rows": rows,
        "summary": summarize(rows),
        "environment": environment_metadata(),
    }
    out = Path(args.out or Path(args.run) / "goodhart.json")
    save_json(report, out)
    logger.info("Goodhart table -> %s", out)


if __name__ == "__main__":
    main()
