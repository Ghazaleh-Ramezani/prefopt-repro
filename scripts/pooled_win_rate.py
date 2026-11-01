#!/usr/bin/env python3
"""Pool a multi-seed win-rate, using the prompt as the resampling unit.

`prefopt.evaluate` bootstraps inside a single generation seed, so a run with
several seeds produces one interval per seed and `prefopt.report` prints their
envelope. That envelope is not a confidence interval for the pooled estimate.
It is wider than the data warrants, because it treats the gap between seed
means as estimation error when averaging over seeds is exactly what removes it.

Pooling the other way — stacking every comparison into one sample of 2N — errs
in the opposite direction. The prompts are shared across seeds, so those
comparisons are not independent and the interval comes out too narrow.

The unit that is independent here is the prompt. This script averages each
prompt's score across seeds, then resamples prompts. The point estimate is
unchanged; only the interval differs, and it is the interval the design
supports.

What it does not fix: with a small number of seeds the per-prompt average
still carries generation noise, and seed-level variance cannot be estimated
from two seeds. Read the seed spread printed alongside the interval before
treating a narrow interval as a settled result.

Reads either artifact: the `results.json` that `prefopt.evaluate` writes, or
the `per_prompt_scores.json` beside it, which is the same scores with the
prompt text and per-order verdicts dropped.

Usage:
    python scripts/pooled_win_rate.py results/small_run/dpo_b0.1/per_prompt_scores.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def pooled_win_rate(
    results_path: Path,
    n_boot: int = 20000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    payload = json.loads(results_path.read_text())

    if "scores" in payload:
        # per_prompt_scores.json: one row per prompt, one column per seed.
        rows = payload["scores"]
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            raise ValueError(f"ragged score rows in {results_path}: {sorted(widths)}")
        seed_win_rates = [s["win_rate"] for s in payload.get("per_seed_summary", [])]
        n_seeds = widths.pop()
    else:
        # results.json as prefopt.evaluate writes it.
        per_seed = payload.get("per_seed")
        if not per_seed:
            raise ValueError(f"{results_path} contains neither 'scores' nor 'per_seed'")
        record_lists = [s["records"] for s in per_seed]
        lengths = {len(r) for r in record_lists}
        if len(lengths) != 1:
            raise ValueError(
                "seeds cover different numbers of prompts "
                f"({sorted(lengths)}); they must share one prompt set to pool this way"
            )
        rows = [
            [records[i]["score"] for records in record_lists]
            for i in range(len(record_lists[0]))
        ]
        seed_win_rates = [s["win_rate"] for s in per_seed]
        n_seeds = len(per_seed)

    # One score per prompt, averaged over seeds. Ties already arrive as 0.5.
    per_prompt = [statistics.mean(row) for row in rows]

    point = statistics.mean(per_prompt)

    rng = random.Random(seed)
    n = len(per_prompt)
    draws = sorted(
        statistics.mean(rng.choices(per_prompt, k=n)) for _ in range(n_boot)
    )
    lo = draws[int((alpha / 2) * n_boot)]
    hi = draws[int((1 - alpha / 2) * n_boot)]

    return {
        "policy": payload.get("policy"),
        "baseline": payload.get("baseline"),
        "judge": payload.get("judge"),
        "n_prompts": n,
        "n_seeds": n_seeds,
        "win_rate": point,
        "ci95": [lo, hi],
        "seed_win_rates": seed_win_rates,
        "p_le_half": sum(1 for d in draws if d <= 0.5) / n_boot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="path to a results.json")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    out = pooled_win_rate(args.results, args.bootstrap, args.alpha, args.seed)

    if args.json:
        print(json.dumps(out, indent=2))
        return

    lo, hi = out["ci95"]
    spread = ", ".join(f"{w:.3f}" for w in out["seed_win_rates"]) or "n/a"
    print(f"policy     {out['policy']}")
    print(f"baseline   {out['baseline']}")
    print(f"judge      {out['judge']}")
    print(f"prompts    {out['n_prompts']} x {out['n_seeds']} seeds")
    print(f"win-rate   {out['win_rate']:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"seeds      {spread}")
    print(f"P(<=0.5)   {out['p_le_half']:.4f}")


if __name__ == "__main__":
    main()
