"""Tie rate, parse rate and slot bias of the pairwise judge, and the decoding
settings the LLM judge sends.

These import the evaluation module, which pulls in `datasets`; they are skipped
on a core-only install.
"""

from __future__ import annotations

import math
import random
import sys
import types

import pytest

pytest.importorskip("datasets")

from prefopt.evaluate import AnthropicJudge, judge_diagnostics, win_rate  # noqa: E402

POLICY, BASELINE = "policy response", "baseline response"


def _records(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"verdict_ab": ab, "verdict_ba": ba} for ab, ba in pairs]


class _BradleyTerryJudge:
    """Names slot A with probability sigmoid(theta_A - theta_B + b), where the
    policy is worth `d` logits more than the baseline."""

    def __init__(self, b: float, d: float, seed: int) -> None:
        self.b, self.d = b, d
        self.rng = random.Random(seed)

    def compare(self, prompt: str, a: str, b: str) -> str:
        delta = self.d if a == POLICY else -self.d
        p_a = 1.0 / (1.0 + math.exp(-(delta + self.b)))
        return "A" if self.rng.random() < p_a else "B"


class _PointwiseJudge:
    """Scores each response on its own, like a reward model."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def compare(self, prompt: str, a: str, b: str) -> str:
        sa, sb = self.scores[a], self.scores[b]
        if sa == sb:
            return "TIE"
        return "A" if sa > sb else "B"


def test_self_consistent_judge_has_no_slot_bias():
    # The policy is named in both orders on every pair: no lean toward a slot,
    # and a positive quality difference.
    out = judge_diagnostics(_records([("A", "B")] * 20))
    assert out["slot_bias"] == pytest.approx(0.0, abs=1e-12)
    assert out["quality_logit"] > 0
    assert out["slot_a_rate"] == 0.5


def test_judge_that_always_names_a_leans_toward_slot_a():
    out = judge_diagnostics(_records([("A", "A")] * 20))
    assert out["slot_bias"] > 0
    assert out["quality_logit"] == pytest.approx(0.0, abs=1e-12)
    assert out["slot_a_rate"] == 1.0


def test_judge_that_always_names_b_has_a_negative_coefficient():
    # Same sign convention as P(A) = sigmoid(theta_A - theta_B + b).
    out = judge_diagnostics(_records([("B", "B")] * 20))
    assert out["slot_bias"] < 0
    assert out["quality_logit"] == pytest.approx(0.0, abs=1e-12)


def test_slot_bias_recovers_a_known_coefficient():
    b_true, d_true, n = 0.8, 0.5, 4000
    judge = _BradleyTerryJudge(b=b_true, d=d_true, seed=0)
    result = win_rate(["p"] * n, [POLICY] * n, [BASELINE] * n, judge)
    diag = result["judge_diagnostics"]
    assert diag["slot_bias"] == pytest.approx(b_true, abs=0.1)
    assert diag["quality_logit"] == pytest.approx(d_true, abs=0.1)
    lo, hi = diag["slot_bias_ci95"]
    assert lo < b_true < hi
    lo, hi = diag["quality_logit_ci95"]
    assert lo < d_true < hi


def test_pointwise_judge_has_no_slot_bias_by_construction():
    # A reward model never sees the slot, so its coefficient is exactly zero
    # whatever the outcomes: the diagnostic is informative only for an LLM judge.
    responses = [f"r{i}" for i in range(12)]
    scores = {r: float(i % 5) for i, r in enumerate(responses)}
    judge = _PointwiseJudge(scores)
    policy, baseline = responses[:6], responses[6:]
    result = win_rate(["p"] * 6, policy, baseline, judge)
    assert result["judge_diagnostics"]["slot_bias"] == pytest.approx(0.0, abs=1e-12)


def test_ties_and_invalid_replies_are_counted_but_do_not_enter_the_coefficient():
    out = judge_diagnostics(
        _records([("A", "B"), ("A", "B"), ("TIE", "B"), ("INVALID", "TIE")])
    )
    assert out["order_ab"]["tie_rate"] == 0.25
    assert out["order_ab"]["invalid_rate"] == 0.25
    assert out["order_ab"]["n_decisive"] == 2
    assert out["order_ba"]["tie_rate"] == 0.25
    assert out["order_ba"]["n_decisive"] == 3
    assert out["verdict_tie_rate"] == 2 / 8
    assert out["parse_rate"] == 7 / 8
    # Only decisive verdicts enter, smoothed by 0.5: 2 of 2 name A with the
    # policy first, 0 of 3 with it second.
    assert out["slot_bias"] == pytest.approx(
        0.5 * (math.log(2.5 / 0.5) + math.log(0.5 / 3.5))
    )


def test_slot_bias_is_undefined_without_decisive_verdicts_in_one_order():
    out = judge_diagnostics(_records([("A", "TIE"), ("B", "TIE")]))
    assert out["slot_bias"] is None
    assert out["slot_bias_ci95"] is None
    assert out["order_ba"]["tie_rate"] == 1.0


def test_empty_records_do_not_divide_by_zero():
    out = judge_diagnostics([])
    assert out["n_pairs"] == 0
    assert out["parse_rate"] is None and out["slot_bias"] is None


def test_bootstrap_interval_is_reproducible():
    records = _records([("A", "B")] * 30 + [("A", "A")] * 10 + [("B", "A")] * 5)
    assert judge_diagnostics(records) == judge_diagnostics(records)


def test_llm_judge_sends_its_temperature_explicitly(monkeypatch):
    # Omitted, the API samples at temperature 1.0 and each verdict is a draw.
    sent: dict = {}

    class _Messages:
        def create(self, **kwargs):
            sent.update(kwargs)
            block = types.SimpleNamespace(type="text", text="B")
            return types.SimpleNamespace(content=[block])

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    judge = AnthropicJudge(model="judge-model")
    assert judge.compare("p", "a", "b") == "B"
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 8

    described = judge.describe()
    assert described["decoding"] == {"temperature": 0.0, "max_tokens": 8}
    assert "{prompt}" in described["prompt"] and described["model"] == "judge-model"
