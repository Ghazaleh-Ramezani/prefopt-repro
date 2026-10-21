"""Win-rate, verdict parsing and rank-correlation logic.

These import the evaluation modules, which pull in `datasets`; they are skipped
on a core-only install.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("datasets")

from prefopt.diagnostics import spearman  # noqa: E402
from prefopt.evaluate import _parse_verdict, win_rate  # noqa: E402


class _FixedJudge:
    """Returns the same verdict for every comparison."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    def compare(self, prompt: str, a: str, b: str) -> str:
        return self.verdict


class _ScriptedJudge:
    """Returns verdicts from a list, in call order."""

    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = list(verdicts)

    def compare(self, prompt: str, a: str, b: str) -> str:
        return self.verdicts.pop(0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A", "A"),
        ("  b\n", "B"),
        ("TIE", "TIE"),
        ("tie.", "TIE"),
        ("Both are fine", "INVALID"),
        ("Answer A", "INVALID"),
        ("", "INVALID"),
        ("A or B", "A"),
    ],
)
def test_verdict_parsing_is_exact_on_the_first_token(text, expected):
    assert _parse_verdict(text) == expected


def test_position_biased_judge_scores_as_ties():
    # Always answering "A" means the policy wins in one order and loses in the
    # other: every pair is a flip, so the win-rate is exactly 0.5 and the judge
    # is never self-consistent.
    out = win_rate(["p"] * 4, ["x"] * 4, ["y"] * 4, _FixedJudge("A"))
    assert out["win_rate"] == 0.5
    assert out["swap_consistency"] == 0.0
    assert out["ties"] == 4


def test_consistent_preference_is_a_win():
    # Policy shown first, judge says A; policy shown second, judge says B.
    out = win_rate(["p"], ["x"], ["y"], _ScriptedJudge(["A", "B"]))
    assert out["win_rate"] == 1.0
    assert out["swap_consistency"] == 1.0


def test_invalid_verdicts_are_ties_not_losses():
    out = win_rate(["p"], ["x"], ["y"], _ScriptedJudge(["INVALID", "INVALID"]))
    assert out["win_rate"] == 0.5
    assert out["losses"] == 0
    assert out["invalid_verdicts"] == 1
    assert out["swap_consistency"] == 0.0


def test_misaligned_inputs_fail_instead_of_truncating():
    with pytest.raises(ValueError, match="must align"):
        win_rate(["p1", "p2"], ["x"], ["y", "y"], _FixedJudge("A"))


def test_spearman_constant_input_is_undefined():
    assert math.isnan(spearman([5, 5, 5, 5], [1, 2, 3, 4]))


def test_spearman_uses_average_ranks_for_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    x = [1, 2, 2, 3, 3, 3, 4]
    y = [2, 1, 4, 3, 6, 5, 7]
    expected = scipy_stats.spearmanr(x, y).correlation
    assert spearman(x, y) == pytest.approx(expected)


def test_spearman_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        spearman([1, 2, 3], [1, 2])


def test_generate_skips_vllm_for_adapter_directories(tmp_path, monkeypatch):
    # A run directory is an adapter; vLLM would fail on it with an error that
    # is not an ImportError, so the HF fallback would never be reached.
    import sys
    import types

    from prefopt import evaluate

    (tmp_path / "adapter_config.json").write_text("{}")
    called = {"vllm": False}

    class _LLM:
        def __init__(self, *a, **k):
            called["vllm"] = True
            raise OSError("no config.json")

    monkeypatch.setitem(
        sys.modules, "vllm", types.SimpleNamespace(LLM=_LLM, SamplingParams=object)
    )

    def _stop(*a, **k):
        raise RuntimeError("reached HF path")

    monkeypatch.setattr("prefopt.modeling.load_for_inference", _stop)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _Tok())

    class _Tok:
        pad_token = "x"

    with pytest.raises(RuntimeError, match="reached HF path"):
        evaluate.generate(str(tmp_path), ["p"], base_model="base")
    assert not called["vllm"]
