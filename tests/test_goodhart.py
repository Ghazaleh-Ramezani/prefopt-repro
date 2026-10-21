"""Checkpoint discovery and gap arithmetic for prefopt.goodhart. No model needed."""

from __future__ import annotations

import json

from prefopt.goodhart import judge_pair, list_checkpoints, summarize


def _adapter(d):
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")


def test_checkpoints_sorted_numerically_and_final_deduplicated(tmp_path):
    run = tmp_path / "dpo"
    _adapter(run)
    for step in (12, 4, 8, 16):
        _adapter(run / f"checkpoint-{step}")
    (run / "checkpoint-16" / "trainer_state.json").write_text(json.dumps({"max_steps": 16}))
    pts = list_checkpoints(run)
    assert [s for s, _ in pts] == [4, 8, 12, 16]  # final weights == checkpoint-16


def test_final_run_appended_when_last_step_not_checkpointed(tmp_path):
    run = tmp_path / "grpo"
    _adapter(run)
    _adapter(run / "checkpoint-5")
    (run / "checkpoint-5" / "trainer_state.json").write_text(json.dumps({"max_steps": 7}))
    pts = list_checkpoints(run)
    assert [s for s, _ in pts] == [5, 7] and pts[-1][1] == run


class _Prefers:
    """Stub judge that prefers whichever response contains `marker`."""

    def __init__(self, marker):
        self.marker = marker

    def compare(self, prompt, a, b):
        ia, ib = self.marker in a, self.marker in b
        return "A" if ia and not ib else "B" if ib and not ia else "TIE"


def test_gap_is_proxy_minus_external():
    prompts = ["p1", "p2", "p3", "p4"]
    policy = ["long padded", "long padded", "short good", "long padded"]
    reference = ["short", "short", "short", "short"]
    # Proxy rewards padding; the external judge rewards "good". A pair where
    # neither response carries the marker is a tie and scores 0.5.
    r = judge_pair(prompts, policy, reference, _Prefers("padded"), _Prefers("good"))
    assert r["proxy_win_rate"] == (3 * 1.0 + 0.5) / 4
    assert r["external_win_rate"] == (1.0 + 3 * 0.5) / 4
    assert r["goodhart_gap"] == 0.25
    assert r["external_swap_consistency"] == 1.0


def test_reference_against_itself_has_zero_gap():
    out = ["a", "b"]
    r = judge_pair(["p", "q"], out, out, _Prefers("x"), _Prefers("y"))
    assert r["proxy_win_rate"] == r["external_win_rate"] == 0.5 and r["goodhart_gap"] == 0.0


def test_summary_tracks_the_widening_gap():
    rows = [
        {"step": 0, "goodhart_gap": 0.0, "kl_from_reference": 0.0},
        {"step": 4, "goodhart_gap": 0.1, "kl_from_reference": 0.01},
        {"step": 8, "goodhart_gap": 0.2, "kl_from_reference": 0.02},
        {"step": 12, "goodhart_gap": 0.4, "kl_from_reference": 0.05},
    ]
    s = summarize(rows)
    assert s["gap_max"] == 0.4 and s["gap_max_step"] == 12
    assert s["spearman_step_gap"] == 1.0 and s["spearman_kl_gap"] == 1.0
