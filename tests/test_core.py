"""Tests that run without a GPU or a model download.

Deliberately narrow: they cover the config plumbing, the reward shaping
functions, the win-rate arithmetic, and the report aggregation — the parts that
can be wrong silently. Training itself is covered by `scripts/smoke_test.sh`.
"""

from __future__ import annotations

import json

import pytest
import yaml

from prefopt.config import load_config, save_config, to_dict
from prefopt.report import collect, to_markdown
from prefopt.reward import length_penalty, repetition_penalty
from prefopt.utils import bootstrap_ci


@pytest.fixture()
def config_dir(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": 17,
                "model": {"base_model": "test/model", "lora_r": 8},
                "train": {"output_dir": "runs/base", "learning_rate": 1e-5},
            }
        )
    )
    (tmp_path / "dpo.yaml").write_text(
        yaml.safe_dump(
            {
                "_base_": "base.yaml",
                "name": "dpo_test",
                "stage": "dpo",
                "method": {"beta": 0.1},
            }
        )
    )
    return tmp_path


def test_base_inheritance_and_override(config_dir):
    cfg = load_config(config_dir / "dpo.yaml")
    assert cfg.model.base_model == "test/model"  # from base
    assert cfg.model.lora_r == 8
    assert cfg.stage == "dpo"  # from child
    assert cfg.method.beta == 0.1


@pytest.mark.parametrize(
    ("override", "attr", "expected"),
    [
        ("method.beta=0.05", lambda c: c.method.beta, 0.05),
        ("model.use_peft=false", lambda c: c.model.use_peft, False),
        ("train.deepspeed=none", lambda c: c.train.deepspeed, None),
        ("method.num_generations=16", lambda c: c.method.num_generations, 16),
        ("name=custom", lambda c: c.name, "custom"),
    ],
)
def test_cli_overrides_are_typed(config_dir, override, attr, expected):
    cfg = load_config(config_dir / "dpo.yaml", [override])
    assert attr(cfg) == expected


def test_unknown_key_fails_loudly(config_dir):
    """A typo in an ablation config must not silently run the default."""
    (config_dir / "bad.yaml").write_text(
        yaml.safe_dump({"_base_": "base.yaml", "method": {"bta": 0.1}})
    )
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(config_dir / "bad.yaml")


def test_resolved_config_roundtrips(config_dir, tmp_path):
    cfg = load_config(config_dir / "dpo.yaml", ["method.beta=0.02"])
    path = save_config(cfg, tmp_path / "run")
    assert yaml.safe_load(path.read_text())["method"]["beta"] == 0.02
    assert to_dict(cfg)["model"]["base_model"] == "test/model"


def test_length_penalty_is_bounded_and_peaks_at_target():
    fn = length_penalty(target_chars=100)
    scores = fn(["x" * 100, "x" * 50, "x" * 10000])
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] < scores[0]
    assert all(-1.0 <= s <= 0.0 for s in scores)


def test_repetition_penalty_flags_loops():
    fn = repetition_penalty(ngram=4)
    varied, looped = fn(["the quick brown fox jumps over the lazy dog today", "a b c d " * 30])
    assert varied > looped
    assert looped < -0.5


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0] * 60 + [0.0] * 40
    mean, lo, hi = bootstrap_ci(values, n_boot=2000, seed=0)
    assert mean == pytest.approx(0.6)
    assert lo < 0.6 < hi
    assert hi - lo < 0.25  # n=100 should not give a wider interval than this


def test_report_collects_runs(tmp_path):
    run = tmp_path / "dpo_b0.1"
    run.mkdir()
    (run / "config.resolved.yaml").write_text(
        yaml.safe_dump({"stage": "dpo", "method": {"beta": 0.1}, "data": {}})
    )
    (run / "results.json").write_text(
        json.dumps(
            {
                "win_rate_mean": 0.61,
                "per_seed": [{"ci95": [0.55, 0.67], "swap_consistency": 0.82}],
            }
        )
    )
    rows = collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["win_rate"] == 0.61
    table = to_markdown(rows)
    assert "0.610" in table and "dpo_b0.1" in table


def test_report_skips_directories_without_a_config(tmp_path):
    (tmp_path / "not_a_run").mkdir()
    assert collect(tmp_path) == []
