"""Re-entering a finished run: --eval-only must not rewrite the run's records,
and must only evaluate the adapter on top of the model it was trained from."""

from __future__ import annotations

import argparse

import pytest
import yaml

pytest.importorskip("trl")

from prefopt.cli import prepare  # noqa: E402
from prefopt.config import load_config  # noqa: E402
from prefopt.train_dpo import load_trained_policy  # noqa: E402

CONFIG = "configs/dpo.yaml"


def _args(out, *overrides):
    return argparse.Namespace(config=CONFIG, overrides=list(overrides), output_dir=str(out))


def test_prepare_without_record_leaves_the_run_files_alone(tmp_path):
    run = tmp_path / "dpo"
    prepare(_args(run))
    resolved = (run / "config.resolved.yaml").read_text()
    environment = (run / "environment.json").read_text()

    prepare(_args(run, "train.per_device_eval_batch_size=2"), record=False)
    assert (run / "config.resolved.yaml").read_text() == resolved
    assert (run / "environment.json").read_text() == environment


def test_eval_only_needs_a_trained_adapter(tmp_path):
    cfg = load_config(CONFIG)
    cfg.train.output_dir = str(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="trained adapter"):
        load_trained_policy(cfg)


def test_eval_only_refuses_a_different_starting_adapter(tmp_path):
    # The DPO adapter is a delta on the SFT-merged weights it trained from;
    # evaluating it on another starting point scores a model never trained.
    run = tmp_path / "dpo"
    run.mkdir()
    (run / "adapter_config.json").write_text("{}")
    (run / "config.resolved.yaml").write_text(yaml.safe_dump({"init_adapter": "runs/sft"}))
    cfg = load_config(CONFIG, ["init_adapter=runs/other_sft"])
    cfg.train.output_dir = str(run)
    with pytest.raises(ValueError, match="was trained from"):
        load_trained_policy(cfg)
