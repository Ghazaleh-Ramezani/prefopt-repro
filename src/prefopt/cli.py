"""Shared argument parsing so all four stages have an identical interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunCfg, load_config, save_config
from .utils import environment_metadata, get_logger, save_json, set_seed

logger = get_logger(__name__)


def build_parser(description: str) -> argparse.ArgumentParser:
    """The arguments every stage takes. A stage can add its own before parsing."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted-path override, e.g. --set method.beta=0.05 (repeatable).",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Overrides train.output_dir."
    )
    return parser


def prepare(args: argparse.Namespace, record: bool = True) -> RunCfg:
    """Resolve the config, seed, and, when `record` is true, write the run's
    config.resolved.yaml and environment.json.

    `record=False` is for re-entering a run that already exists (evaluating a
    saved adapter): its resolved config is what later stages read to rebuild
    the model, and its environment record is the training environment, so
    neither is overwritten.
    """
    cfg = load_config(args.config, args.overrides)
    if args.output_dir:
        cfg.train.output_dir = args.output_dir

    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    if record:
        save_config(cfg, cfg.train.output_dir)
        save_json(environment_metadata(), Path(cfg.train.output_dir) / "environment.json")

    logger.info("Run '%s' (stage=%s, seed=%d)", cfg.name, cfg.stage, cfg.seed)
    logger.info("Output: %s", cfg.train.output_dir)
    return cfg


def parse_and_prepare(description: str) -> RunCfg:
    return prepare(build_parser(description).parse_args())
