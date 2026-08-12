"""Shared argument parsing so all four stages have an identical interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunCfg, load_config, save_config
from .utils import environment_metadata, get_logger, save_json, set_seed

logger = get_logger(__name__)


def parse_and_prepare(description: str) -> RunCfg:
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
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.output_dir:
        cfg.train.output_dir = args.output_dir

    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    save_config(cfg, cfg.train.output_dir)
    save_json(environment_metadata(), Path(cfg.train.output_dir) / "environment.json")

    logger.info("Run '%s' (stage=%s, seed=%d)", cfg.name, cfg.stage, cfg.seed)
    logger.info("Output: %s", cfg.train.output_dir)
    return cfg
