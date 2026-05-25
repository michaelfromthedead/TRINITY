"""Spin Physics Research — main entry point."""

import argparse
from src.utils.helpers import load_config, setup_logging


def run(config_path: str) -> None:
    """Execute the spin dynamics simulation pipeline."""
    cfg = load_config(config_path)
    logger = setup_logging(cfg.get("log_level", "INFO"))
    logger.info("Spin Physics simulation starting.")
    # TODO (placeholder): invoke simulation kernel
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spin Physics Simulation")
    parser.add_argument("--config", required=True, help="Path to config TOML")
    args = parser.parse_args()
    run(args.config)
