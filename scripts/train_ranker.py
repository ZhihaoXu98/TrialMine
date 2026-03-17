"""Train the LightGBM metadata blender / ranker.

Usage:
    python scripts/train_ranker.py [--config PATH]

Config: configs/training/ranker.yaml
Logs experiment to MLflow. Saves model to models/ranker/v{version}/.
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM ranker")
    parser.add_argument("--config", default="configs/training/ranker.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # TODO: load config YAML (hyperparams must come from config, never hardcoded)
    # TODO: load feature matrix from data/processed/features/
    # TODO: mlflow.start_run(), log params from config
    # TODO: train LGBMRanker or LGBMClassifier
    # TODO: log metrics, save model artifact to MLflow + models/ranker/v{n}/
    # TODO: create model card at docs/model-cards/ranker.md
    logger.info("train_ranker.py not yet implemented")
    sys.exit(1)


if __name__ == "__main__":
    main()
