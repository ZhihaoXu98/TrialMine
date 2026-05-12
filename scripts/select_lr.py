"""Select the winning learning rate from an MLflow sweep and persist it.

Reads runs from experiment 'trialmind-embeddings' whose run_tag starts with
'sweep-bs128-lr', picks the one with the highest val_retrieval_cosine_ndcg@10,
and rewrites the training.learning_rate line in configs/training/embeddings.yaml
via a surgical regex edit (preserves comments + field order).
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training/embeddings.yaml")
METRIC = "val_retrieval_cosine_ndcg@10"


def main() -> None:
    """Pick the highest-NDCG LR from MLflow and write it back to the YAML."""
    parser = argparse.ArgumentParser(description="Select winning LR from MLflow sweep")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--experiment", default="trialmind-embeddings")
    parser.add_argument("--tag-prefix", default="sweep-bs128-lr")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print winner but do not modify the config")
    args = parser.parse_args()

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        logger.error("Experiment %s not found", args.experiment)
        return
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.run_tag LIKE '{args.tag_prefix}%'",
    )

    scored: list[tuple[float, float, str]] = []
    for r in runs:
        tag = r.data.tags.get("run_tag", "")
        try:
            lr = float(tag[len(args.tag_prefix):])
        except ValueError:
            continue
        ndcg = r.data.metrics.get(METRIC)
        if ndcg is not None:
            scored.append((lr, ndcg, r.info.run_id))

    if not scored:
        logger.error("No qualifying runs (tag_prefix=%s, metric=%s)",
                     args.tag_prefix, METRIC)
        return

    scored.sort(key=lambda x: -x[1])
    winner_lr, winner_ndcg, winner_id = scored[0]
    logger.info("WINNER: lr=%g  NDCG@10=%.4f  run_id=%s",
                winner_lr, winner_ndcg, winner_id)
    if len(scored) > 1:
        ru_lr, ru_ndcg, _ = scored[1]
        logger.info("Runner-up: lr=%g  NDCG@10=%.4f  gap=%+.4f",
                    ru_lr, ru_ndcg, winner_ndcg - ru_ndcg)

    if args.dry_run:
        logger.info("--dry-run: not writing config")
        return

    content = args.config.read_text()
    new_content, n = re.subn(
        r"^(\s*learning_rate:\s*).*$",
        rf"\g<1>{winner_lr:g}",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        logger.error("Could not find 'learning_rate:' line in %s", args.config)
        return
    args.config.write_text(new_content)
    logger.info("Wrote training.learning_rate=%g to %s", winner_lr, args.config)


if __name__ == "__main__":
    main()
