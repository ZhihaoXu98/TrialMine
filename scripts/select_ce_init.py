"""Select the winning init (warm vs cold) from an MLflow CE sweep and persist it.

Phase A6 of docs/fix_CE.md. Mirrors scripts/select_lr.py but writes a quoted
string back to `model.name` (a path or HF repo ID), not a numeric `learning_rate`.

Reads runs from experiment 'trialmind-cross-encoder' with run_tag
'ce-sweep-warm' / 'ce-sweep-cold', picks the higher pooled_ndcg@10
(falling back to `eval_pooled_ndcg@10` if the HF Trainer's MLflow
callback prefixed it), and rewrites the `model.name` line in
configs/training/cross_encoder.yaml via a surgical regex edit
(preserves the trailing comment + field order).

Logs the verdict with the literal string `selected init:` so the Phase B1
verify one-liner (`grep "selected init" cloud_run_ce.log`) can pick it up.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import mlflow
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training/cross_encoder.yaml")
EXPERIMENT = "trialmind-cross-encoder"
WARM_PATH = "models/cross-encoder/fine-tuned"
COLD_PATH = "michiyasunaga/BioLinkBERT-base"
# Try both names — HF Trainer's MLflow callback may prefix with `eval_`.
METRIC_CANDIDATES = ("pooled_ndcg@10", "eval_pooled_ndcg@10")


def _resolve_tracking_uri(config_path: Path) -> str:
    """Pick the MLflow tracking URI to read from.

    Order of precedence:
    1. MLFLOW_TRACKING_URI env var (set by cloud_run_ce.sh export).
    2. `mlflow.tracking_uri` field in the CE config YAML.
    3. MLflow's library default (./mlruns/).

    Without this resolution, a standalone debug invocation of this script
    would read from ./mlruns/ even though training wrote to sqlite:///mlflow.db.
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text())
        uri = cfg.get("mlflow", {}).get("tracking_uri")
        if uri:
            return uri
    return mlflow.get_tracking_uri()


def _pick_metric(metrics: dict) -> float | None:
    for key in METRIC_CANDIDATES:
        if key in metrics:
            return float(metrics[key])
    return None


def main() -> int:
    """Pick the higher-NDCG init from MLflow and write it to the CE yaml.

    Returns:
        Process exit code (0 on success, 1 on missing runs).
    """
    parser = argparse.ArgumentParser(description="Select winning CE init from MLflow sweep")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print winner but do not modify the config",
    )
    args = parser.parse_args()

    tracking_uri = _resolve_tracking_uri(args.config)
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    if exp is None:
        logger.error("Experiment %s not found", args.experiment)
        return 1
    # MLflow's filter parser doesn't support IN(...) for tag values
    # (mlflow.exceptions.MlflowException: "Expected a quoted string value
    # for tag"). LIKE 'ce-sweep-%' is the workaround used by select_lr.py.
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.run_tag LIKE 'ce-sweep-%'",
    )

    # If multiple runs share a tag (re-runs), keep the most recent by start_time.
    best_per_tag: dict[str, tuple[float, str, int]] = {}
    for r in runs:
        tag = r.data.tags.get("run_tag", "")
        if tag not in {"ce-sweep-warm", "ce-sweep-cold"}:
            continue
        ndcg = _pick_metric(r.data.metrics)
        if ndcg is None:
            logger.warning(
                "Skipping run_id=%s (tag=%s) — no %s metric logged",
                r.info.run_id,
                tag,
                " or ".join(METRIC_CANDIDATES),
            )
            continue
        start = r.info.start_time
        prior = best_per_tag.get(tag)
        if prior is None or start > prior[2]:
            best_per_tag[tag] = (ndcg, r.info.run_id, start)

    if not best_per_tag:
        logger.error(
            "No qualifying runs (experiment=%s, tags={ce-sweep-warm, ce-sweep-cold}, metric=%s)",
            args.experiment,
            " or ".join(METRIC_CANDIDATES),
        )
        return 1

    warm = best_per_tag.get("ce-sweep-warm")
    cold = best_per_tag.get("ce-sweep-cold")
    logger.info(
        "Sweep results — warm: %s, cold: %s",
        f"ndcg@10={warm[0]:.4f} run={warm[1]}" if warm else "MISSING",
        f"ndcg@10={cold[0]:.4f} run={cold[1]}" if cold else "MISSING",
    )

    # Pick winner. If only one arm has metrics, it wins by default.
    if warm and cold:
        winner_tag = "warm" if warm[0] >= cold[0] else "cold"
        gap = warm[0] - cold[0] if warm[0] >= cold[0] else cold[0] - warm[0]
    elif warm:
        winner_tag = "warm"
        gap = 0.0
    else:
        winner_tag = "cold"
        gap = 0.0

    winner_value = WARM_PATH if winner_tag == "warm" else COLD_PATH
    winner_score = (warm if winner_tag == "warm" else cold)[0]

    # Log with the literal phrase 'selected init:' so the Phase B1 grep finds it.
    logger.info(
        "selected init: %s  (pooled_ndcg@10=%.4f, gap vs runner-up=%+.4f, model.name=%s)",
        winner_tag,
        winner_score,
        gap,
        winner_value,
    )

    if args.dry_run:
        logger.info("--dry-run: not writing config")
        return 0

    content = args.config.read_text()
    # Match the model.name line specifically (mlflow.experiment_name has the
    # `experiment_` prefix in front of `name:`, so this regex won't match it).
    new_content, n = re.subn(
        r'^(\s*name:\s*)"[^"]*"(.*)$',
        rf'\g<1>"{winner_value}"\g<2>',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0:
        logger.error("Could not find 'name:' line in %s", args.config)
        return 1
    args.config.write_text(new_content)
    logger.info("Wrote model.name=%r to %s", winner_value, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
