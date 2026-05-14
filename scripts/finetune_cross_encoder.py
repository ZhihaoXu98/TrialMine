"""Fine-tune BioLinkBERT as a cross-encoder re-ranker for clinical trials.

v2 (Phase A5 of docs/fix_CE.md):
- Loss is chosen by `config["loss"]["type"]`. MarginMSELoss consumes graded
  (query, positive, negative, label) triplets where `label` is grade(pos) -
  grade(neg). BinaryCrossEntropyLoss preserves v1's behaviour for revert.
- `config["model"]["name"]` can be either a HF repo (cold start, e.g.
  "michiyasunaga/BioLinkBERT-base") or a local checkpoint directory
  (warm-start from v1, e.g. "models/cross-encoder/fine-tuned"). The
  CrossEncoder init accepts both transparently.
- `config["evaluator"]["type"] == "pooled"` swaps the v1 CERerankingEvaluator
  (saturated at 0.993) for src/TrialMine/evaluation/ce_pooled_evaluator.py
  which scores top-K candidates per val query and reports pooled NDCG/MRR.
- `--override KEY.PATH=VALUE` (repeatable) mutates the loaded config —
  used by the warm-vs-cold sweep arms in cloud_run_ce.sh to override
  `model.name` and `mlflow.run_tag` without touching the YAML.

Usage:
    python scripts/finetune_cross_encoder.py
    python scripts/finetune_cross_encoder.py --dry-run
    python scripts/finetune_cross_encoder.py --override model.name=michiyasunaga/BioLinkBERT-base
    python scripts/finetune_cross_encoder.py --override training.learning_rate=4e-5

Config: configs/training/cross_encoder.yaml
Input:  data/training/ce_graded_{train,val}.jsonl  (v2 graded triplets)
        data/training/{train,val}_pairs.jsonl       (v1 legacy, BCE path)
Output: models/cross-encoder/fine-tuned-v2/         (v2 default)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training/cross_encoder.yaml")

# Source label files for the pooled evaluator. Mirrors the canonical list
# in scripts/build_ce_graded_training.py:SOURCE_LABEL_FILES — if that list
# is edited, edit this one too. Duplicated rather than imported because
# `scripts/` is not on the Python path at runtime.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_LABEL_FILES = (
    PROJECT_ROOT / "data" / "evaluation" / "labeled_queries.jsonl",
    PROJECT_ROOT / "data" / "evaluation" / "test_labels.jsonl",
    PROJECT_ROOT / "data" / "evaluation" / "train_labels_extra.jsonl",
    PROJECT_ROOT / "data" / "evaluation" / "test_labels_v2.jsonl",
)


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    """Load training config from YAML.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply --override KEY.PATH=VALUE strings to a config dict in-place.

    Dotted keys address nested dict paths; intermediate dicts are created
    if absent. Values are cast best-effort: bool -> int -> float -> str.
    "true"/"false" (case-insensitive) become real booleans BEFORE the numeric
    casts (so "0"/"1" stay int). Same semantics as the bi-encoder runbook.

    Args:
        config: The loaded config dict.
        overrides: List of "key.path=value" strings from --override flags.

    Returns:
        The (mutated) config dict.
    """
    def _cast(s: str):
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        for fn in (int, float):
            try:
                return fn(s)
            except ValueError:
                pass
        return s

    def _set(d: dict, path: str, value) -> None:
        keys = path.split(".")
        cur = d
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value

    for spec in overrides:
        if "=" not in spec:
            logger.warning("Ignoring malformed --override (no '='): %s", spec)
            continue
        key, _, raw = spec.partition("=")
        value = _cast(raw)
        _set(config, key, value)
        logger.info("Override applied: %s = %r (%s)", key, value, type(value).__name__)

    return config


# ── Loss dispatch (with fail-fast smoke import) ──────────────────────────────


def resolve_loss_class(loss_type: str):
    """Smoke-import the requested loss class. Fail fast with a clear message
    if it isn't available — never silently fall back to a different loss.

    Args:
        loss_type: One of "MarginMSELoss", "BinaryCrossEntropyLoss".

    Returns:
        The loss class (not an instance).

    Raises:
        ValueError: if `loss_type` is unrecognised.
        ImportError: if the requested class is not in the installed
            sentence-transformers version, with a hint on the upgrade path.
    """
    if loss_type == "MarginMSELoss":
        try:
            from sentence_transformers.cross_encoder.losses import MarginMSELoss
        except ImportError as e:
            raise ImportError(
                "MarginMSELoss is not available in the installed sentence-transformers. "
                "Upgrade to >=3.1 (pip install -U sentence-transformers). "
                "Do NOT fall back to the bi-encoder MarginMSELoss — the CE forward "
                "signature differs."
            ) from e
        return MarginMSELoss
    if loss_type == "BinaryCrossEntropyLoss":
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        return BinaryCrossEntropyLoss
    raise ValueError(
        f"Unsupported loss.type={loss_type!r}. "
        "Allowed: 'MarginMSELoss' (v2 graded) or 'BinaryCrossEntropyLoss' (v1 legacy)."
    )


# ── Device detection ─────────────────────────────────────────────────────────


def detect_device() -> str:
    """Detect the best available torch device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Detected device: %s", device)
    return device


# ── Data loading ─────────────────────────────────────────────────────────────


def load_training_data(config: dict) -> tuple[Dataset, Dataset]:
    """Load train/val datasets — schema depends on the loss type.

    - MarginMSELoss: graded triplets with columns
      `query` / `positive` / `negative` / `label` (margin in {1.0, 2.0, 3.0}).
      Audit columns (query_id, doc_pos_nct_id, …) are dropped.
    - BinaryCrossEntropyLoss: each triplet is split into two binary rows
      (sentence1, sentence2, label=1 or 0) — legacy v1 behaviour.

    Args:
        config: Full config dict.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    loss_type = config["loss"]["type"]
    data_config = config["data"]
    if loss_type == "MarginMSELoss":
        return _load_triplet_data(data_config)
    return _load_binary_pair_data(data_config)


def _load_triplet_data(data_config: dict) -> tuple[Dataset, Dataset]:
    """Load (query, positive, negative, label) triplets for MarginMSELoss."""
    def load(filepath: str) -> Dataset:
        rows: list[dict] = []
        skipped = 0
        with open(filepath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                query = rec.get("query", "")
                positive = rec.get("positive", "")
                negative = rec.get("negative", "")
                label = rec.get("label")
                if not query or not positive or not negative or label is None:
                    skipped += 1
                    continue
                rows.append(
                    {
                        "query": query,
                        "positive": positive,
                        "negative": negative,
                        "label": float(label),
                    }
                )
        if skipped:
            logger.info("Skipped %d malformed/incomplete rows in %s", skipped, filepath)
        return Dataset.from_list(rows)

    train_ds = load(data_config["train_file"])
    val_ds = load(data_config["val_file"])
    logger.info(
        "Loaded triplets: %d train, %d val (columns: query/positive/negative/label)",
        len(train_ds),
        len(val_ds),
    )
    return train_ds, val_ds


def _load_binary_pair_data(data_config: dict) -> tuple[Dataset, Dataset]:
    """Load triplets and split into binary (sentence1, sentence2, label) — v1 path."""
    def load_and_convert(filepath: str) -> Dataset:
        rows: list[dict] = []
        skipped = 0
        with open(filepath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                query = rec.get("query", "")
                positive = rec.get("positive", "")
                negative = rec.get("negative", "")
                if not query or not positive:
                    skipped += 1
                    continue
                rows.append({"sentence1": query, "sentence2": positive, "label": 1.0})
                if negative and negative.strip():
                    rows.append({"sentence1": query, "sentence2": negative, "label": 0.0})
        if skipped:
            logger.info("Skipped %d malformed rows in %s", skipped, filepath)
        return Dataset.from_list(rows)

    train_ds = load_and_convert(data_config["train_file"])
    val_ds = load_and_convert(data_config["val_file"])
    logger.info(
        "Loaded binary pairs: %d train, %d val (columns: sentence1/sentence2/label)",
        len(train_ds),
        len(val_ds),
    )
    return train_ds, val_ds


# ── Evaluator ────────────────────────────────────────────────────────────────


def build_evaluator(config: dict):
    """Construct the evaluator the trainer will use for periodic scoring.

    Dispatch on `config["evaluator"]["type"]`:
    - "pooled" → CEPooledEvaluator (production-shaped, see Phase A5 writeup).
    - default  → v1 CERerankingEvaluator (saturated baseline, kept for revert).

    Both respect `config["evaluation"]["max_val_samples"]` as a soft cap.
    """
    evaluator_cfg = config.get("evaluator", {})
    evaluator_type = evaluator_cfg.get("type", "ce_reranking")
    max_val_samples = config["evaluation"]["max_val_samples"]

    if evaluator_type == "pooled":
        from TrialMine.evaluation.ce_pooled_evaluator import CEPooledEvaluator
        return CEPooledEvaluator(
            val_pair_file=config["data"]["val_file"],
            source_label_files=SOURCE_LABEL_FILES,
            pool_top_k=evaluator_cfg.get("pool_top_k", 20),
            max_val_pairs=max_val_samples,
            batch_size=64,
            name="pooled",
        )

    # Legacy v1 path: CERerankingEvaluator over 1-pos/1-neg triplets.
    return _build_ce_reranking_evaluator(
        val_file=config["data"]["val_file"],
        max_samples=max_val_samples,
    )


def _build_ce_reranking_evaluator(
    val_file: str,
    max_samples: int,
    seed: int = 42,
) -> CERerankingEvaluator:
    """v1 CERerankingEvaluator. Kept for the BCE legacy / revert path."""
    val_rows: list[dict] = []
    with open(val_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("negative", "").strip():
                continue
            val_rows.append(row)

    rng = random.Random(seed)
    if len(val_rows) > max_samples:
        val_rows = rng.sample(val_rows, max_samples)

    samples = [
        {
            "query": row["query"],
            "positive": [row["positive"]],
            "negative": [row["negative"]],
        }
        for row in val_rows
    ]
    logger.info("CE reranking evaluator (legacy): %d samples", len(samples))
    return CERerankingEvaluator(
        samples=samples,
        name="CERerankingEvaluator",
        batch_size=64,
    )


# ── Training example inspection ──────────────────────────────────────────────


def print_training_examples(train_ds: Dataset) -> None:
    """Print a small sample of training rows for data-quality sanity.

    Handles both schemas — graded triplets (query/positive/negative/label-margin)
    and binary pairs (sentence1/sentence2/label).
    """
    columns = set(train_ds.column_names)
    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print(f"  Total rows: {len(train_ds):,}")
    print(f"  Columns:    {sorted(columns)}")
    print("=" * 80)

    if {"query", "positive", "negative", "label"} <= columns:
        # Graded triplet schema (v2)
        from collections import Counter
        margins = Counter(int(r["label"]) for r in train_ds)
        print("  Margin distribution:")
        for k in sorted(margins):
            print(f"    label={float(k)}  count={margins[k]:,}")
        print("\n--- 3 TRIPLET examples (margin = grade_pos - grade_neg) ---")
        for i, row in enumerate(train_ds.select(range(min(3, len(train_ds))))):
            print(f"\n  [{i + 1}] margin={row['label']}")
            print(f"      Query: {row['query'][:100]}")
            print(f"      POS:   {row['positive'][:120]}")
            print(f"      NEG:   {row['negative'][:120]}")
    else:
        # Binary pair schema (v1)
        positives = [r for r in train_ds if r["label"] == 1.0]
        negatives = [r for r in train_ds if r["label"] == 0.0]
        print(f"  Positive pairs: {len(positives):,}")
        print(f"  Negative pairs: {len(negatives):,}")
        print(f"  Balance:        {len(positives) / len(train_ds) * 100:.1f}% positive")
        print("\n--- 3 POSITIVE examples (label=1.0) ---")
        for i, row in enumerate(positives[:3]):
            print(f"\n  [{i + 1}] Query: {row['sentence1'][:100]}")
            print(f"      Trial: {row['sentence2'][:120]}...")
        print("\n--- 3 NEGATIVE examples (label=0.0) ---")
        for i, row in enumerate(negatives[:3]):
            print(f"\n  [{i + 1}] Query: {row['sentence1'][:100]}")
            print(f"      Trial: {row['sentence2'][:120]}...")
    print()


# ── Metadata saving ──────────────────────────────────────────────────────────


def save_metadata(
    output_dir: Path,
    config: dict,
    train_size: int,
    val_size: int,
    device: str,
    training_time_minutes: float,
    eval_results: dict | None = None,
) -> None:
    """Save training metadata to a JSON file alongside the model.

    Args:
        output_dir: Model output directory.
        config: Full training config.
        train_size: Number of training rows (triplets or binary pairs).
        val_size: Number of validation rows.
        device: Device used for training.
        training_time_minutes: Total training time.
        eval_results: Final evaluation metrics (if available).
    """
    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "base_model": config["model"]["name"],
        "model_type": "cross-encoder",
        "num_labels": config["model"]["num_labels"],
        "dataset_size": {"train": train_size, "val": val_size},
        "hyperparameters": {
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "warmup_ratio": config["training"]["warmup_ratio"],
            "weight_decay": config["training"]["weight_decay"],
            "loss": config["loss"]["type"],
            "max_length": config["model"]["max_length"],
        },
        "evaluator": config.get("evaluator", {}),
        "metric_for_best_model": config["training"]["metric_for_best_model"],
        "device": device,
        "training_time_minutes": round(training_time_minutes, 1),
        "eval_metrics": eval_results or {},
    }

    if device == "cuda" and torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(0)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved training metadata to %s", metadata_path)


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune BioLinkBERT as a cross-encoder re-ranker",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config, dataset stats, and examples without training",
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None,
        help="Resume training from a checkpoint directory",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help=(
            "Override a config field. Repeatable. Examples: "
            "--override model.name=michiyasunaga/BioLinkBERT-base "
            "--override training.learning_rate=4e-5 "
            "--override mlflow.run_tag=ce-sweep-cold"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the cross-encoder fine-tuning pipeline."""
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    # ── Smoke-import the loss BEFORE any heavy work — fail fast on missing dep ─
    loss_type = config["loss"]["type"]
    loss_cls = resolve_loss_class(loss_type)
    logger.info("Loss resolved: %s.%s", loss_cls.__module__, loss_cls.__name__)

    # ── Device detection ─────────────────────────────────────────────────
    device = detect_device()

    # Auto-reduce epochs on CPU
    if device == "cpu":
        original_epochs = config["training"]["epochs"]
        config["training"]["epochs"] = 1
        logger.info(
            "CPU detected: reducing epochs from %d to 1", original_epochs,
        )

    # ── Load model ───────────────────────────────────────────────────────
    # `model_name` may be a HF repo ID (cold start) OR a local path (warm-start
    # from v1 binary CE). CrossEncoder.__init__ accepts both transparently.
    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]
    max_length = config["model"]["max_length"]

    logger.info("Loading base model: %s (num_labels=%d)", model_name, num_labels)
    model = CrossEncoder(
        model_name,
        num_labels=num_labels,
        max_length=max_length,
        device=device,
    )
    logger.info("Cross-encoder loaded. max_length=%d", max_length)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading training data ...")
    train_ds, val_ds = load_training_data(config)

    # ── Print examples for data quality check ────────────────────────────
    print_training_examples(train_ds)

    # ── Training stats ───────────────────────────────────────────────────
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    steps_per_epoch = len(train_ds) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])

    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"  Base model:     {model_name}")
    print(f"  Model type:     cross-encoder (num_labels={num_labels})")
    print(f"  Device:         {device}")
    print(f"  Train rows:     {len(train_ds):,}")
    print(f"  Val rows:       {len(val_ds):,}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Epochs:         {epochs}")
    print(f"  Steps/epoch:    {steps_per_epoch:,}")
    print(f"  Total steps:    {total_steps:,}")
    print(f"  Warmup steps:   {warmup_steps:,}")
    print(f"  Learning rate:  {config['training']['learning_rate']}")
    print(f"  Max length:     {max_length}")
    print(f"  Loss:           {loss_type}")
    print(f"  Evaluator:      {config.get('evaluator', {}).get('type', 'ce_reranking')}")
    print(f"  Best metric:    {config['training']['metric_for_best_model']}")
    print(f"  Eval every:     {config['training']['eval_steps']} steps")
    print(f"  Log every:      {config['training']['logging_steps']} steps")
    print(f"  Output:         {config['model']['output_dir']}")
    print()

    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        est_speed = 0.03 if "A100" in gpu_name else 0.10
        est_hours = total_steps * est_speed / 3600
        print(f"  GPU:            {gpu_name}")
        print(f"  Estimated time: ~{est_hours:.1f} hours")
    elif device == "cpu":
        est_hours = total_steps * 1.0 / 3600
        print(f"  Estimated time (CPU): ~{est_hours:.1f} hours")

    print()

    # ── Loss function ────────────────────────────────────────────────────
    loss = loss_cls(model=model)
    logger.info("Loss instantiated: %s", loss.__class__.__name__)

    # ── Evaluator ────────────────────────────────────────────────────────
    logger.info(
        "Building evaluator (type=%s, max_val_samples=%d) ...",
        config.get("evaluator", {}).get("type", "ce_reranking"),
        config["evaluation"]["max_val_samples"],
    )
    evaluator = build_evaluator(config)
    logger.info("Evaluator instantiated: %s", evaluator.__class__.__name__)

    if args.dry_run:
        logger.info("DRY RUN — exiting before training.")
        return

    # ── MLflow setup ─────────────────────────────────────────────────────
    mlflow_config = config["mlflow"]
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_config["tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_config["experiment_name"]
    os.environ["MLFLOW_RUN_TAGS"] = json.dumps({
        "stage": mlflow_config["run_tag"],
        "model": model_name,
        "model_type": "cross-encoder",
        "device": device,
    })

    # ── Training arguments ───────────────────────────────────────────────
    output_dir = config["model"]["output_dir"]
    training_args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        fp16=config["training"].get("fp16", False),
        eval_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        logging_steps=config["training"]["logging_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        # IMPORTANT: this key must match a key returned by `evaluator(model)`.
        # Pooled evaluator emits "pooled_ndcg@5/@10/_mrr"; legacy CE
        # reranking emits "CERerankingEvaluator_ndcg@10" etc.
        metric_for_best_model=config["training"]["metric_for_best_model"],
        report_to="mlflow",
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
        evaluator=evaluator,
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("Starting training ...")
    start_time = time.time()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60
    logger.info("Training completed in %.1f minutes", elapsed_min)

    # ── Save best model ──────────────────────────────────────────────────
    final_output = Path(output_dir)
    final_output.mkdir(parents=True, exist_ok=True)
    model.save(str(final_output))
    logger.info("Best model saved to %s", final_output)

    # ── Final evaluation ─────────────────────────────────────────────────
    logger.info("Running final evaluation ...")
    eval_results = evaluator(model)
    logger.info("Final eval results: %s", eval_results)

    # ── Save metadata ────────────────────────────────────────────────────
    save_metadata(
        output_dir=final_output,
        config=config,
        train_size=len(train_ds),
        val_size=len(val_ds),
        device=device,
        training_time_minutes=elapsed_min,
        eval_results=eval_results,
    )

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Duration:       {elapsed_min:.1f} minutes")
    print(f"  Model saved:    {final_output}")

    # Surface whichever metric set the active evaluator emitted.
    headline_keys = (
        "pooled_ndcg@5",
        "pooled_ndcg@10",
        "pooled_mrr",
        "CERerankingEvaluator_ndcg@10",
        "CERerankingEvaluator_map",
        "CERerankingEvaluator_mrr@10",
    )
    for k in headline_keys:
        if k in eval_results:
            print(f"  {k:<32} {eval_results[k]}")
    print()
    print("Next steps:")
    print("  1. make mlflow  — view training curves")
    print(f"  2. Evaluate on labeled data: python scripts/evaluate_cross_encoder.py --model {output_dir}")
    print(f"  3. Demo before/after re-ranking: python scripts/demo_reranker.py --model {output_dir}")


if __name__ == "__main__":
    main()
