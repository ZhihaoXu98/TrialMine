"""Fine-tune BioLinkBERT as a cross-encoder re-ranker for clinical trials.

Converts (query, positive, negative) triplets to binary-labeled pairs and
trains with BinaryCrossEntropyLoss. Evaluates with CERerankingEvaluator
during training. Logs everything to MLflow.

Usage:
    python scripts/finetune_cross_encoder.py
    python scripts/finetune_cross_encoder.py --dry-run
    python scripts/finetune_cross_encoder.py --config configs/training/cross_encoder.yaml

Config: configs/training/cross_encoder.yaml
Input:  data/training/train_pairs.jsonl, data/training/val_pairs.jsonl
Output: models/cross-encoder/fine-tuned/
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
    """Load triplets and convert to binary-labeled pairs for BCE.

    Each triplet (query, positive, negative) becomes two rows:
        - (query, positive, label=1.0)
        - (query, negative, label=0.0)

    Filters out triplets with empty negative fields.

    Args:
        config: Full config dict.

    Returns:
        Tuple of (train_dataset, val_dataset) as HuggingFace Datasets.
    """
    data_config = config["data"]

    def load_and_convert(filepath: str) -> Dataset:
        """Load JSONL triplets and convert to labeled pairs."""
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

                # Positive pair
                rows.append({
                    "sentence1": query,
                    "sentence2": positive,
                    "label": 1.0,
                })

                # Negative pair (skip if empty)
                if negative and negative.strip():
                    rows.append({
                        "sentence1": query,
                        "sentence2": negative,
                        "label": 0.0,
                    })

        if skipped:
            logger.info("Skipped %d malformed rows in %s", skipped, filepath)

        return Dataset.from_list(rows)

    train_ds = load_and_convert(data_config["train_file"])
    val_ds = load_and_convert(data_config["val_file"])

    logger.info(
        "Loaded %d train pairs, %d val pairs (from triplet conversion)",
        len(train_ds), len(val_ds),
    )

    return train_ds, val_ds


# ── Evaluator ────────────────────────────────────────────────────────────────


def build_evaluator(
    val_file: str,
    max_samples: int,
    seed: int = 42,
) -> CERerankingEvaluator:
    """Build a CERerankingEvaluator from validation triplets.

    Subsamples to max_samples for speed. Each sample has one positive
    and one negative document for the cross-encoder to re-rank.

    Args:
        val_file: Path to validation JSONL (triplet format).
        max_samples: Maximum validation triplets for the evaluator.
        seed: Random seed for subsampling.

    Returns:
        Configured CERerankingEvaluator.
    """
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

    logger.info("CE reranking evaluator: %d samples", len(samples))

    return CERerankingEvaluator(
        samples=samples,
        name="CERerankingEvaluator",
        batch_size=64,
    )


# ── Training example inspection ──────────────────────────────────────────────


def print_training_examples(train_ds: Dataset) -> None:
    """Print example positive and negative pairs for data quality verification.

    Args:
        train_ds: Training dataset with sentence1, sentence2, label columns.
    """
    positives = [r for r in train_ds if r["label"] == 1.0]
    negatives = [r for r in train_ds if r["label"] == 0.0]

    print("\n" + "=" * 80)
    print(f"DATA QUALITY CHECK")
    print(f"  Total pairs:    {len(train_ds):,}")
    print(f"  Positive pairs: {len(positives):,}")
    print(f"  Negative pairs: {len(negatives):,}")
    print(f"  Balance:        {len(positives) / len(train_ds) * 100:.1f}% positive")
    print("=" * 80)

    print("\n--- 3 POSITIVE examples (label=1.0) ---")
    for i, row in enumerate(positives[:3]):
        print(f"\n  [{i + 1}] Query:    {row['sentence1'][:100]}")
        print(f"      Trial:    {row['sentence2'][:120]}...")

    print("\n--- 3 NEGATIVE examples (label=0.0) ---")
    for i, row in enumerate(negatives[:3]):
        print(f"\n  [{i + 1}] Query:    {row['sentence1'][:100]}")
        print(f"      Trial:    {row['sentence2'][:120]}...")

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
        train_size: Number of training pairs.
        val_size: Number of validation pairs.
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
    return parser.parse_args()


def main() -> None:
    """Run the cross-encoder fine-tuning pipeline."""
    args = parse_args()
    config = load_config(args.config)

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
    print(f"  Train pairs:    {len(train_ds):,}")
    print(f"  Val pairs:      {len(val_ds):,}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Epochs:         {epochs}")
    print(f"  Steps/epoch:    {steps_per_epoch:,}")
    print(f"  Total steps:    {total_steps:,}")
    print(f"  Warmup steps:   {warmup_steps:,}")
    print(f"  Learning rate:  {config['training']['learning_rate']}")
    print(f"  Max length:     {max_length}")
    print(f"  Loss:           {config['loss']['type']}")
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

    if args.dry_run:
        logger.info("DRY RUN — exiting before training.")
        return

    # ── Loss function ────────────────────────────────────────────────────
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    loss = BinaryCrossEntropyLoss(model=model)

    # ── Evaluator ────────────────────────────────────────────────────────
    logger.info(
        "Building CE reranking evaluator (subsampled to %d) ...",
        config["evaluation"]["max_val_samples"],
    )
    evaluator = build_evaluator(
        val_file=config["data"]["val_file"],
        max_samples=config["evaluation"]["max_val_samples"],
    )

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

    # Extract key metrics
    ndcg = eval_results.get("CERerankingEvaluator_ndcg@10", "N/A")
    map_score = eval_results.get("CERerankingEvaluator_map", "N/A")
    mrr_score = eval_results.get("CERerankingEvaluator_mrr@10", "N/A")

    print(f"  NDCG@10:        {ndcg}")
    print(f"  MAP:            {map_score}")
    print(f"  MRR@10:         {mrr_score}")
    print()
    print("Next steps:")
    print("  1. make mlflow  — view training curves")
    print("  2. Evaluate on labeled data:")
    print("     python scripts/evaluate_cross_encoder.py --model models/cross-encoder/fine-tuned")
    print("  3. Demo before/after re-ranking:")
    print("     python scripts/demo_reranker.py --model models/cross-encoder/fine-tuned")


if __name__ == "__main__":
    main()
