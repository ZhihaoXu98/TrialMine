"""Fine-tune BioLinkBERT as a bi-encoder for clinical trial retrieval.

Uses MultipleNegativesRankingLoss with hard negatives from our generated
training data. Evaluates with InformationRetrievalEvaluator during training.
Logs everything to MLflow.

Usage:
    python scripts/finetune_embeddings.py
    python scripts/finetune_embeddings.py --dry-run
    python scripts/finetune_embeddings.py --config configs/training/embeddings.yaml

Config: configs/training/embeddings.yaml
Input:  data/training/train_pairs.jsonl, data/training/val_pairs.jsonl
Output: models/embeddings/fine-tuned/
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
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training/embeddings.yaml")


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


def load_training_data(config: dict) -> tuple:
    """Load train and val datasets from JSONL files.

    Filters out rows with empty negative fields. Renames columns
    to match sentence-transformers convention (anchor, positive, negative).

    Args:
        config: Full config dict.

    Returns:
        Tuple of (train_dataset, val_dataset) as HuggingFace Datasets.
    """
    data_config = config["data"]
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_config["train_file"],
            "val": data_config["val_file"],
        },
    )

    train_ds = dataset["train"]
    val_ds = dataset["val"]

    logger.info("Raw dataset: %d train, %d val", len(train_ds), len(val_ds))

    # Filter out rows with empty negatives
    train_ds = train_ds.filter(lambda x: x["negative"] and len(x["negative"].strip()) > 0)
    val_ds = val_ds.filter(lambda x: x["negative"] and len(x["negative"].strip()) > 0)

    logger.info("After filtering empty negatives: %d train, %d val", len(train_ds), len(val_ds))

    # Rename columns for sentence-transformers: anchor, positive, negative
    train_ds = train_ds.rename_column("query", "anchor")
    val_ds = val_ds.rename_column("query", "anchor")

    # Keep only the columns the loss function needs
    train_ds = train_ds.select_columns(["anchor", "positive", "negative"])
    val_ds = val_ds.select_columns(["anchor", "positive", "negative"])

    return train_ds, val_ds


# ── Evaluator ────────────────────────────────────────────────────────────────


def build_evaluator(
    val_file: str,
    max_samples: int,
    seed: int = 42,
) -> InformationRetrievalEvaluator:
    """Build an InformationRetrievalEvaluator from validation data.

    Subsamples to max_samples for speed. Uses NCT IDs as corpus keys.

    Args:
        val_file: Path to validation JSONL.
        max_samples: Maximum validation examples for the evaluator.
        seed: Random seed for subsampling.

    Returns:
        Configured InformationRetrievalEvaluator.
    """
    # Load raw val data for NCT IDs (not available in the renamed dataset)
    val_rows: list[dict] = []
    with open(val_file) as f:
        for line in f:
            try:
                val_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Filter empty negatives
    val_rows = [r for r in val_rows if r.get("negative", "").strip()]

    # Subsample
    rng = random.Random(seed)
    if len(val_rows) > max_samples:
        val_rows = rng.sample(val_rows, max_samples)

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for i, row in enumerate(val_rows):
        qid = f"q{i}"
        # Use nct_id + index to handle duplicate nct_ids
        cid = f"{row.get('nct_id', 'unknown')}_{i}"
        queries[qid] = row["query"]
        corpus[cid] = row["positive"]
        relevant_docs[qid] = {cid}

    logger.info(
        "IR evaluator: %d queries, %d corpus documents",
        len(queries), len(corpus),
    )

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        name="val-retrieval",
        batch_size=64,
    )


# ── Training example inspection ──────────────────────────────────────────────


def print_training_examples(train_ds, n: int = 5) -> None:
    """Print first N training examples for sanity checking.

    Args:
        train_ds: Training dataset.
        n: Number of examples to print.
    """
    print("\n" + "=" * 80)
    print(f"FIRST {n} TRAINING EXAMPLES")
    print("=" * 80)

    for i in range(min(n, len(train_ds))):
        row = train_ds[i]
        print(f"\n--- Example {i + 1} ---")
        print(f"  Anchor:   {row['anchor'][:100]}")
        print(f"  Positive: {row['positive'][:100]}...")
        print(f"  Negative: {row['negative'][:100]}...")

    print()


# ── Metadata saving ──────────────────────────────────────────────────────────


def save_metadata(
    output_dir: Path,
    config: dict,
    train_size: int,
    val_size: int,
    device: str,
    eval_results: dict | None = None,
) -> None:
    """Save training metadata to a JSON file alongside the model.

    Args:
        output_dir: Model output directory.
        config: Full training config.
        train_size: Number of training examples.
        val_size: Number of validation examples.
        device: Device used for training.
        eval_results: Final evaluation metrics (if available).
    """
    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "base_model": config["model"]["name"],
        "dataset_size": {"train": train_size, "val": val_size},
        "hyperparameters": {
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "warmup_ratio": config["training"]["warmup_ratio"],
            "weight_decay": config["training"]["weight_decay"],
            "loss": config["loss"]["type"],
            "loss_scale": config["loss"]["scale"],
            "max_seq_length": config["model"]["max_seq_length"],
        },
        "device": device,
        "eval_metrics": eval_results or {},
    }

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
        description="Fine-tune BioLinkBERT for clinical trial retrieval",
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
    """Run the fine-tuning pipeline."""
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
    logger.info("Loading base model: %s", model_name)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = config["model"]["max_seq_length"]
    logger.info(
        "Model loaded. Dimension: %d, max_seq_length: %d",
        model.get_sentence_embedding_dimension(),
        model.max_seq_length,
    )

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading training data ...")
    train_ds, val_ds = load_training_data(config)

    # ── Print examples for sanity check ──────────────────────────────────
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
    print(f"  Device:         {device}")
    print(f"  Train examples: {len(train_ds):,}")
    print(f"  Val examples:   {len(val_ds):,}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Epochs:         {epochs}")
    print(f"  Steps/epoch:    {steps_per_epoch:,}")
    print(f"  Total steps:    {total_steps:,}")
    print(f"  Warmup steps:   {warmup_steps:,}")
    print(f"  Learning rate:  {config['training']['learning_rate']}")
    print(f"  Loss:           {config['loss']['type']} (scale={config['loss']['scale']})")
    print(f"  Eval every:     {config['training']['eval_steps']} steps")
    print(f"  Log every:      {config['training']['logging_steps']} steps")
    print(f"  Output:         {config['model']['output_dir']}")
    print()

    if device == "cpu":
        est_hours = total_steps * 1.5 / 3600
        print(f"  Estimated time (CPU): ~{est_hours:.1f} hours")
    elif device == "mps":
        est_hours = total_steps * 0.4 / 3600
        print(f"  Estimated time (MPS): ~{est_hours:.1f} hours")

    print()

    if args.dry_run:
        logger.info("DRY RUN — exiting before training.")
        return

    # ── Loss function ────────────────────────────────────────────────────
    loss = MultipleNegativesRankingLoss(
        model=model,
        scale=config["loss"]["scale"],
    )

    # ── Evaluator ────────────────────────────────────────────────────────
    logger.info("Building IR evaluator (subsampled to %d) ...",
                config["evaluation"]["max_val_samples"])
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
        "device": device,
    })

    # ── Training arguments ───────────────────────────────────────────────
    output_dir = config["model"]["output_dir"]
    training_args = SentenceTransformerTrainingArguments(
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
    trainer = SentenceTransformerTrainer(
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
    logger.info("Training completed in %.1f minutes", elapsed / 60)

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
        eval_results=eval_results,
    )

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Duration:       {elapsed / 60:.1f} minutes")
    print(f"  Model saved:    {final_output}")

    # Extract key metrics
    ndcg = eval_results.get("val-retrieval_cosine_ndcg@10", "N/A")
    mrr = eval_results.get("val-retrieval_cosine_mrr@10", "N/A")
    recall_10 = eval_results.get("val-retrieval_cosine_recall@10", "N/A")
    recall_1 = eval_results.get("val-retrieval_cosine_recall@1", "N/A")

    print(f"  NDCG@10:        {ndcg}")
    print(f"  MRR@10:         {mrr}")
    print(f"  Recall@1:       {recall_1}")
    print(f"  Recall@10:      {recall_10}")
    print()
    print("Next steps:")
    print("  1. make mlflow  — view training curves")
    print(f"  2. Rebuild FAISS index with fine-tuned model:")
    print(f"     Update model path in TrialEmbedder or scripts/build_index.py")
    print(f"  3. Re-run: python scripts/compare_methods.py  — compare against baseline")


if __name__ == "__main__":
    main()
