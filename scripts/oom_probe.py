"""Probe whether the configured batch size fits in GPU memory.

Runs 5 forward + backward passes of BioLinkBERT-base in fp16 at the batch
size from configs/training/embeddings.yaml (or --batch override) and reports
peak CUDA memory. Exits 0 on success, 1 on OOM (or if CUDA is unavailable).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training/embeddings.yaml")


def main() -> int:
    """Run the OOM probe and return an exit code (0=pass, 1=fail)."""
    parser = argparse.ArgumentParser(description="OOM probe for fine-tuning")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    batch: int = args.batch or cfg["training"]["batch_size"]
    model_name: str = cfg["model"]["name"]
    seq_len: int = cfg["model"]["max_seq_length"]

    if not torch.cuda.is_available():
        logger.error("CUDA unavailable; OOM probe is meaningless on CPU/MPS. Exit 1.")
        return 1

    device = torch.device("cuda")
    logger.info("Probing: model=%s batch=%d seq_len=%d steps=%d",
                model_name, batch, seq_len, args.steps)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    dummy = ["clinical trial " * 50] * batch

    try:
        for step in range(args.steps):
            optim.zero_grad()
            inp = tokenizer(dummy, padding="max_length", truncation=True,
                            max_length=seq_len, return_tensors="pt").to(device)
            out = model(**inp)
            out.last_hidden_state.mean().backward()
            optim.step()
            logger.info("Step %d/%d OK, peak %.2f GB", step + 1, args.steps,
                        torch.cuda.max_memory_allocated() / 1e9)
    except torch.cuda.OutOfMemoryError:
        logger.error("OOM at step %d, batch=%d", step + 1, batch)
        return 1

    logger.info("PROBE PASSED. Peak %.2f GB at batch=%d",
                torch.cuda.max_memory_allocated() / 1e9, batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
