"""Lightweight FAISS index builder — minimal memory footprint.

Skips sentence-transformers and the full TrialMine package to fit
in memory-constrained environments. Uses raw transformers + torch.

Usage:
    python scripts/build_faiss.py [--db PATH] [--output PATH]
                                  [--model NAME] [--batch-size N] [--chunk-size N]
"""

import argparse
import gc
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

# Disable tokenizer parallelism to avoid forked-process segfaults with FAISS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings, respecting the attention mask."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def build_text(title, conditions_json, summary):
    """Build embedding text from trial fields."""
    parts = []
    if title:
        parts.append(title)
    if conditions_json:
        conds = json.loads(conditions_json)
        if conds:
            parts.append(" ".join(conds))
    if summary:
        parts.append(summary)
    text = " [SEP] ".join(parts) if parts else ""
    return text[:2048]


def main():
    parser = argparse.ArgumentParser(description="Build FAISS semantic index")
    parser.add_argument("--db", default="data/trials.db")
    parser.add_argument("--output", default="data/trial_embeddings.faiss")
    parser.add_argument("--model", default="michiyasunaga/BioLinkBERT-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=2000)
    args = parser.parse_args()

    # Load tokenizer and model
    logger.info("Loading model %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    dim = model.config.hidden_size
    logger.info("Model loaded, dim=%d", dim)

    # Count trials
    conn = sqlite3.connect(args.db)
    n = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    logger.info("Found %d trials", n)

    # Build index incrementally
    faiss_index = faiss.IndexFlatIP(dim)
    all_trial_ids = []
    title_lookup = {}

    t0 = time.time()
    for offset in range(0, n, args.chunk_size):
        end = min(offset + args.chunk_size, n)
        logger.info("Processing %d-%d / %d ...", offset, end, n)

        rows = conn.execute(
            "SELECT nct_id, title, conditions, brief_summary FROM trials LIMIT ? OFFSET ?",
            (args.chunk_size, offset),
        ).fetchall()

        texts = []
        for nct_id, title, conds_json, summary in rows:
            all_trial_ids.append(nct_id)
            title_lookup[nct_id] = (title or "N/A")[:90]
            texts.append(build_text(title, conds_json, summary))
        del rows

        # Encode in sub-batches
        all_embs = []
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i : i + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = model(**encoded)
            embs = mean_pooling(output, encoded["attention_mask"])
            # L2 normalize
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy().astype(np.float32))
            del encoded, output, embs

        chunk_embs = np.vstack(all_embs)
        faiss_index.add(chunk_embs)
        del texts, all_embs, chunk_embs
        gc.collect()

    elapsed = time.time() - t0
    conn.close()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path = str(output_path.with_suffix(".json"))

    faiss.write_index(faiss_index, str(output_path))
    with open(mapping_path, "w") as f:
        json.dump(all_trial_ids, f)

    print(f"\n{'='*55}")
    print("  SEMANTIC INDEX COMPLETE")
    print(f"{'='*55}")
    print(f"  Trials embedded : {n:,}")
    print(f"  Embedding time  : {elapsed:.1f}s ({n / elapsed:.1f} trials/sec)")
    print(f"  Embedding dim   : {dim}")
    print(f"  Index size      : {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  FAISS index     : {output_path}")
    print(f"  ID mapping      : {mapping_path}")
    print(f"{'='*55}")

    # Test query
    test_query = "immunotherapy for melanoma that has spread"
    encoded = tokenizer([test_query], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    query_emb = mean_pooling(output, encoded["attention_mask"])
    query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1).numpy().astype(np.float32)

    scores, indices = faiss_index.search(query_emb, 5)
    print(f"\n  Semantic test query: '{test_query}'\n")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx >= 0:
            nct_id = all_trial_ids[idx]
            print(f"  {i}. [{nct_id}] (cosine={score:.4f})")
            print(f"     {title_lookup.get(nct_id, 'N/A')}")
            print()


if __name__ == "__main__":
    main()
