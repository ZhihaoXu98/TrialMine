"""Generate training data for BioLinkBERT fine-tuning.

Produces (query, positive, negative) triplets from three sources:
1. Metadata-derived pairs (conditions, interventions, phases)
2. Synthetic patient queries via Claude Haiku API
3. Hard negatives mined from same-condition trials

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --skip-synthetic
    python scripts/generate_training_data.py --dry-run
    python scripts/generate_training_data.py --resume   # resume interrupted API calls

Config: configs/training_data.yaml
Output: data/training/train_pairs.jsonl, data/training/val_pairs.jsonl
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from TrialMine.data.models import Trial
from TrialMine.data.store import load_trials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = Path("data/trials.db")
CONFIG_PATH = Path("configs/training_data.yaml")


# ── Trial text preparation ───────────────────────────────────────────────────
# Replicates TrialEmbedder.prepare_trial_text() without loading the model.
# Must stay in sync with src/TrialMine/models/embeddings.py.

MAX_TRIAL_TEXT_CHARS = 2048


def prepare_trial_text(trial: Trial) -> str:
    """Build the text representation used for embedding.

    Matches TrialEmbedder.prepare_trial_text() exactly:
    title [SEP] conditions [SEP] brief_summary, truncated to 2048 chars.

    Args:
        trial: A Trial object.

    Returns:
        Concatenated text string.
    """
    parts: list[str] = []
    if trial.title:
        parts.append(trial.title)
    if trial.conditions:
        parts.append(" ".join(trial.conditions))
    if trial.brief_summary:
        parts.append(trial.brief_summary)
    text = " [SEP] ".join(parts) if parts else ""
    if len(text) > MAX_TRIAL_TEXT_CHARS:
        text = text[:MAX_TRIAL_TEXT_CHARS]
    return text


# ── Cancer type classification ───────────────────────────────────────────────


def load_config(config_path: Path) -> dict:
    """Load the training data YAML config.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def classify_cancer_type(
    conditions: list[str],
    cancer_types: dict[str, list[str]],
) -> str:
    """Classify a trial into a cancer type group.

    Checks condition strings against keyword patterns (case-insensitive).
    Returns the first matching group, or 'other'.

    Args:
        conditions: List of condition strings from the trial.
        cancer_types: Mapping of group_name -> list of keyword patterns.

    Returns:
        Cancer type group name.
    """
    conditions_lower = " ".join(conditions).lower()
    for group_name, keywords in cancer_types.items():
        for keyword in keywords:
            if keyword.lower() in conditions_lower:
                return group_name
    return "other"


def build_cancer_group_index(
    trials: list[Trial],
    cancer_types: dict[str, list[str]],
) -> dict[str, list[Trial]]:
    """Index all trials by cancer type group.

    Args:
        trials: All trials from the database.
        cancer_types: Mapping from config.

    Returns:
        Dict mapping cancer_group -> list of Trial objects.
    """
    index: dict[str, list[Trial]] = defaultdict(list)
    for trial in trials:
        group = classify_cancer_type(trial.conditions, cancer_types)
        index[group].append(trial)
    return dict(index)


def stratified_sample(
    trials_by_group: dict[str, list[Trial]],
    max_per_group: int,
    rng: random.Random,
) -> dict[str, list[Trial]]:
    """Sample trials with a per-group cap for balanced representation.

    Args:
        trials_by_group: Cancer group index.
        max_per_group: Maximum trials to keep per group.
        rng: Seeded Random instance.

    Returns:
        Sampled cancer group index.
    """
    sampled: dict[str, list[Trial]] = {}
    for group, group_trials in trials_by_group.items():
        if len(group_trials) <= max_per_group:
            sampled[group] = list(group_trials)
        else:
            sampled[group] = rng.sample(group_trials, max_per_group)
    return sampled


# ── Condition keyword index (for hard negatives) ────────────────────────────


def build_condition_index(
    trials: list[Trial],
) -> dict[str, list[str]]:
    """Map condition keywords to NCT IDs for hard negative mining.

    Splits each condition string into lowercase words and indexes by word.

    Args:
        trials: List of Trial objects.

    Returns:
        Dict mapping keyword -> list of nct_ids.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for trial in trials:
        seen_keywords: set[str] = set()
        for condition in trial.conditions:
            for word in condition.lower().split():
                # Skip short/common words
                if len(word) < 3:
                    continue
                if word not in seen_keywords:
                    seen_keywords.add(word)
                    index[word].append(trial.nct_id)
    return dict(index)


# ── Source 1: Metadata-derived pairs ─────────────────────────────────────────


def generate_metadata_pairs(
    trials_by_group: dict[str, list[Trial]],
) -> list[dict]:
    """Generate (query, positive) pairs from trial metadata fields.

    For each trial, produces pairs from:
    - Each condition -> trial_text
    - Each intervention -> trial_text
    - condition + intervention -> trial_text
    - condition + phase -> trial_text

    Args:
        trials_by_group: Sampled cancer group index.

    Returns:
        List of dicts with keys: query, positive, nct_id, source, cancer_group.
    """
    pairs: list[dict] = []
    total_trials = sum(len(ts) for ts in trials_by_group.values())
    processed = 0

    for group, group_trials in trials_by_group.items():
        for trial in group_trials:
            trial_text = prepare_trial_text(trial)
            if not trial_text.strip():
                continue

            nct_id = trial.nct_id
            base = {"positive": trial_text, "nct_id": nct_id, "source": "metadata", "cancer_group": group}

            # Condition-based pairs
            for condition in trial.conditions:
                pairs.append({"query": condition, **base})

            # Intervention-based pairs
            for intervention in trial.interventions:
                pairs.append({"query": intervention, **base})

            # Condition + intervention
            if trial.conditions and trial.interventions:
                pairs.append({
                    "query": f"{trial.interventions[0]} for {trial.conditions[0]}",
                    **base,
                })

            # Condition + phase
            if trial.conditions and trial.phase:
                pairs.append({
                    "query": f"{trial.phase.lower()} {trial.conditions[0].lower()} trial",
                    **base,
                })

            processed += 1
            if processed % 5000 == 0:
                logger.info(
                    "Source 1 progress: %d / %d trials processed (%d pairs so far)",
                    processed, total_trials, len(pairs),
                )

    logger.info("Source 1 complete: %d metadata pairs from %d trials", len(pairs), processed)
    return pairs


# ── Source 2: Synthetic patient queries (Claude API) ─────────────────────────

SYSTEM_PROMPT = "You generate realistic patient search queries for clinical trials."

USER_PROMPT_TEMPLATE = """Write a realistic 1-2 sentence search query that a PATIENT (not a doctor) \
might type when looking for this trial. Use simple patient language. \
DO NOT use medical jargon or acronyms.

Good examples:
- 'lung cancer treatment options after chemo stopped working'
- 'is there a trial for breast cancer near Chicago'
- 'my colon cancer came back, what trials can I try'

Trial: {title}
Conditions: {conditions}
Eligibility (first 300 chars): {eligibility}

Respond with ONLY the patient query. Nothing else."""


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load NCT IDs already processed from checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint JSONL file.

    Returns:
        Set of NCT IDs already processed.
    """
    done: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add(row["nct_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("Loaded checkpoint: %d queries already generated", len(done))
    return done


def generate_synthetic_queries(
    trials_by_group: dict[str, list[Trial]],
    config: dict,
    resume: bool = False,
) -> list[dict]:
    """Generate patient-language queries via Claude Haiku API.

    Samples trials stratified by cancer type, calls the API for each,
    saves progress to a checkpoint file every N queries.

    Args:
        trials_by_group: Sampled cancer group index.
        config: Full config dict.
        resume: If True, skip already-processed NCT IDs from checkpoint.

    Returns:
        List of dicts with keys: query, positive, nct_id, source, cancer_group.
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set. Skipping synthetic query generation.")
        return []

    syn_config = config["synthetic"]
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config["output"]["checkpoint_file"]
    synthetic_path = output_dir / config["output"]["synthetic_file"]

    # Sample trials for synthetic generation (stratified)
    sample_count = syn_config["sample_count"]
    rng = random.Random(config["sampling"]["random_seed"] + 1)
    all_trials: list[tuple[str, Trial]] = []
    for group, group_trials in trials_by_group.items():
        for trial in group_trials:
            all_trials.append((group, trial))
    rng.shuffle(all_trials)
    all_trials = all_trials[:sample_count]

    # Resume support
    done_ids: set[str] = set()
    if resume:
        done_ids = load_checkpoint(checkpoint_path)

    client = anthropic.Anthropic(api_key=api_key)
    model = syn_config["model"]
    max_rps = syn_config["max_requests_per_second"]
    checkpoint_interval = syn_config["checkpoint_interval"]
    min_delay = 1.0 / max_rps

    pairs: list[dict] = []
    # Load existing checkpoint pairs if resuming
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    pending = [(group, trial) for group, trial in all_trials if trial.nct_id not in done_ids]
    logger.info(
        "Source 2: generating %d synthetic queries (%d already done, %d pending)",
        sample_count, len(done_ids), len(pending),
    )

    checkpoint_file = open(checkpoint_path, "a")
    try:
        for i, (group, trial) in enumerate(pending, 1):
            trial_text = prepare_trial_text(trial)
            if not trial_text.strip():
                continue

            user_prompt = USER_PROMPT_TEMPLATE.format(
                title=trial.title or "",
                conditions=", ".join(trial.conditions) if trial.conditions else "N/A",
                eligibility=(trial.eligibility_criteria or "")[:300],
            )

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=150,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                patient_query = response.content[0].text.strip().strip("'\"")
            except Exception:
                logger.exception("API call failed for %s, skipping", trial.nct_id)
                continue

            row = {
                "query": patient_query,
                "positive": trial_text,
                "nct_id": trial.nct_id,
                "source": "synthetic",
                "cancer_group": group,
            }
            pairs.append(row)

            # Checkpoint
            checkpoint_file.write(json.dumps(row) + "\n")
            if i % checkpoint_interval == 0:
                checkpoint_file.flush()
                logger.info("Source 2 progress: %d / %d queries generated", i, len(pending))

            # Rate limiting
            time.sleep(min_delay)
    finally:
        checkpoint_file.close()

    # Also save synthetic queries separately for inspection
    with open(synthetic_path, "w") as f:
        for row in pairs:
            f.write(json.dumps(row) + "\n")

    logger.info("Source 2 complete: %d synthetic patient queries", len(pairs))
    return pairs


# ── Source 3: Hard negatives ─────────────────────────────────────────────────


def mine_hard_negatives(
    pairs: list[dict],
    nct_to_trial: dict[str, Trial],
    condition_index: dict[str, list[str]],
    negatives_per_positive: int,
    rng: random.Random,
) -> list[dict]:
    """Add hard negatives to each (query, positive) pair.

    For each positive pair, finds trials sharing at least one condition
    keyword but with different NCT ID. Prefers trials with different
    interventions (harder negatives).

    Args:
        pairs: List of (query, positive) pair dicts.
        nct_to_trial: Mapping of nct_id -> Trial object.
        condition_index: Keyword -> list of nct_ids index.
        negatives_per_positive: Number of negatives to mine per pair.
        rng: Seeded Random instance.

    Returns:
        List of dicts with added 'negative' key (trial text of hard negative).
    """
    # Pre-compute intervention sets for faster lookup
    nct_interventions: dict[str, set[str]] = {}
    for nct_id, trial in nct_to_trial.items():
        nct_interventions[nct_id] = {i.lower() for i in trial.interventions}

    triplets: list[dict] = []
    no_neg_found = 0
    total = len(pairs)

    for idx, pair in enumerate(pairs):
        positive_nct = pair["nct_id"]
        positive_trial = nct_to_trial.get(positive_nct)
        if positive_trial is None:
            continue

        # Collect candidate NCT IDs sharing condition keywords
        candidates: set[str] = set()
        for condition in positive_trial.conditions:
            for word in condition.lower().split():
                if len(word) < 3:
                    continue
                for nct_id in condition_index.get(word, []):
                    if nct_id != positive_nct:
                        candidates.add(nct_id)

        if not candidates:
            no_neg_found += 1
            continue

        # Partition: prefer different-intervention candidates (harder)
        positive_interventions = nct_interventions.get(positive_nct, set())
        hard_candidates: list[str] = []
        easy_candidates: list[str] = []
        for nct_id in candidates:
            cand_interventions = nct_interventions.get(nct_id, set())
            if not cand_interventions & positive_interventions:
                hard_candidates.append(nct_id)
            else:
                easy_candidates.append(nct_id)

        # Pick negatives: hard first, then easy to fill remainder
        selected: list[str] = []
        if len(hard_candidates) >= negatives_per_positive:
            selected = rng.sample(hard_candidates, negatives_per_positive)
        else:
            selected = list(hard_candidates)
            remaining = negatives_per_positive - len(selected)
            if easy_candidates:
                selected += rng.sample(easy_candidates, min(remaining, len(easy_candidates)))

        for neg_nct_id in selected:
            neg_trial = nct_to_trial.get(neg_nct_id)
            if neg_trial is None:
                continue
            neg_text = prepare_trial_text(neg_trial)
            if not neg_text.strip():
                continue
            triplets.append({
                "query": pair["query"],
                "positive": pair["positive"],
                "negative": neg_text,
                "nct_id": pair["nct_id"],
                "source": pair["source"],
                "cancer_group": pair["cancer_group"],
            })

        if (idx + 1) % 10000 == 0:
            logger.info(
                "Source 3 progress: %d / %d pairs processed (%d triplets)",
                idx + 1, total, len(triplets),
            )

    logger.info(
        "Source 3 complete: %d triplets from %d pairs (%d pairs had no candidates)",
        len(triplets), total, no_neg_found,
    )
    return triplets


# ── Split and save ───────────────────────────────────────────────────────────


def split_and_save(
    triplets: list[dict],
    val_fraction: float,
    output_dir: Path,
    train_file: str,
    val_file: str,
    rng: random.Random,
) -> tuple[int, int]:
    """Split triplets by trial (NCT ID) and write JSONL files.

    All pairs for a given NCT ID go to the same split to prevent leakage.

    Args:
        triplets: List of triplet dicts.
        val_fraction: Fraction of trials for validation.
        output_dir: Output directory path.
        train_file: Training file name.
        val_file: Validation file name.
        rng: Seeded Random instance.

    Returns:
        Tuple of (train_count, val_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split by unique NCT IDs
    all_nct_ids = list({t["nct_id"] for t in triplets})
    rng.shuffle(all_nct_ids)
    val_count = max(1, int(len(all_nct_ids) * val_fraction))
    val_nct_ids = set(all_nct_ids[:val_count])

    train_data: list[dict] = []
    val_data: list[dict] = []
    for t in triplets:
        if t["nct_id"] in val_nct_ids:
            val_data.append(t)
        else:
            train_data.append(t)

    rng.shuffle(train_data)
    rng.shuffle(val_data)

    train_path = output_dir / train_file
    val_path = output_dir / val_file

    with open(train_path, "w") as f:
        for row in train_data:
            f.write(json.dumps(row) + "\n")

    with open(val_path, "w") as f:
        for row in val_data:
            f.write(json.dumps(row) + "\n")

    logger.info("Saved %d train, %d val to %s", len(train_data), len(val_data), output_dir)
    return len(train_data), len(val_data)


# ── Summary printing ─────────────────────────────────────────────────────────


def print_summary(
    metadata_pairs: list[dict],
    synthetic_pairs: list[dict],
    triplets: list[dict],
    trials_by_group: dict[str, list[Trial]],
    train_count: int,
    val_count: int,
) -> None:
    """Print generation summary with counts and examples.

    Args:
        metadata_pairs: Source 1 pairs.
        synthetic_pairs: Source 2 pairs.
        triplets: Final triplets with hard negatives.
        trials_by_group: Cancer group index used.
        train_count: Number of training examples saved.
        val_count: Number of validation examples saved.
    """
    print("\n" + "=" * 80)
    print("TRAINING DATA GENERATION SUMMARY")
    print("=" * 80)

    print(f"\nSource 1 (metadata):  {len(metadata_pairs):>8,} pairs")
    print(f"Source 2 (synthetic): {len(synthetic_pairs):>8,} pairs")
    total_pairs = len(metadata_pairs) + len(synthetic_pairs)
    print(f"Total pairs:          {total_pairs:>8,}")
    print(f"Total triplets:       {len(triplets):>8,} (with hard negatives)")
    print(f"\nTrain: {train_count:,}  |  Val: {val_count:,}")

    print(f"\nCancer type distribution ({len(trials_by_group)} groups):")
    print(f"  {'Group':<20} {'Trials':>8} {'Pairs':>8}")
    print(f"  {'-' * 38}")
    group_pair_counts: dict[str, int] = defaultdict(int)
    for p in metadata_pairs + synthetic_pairs:
        group_pair_counts[p["cancer_group"]] += 1
    for group in sorted(trials_by_group.keys()):
        trial_ct = len(trials_by_group[group])
        pair_ct = group_pair_counts.get(group, 0)
        print(f"  {group:<20} {trial_ct:>8,} {pair_ct:>8,}")

    # Example pairs from each source
    for source_name, source_pairs in [("metadata", metadata_pairs), ("synthetic", synthetic_pairs)]:
        print(f"\nExample {source_name} pairs:")
        samples = source_pairs[:5] if len(source_pairs) >= 5 else source_pairs
        for i, p in enumerate(samples, 1):
            query_preview = p["query"][:80]
            positive_preview = p["positive"][:60]
            print(f"  {i}. Query: {query_preview}")
            print(f"     Positive: {positive_preview}...")
            print()


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate training data for BioLinkBERT fine-tuning",
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH,
        help="Path to config YAML (default: configs/training_data.yaml)",
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip Claude API synthetic query generation",
    )
    parser.add_argument(
        "--skip-negatives", action="store_true",
        help="Skip hard negative mining (output pairs only, no triplets)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted synthetic query generation from checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics without writing output files",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training data generation pipeline."""
    args = parse_args()
    config = load_config(args.config)

    # ── Load trials ──────────────────────────────────────────────────────
    logger.info("Loading trials from %s ...", DB_PATH)
    all_trials = load_trials(DB_PATH)
    logger.info("Loaded %d trials", len(all_trials))

    # ── Classify and sample ──────────────────────────────────────────────
    cancer_types = config["cancer_types"]
    trials_by_group = build_cancer_group_index(all_trials, cancer_types)

    logger.info("Cancer type distribution (before sampling):")
    for group in sorted(trials_by_group.keys()):
        logger.info("  %s: %d trials", group, len(trials_by_group[group]))

    rng = random.Random(config["sampling"]["random_seed"])
    max_per_group = config["sampling"]["max_trials_per_cancer_group"]
    sampled = stratified_sample(trials_by_group, max_per_group, rng)
    total_sampled = sum(len(ts) for ts in sampled.values())
    logger.info("Sampled %d trials (capped at %d per group)", total_sampled, max_per_group)

    if args.dry_run:
        logger.info("DRY RUN — estimating pair counts:")
        estimated_pairs = 0
        for group, group_trials in sampled.items():
            group_pairs = 0
            for trial in group_trials:
                group_pairs += len(trial.conditions)
                group_pairs += len(trial.interventions)
                if trial.conditions and trial.interventions:
                    group_pairs += 1
                if trial.conditions and trial.phase:
                    group_pairs += 1
            estimated_pairs += group_pairs
            logger.info("  %s: %d trials -> ~%d pairs", group, len(group_trials), group_pairs)
        logger.info("Estimated total metadata pairs: %d", estimated_pairs)
        logger.info("Estimated synthetic pairs: %d", config["synthetic"]["sample_count"])
        logger.info("Estimated triplets: ~%d", estimated_pairs * config["hard_negatives"]["negatives_per_positive"])
        return

    # ── Build lookup structures ──────────────────────────────────────────
    # Flat list of all sampled trials for condition index
    all_sampled_trials = [t for ts in sampled.values() for t in ts]
    nct_to_trial = {t.nct_id: t for t in all_sampled_trials}
    condition_index = build_condition_index(all_sampled_trials)
    logger.info("Condition index: %d keywords", len(condition_index))

    # ── Source 1: Metadata pairs ─────────────────────────────────────────
    logger.info("Generating Source 1: metadata-derived pairs ...")
    metadata_pairs = generate_metadata_pairs(sampled)

    # ── Source 2: Synthetic queries ──────────────────────────────────────
    synthetic_pairs: list[dict] = []
    if not args.skip_synthetic:
        logger.info("Generating Source 2: synthetic patient queries ...")
        synthetic_pairs = generate_synthetic_queries(sampled, config, resume=args.resume)
    else:
        logger.info("Skipping Source 2 (synthetic queries)")

    # ── Source 3: Hard negatives ─────────────────────────────────────────
    all_pairs = metadata_pairs + synthetic_pairs
    if not args.skip_negatives:
        logger.info("Mining Source 3: hard negatives ...")
        negatives_per = config["hard_negatives"]["negatives_per_positive"]
        triplets = mine_hard_negatives(all_pairs, nct_to_trial, condition_index, negatives_per, rng)
    else:
        logger.info("Skipping Source 3 (hard negatives) — saving pairs only")
        # Save as triplets without negative field for consistent format
        triplets = [{**p, "negative": ""} for p in all_pairs]

    # ── Split and save ───────────────────────────────────────────────────
    output_dir = Path(config["output"]["dir"])
    val_fraction = config["split"]["val_fraction"]
    train_count, val_count = split_and_save(
        triplets,
        val_fraction,
        output_dir,
        config["output"]["train_file"],
        config["output"]["val_file"],
        rng,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print_summary(metadata_pairs, synthetic_pairs, triplets, sampled, train_count, val_count)


if __name__ == "__main__":
    main()
