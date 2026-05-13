"""Eligibility hard-filter logic, shared between orchestrator (live serving)
and evaluation scripts.

Keeps the "what counts as a hard mismatch" rule in one place so live pipeline
and offline eval always agree on which trials would be dropped.

Background (Phase C4):
    The eligibility parser's overall verdict rolls Met/Unmet/Unknown across
    all criteria. But criteria-level precision varies wildly. Closed-vocab
    fields (age, sex, drug-vocabulary excluded_prior_treatments) have high
    precision; open-vocab fields (required_conditions extracted via SciSpacy,
    required_prior_treatments via literal string match) have ~70 % precision
    per CLAUDE.md Decision 17 — they often false-positive Unmet on boilerplate
    text or drug-class equivalences.

    Filtering on the overall verdict drops too many trials. Filtering only on
    high-precision criteria drops the right ones (sex/age mismatches for
    pediatric-patient adult trials, drug-name-exact excluded treatments).
"""

from __future__ import annotations

# Closed-vocab criteria with high parser precision. Hard-Unmet here legitimately
# drops the trial. Other criteria (required_conditions, required_prior_treatments)
# are surfaced in the verdict annotation but do NOT trigger drops.
HARD_FILTER_CRITERIA: tuple[str, ...] = (
    "age",
    "sex",
    "excluded_prior_treatments",
)


def is_hard_unmet(elig: dict | None) -> bool:
    """True iff any HARD_FILTER_CRITERIA criterion returned verdict 'Unmet'.

    Returns False for missing eligibility, missing criteria dict, or all-soft
    Unmets. The intent is "drop only when we have a regex-clean signal."
    """
    if not elig or "criteria" not in elig:
        return False
    criteria = elig["criteria"]
    for name in HARD_FILTER_CRITERIA:
        c = criteria.get(name)
        if isinstance(c, dict) and c.get("verdict") == "Unmet":
            return True
    return False


def apply_hard_filter(
    ranked: list[dict],
    eligibility_results: list[dict | None],
) -> tuple[list[dict], list[dict | None], int, bool]:
    """Drop trials whose eligibility check is hard-Unmet.

    Safety net: if the filter would empty the result set entirely, the
    unfiltered lists are returned with ``safety_net_triggered=True``. This
    prevents "no results" outcomes when the eligibility parser is broadly
    unreliable (e.g., trials with empty eligibility sections) or the patient
    profile is highly constrained.

    Args:
        ranked: Trial dicts in pipeline-ranked order. Each must have an
            ``nct_id`` key for identification (not used by the filter, but
            needed by callers).
        eligibility_results: Parallel list (same length as ``ranked``) of
            eligibility verdict dicts from ``check_trial_eligibility``, or
            ``None`` when the check was skipped. Must align by index with
            ``ranked``.

    Returns:
        Tuple of:
          - filtered_ranked: ``ranked`` minus dropped trials (or unchanged on
            safety net)
          - filtered_eligibility: parallel-aligned eligibility verdicts (or
            unchanged on safety net)
          - n_dropped: number of trials dropped (0 on safety net)
          - safety_net_triggered: True when the filter would have emptied the
            result set; both lists are returned unchanged in that case
    """
    if len(ranked) != len(eligibility_results):
        raise ValueError(
            f"ranked ({len(ranked)}) and eligibility_results "
            f"({len(eligibility_results)}) must align by index"
        )
    keep_indices = [i for i in range(len(ranked)) if not is_hard_unmet(eligibility_results[i])]
    n_dropped = len(ranked) - len(keep_indices)
    if not keep_indices and ranked:
        return ranked, eligibility_results, 0, True
    return (
        [ranked[i] for i in keep_indices],
        [eligibility_results[i] for i in keep_indices],
        n_dropped,
        False,
    )
