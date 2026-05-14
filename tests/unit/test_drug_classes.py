"""Tests for the hand-curated UMLS drug -> drug-class CUI table.

Two tiers:

* **Fast** (always run, no linker needed): shape / format / minimum-
  size assertions on the static table.
* **Slow** (``@pytest.mark.slow`` + filesystem-gated): canonical Q413 /
  Q416 lookups that resolve CUIs through the scispacy UMLS linker at
  test time, so the assertions stay correct if the KB is upgraded.

The slow tests skip cleanly when scispacy's UMLS KB is not cached
locally — see :data:`_KB_CACHED`. We check the filesystem directly
because scispacy 0.6.2's :func:`scispacy.candidate_generation.UmlsLinkerPaths.is_locally_cached`
returns ``False`` even when the KB is fully present (confirmed in the
Phase A2 verification run on 2026-05-14).
"""

from __future__ import annotations

import pathlib
import re

import pytest

from TrialMine.features.concepts import link_to_cuis
from TrialMine.features.drug_classes import DRUG_TO_CLASS_CUIS

# Filesystem gate for tests that need scispacy's UMLS KB.
_SCISPACY_KB_DIR = pathlib.Path.home() / ".scispacy" / "datasets"
_KB_CACHED = _SCISPACY_KB_DIR.exists() and any(_SCISPACY_KB_DIR.iterdir())

# UMLS CUI: a "C" followed by 7 digits.
_CUI_RE = re.compile(r"^C\d{7}$")

_requires_kb = pytest.mark.skipif(
    not _KB_CACHED,
    reason=("scispacy UMLS KB not downloaded; run Phase A2 of docs/fix_parser_umls.md first"),
)


# --------------------------------------------------------------------------- #
# Fast tests (always run -- no linker needed)                                  #
# --------------------------------------------------------------------------- #


def test_table_has_minimum_entries() -> None:
    """Table must cover the canonical Q413 + Q416 cases plus breadth.

    The >= 20 floor comes from docs/fix_parser_umls.md A4. Hitting it
    forces curation across at least 3 drug families (single-family tables
    don't generalize to the complex+vague held-out set).
    """
    assert len(DRUG_TO_CLASS_CUIS) >= 20, (
        f"DRUG_TO_CLASS_CUIS has only {len(DRUG_TO_CLASS_CUIS)} entries; "
        "expected >= 20 (see docs/fix_parser_umls.md A4 EXTEND HERE recipe)"
    )


def test_every_key_is_well_formed_cui() -> None:
    """All keys must look like UMLS CUIs (``C`` followed by 7 digits)."""
    for cui in DRUG_TO_CLASS_CUIS:
        assert _CUI_RE.match(cui), f"Key {cui!r} is not a well-formed UMLS CUI"


def test_every_value_is_frozenset_of_well_formed_cuis() -> None:
    """Values must be immutable frozensets of well-formed CUIs."""
    for drug_cui, class_cuis in DRUG_TO_CLASS_CUIS.items():
        assert isinstance(class_cuis, frozenset), (
            f"Value for {drug_cui!r} is {type(class_cuis).__name__}, not frozenset"
        )
        for class_cui in class_cuis:
            assert _CUI_RE.match(class_cui), (
                f"Class CUI {class_cui!r} (under {drug_cui!r}) is not well-formed"
            )


def test_no_empty_class_sets() -> None:
    """Every drug must map to at least one class CUI."""
    for drug_cui, class_cuis in DRUG_TO_CLASS_CUIS.items():
        assert class_cuis, f"Drug {drug_cui!r} has empty class CUI set"


# --------------------------------------------------------------------------- #
# Slow tests (gated on KB cache -- linker-dependent)                           #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@_requires_kb
def test_q413_osimertinib_egfr_tki_class_membership() -> None:
    """Q413 canonical: scispacy CUI for ``osimertinib`` is a key in the
    table, and its class set overlaps the scispacy CUI(s) for ``EGFR
    tyrosine kinase inhibitor``.

    Both CUIs resolve at test time via :func:`link_to_cuis` (no
    hardcoded strings) so the assertion stays correct if scispacy's KB
    is upgraded later.
    """
    osi_links = link_to_cuis("osimertinib")
    assert osi_links, "scispacy KB returned no CUI for 'osimertinib'"
    osi_cui = osi_links[0].cui

    egfr_links = link_to_cuis("EGFR tyrosine kinase inhibitor")
    assert egfr_links, "scispacy KB returned no CUI for 'EGFR tyrosine kinase inhibitor'"
    egfr_class_cuis = {link.cui for link in egfr_links}

    assert osi_cui in DRUG_TO_CLASS_CUIS, (
        f"osimertinib CUI {osi_cui!r} is not a key in DRUG_TO_CLASS_CUIS"
    )
    assert DRUG_TO_CLASS_CUIS[osi_cui] & egfr_class_cuis, (
        f"DRUG_TO_CLASS_CUIS[{osi_cui!r}]={set(DRUG_TO_CLASS_CUIS[osi_cui])} "
        f"does not overlap EGFR-TKI class CUIs {egfr_class_cuis}"
    )


@pytest.mark.slow
@_requires_kb
def test_q416_trastuzumab_her2_class_membership() -> None:
    """Q416 canonical: scispacy CUI for ``trastuzumab`` is a key in the
    table, and its class set overlaps the scispacy CUI(s) for ``HER2
    inhibitor``.

    Uses the ``HER2 inhibitor`` phrasing — ``HER2-targeted therapy``
    does NOT resolve to a drug-type CUI in scispacy's 2022 KB (Finding
    #2 in docs/fix_parser_umls.md A4).
    """
    tras_links = link_to_cuis("trastuzumab")
    assert tras_links, "scispacy KB returned no CUI for 'trastuzumab'"
    tras_cui = tras_links[0].cui

    her2_links = link_to_cuis("HER2 inhibitor")
    assert her2_links, "scispacy KB returned no CUI for 'HER2 inhibitor'"
    her2_class_cuis = {link.cui for link in her2_links}

    assert tras_cui in DRUG_TO_CLASS_CUIS, (
        f"trastuzumab CUI {tras_cui!r} is not a key in DRUG_TO_CLASS_CUIS"
    )
    assert DRUG_TO_CLASS_CUIS[tras_cui] & her2_class_cuis, (
        f"DRUG_TO_CLASS_CUIS[{tras_cui!r}]={set(DRUG_TO_CLASS_CUIS[tras_cui])} "
        f"does not overlap HER2-inhibitor class CUIs {her2_class_cuis}"
    )


@pytest.mark.slow
@_requires_kb
def test_brand_name_aliasing_via_shared_class() -> None:
    """Brand (``Tagrisso``) and generic (``osimertinib``) have DIFFERENT
    direct CUIs in scispacy's 2022 KB, so direct-CUI overlap can't
    bridge them. The table gives both CUIs the SAME class-value set so
    :func:`TrialMine.features.concepts.match_via_cui` resolves them via
    the drug-class path.
    """
    brand_links = link_to_cuis("Tagrisso")
    generic_links = link_to_cuis("osimertinib")
    assert brand_links and generic_links, "linker returned no CUI for one of the inputs"

    brand_cui = brand_links[0].cui
    generic_cui = generic_links[0].cui

    # This is the design premise the table's brand-key entries work around.
    # If the KB ever consolidates them, the workaround becomes dead weight.
    assert brand_cui != generic_cui, (
        "Tagrisso and osimertinib unexpectedly share a CUI — the brand-"
        "aliasing workaround may no longer be needed (review the table)"
    )

    assert brand_cui in DRUG_TO_CLASS_CUIS, f"brand CUI {brand_cui!r} missing from table"
    assert generic_cui in DRUG_TO_CLASS_CUIS, f"generic CUI {generic_cui!r} missing from table"
    assert DRUG_TO_CLASS_CUIS[brand_cui] & DRUG_TO_CLASS_CUIS[generic_cui], (
        f"brand class set {set(DRUG_TO_CLASS_CUIS[brand_cui])} doesn't overlap "
        f"generic class set {set(DRUG_TO_CLASS_CUIS[generic_cui])}"
    )
