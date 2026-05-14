"""Unit tests for :mod:`TrialMine.features.concepts`.

Two tiers:

* **Fast** (always run): pure-function tests of :class:`ConceptNormalizer`
  and the abbreviation-expansion regex used by :func:`link_to_cuis`.
* **Slow** (``@pytest.mark.slow``, filesystem-gated on scispacy KB):
  end-to-end checks that abbreviation expansion lets the UMLS linker
  find drug-class CUIs that would otherwise miss.
"""

from __future__ import annotations

import pathlib

import pytest

from TrialMine.features.concepts import (
    ConceptNormalizer,
    _expand_abbreviations,
)

# Filesystem gate for tests that need scispacy's UMLS KB. scispacy 0.6.2's
# UmlsLinkerPaths.is_locally_cached() is unreliable (returns False even
# when the KB is fully cached); check the cache dir directly.
_SCISPACY_KB_DIR = pathlib.Path.home() / ".scispacy" / "datasets"
_KB_CACHED = _SCISPACY_KB_DIR.exists() and any(_SCISPACY_KB_DIR.iterdir())

_requires_kb = pytest.mark.skipif(
    not _KB_CACHED,
    reason=("scispacy UMLS KB not downloaded; run Phase A2 of docs/fix_parser_umls.md first"),
)


def test_normalize_known_term_returns_medical_form() -> None:
    """Lay phrase in the synonym map → broader medical term."""
    nz = ConceptNormalizer()
    assert nz.normalize("stomach cancer") == "gastric neoplasm"
    assert nz.normalize("breast cancer") == "breast neoplasm"
    # Case-insensitive
    assert nz.normalize("Lung Cancer") == "lung neoplasm"


def test_normalize_unknown_term_returns_input_unchanged() -> None:
    """No mapping → input is echoed back verbatim (single-hop contract)."""
    nz = ConceptNormalizer()
    assert nz.normalize("hepatocellular carcinoma") == "hepatocellular carcinoma"
    assert nz.normalize("xyz123") == "xyz123"
    # Whitespace-only stays as-is
    assert nz.normalize("") == ""


def test_expand_query_returns_two_variants_when_lay_match_present() -> None:
    """Query with a lay phrase → ``[original, expanded]``; original first."""
    nz = ConceptNormalizer()
    out = nz.expand_query("my mom has stomach cancer")
    assert len(out) == 2
    assert out[0] == "my mom has stomach cancer"
    assert "gastric neoplasm" in out[1]

    # Leftmost-first match wins: "cancer spread" lands at offset 0 and is
    # consumed before the regex reaches the trailing "spread to bone", so
    # the expansion is "metastatic disease to bone" (not "bone metastasis").
    # The contract we care about is *that an expansion happened*.
    out2 = nz.expand_query("cancer spread to bone")
    assert len(out2) == 2 and out2[0] != out2[1]
    assert "metastatic" in out2[1]


def test_expand_query_returns_single_entry_when_no_match() -> None:
    """Query with no lay phrases → single-entry list."""
    nz = ConceptNormalizer()
    out = nz.expand_query("hepatocellular carcinoma stage IV")
    assert out == ["hepatocellular carcinoma stage IV"]
    # Empty / whitespace-only handled
    assert nz.expand_query("") == [""]


# --------------------------------------------------------------------------- #
# Fast tests for the abbreviation expansion regex (pure function — no KB).    #
# --------------------------------------------------------------------------- #


def test_expand_abbreviations_TKI_word_boundary() -> None:
    """TKI / TKIs / 'EGFR TKI' / 'TKI:' all expand to the spelled-out form."""
    assert _expand_abbreviations("TKI") == "tyrosine kinase inhibitor"
    assert _expand_abbreviations("TKIs") == "tyrosine kinase inhibitor"
    assert _expand_abbreviations("EGFR TKI") == "EGFR tyrosine kinase inhibitor"
    assert _expand_abbreviations("no prior EGFR TKI") == "no prior EGFR tyrosine kinase inhibitor"
    # Punctuation boundaries
    assert _expand_abbreviations("TKI:") == "tyrosine kinase inhibitor:"
    # Inside another token must NOT expand
    assert _expand_abbreviations("TKIs2") == "TKIs2"
    assert _expand_abbreviations("aTKIb") == "aTKIb"


def test_expand_abbreviations_ICI_and_PARP_inhibitor_shorthand() -> None:
    """ICI / ICIs and PARP-i / PARPi / PARP-I / PARPI all expand correctly."""
    assert _expand_abbreviations("ICI") == "immune checkpoint inhibitor"
    assert _expand_abbreviations("ICIs") == "immune checkpoint inhibitor"
    assert (
        _expand_abbreviations("post ICI progression")
        == "post immune checkpoint inhibitor progression"
    )

    assert _expand_abbreviations("PARP-i") == "PARP inhibitor"
    assert _expand_abbreviations("PARPi") == "PARP inhibitor"
    assert _expand_abbreviations("PARP-I") == "PARP inhibitor"
    assert _expand_abbreviations("PARPI") == "PARP inhibitor"


def test_expand_abbreviations_does_not_expand_bare_protein_names() -> None:
    """Bare PARP / EGFR / HER2 / CDK4/6 stay unchanged — they are protein /
    gene names, not drug classes. Expanding them would create false
    drug-class matches against e.g. the gene CUI for PARP.
    """
    assert _expand_abbreviations("PARP") == "PARP"
    assert _expand_abbreviations("PARP failure") == "PARP failure"
    assert _expand_abbreviations("EGFR") == "EGFR"
    assert _expand_abbreviations("EGFR positive") == "EGFR positive"
    assert _expand_abbreviations("HER2") == "HER2"
    assert _expand_abbreviations("CDK4/6") == "CDK4/6"


def test_expand_abbreviations_idempotent_on_spelled_out_forms() -> None:
    """Already-spelled-out drug-class phrases pass through unchanged.

    Important property: calling :func:`_expand_abbreviations` twice on
    any input yields the same result as calling it once. (We rely on
    this for the LRU cache in :func:`link_to_cuis` to behave sanely.)
    """
    inputs = [
        "tyrosine kinase inhibitor",
        "EGFR tyrosine kinase inhibitor",
        "immune checkpoint inhibitor",
        "PARP inhibitor",
        "no prior osimertinib",
        "",
    ]
    for s in inputs:
        once = _expand_abbreviations(s)
        twice = _expand_abbreviations(once)
        assert once == twice, f"expansion not idempotent on {s!r}: {once!r} != {twice!r}"


# --------------------------------------------------------------------------- #
# Slow tests (gated on KB cache) — abbreviation expansion via link_to_cuis.   #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@_requires_kb
def test_link_to_cuis_resolves_TKI_after_expansion() -> None:
    """link_to_cuis('EGFR TKI') should overlap link_to_cuis('EGFR
    tyrosine kinase inhibitor') — the abbreviation pre-expansion makes
    them equivalent through the linker.
    """
    from TrialMine.features.concepts import link_to_cuis

    abbrev_cuis = {link.cui for link in link_to_cuis("EGFR TKI")}
    spelled_cuis = {link.cui for link in link_to_cuis("EGFR tyrosine kinase inhibitor")}

    assert abbrev_cuis, "'EGFR TKI' produced no drug-type CUIs after expansion"
    assert abbrev_cuis & spelled_cuis, (
        f"'EGFR TKI' CUIs {abbrev_cuis} don't overlap 'EGFR tyrosine "
        f"kinase inhibitor' CUIs {spelled_cuis}"
    )


@pytest.mark.slow
@_requires_kb
def test_link_to_cuis_resolves_PARP_inhibitor_shorthand() -> None:
    """link_to_cuis('PARPi') overlaps link_to_cuis('PARP inhibitor')."""
    from TrialMine.features.concepts import link_to_cuis

    abbrev_cuis = {link.cui for link in link_to_cuis("PARPi")}
    spelled_cuis = {link.cui for link in link_to_cuis("PARP inhibitor")}

    assert abbrev_cuis, "'PARPi' produced no drug-type CUIs after expansion"
    assert abbrev_cuis & spelled_cuis, (
        f"'PARPi' CUIs {abbrev_cuis} don't overlap 'PARP inhibitor' CUIs {spelled_cuis}"
    )


@pytest.mark.slow
@_requires_kb
def test_link_to_cuis_does_not_link_bare_PARP_as_drug() -> None:
    """Bare 'PARP' (the enzyme name) must NOT resolve to a drug-type
    CUI. The abbreviation expansion is deliberately constrained to
    forms ending in ``-i`` / ``i`` / ``-I`` / ``I``; leaving the enzyme
    name alone prevents 'PARP failure' eligibility text from
    false-matching olaparib via the drug-class table.
    """
    from TrialMine.features.concepts import link_to_cuis

    bare_cuis = {link.cui for link in link_to_cuis("PARP")}
    # Implementation may still return *some* concept (e.g., the
    # gene/enzyme CUI), but it must not surface a drug-class CUI that
    # would clobber the drug-class matching path. We assert the bare
    # input does not produce the same drug-class CUI as the
    # inhibitor form.
    inhibitor_cuis = {link.cui for link in link_to_cuis("PARP inhibitor")}
    assert not (bare_cuis & inhibitor_cuis), (
        f"bare 'PARP' CUIs {bare_cuis} unexpectedly overlap "
        f"'PARP inhibitor' CUIs {inhibitor_cuis} — the abbreviation "
        f"expansion may be over-eager"
    )


# --------------------------------------------------------------------------- #
# Phase A6: link_to_cuis + match_via_cui coverage tests.                      #
# --------------------------------------------------------------------------- #

_DRUG_SEMANTIC_TYPES_FOR_TEST = frozenset({"T121", "T200", "T109", "T123", "T129"})


@pytest.mark.slow
@_requires_kb
def test_link_known_drug_resolves_to_cui() -> None:
    """osimertinib resolves to >= 1 ConceptLink with a drug-relevant type.

    Asserts membership in the drug-type semantic-type set rather than
    a specific CUI, so the test stays correct if scispacy ranks
    candidates differently across KB versions.
    """
    from TrialMine.features.concepts import link_to_cuis

    links = link_to_cuis("osimertinib")
    assert len(links) >= 1
    assert any(t in _DRUG_SEMANTIC_TYPES_FOR_TEST for link in links for t in link.semantic_types), (
        f"no drug-type semantic type in osimertinib links: {links}"
    )


@pytest.mark.slow
@_requires_kb
def test_link_drug_class_resolves_to_cui() -> None:
    """ "EGFR tyrosine kinase inhibitor" resolves to >= 1 ConceptLink.

    Canonical name should reference EGFR or "kinase inhibitor" — we
    don't assert a specific CUI because scispacy returns multiple
    plausible candidates (the test from A3 surfaced 3) and the
    ordering isn't stable across versions.
    """
    from TrialMine.features.concepts import link_to_cuis

    links = link_to_cuis("EGFR tyrosine kinase inhibitor")
    assert len(links) >= 1
    assert any(
        ("egfr" in link.canonical_name.lower())
        or ("kinase inhibitor" in link.canonical_name.lower())
        for link in links
    ), f"no EGFR/kinase-inhibitor canonical name: {[link.canonical_name for link in links]}"


@pytest.mark.slow
@_requires_kb
def test_link_filters_out_anatomy() -> None:
    """'kidney' returns zero drug-type ConceptLinks.

    scispacy may extract anatomy/disease entities for "kidney"; the
    semantic-type filter in :func:`link_to_cuis` drops anything not
    in :data:`_DRUG_SEMANTIC_TYPES`. The returned tuple should be
    empty (no surviving drug-relevant entities).
    """
    from TrialMine.features.concepts import link_to_cuis

    links = link_to_cuis("kidney")
    assert links == (), f"expected no drug-type CUIs for 'kidney', got {links}"


@pytest.mark.slow
@_requires_kb
def test_link_to_cuis_returns_tuple() -> None:
    """Result is a tuple — immutable + hashable for downstream caching."""
    from TrialMine.features.concepts import link_to_cuis

    result = link_to_cuis("osimertinib")
    assert isinstance(result, tuple)
    # Hashable: hash() raises TypeError on unhashable input. If we get
    # here without an exception, the result is hashable.
    hash(result)


@pytest.mark.slow
@_requires_kb
def test_link_to_cuis_caches() -> None:
    """Second call with the same arg returns in < 5 ms (LRU cache hit).

    Times the second call after priming. 5 ms gives plenty of headroom
    above the typical microsecond LRU dict lookup but tolerates CI
    machine load variance.
    """
    import time

    from TrialMine.features.concepts import link_to_cuis

    # Prime the cache (may load the linker; not timed).
    _ = link_to_cuis("osimertinib")

    t0 = time.perf_counter()
    _ = link_to_cuis("osimertinib")
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 5.0, f"cache hit took {dt_ms:.3f} ms, expected < 5 ms"


@pytest.mark.slow
@_requires_kb
def test_match_via_cui_alias() -> None:
    """Direct CUI overlap path: same surface form on both sides.

    Note: the runbook originally framed this as a "Tagrisso ↔
    osimertinib" alias case, but scispacy's 2022 KB assigns DIFFERENT
    direct CUIs to brand vs generic names (C4058817 vs C4058811),
    so brand/generic uses the shared-class path in
    :data:`DRUG_TO_CLASS_CUIS` (covered by
    ``test_brand_name_aliasing_via_shared_class`` in
    ``test_drug_classes.py``). Here we exercise the direct-overlap
    path with a case where it actually fires.
    """
    from TrialMine.features.concepts import match_via_cui

    ok, matched = match_via_cui("osimertinib", "patient took osimertinib previously")
    assert ok
    assert matched, "matched CUI list should be non-empty when direct overlap fires"


@pytest.mark.slow
@_requires_kb
def test_match_via_cui_drug_to_class() -> None:
    """osimertinib ↔ "EGFR tyrosine kinase inhibitor" via DRUG_TO_CLASS_CUIS.

    Drug-class overlap path: osimertinib's direct CUI is a key in
    DRUG_TO_CLASS_CUIS, whose value set overlaps the class CUIs the
    linker returns for "EGFR tyrosine kinase inhibitor".
    """
    from TrialMine.features.concepts import match_via_cui

    ok, matched = match_via_cui("osimertinib", "EGFR tyrosine kinase inhibitor")
    assert ok
    assert matched


@pytest.mark.slow
@_requires_kb
def test_match_via_cui_unrelated_drugs() -> None:
    """osimertinib ↔ trastuzumab returns False (drugs in different classes).

    Neither direct CUI overlap (different drug CUIs) nor class overlap
    (EGFR-TKI vs HER2-inhibitor classes don't intersect) should fire.
    """
    from TrialMine.features.concepts import match_via_cui

    ok, matched = match_via_cui("osimertinib", "trastuzumab")
    assert not ok
    assert matched == []


@pytest.mark.slow
@_requires_kb
def test_match_via_cui_empty_input() -> None:
    """Empty / whitespace-only input on either side returns (False, [])."""
    from TrialMine.features.concepts import match_via_cui

    assert match_via_cui("", "osimertinib") == (False, [])
    assert match_via_cui("osimertinib", "") == (False, [])
    assert match_via_cui("", "") == (False, [])
    assert match_via_cui("   ", "osimertinib") == (False, [])
