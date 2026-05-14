"""Unit tests for :class:`TrialMine.features.concepts.ConceptNormalizer`.

Pure-function tests — no I/O, no models. Fast.
"""

from __future__ import annotations

from TrialMine.features.concepts import ConceptNormalizer


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
