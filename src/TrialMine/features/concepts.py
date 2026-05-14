"""Medical-concept normalisation, query expansion, and UMLS linking.

Three layers, each solving a different problem:

1. :class:`ConceptNormalizer` — a hand-built lay→medical synonym map for
   patient queries. Used to expand a query into multiple variants for BM25
   retrieval. Cheap to ship, brittle, ~30 entries.

2. :func:`link_to_cuis` — UMLS canonicalization via SciSpacy
   ``EntityLinker``. Maps drug surface forms (and other biomedical
   entities) to UMLS CUIs from scispacy's 2022 KB snapshot. Filtered to
   drug-relevant semantic types (T121 / T200 / T109 / T123 / T129) so
   anatomy or disease concepts don't pollute drug matching. Lazy-loaded
   via a module-level singleton; first call pays ~30–45 s after the KB
   is cached at ``~/.scispacy/datasets/``; subsequent calls are
   essentially free (LRU-cached up to 2000 distinct spans).

3. :func:`match_via_cui` — checks whether two free-text spans refer to
   the same drug or drug class. Combines direct CUI overlap (alias
   matching: ``"Tagrisso"`` ↔ ``"osimertinib"`` both link to
   ``C4058811``) with a hand-curated drug-name ↔ drug-class bridge in
   :data:`TrialMine.features.drug_classes.DRUG_TO_CLASS_CUIS`
   (e.g. osimertinib ↔ "EGFR tyrosine kinase inhibitor"). The class
   table substitutes for UMLS hierarchy traversal, which scispacy's KB
   doesn't expose (the ``Entity`` NamedTuple has no parent relations).

The layers don't overlap: ``ConceptNormalizer`` operates on lay language
(a patient's free-text query); UMLS linking operates on canonical
medical text (trial eligibility, parsed drug entities).
"""

from __future__ import annotations

import functools
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lay → medical synonym map                                                    #
# --------------------------------------------------------------------------- #

# Design notes:
# - Each entry maps a lay phrase the patient might use to a *broader* medical
#   term, never a narrower one. Mapping "liver cancer" → "hepatocellular
#   carcinoma" would lose recall on cholangiocarcinoma trials, so we map to
#   "hepatic neoplasm" instead.
# - No chained mappings (e.g. "immuno" → "immunotherapy" → narrower).
#   Resolution is one hop at most.
# - Entries that would replace terms trials use verbatim ("tumor" → "neoplasm",
#   "advanced" → "advanced stage") are intentionally excluded — they would
#   hurt BM25 precision without helping semantic search.
# - For query expansion, we ADD the medical form alongside the original; we do
#   not REPLACE. That keeps BM25 matches on the original phrasing.

_SYNONYMS: dict[str, str] = {
    # Site-specific cancers
    "stomach cancer": "gastric neoplasm",
    "breast cancer": "breast neoplasm",
    "lung cancer": "lung neoplasm",
    "brain cancer": "central nervous system neoplasm",
    "skin cancer": "cutaneous neoplasm",
    "blood cancer": "hematologic neoplasm",
    "liver cancer": "hepatic neoplasm",
    "kidney cancer": "renal neoplasm",
    "bone cancer": "bone neoplasm",
    "colon cancer": "colorectal neoplasm",
    "throat cancer": "head and neck neoplasm",
    "mouth cancer": "oral neoplasm",
    "ovarian cancer": "ovarian neoplasm",
    "pancreatic cancer": "pancreatic neoplasm",
    "prostate cancer": "prostatic neoplasm",
    "thyroid cancer": "thyroid neoplasm",
    "bladder cancer": "urinary bladder neoplasm",
    "cervical cancer": "cervical neoplasm",
    "uterine cancer": "endometrial neoplasm",
    "esophageal cancer": "esophageal neoplasm",
    # Disease state / progression
    "cancer that came back": "recurrent neoplasm",
    "cancer that returned": "recurrent neoplasm",
    "cancer came back": "recurrent neoplasm",
    "cancer that spread": "metastatic neoplasm",
    "cancer spread": "metastatic disease",
    "cancer has spread": "metastatic disease",
    "metastasized": "metastatic",
    "spread to bone": "bone metastasis",
    "spread to brain": "brain metastasis",
    "spread to liver": "hepatic metastasis",
    "spread to lung": "pulmonary metastasis",
    "early stage": "stage I",
    "late stage": "stage IV",
    "rectal cancer": "rectal neoplasm",
    # Treatment vocabulary
    "chemo": "chemotherapy",
    "radiation": "radiation therapy",
    "immuno": "immunotherapy",
    "targeted therapy": "molecular targeted therapy",
    "stem cell transplant": "hematopoietic stem cell transplantation",
    "bone marrow transplant": "hematopoietic stem cell transplantation",
    # Stage notation — patients write Arabic, oncology writes Roman
    "stage 1": "stage I",
    "stage 2": "stage II",
    "stage 3": "stage III",
    "stage 4": "stage IV",
    # Symptom / clinical
    "lump": "mass",
}


class ConceptNormalizer:
    """Lay → medical synonym expansion for patient queries.

    Two operations:

    * :meth:`normalize` — return the canonical medical form for an exact
      phrase, or the original term if no mapping exists.
    * :meth:`expand_query` — produce a list of query variants with lay
      phrases substituted by their medical equivalents. Always includes
      the original query first.

    The synonym map is a fixed ~30-entry dict tuned for oncology patient
    language. Replacement is one hop only (no chaining). For UMLS-backed
    drug-class equivalences on canonical medical text, use
    :func:`link_to_cuis` and :func:`match_via_cui` instead — they
    operate on a different layer (parsed trial eligibility entities,
    not lay patient queries).
    """

    def __init__(self, synonyms: dict[str, str] | None = None) -> None:
        """Initialise with the default synonym map or an injected one.

        Args:
            synonyms: Optional override for unit tests. Keys must be
                lowercase. If ``None``, the default map is used.
        """
        self.synonyms: dict[str, str] = dict(synonyms) if synonyms else dict(_SYNONYMS)
        # Pre-compile a single regex that matches any lay phrase as a whole-word
        # alternation, longest first so multi-word keys take priority.
        sorted_keys = sorted(self.synonyms.keys(), key=len, reverse=True)
        self._pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b",
            flags=re.IGNORECASE,
        )

    def normalize(self, term: str) -> str:
        """Return the canonical medical form for ``term``.

        If ``term`` (lowercased, stripped) is in the synonym map, return the
        medical form. Otherwise return the input unchanged. This is a single-
        hop mapping; the result is never re-normalised.

        Args:
            term: A lay phrase, e.g. ``"stomach cancer"``.

        Returns:
            The medical equivalent, or ``term`` itself if unknown.
        """
        key = term.strip().lower()
        return self.synonyms.get(key, term)

    def expand_query(self, query: str) -> list[str]:
        """Expand a query with synonym variants for broader retrieval.

        The original query is always returned first. If any lay phrase in
        the query matches a synonym key, a variant query is produced with
        every match substituted by its medical form. Matching is
        case-insensitive on whole words; the longest matching phrase wins
        when keys overlap.

        Args:
            query: A patient's free-text query.

        Returns:
            A deduped list ``[original, expanded?]``. Length 1 if no
            synonyms matched, length 2 otherwise.
        """
        if not query or not query.strip():
            return [query] if query is not None else []

        def _replace(match: re.Match[str]) -> str:
            return self.synonyms[match.group(0).lower()]

        expanded = self._pattern.sub(_replace, query)

        if expanded == query:
            return [query]
        return [query, expanded]


# --------------------------------------------------------------------------- #
# UMLS linking via SciSpacy EntityLinker                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConceptLink:
    """One UMLS concept linked from a text span.

    Attributes:
        cui: UMLS Concept Unique Identifier (e.g. ``"C4058811"``).
        canonical_name: Canonical surface form for this CUI in the
            scispacy 2022 KB (e.g. ``"osimertinib"``).
        score: Linker confidence in the link, on the
            ``threshold..1.0`` scale set in :func:`_get_linker_nlp`.
        semantic_types: UMLS semantic-type codes for this concept
            (e.g. ``("T109", "T121")``). Drug-relevant codes live in
            :data:`_DRUG_SEMANTIC_TYPES`.
    """

    cui: str
    canonical_name: str
    score: float
    semantic_types: tuple[str, ...]


# UMLS semantic types we accept as "drug-relevant". Anything not landing
# in one of these is dropped by :func:`link_to_cuis` so anatomy / disease /
# lab-test concepts can't false-match drug names downstream.
_DRUG_SEMANTIC_TYPES: frozenset[str] = frozenset(
    {
        "T121",  # Pharmacologic Substance
        "T200",  # Clinical Drug
        "T109",  # Organic Chemical
        "T123",  # Biologically Active Substance
        "T129",  # Immunologic Factor (monoclonal antibodies)
    }
)


# Oncology abbreviation expansions, applied to text BEFORE NER + UMLS
# linking. scispacy's 2022 UMLS KB indexes spelled-out drug-class
# concepts ("tyrosine kinase inhibitor", "immune checkpoint inhibitor",
# "PARP inhibitor") but NOT their common abbreviations ("TKI", "ICI",
# "PARP-i"), so eligibility text written in shorthand silently fails
# to link. Each pattern is word-boundary anchored (``\b``) to avoid
# expanding inside other tokens. Bare protein / gene names (PARP,
# EGFR, HER2, CDK4/6) are NOT expanded — they're not drug classes and
# would create false matches.
_ABBREVIATION_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # TKI / TKIs → "tyrosine kinase inhibitor". When the source text
    # contains "EGFR TKI", the expansion produces "EGFR tyrosine kinase
    # inhibitor", which scispacy resolves to the EGFR-inhibitor class
    # CUI (C5574906). Standalone "TKI" → "tyrosine kinase inhibitor"
    # resolves to the broader TKI class CUI (C1268567).
    (re.compile(r"\bTKIs?\b", re.IGNORECASE), "tyrosine kinase inhibitor"),
    # ICI / ICIs → "immune checkpoint inhibitor". Resolves to class
    # CUI C4684977 used by pembrolizumab / nivolumab / atezolizumab /
    # durvalumab / ipilimumab.
    (re.compile(r"\bICIs?\b", re.IGNORECASE), "immune checkpoint inhibitor"),
    # PARP-i / PARPi / PARP-I / PARPI → "PARP inhibitor". Resolves to
    # class CUI C1882413 used by olaparib / niraparib / rucaparib /
    # talazoparib. Bare "PARP" is intentionally NOT expanded — it's
    # the enzyme name, not the drug class.
    (re.compile(r"\bPARP-?[iI]\b"), "PARP inhibitor"),
)


def _expand_abbreviations(text: str) -> str:
    """Expand common oncology abbreviations to scispacy-resolvable forms.

    Applied inside :func:`link_to_cuis` before NER. scispacy's 2022 UMLS
    KB doesn't index "TKI" / "ICI" / "PARP-i" as drug-class concepts but
    DOES index the spelled-out forms. Without this expansion,
    eligibility text written in shorthand silently fails to link.

    Word-boundary anchored to avoid expanding inside other tokens.
    Bare protein / gene names (PARP, EGFR, HER2, CDK4/6) are NOT
    expanded — they're not drug classes and would produce false
    matches if they were.

    Known limitation (recorded 2026-05-14 during Phase A5b verification):
    expanding the abbreviation is necessary but not always sufficient —
    scispacy's 2022 KB NER fails to extract "PARP inhibitor" as a single
    drug-class entity when it appears in some sentence contexts (e.g.
    "no prior PARP inhibitor"), even though the bare phrase
    "PARP inhibitor" resolves cleanly to C1882413. The TKI and ICI
    cases don't hit this quirk in our probes. Impact is narrow because
    (a) the patient side usually says the drug name (olaparib /
    niraparib), and (b) the trial side comes from
    ``parsed_eligibility`` which stores pre-extracted entity strings,
    not raw sentences. Phase B2 will measure whether this matters; if
    it does, the next intervention is a direct-CUI fallback in
    :func:`link_to_cuis` that recognizes "PARP inhibitor" (and
    similar) as a fixed string regardless of NER outcome.
    """
    for pattern, replacement in _ABBREVIATION_EXPANSIONS:
        text = pattern.sub(replacement, text)
    return text


# Process-wide lazy singleton: the spaCy pipeline with scispacy_linker
# attached. First access pays the ~30–45 s load (after the KB is cached
# locally per Phase A2 of docs/fix_parser_umls.md). Subsequent accesses
# are essentially free.
_LINKER_NLP: spacy.Language | None = None


def _get_linker_nlp() -> spacy.Language:
    """Lazy-load and cache the spaCy + UMLS linker pipeline.

    On first call: imports ``spacy`` and ``scispacy.linking`` (the latter
    has the side effect of registering the ``"scispacy_linker"`` component
    with spaCy — do not strip it as unused), loads ``en_core_sci_lg``,
    attaches the linker pipe configured for UMLS with a 0.85 confidence
    threshold, caches the result, and logs the wall-clock load time so it
    surfaces in API startup logs. Subsequent calls return the cached
    pipeline.
    """
    global _LINKER_NLP
    if _LINKER_NLP is not None:
        return _LINKER_NLP

    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401 - side-effect import

    t0 = time.time()
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "max_entities_per_mention": 5,
            "threshold": 0.85,
        },
    )
    logger.info(
        "Loaded SciSpacy + UMLS linker (en_core_sci_lg + scispacy_linker) in %.1fs",
        time.time() - t0,
    )
    _LINKER_NLP = nlp
    return _LINKER_NLP


@functools.lru_cache(maxsize=2000)
def link_to_cuis(text: str) -> tuple[ConceptLink, ...]:
    """Link a text span to drug-relevant UMLS concepts.

    Pre-expands common oncology abbreviations via
    :func:`_expand_abbreviations` (TKI → tyrosine kinase inhibitor; ICI
    → immune checkpoint inhibitor; PARP-i → PARP inhibitor) so that
    eligibility text written in shorthand still resolves to drug-class
    CUIs. scispacy's 2022 UMLS KB doesn't index those abbreviations
    natively.

    Then runs the cached spaCy + UMLS linker on the expanded text,
    iterates the detected entities, looks up each candidate CUI in
    scispacy's KB, and emits a :class:`ConceptLink` for each entity
    whose semantic types overlap :data:`_DRUG_SEMANTIC_TYPES`. Non-drug
    entities (anatomy, diseases, procedures, lab tests) are dropped so
    they can't false-match drug names downstream.

    Empty or whitespace-only input returns ``()`` without loading
    the pipeline — so a no-op query doesn't pay the ~30–45 s linker
    load.

    Returns a tuple (immutable + hashable) so the surrounding
    :func:`functools.lru_cache` retains identity across calls. Cache
    size 2000 means we keep CUIs for the last 2000 distinct text
    spans; older entries are evicted LRU.
    """
    if not text or not text.strip():
        return ()

    expanded = _expand_abbreviations(text)

    nlp = _get_linker_nlp()
    linker = nlp.get_pipe("scispacy_linker")
    doc = nlp(expanded)

    links: list[ConceptLink] = []
    for ent in doc.ents:
        for cui, score in ent._.kb_ents or []:
            entity = linker.kb.cui_to_entity.get(cui)
            if entity is None:
                continue
            types = tuple(entity.types or ())
            if not any(t in _DRUG_SEMANTIC_TYPES for t in types):
                continue
            links.append(
                ConceptLink(
                    cui=cui,
                    canonical_name=entity.canonical_name,
                    score=float(score),
                    semantic_types=types,
                )
            )
    return tuple(links)


def match_via_cui(text_a: str, text_b: str) -> tuple[bool, list[str]]:
    """Check whether two text spans refer to the same drug or drug class.

    Two paths to a match:

    1. **Direct CUI overlap** — the spans share at least one UMLS CUI
       (alias matching: ``"Tagrisso"`` and ``"osimertinib"`` both link
       to ``C4058811``).
    2. **Drug-class overlap** — one span's CUI is a key in
       :data:`TrialMine.features.drug_classes.DRUG_TO_CLASS_CUIS` whose
       class CUIs overlap the other span's CUIs. This bridges
       drug-name ↔ drug-class equivalences that UMLS doesn't capture
       as aliases (e.g. osimertinib ↔ "EGFR TKI"). The class-side
       intersection (A's classes ∩ B's classes) is also checked, so
       two different drugs in the same class match each other.

    The drug-class table is imported lazily inside this function so
    this module can be used before
    :mod:`TrialMine.features.drug_classes` lands (Phase A4 of
    ``docs/fix_parser_umls.md``). On :class:`ImportError`, only the
    direct-CUI path is attempted.

    Returns:
        Tuple ``(matched, matched_cuis)``:

        - ``matched`` is True iff any of the three overlap paths fired.
        - ``matched_cuis`` is the sorted list of CUIs that triggered
          the match. Empty list when ``matched`` is False.
    """
    cuis_a = {link.cui for link in link_to_cuis(text_a)}
    cuis_b = {link.cui for link in link_to_cuis(text_b)}

    direct = cuis_a & cuis_b
    if direct:
        matched_cuis = sorted(direct)
        logger.info(
            "UMLS match %r <-> %r via direct CUI overlap: %s",
            text_a,
            text_b,
            matched_cuis,
        )
        return True, matched_cuis

    try:
        from TrialMine.features.drug_classes import DRUG_TO_CLASS_CUIS
    except ImportError:
        logger.warning(
            "TrialMine.features.drug_classes not importable; "
            "match_via_cui falling back to direct-CUI overlap only"
        )
        return False, []

    classes_a = {
        class_cui for cui in cuis_a for class_cui in DRUG_TO_CLASS_CUIS.get(cui, frozenset())
    }
    classes_b = {
        class_cui for cui in cuis_b for class_cui in DRUG_TO_CLASS_CUIS.get(cui, frozenset())
    }

    # Three overlap directions:
    #   1. A's drug classes intersect B's drug CUIs.
    #   2. B's drug classes intersect A's drug CUIs.
    #   3. A's drug classes intersect B's drug classes (drug-class to
    #      drug-class — e.g. osimertinib vs erlotinib, both EGFR TKI).
    all_matches = (classes_a & cuis_b) | (classes_b & cuis_a) | (classes_a & classes_b)
    if all_matches:
        matched_cuis = sorted(all_matches)
        logger.info(
            "UMLS match %r <-> %r via drug-class overlap: %s",
            text_a,
            text_b,
            matched_cuis,
        )
        return True, matched_cuis

    return False, []
