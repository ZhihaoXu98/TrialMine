"""Medical-concept normalisation and query expansion.

Two layers, each solving a different problem:

1. :class:`ConceptNormalizer` — a hand-built lay→medical synonym map for
   patient queries. Used to expand a query into multiple variants for BM25
   retrieval. Cheap to ship, brittle, ~30 entries.

2. :func:`extract_concepts` and :func:`normalise_concept` — long-term UMLS
   linking via SciSpacy ``EntityLinker``. Skeletons remain for a future PR;
   they will return UMLS CUIs (e.g. ``"C0007115"``), not free-text synonyms.

The two layers do not overlap: the ``ConceptNormalizer`` operates on lay
language (a patient's free-text query); UMLS linking operates on canonical
medical text (trial documents, parsed eligibility entities).
"""

from __future__ import annotations

import logging
import re

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
    language. Replacement is one hop only (no chaining). For broader
    coverage, use SciSpacy ``EntityLinker`` + UMLS via
    :func:`normalise_concept` (TODO).
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
# UMLS linking — skeletons for a future PR                                    #
# --------------------------------------------------------------------------- #


def extract_concepts(text: str) -> list[dict]:
    """Extract medical concepts from text using a SciSpacy NER model.

    Args:
        text: Input clinical text (patient description or trial excerpt).

    Returns:
        List of dicts with keys: text, label, cui, start, end.
    """
    # TODO: load en_core_sci_lg or en_ner_bc5cdr_md, run NER, link to UMLS
    raise NotImplementedError("UMLS extract_concepts is a future PR")


def normalise_concept(concept_text: str) -> str | None:
    """Map a concept surface form to a canonical UMLS CUI.

    Args:
        concept_text: Surface form of the concept (e.g. 'non-small cell lung cancer').

    Returns:
        UMLS CUI string, or None if no match found.
    """
    # TODO: use scispacy EntityLinker or UMLS API with UMLS_API_KEY
    raise NotImplementedError("UMLS normalise_concept is a future PR")
