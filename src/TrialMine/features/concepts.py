"""Medical concept extraction and normalisation using SciSpacy + UMLS.

Extracts and normalises:
- Cancer types and subtypes → UMLS CUIs
- Biomarkers and mutations (e.g. EGFR, BRCA1)
- Drugs and intervention types
- Anatomical sites

Used to enrich patient queries and trial documents for better matching.
"""

import logging

logger = logging.getLogger(__name__)


def extract_concepts(text: str) -> list[dict]:
    """Extract medical concepts from text using a SciSpacy NER model.

    Args:
        text: Input clinical text (patient description or trial excerpt).

    Returns:
        List of dicts with keys: text, label, cui, start, end.
    """
    # TODO: load en_core_sci_lg or en_ner_bc5cdr_md, run NER, link to UMLS
    raise NotImplementedError


def normalise_concept(concept_text: str) -> str | None:
    """Map a concept surface form to a canonical UMLS CUI.

    Args:
        concept_text: Surface form of the concept (e.g. 'non-small cell lung cancer').

    Returns:
        UMLS CUI string, or None if no match found.
    """
    # TODO: use scispacy EntityLinker or UMLS API with UMLS_API_KEY
    raise NotImplementedError
