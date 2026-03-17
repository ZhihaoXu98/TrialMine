"""LangGraph node: QueryParser.

Transforms a raw patient description into a structured query containing:
- Extracted cancer type(s) and stage
- Key biomarkers / mutations mentioned
- Patient constraints (age, sex, location preferences)
- Reformulated search query for downstream retrieval
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class QueryParser:
    """Extracts structured clinical intent from free-text patient descriptions."""

    def __init__(self, llm: Any) -> None:
        """Initialise with a LangChain-compatible LLM.

        Args:
            llm: LangChain LLM instance (e.g. ChatAnthropic).
        """
        # TODO: set up structured output chain with a Pydantic output schema
        self.llm = llm

    async def parse(self, patient_description: str) -> dict:
        """Parse a patient description into a structured query dict.

        Args:
            patient_description: Raw free-text input from the patient.

        Returns:
            Dict with keys: cancer_type, stage, biomarkers, age, sex,
            location, reformulated_query.
        """
        # TODO: call LLM with structured output, return parsed dict
        raise NotImplementedError
