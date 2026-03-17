"""End-to-end LangGraph pipeline wiring QueryParser → SearchOrchestrator.

Builds and compiles the LangGraph StateGraph that defines the full
query-to-results flow. This is the single entry point for the agent system.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(llm: Any) -> Any:
    """Construct and compile the LangGraph search pipeline.

    Args:
        llm: LangChain LLM instance (ChatAnthropic recommended).

    Returns:
        Compiled LangGraph runnable ready for .ainvoke().
    """
    # TODO: define StateGraph, add QueryParser and SearchOrchestrator nodes,
    #       wire edges, compile with checkpointer for streaming support
    raise NotImplementedError


async def search(patient_description: str, pipeline: Any) -> dict:
    """Run the full pipeline for a patient description.

    Args:
        patient_description: Raw free-text input from the patient.
        pipeline: Compiled LangGraph runnable from build_pipeline().

    Returns:
        Final state dict with ranked_trials and explanations.
    """
    # TODO: pipeline.ainvoke({"patient_description": patient_description})
    raise NotImplementedError
