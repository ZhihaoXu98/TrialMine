"""LangGraph node: SearchOrchestrator.

The central agent that:
1. Receives structured query from QueryParser
2. Calls tools (search_trials, check_eligibility, explain_trial) as needed
3. Compiles the final ranked list with explanations
4. Decides when to stop (sufficient results found or max iterations reached)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SearchOrchestrator:
    """LangGraph agent that orchestrates trial search and explanation."""

    def __init__(self, llm: Any, tools: list) -> None:
        """Initialise with LLM and tool list.

        Args:
            llm: LangChain LLM instance bound with tools.
            tools: List of callable tool functions.
        """
        # TODO: bind tools to LLM, set up ReAct-style agent loop
        self.llm = llm
        self.tools = tools

    async def run(self, structured_query: dict) -> dict:
        """Execute the orchestration loop for a structured query.

        Args:
            structured_query: Output from QueryParser.parse().

        Returns:
            Dict with ranked_trials (list) and explanations (dict[nct_id, str]).
        """
        # TODO: implement LangGraph node logic with tool dispatch
        raise NotImplementedError
