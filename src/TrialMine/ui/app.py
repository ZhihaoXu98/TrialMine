"""Streamlit patient-facing UI for TrialMine.

Pages / sections:
1. Search — patient describes their situation, sees ranked trial cards
2. Trial Detail — expanded view of a single trial with eligibility breakdown
3. About — explanation of how the system works

Communicates with the FastAPI backend via httpx.
"""

import logging

logger = logging.getLogger(__name__)

# TODO: set page config (title, layout="wide", favicon)
# TODO: implement search form (st.text_area for description, st.slider for max_results)
# TODO: implement trial card component (title, status, phase, relevance score, explanation)
# TODO: implement trial detail page with full eligibility criteria
# TODO: add st.session_state management for search history


def main() -> None:
    """Entry point for `streamlit run` and `trialmine-ui`."""
    # TODO: import streamlit, render app
    raise NotImplementedError


if __name__ == "__main__":
    main()
