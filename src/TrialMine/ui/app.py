"""Streamlit patient-facing UI for TrialMine.

Communicates with the FastAPI backend via httpx.
Run with: streamlit run src/TrialMine/ui/app.py
"""

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TrialMine", layout="wide")
st.title("TrialMine — Find Clinical Trials")
st.caption("ML-powered clinical trial search engine for oncology")

# ── Sidebar filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Settings")
    search_method = st.radio(
        "Search Method",
        options=["hybrid", "bm25", "semantic"],
        format_func=lambda x: {"hybrid": "Hybrid (BM25 + Semantic)", "bm25": "BM25 (Keyword)", "semantic": "Semantic (Embedding)"}[x],
    )

    st.header("Filters")
    status_filter = st.selectbox(
        "Status",
        ["Any", "RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "NOT_YET_RECRUITING"],
    )
    phase_filter = st.selectbox(
        "Phase",
        ["Any", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 1/Phase 2", "Phase 2/Phase 3"],
    )
    top_k = st.slider("Max results", 5, 100, 20)

# ── Search bar ───────────────────────────────────────────────────────────────
query = st.text_input(
    "Search",
    placeholder="Describe your condition or what you're looking for...",
    label_visibility="collapsed",
)

# ── Example buttons ──────────────────────────────────────────────────────────
cols = st.columns(3)
examples = [
    "breast cancer phase 3",
    "lung cancer immunotherapy recruiting",
    "pediatric leukemia trials",
]
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        query = example

# ── Status badges ────────────────────────────────────────────────────────────
_STATUS_COLORS = {
    "RECRUITING": "green",
    "ACTIVE_NOT_RECRUITING": "blue",
    "COMPLETED": "gray",
    "NOT_YET_RECRUITING": "orange",
}

_SOURCE_LABELS = {
    "bm25_only": "keyword",
    "semantic_only": "semantic",
    "both": "keyword + semantic",
}


def _status_badge(status: str | None) -> str:
    color = _STATUS_COLORS.get(status or "", "gray")
    return f":{color}[{status or 'Unknown'}]"


def _phase_badge(phase: str | None) -> str:
    return f"**{phase}**" if phase else ""


def _source_tag(source: str | None, method: str) -> str:
    """Return a tag showing how this result was found."""
    if method == "bm25":
        return ":blue[keyword]"
    if method == "semantic":
        return ":violet[semantic]"
    # Hybrid — show source
    label = _SOURCE_LABELS.get(source or "", "")
    if source == "bm25_only":
        return ":blue[keyword]"
    if source == "semantic_only":
        return ":violet[semantic]"
    if source == "both":
        return ":blue[keyword] + :violet[semantic]"
    return ""


# ── Run search ───────────────────────────────────────────────────────────────
if query:
    filters = {}
    if status_filter != "Any":
        filters["status"] = status_filter
    if phase_filter != "Any":
        filters["phase"] = phase_filter

    payload = {
        "query": query,
        "top_k": top_k,
        "filters": filters or None,
        "method": search_method,
    }

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{API_BASE}/api/v1/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        st.error("Cannot connect to the API server. Is it running? (`make serve`)")
        st.stop()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error: {exc.response.status_code} — {exc.response.text}")
        st.stop()

    results = data["results"]
    method_label = {
        "bm25": "BM25",
        "semantic": "Semantic",
        "hybrid": "Hybrid",
    }.get(data.get("search_method", ""), "")

    st.markdown(
        f"**{data['total']} results** for *\"{data['query']}\"* "
        f"via **{method_label}** search "
        f"({data['search_time_ms']:.0f} ms)"
    )

    if not results:
        st.info("No matching trials found. Try broadening your search.")

    # ── Result cards ─────────────────────────────────────────────────────────
    for r in results:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{r['title'][:120]}**")
                condition_tags = " · ".join(r["conditions"][:5]) if r["conditions"] else ""
                st.markdown(f"{condition_tags}")
            with right:
                st.markdown(_status_badge(r["status"]))
                st.markdown(_phase_badge(r["phase"]))

            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.caption(f"Score: {r['score']:.4f}")
            with meta_col2:
                st.markdown(_source_tag(r.get("source"), data.get("search_method", "")))
            with meta_col3:
                if r.get("url"):
                    st.caption(f"[{r['nct_id']}]({r['url']})")
                else:
                    st.caption(r["nct_id"])


def main() -> None:
    """Entry point for `trialmine-ui`."""
    pass  # Streamlit runs the file directly; nothing else needed


if __name__ == "__main__":
    main()
