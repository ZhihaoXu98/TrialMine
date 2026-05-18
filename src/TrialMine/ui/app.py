"""Streamlit patient-facing UI for TrialMine.

Demo-quality UI for the agent-powered clinical trial search.
Run with: streamlit run src/TrialMine/ui/app.py
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import streamlit as st

logger = logging.getLogger(__name__)

# Driven by env so the same image can talk to localhost (dev) or the
# `api` service hostname (Docker compose).
API_BASE = os.environ.get("API_URL", "http://localhost:8000")
SEARCH_TIMEOUT_S = 60.0

EXAMPLE_QUERIES: list[str] = [
    "Stage 3 lung cancer, tried chemo",
    "HER2+ breast cancer immunotherapy",
    "Pediatric leukemia trials",
    "MSI-high colorectal cancer",
    "Pancreatic cancer, new options",
]

# emoji + Streamlit color name + display label
STATUS_DISPLAY: dict[str, tuple[str, str, str]] = {
    "RECRUITING": ("🟢", "green", "Recruiting"),
    "ACTIVE_NOT_RECRUITING": ("🟡", "orange", "Active, not recruiting"),
    "NOT_YET_RECRUITING": ("🟠", "orange", "Not yet recruiting"),
    "ENROLLING_BY_INVITATION": ("🟡", "orange", "Enrolling by invitation"),
    "AVAILABLE": ("🟢", "green", "Available"),
    "COMPLETED": ("⚪", "gray", "Completed"),
    "TERMINATED": ("🔴", "red", "Terminated"),
    "WITHDRAWN": ("🔴", "red", "Withdrawn"),
    "SUSPENDED": ("🔴", "red", "Suspended"),
}

STEP_DISPLAY: dict[str, str] = {
    "parse_query": "🧠 Understood your situation",
    "normalize": "🔤 Normalized terminology",
    "build_query": "🔍 Built search query",
    "build_filters": "🎚️  Set filters",
    "retrieve": "📚 Searched trial database",
    "check_eligibility": "✅ Checked eligibility",
    "explain": "💬 Generated explanations",
    "fallback_search": "⚠️  Fell back to basic search",
    "timeout": "⏱️  Timed out",
    "execute_search_error": "❌ Primary search failed",
    "fallback_search_error": "❌ Fallback also failed",
    "pipeline_error": "❌ Pipeline error",
}

VERDICT_ICON = {"Met": "✅", "Unmet": "❌", "Unknown": "❓"}


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrialMine — AI Clinical Trial Search",
    page_icon="🔬",
    layout="centered",  # narrower column reads better for cards
    initial_sidebar_state="expanded",
)


# ── Visual polish (CSS injection) ────────────────────────────────────────────
#
# Streamlit's default styling is functional but cramped; the result list is the
# load-bearing surface so it gets the most attention. We keep the rules
# attribute-selector-based and avoid Streamlit's auto-generated class names —
# the data-testid hooks are stable across releases.
_CUSTOM_CSS = """
<style>
    /* Tighten top padding and constrain content width for readability. */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 880px;
    }

    /* Hero block */
    .tm-hero {
        margin-bottom: 1.5rem;
    }
    .tm-hero h1 {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0 0 0.35rem 0;
        line-height: 1.1;
    }
    .tm-hero p {
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.02rem;
        margin: 0;
    }

    /* Section label (uppercase, muted) */
    .tm-section-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.25rem 0 0.5rem 0;
    }

    /* Suggestion chips — make Streamlit buttons look like pills. */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(255, 255, 255, 0.025) !important;
        padding: 0.35rem 0.95rem !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
        color: rgba(255, 255, 255, 0.82) !important;
        transition: all 0.12s ease !important;
        white-space: nowrap;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
        background: rgba(255, 255, 255, 0.07) !important;
        border-color: rgba(255, 255, 255, 0.28) !important;
        color: white !important;
    }

    /* Primary search button — slightly bigger, weighted */
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
    }

    /* Result cards (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(255, 255, 255, 0.015);
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(255, 255, 255, 0.18) !important;
        background: rgba(255, 255, 255, 0.03);
    }

    /* Custom badge styles (rendered inline via st.markdown) */
    .tm-badge {
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.4rem;
        white-space: nowrap;
        letter-spacing: 0.01em;
    }
    .tm-badge-recruiting {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
    .tm-badge-active {
        background: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.25);
    }
    .tm-badge-completed {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.22);
    }
    .tm-badge-stopped {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .tm-badge-phase {
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .tm-badge-met {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
    .tm-badge-unmet {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .tm-badge-unknown {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.22);
    }
    .tm-badge-match-top {
        background: rgba(168, 85, 247, 0.14);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.28);
    }
    .tm-badge-match-good {
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.22);
    }
    .tm-badge-match-fair {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.22);
    }

    /* Condition chips — subtler than the badges */
    .tm-chip {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 5px;
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.7);
    }

    /* Card title link */
    .tm-card-title a {
        color: white;
        text-decoration: none;
        font-weight: 600;
    }
    .tm-card-title a:hover {
        text-decoration: underline;
        text-decoration-color: rgba(255,255,255,0.4);
    }

    /* Card metadata row (NCT + rank) */
    .tm-card-meta {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.78rem;
        font-family: 'SF Mono', 'Monaco', 'Menlo', monospace;
        margin: 0.15rem 0 0.6rem 0;
    }

    /* Patient profile callout */
    .tm-profile-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(168, 85, 247, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .tm-profile-card .tm-profile-label {
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.78rem;
        margin-bottom: 0.45rem;
        font-weight: 500;
    }
    .tm-profile-card .tm-profile-slot {
        display: inline-block;
        margin-right: 1.2rem;
        font-size: 0.9rem;
    }
    .tm-profile-card .tm-profile-slot strong {
        color: rgba(255, 255, 255, 0.55);
        font-weight: 500;
        margin-right: 0.3rem;
    }

    /* Hide Streamlit's default "Made with Streamlit" footer + main menu */
    [data-testid="stToolbar"] { display: none; }
    footer { visibility: hidden; height: 0; }
    #MainMenu { visibility: hidden; }

    /* Sidebar polish */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.15);
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1.0rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        margin-bottom: 0.5rem;
    }

    /* Disclaimer at bottom */
    .tm-disclaimer {
        margin-top: 3rem;
        padding: 0.85rem 1rem;
        background: rgba(239, 68, 68, 0.04);
        border-left: 3px solid rgba(239, 68, 68, 0.35);
        border-radius: 4px;
        font-size: 0.78rem;
        color: rgba(255, 255, 255, 0.55);
        line-height: 1.5;
    }

    /* Empty state */
    .tm-empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: rgba(255, 255, 255, 0.55);
    }
    .tm-empty-state .tm-empty-icon {
        font-size: 2.5rem;
        margin-bottom: 0.6rem;
        opacity: 0.6;
    }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# Map ES status strings → CSS modifier suffix on `.tm-badge-*`.
_STATUS_BADGE_CLASS: dict[str, str] = {
    "RECRUITING": "recruiting",
    "ACTIVE_NOT_RECRUITING": "active",
    "NOT_YET_RECRUITING": "active",
    "ENROLLING_BY_INVITATION": "active",
    "AVAILABLE": "recruiting",
    "COMPLETED": "completed",
    "TERMINATED": "stopped",
    "WITHDRAWN": "stopped",
    "SUSPENDED": "stopped",
}


def _match_label(idx: int, total: int) -> tuple[str, str]:
    """Honest rank-based match label. Replaces the misleading 'Match 100%'
    with semantic tiers — top-3 = 'Top match', next 30 % = 'Strong match',
    next 40 % = 'Good match', tail = 'Moderate match'. Returns
    (label_text, css_class_suffix)."""
    if idx == 0:
        return ("Top match", "top")
    pct = (idx + 1) / max(total, 1)
    if pct <= 0.3:
        return ("Strong match", "top")
    if pct <= 0.7:
        return ("Good match", "good")
    return ("Moderate match", "fair")


# ── Session state ────────────────────────────────────────────────────────────


def _init_state() -> None:
    """Seed session_state with the keys we depend on across reruns."""
    defaults: dict[str, Any] = {
        "results": None,  # full SearchResponse dict, or None
        "results_error": None,  # human-readable error string, or None
        "search_input": "",  # bound to the text input widget
        "pending_query": None,  # set by submit/example handlers; consumed below
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ── HTTP client (cached across reruns) ───────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _client() -> httpx.Client:
    """Persistent httpx client, shared across reruns within a session."""
    return httpx.Client(base_url=API_BASE, timeout=SEARCH_TIMEOUT_S)


def _run_search(
    query: str,
    *,
    use_agent: bool,
    top_k: int,
    filters: dict[str, str] | None,
) -> None:
    """Hit POST /api/v1/search and store response (or friendly error) in session_state."""
    payload: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "use_agent": use_agent,
        "method": "hybrid",
    }
    if filters and not use_agent:
        payload["filters"] = filters

    try:
        resp = _client().post("/api/v1/search", json=payload)
    except httpx.ConnectError:
        st.session_state.results = None
        st.session_state.results_error = (
            "**Cannot reach the search service** at `localhost:8000`. "
            "Start it with `make serve` (and ensure Elasticsearch is up: `docker start es`)."
        )
        return
    except httpx.TimeoutException:
        st.session_state.results = None
        st.session_state.results_error = (
            f"**Search timed out** after {SEARCH_TIMEOUT_S:.0f} s. "
            "The API may be cold-starting (cross-encoder takes ~5 s on first call). Try again."
        )
        return
    except Exception as exc:  # network reset, DNS, etc. — never crash the UI
        logger.exception("Unexpected request error")
        st.session_state.results = None
        st.session_state.results_error = f"**Unexpected error:** {type(exc).__name__}: {exc}"
        return

    if resp.status_code == 503:
        try:
            detail = resp.json().get("detail", resp.text[:200])
        except Exception:
            detail = resp.text[:200]
        st.session_state.results = None
        st.session_state.results_error = (
            f"**Search service is in degraded mode (503).** {detail} "
            "Try again in a moment, or toggle off the AI agent for plain search."
        )
        return

    if resp.status_code >= 400:
        st.session_state.results = None
        st.session_state.results_error = f"**API error {resp.status_code}** — {resp.text[:200]}"
        return

    st.session_state.results = resp.json()
    st.session_state.results_error = None


# ── Render helpers ───────────────────────────────────────────────────────────


def _status_pill(status: str | None) -> str:
    emoji, color, label = STATUS_DISPLAY.get(
        (status or "").upper(), ("⚪", "gray", status or "Unknown")
    )
    return f"{emoji} :{color}[**{label}**]"


def _phase_pill(phase: str | None) -> str:
    if not phase:
        return ""
    return f":blue[**{phase}**]"


def _condition_chips(conditions: list[str], n: int = 5) -> str:
    """Render up to `n` conditions as inline violet chips."""
    if not conditions:
        return ""
    return "  ".join(f":violet-background[{c}]" for c in conditions[:n])


def _match_score(idx: int, total: int) -> float:
    """Rank-based match: top = 1.0, last in top-K = 0.5, linear interpolation.

    Avoids the cross-pipeline mess of mixing RRF, LightGBM, and BM25 scores —
    the only signal the user actually cares about is *relative* rank.
    """
    if total <= 1:
        return 1.0
    return 1.0 - 0.5 * (idx / max(1, total - 1))


def _render_eligibility(eligibility: dict | None) -> None:
    if eligibility is None:
        st.info("Eligibility wasn't checked for this trial — only the top candidates are checked.")
        return
    if "error" in eligibility and "verdict" not in eligibility:
        st.warning(f"Eligibility check failed: {eligibility['error']}")
        return

    verdict = eligibility.get("verdict", "Unknown")
    parse_conf = eligibility.get("parse_confidence")
    icon = VERDICT_ICON.get(verdict, "❓")

    header = f"**Overall: {icon} {verdict}**"
    if isinstance(parse_conf, int | float):
        header += f" &nbsp; *(parse confidence: {parse_conf:.0%})*"
    st.markdown(header)

    criteria = eligibility.get("criteria") or {}
    if not criteria:
        st.caption("No criterion-level breakdown available.")
        return

    for name, info in criteria.items():
        if not isinstance(info, dict):
            continue
        c_verdict = info.get("verdict", "Unknown")
        c_icon = VERDICT_ICON.get(c_verdict, "❓")
        reason = info.get("reason") or info.get("note") or info.get("explanation") or ""
        if reason:
            st.markdown(f"- {c_icon} **{name}** — {reason}")
        else:
            st.markdown(f"- {c_icon} **{name}**")


def _render_full_details(r: dict) -> None:
    rows: list[tuple[str, str]] = []
    if r.get("conditions"):
        rows.append(("Conditions", "; ".join(r["conditions"])))
    if r.get("phase"):
        rows.append(("Phase", str(r["phase"])))
    if r.get("status"):
        rows.append(("Status", str(r["status"])))
    if r.get("source"):
        rows.append(("Found via", str(r["source"]).replace("_", " ")))
    if r.get("bm25_rank") is not None:
        rows.append(("BM25 rank", str(r["bm25_rank"])))
    if r.get("semantic_rank") is not None:
        rows.append(("Semantic rank", str(r["semantic_rank"])))
    if r.get("score") is not None:
        rows.append(("Internal score", f"{r['score']:.4f}"))

    for label, value in rows:
        st.markdown(f"**{label}:** {value}")

    if r.get("url"):
        st.link_button("View on ClinicalTrials.gov ↗", r["url"])


def _render_card(r: dict, idx: int, total: int) -> None:
    with st.container(border=True):
        # ── Top badges row: match tier, status, phase, eligibility verdict ──
        badge_parts: list[str] = []

        # Match tier (honest semantic label, not a fake %)
        match_text, match_class = _match_label(idx, total)
        badge_parts.append(
            f'<span class="tm-badge tm-badge-match-{match_class}">{match_text}</span>'
        )

        # Status — color-coded, glanceable
        status = (r.get("status") or "").upper()
        if status:
            cls = _STATUS_BADGE_CLASS.get(status, "completed")
            label = STATUS_DISPLAY.get(status, ("⚪", "gray", status.title()))[2]
            badge_parts.append(f'<span class="tm-badge tm-badge-{cls}">{label}</span>')

        # Phase
        if r.get("phase"):
            badge_parts.append(f'<span class="tm-badge tm-badge-phase">{r["phase"]}</span>')

        # Eligibility verdict — only when the agent path computed one
        elig = r.get("eligibility") or {}
        verdict = elig.get("verdict") if isinstance(elig, dict) else None
        if verdict in ("Met", "Unmet", "Unknown"):
            cls = {"Met": "met", "Unmet": "unmet", "Unknown": "unknown"}[verdict]
            icon = VERDICT_ICON[verdict]
            badge_parts.append(
                f'<span class="tm-badge tm-badge-{cls}">{icon} Eligibility: {verdict}</span>'
            )

        st.markdown("".join(badge_parts), unsafe_allow_html=True)

        # ── Title (linked when URL present) + metadata row ──
        title = r.get("title") or "(untitled)"
        url = r.get("url")
        if url:
            title_html = (
                f'<div class="tm-card-title"><a href="{url}" target="_blank">{title}</a></div>'
            )
        else:
            title_html = f'<div class="tm-card-title"><span>{title}</span></div>'
        st.markdown(title_html, unsafe_allow_html=True)

        nct = r.get("nct_id", "")
        st.markdown(
            f'<div class="tm-card-meta">{nct} &nbsp;·&nbsp; rank #{idx + 1} of {total}</div>',
            unsafe_allow_html=True,
        )

        # ── Condition chips ──
        conditions = r.get("conditions") or []
        if conditions:
            chips_html = "".join(f'<span class="tm-chip">{c}</span>' for c in conditions[:6])
            st.markdown(chips_html, unsafe_allow_html=True)

        # ── Warnings (skip "fallback path" — already a top-level banner) ──
        for w in r.get("warnings") or []:
            if w == "fallback path":
                continue
            st.warning(f"⚠️ {w}")

        # ── Expandables ──
        if r.get("explanation"):
            with st.expander("Why this matches"):
                st.write(r["explanation"])

        if r.get("eligibility") is not None:
            with st.expander("Eligibility breakdown"):
                _render_eligibility(r["eligibility"])

        with st.expander("Full trial details"):
            _render_full_details(r)


def _render_agent_trace(trace: list[dict]) -> None:
    """Friendly per-step rendering of the LangGraph agent's reasoning."""
    if not trace:
        st.caption("No trace recorded.")
        return

    for entry in trace:
        step = entry.get("step", "")
        duration = entry.get("duration_ms", 0) or 0
        decisions = entry.get("decisions", {}) or {}
        title = STEP_DISPLAY.get(step, f"⚙️ {step}")

        st.markdown(f"**{title}** &nbsp; `{duration:.0f} ms`")

        if step == "parse_query":
            shown = False
            for key in (
                "condition",
                "condition_stage",
                "age",
                "sex",
                "biomarkers",
                "prior_treatments",
                "preferences",
                "location",
            ):
                v = decisions.get(key)
                if v in (None, [], ""):
                    continue
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                st.markdown(f"- **{key.replace('_', ' ').title()}:** {v}")
                shown = True
            if not shown:
                st.caption("No structured slots extracted — used raw query.")

        elif step == "normalize":
            input_c = decisions.get("input_condition")
            norm = decisions.get("normalized") or ""
            if input_c and norm and str(input_c).lower() != norm.lower():
                st.markdown(f"- `{input_c}` → `{norm}`")
            elif norm:
                st.markdown(f"- `{norm}`")
            else:
                st.caption("Nothing to normalize.")

        elif step == "build_query":
            q = decisions.get("query")
            if q:
                st.code(q, language="text")

        elif step == "build_filters":
            fs = decisions.get("filters") or {}
            if fs:
                for k, v in fs.items():
                    st.markdown(f"- **{k}:** `{v}`")
            else:
                st.caption("No filters applied.")

        elif step == "retrieve":
            n = decisions.get("n_results", 0)
            kind = decisions.get("pipeline", "?")
            st.markdown(f"- Pipeline: `{kind}`, returned {n} candidates")
            top_ids = decisions.get("top_nct_ids") or []
            if top_ids:
                st.caption("Top: " + ", ".join(top_ids))

        elif step == "check_eligibility":
            n = decisions.get("n_checked", 0)
            verdicts = decisions.get("verdicts") or []
            if verdicts:
                joined = " ".join(VERDICT_ICON.get(v or "", "❓") for v in verdicts)
                st.markdown(f"- Checked top {n}: {joined}")
            else:
                st.caption(f"Checked {n} trials.")

        elif step == "explain":
            n = decisions.get("n_explained", 0)
            st.caption(f"Built explanations for {n} trials.")

        else:
            for k, v in decisions.items():
                st.markdown(f"- **{k}:** {v}")

        st.markdown("---")


def _trace_durations(trace: list[dict]) -> dict[str, float]:
    """Sum durations per step name across the trace."""
    out: dict[str, float] = {}
    for entry in trace or []:
        step = entry.get("step", "")
        dur = entry.get("duration_ms", 0) or 0
        out[step] = out.get(step, 0) + dur
    return out


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")
    st.caption("Configure how trials are retrieved and ranked.")

    use_agent = st.toggle(
        "AI Agent",
        value=True,
        help=(
            "**On**: an LLM extracts patient details from natural language, applies "
            "smart filters, and runs per-trial eligibility checks. "
            "**Off**: plain BM25 + semantic hybrid search."
        ),
    )

    if use_agent:
        st.caption(
            ":violet[The AI infers status and phase from your query — no manual filtering needed.]"
        )
        status_filter = "Any"
        phase_filter = "Any"
    else:
        st.markdown(
            '<div class="tm-section-label" style="margin-top:1rem;">Manual filters</div>',
            unsafe_allow_html=True,
        )
        status_filter = st.selectbox(
            "Status",
            ["Any", "RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "COMPLETED"],
            index=1,
        )
        phase_filter = st.selectbox(
            "Phase",
            [
                "Any",
                "Phase 1",
                "Phase 2",
                "Phase 3",
                "Phase 4",
                "Phase 1/Phase 2",
                "Phase 2/Phase 3",
            ],
        )

    st.markdown(
        '<div class="tm-section-label" style="margin-top:1rem;">Result count</div>',
        unsafe_allow_html=True,
    )
    top_k = st.slider(
        "Max results", min_value=5, max_value=50, value=20, step=5, label_visibility="collapsed"
    )

    # Stats — only after a successful search
    if st.session_state.results:
        data = st.session_state.results
        st.markdown("---")
        st.markdown(
            '<div class="tm-section-label" style="margin-top:0;">Search statistics</div>',
            unsafe_allow_html=True,
        )

        n = data.get("total", 0)
        elapsed_s = (data.get("search_time_ms") or 0) / 1000.0
        c1, c2 = st.columns(2)
        c1.metric("Trials found", f"{n}")
        c2.metric("Total time", f"{elapsed_s:.2f} s")

        # Timing breakdown — derive from agent_trace (agent path) or timings dict (legacy)
        trace = data.get("agent_trace") or []
        if trace:
            durs = _trace_durations(trace)
            parts: list[str] = []
            if durs.get("parse_query"):
                parts.append(f"Parse: {durs['parse_query']:.0f} ms")
            if durs.get("retrieve"):
                parts.append(f"Retrieve: {durs['retrieve']:.0f} ms")
            elif durs.get("fallback_search"):
                parts.append(f"Fallback: {durs['fallback_search']:.0f} ms")
            if durs.get("check_eligibility"):
                parts.append(f"Eligibility: {durs['check_eligibility']:.0f} ms")
            if parts:
                st.caption(" · ".join(parts))
        elif data.get("timings"):
            t = data["timings"]
            parts = []
            if t.get("bm25_ms"):
                parts.append(f"BM25: {t['bm25_ms']:.0f} ms")
            if t.get("semantic_ms"):
                parts.append(f"Semantic: {t['semantic_ms']:.0f} ms")
            if t.get("cross_encoder_ms"):
                parts.append(f"Rerank: {t['cross_encoder_ms']:.0f} ms")
            if parts:
                st.caption(" · ".join(parts))

        if trace:
            with st.expander("Agent reasoning trace"):
                _render_agent_trace(trace)


# ── Main: header ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="tm-hero">
        <h1>🔬 TrialMine</h1>
        <p>AI-powered clinical trial search for oncology.
           Describe your situation in plain language.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Main: search bar + examples ──────────────────────────────────────────────


def _trigger_form() -> None:
    """Form-submit handler: queue whatever's currently in the text input."""
    st.session_state.pending_query = st.session_state.search_input


def _trigger_example(example: str) -> None:
    """Example-button handler: fill the input AND queue the search."""
    st.session_state.search_input = example
    st.session_state.pending_query = example


with st.form("search_form", clear_on_submit=False):
    st.text_input(
        "Your search",
        placeholder=(
            "e.g.  I'm 55 with stage 3 lung cancer who has tried chemo and want a phase 2 trial"
        ),
        label_visibility="collapsed",
        key="search_input",
    )
    st.form_submit_button(
        "Search trials",
        type="primary",
        on_click=_trigger_form,
    )

st.markdown('<div class="tm-section-label">Or try an example</div>', unsafe_allow_html=True)
ex_cols = st.columns(len(EXAMPLE_QUERIES))
for i, ex in enumerate(EXAMPLE_QUERIES):
    ex_cols[i].button(
        ex,
        key=f"ex_{i}",
        use_container_width=True,
        on_click=_trigger_example,
        args=(ex,),
    )


# ── Process pending search ───────────────────────────────────────────────────

if st.session_state.pending_query:
    q = st.session_state.pending_query
    st.session_state.pending_query = None  # consume

    if q and q.strip():
        # Recompute filters from sidebar state (may have changed since last search)
        filters: dict[str, str] = {}
        if not use_agent:
            if status_filter != "Any":
                filters["status"] = status_filter
            if phase_filter != "Any":
                filters["phase"] = phase_filter

        with st.spinner("🔎 Searching trials..."):
            _run_search(
                q.strip(),
                use_agent=use_agent,
                top_k=top_k,
                filters=filters or None,
            )

st.divider()


# ── Results / errors ─────────────────────────────────────────────────────────

if st.session_state.results_error:
    st.error(st.session_state.results_error)

elif st.session_state.results:
    data = st.session_state.results
    results: list[dict] = data.get("results") or []

    elapsed_s = (data.get("search_time_ms") or 0) / 1000.0
    n = len(results)
    qtext = data.get("query", "")
    plural = "s" if n != 1 else ""
    st.markdown(
        f"""
        <div style="margin: 1.5rem 0 0.25rem 0;">
            <span style="font-size: 1.3rem; font-weight: 600;">
                {n} trial{plural} found
            </span>
            <span style="color: rgba(255,255,255,0.45); font-size: 0.9rem;
                         margin-left: 0.6rem;">
                for &ldquo;{qtext}&rdquo; &nbsp;·&nbsp; {elapsed_s:.1f} s
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Patient profile callout (agent path only) ──
    profile = data.get("patient_profile") or {}
    profile_keys = (
        "condition",
        "condition_stage",
        "age",
        "sex",
        "biomarkers",
        "prior_treatments",
        "preferences",
        "location",
    )
    if any(profile.get(k) for k in profile_keys):
        slot_pairs: list[tuple[str, str]] = []
        if profile.get("condition"):
            slot_pairs.append(("Condition", str(profile["condition"])))
        if profile.get("condition_stage"):
            slot_pairs.append(("Stage", str(profile["condition_stage"])))
        if profile.get("age") is not None:
            slot_pairs.append(("Age", str(profile["age"])))
        if profile.get("sex"):
            slot_pairs.append(("Sex", str(profile["sex"])))
        if profile.get("biomarkers"):
            slot_pairs.append(("Biomarkers", ", ".join(profile["biomarkers"])))
        if profile.get("prior_treatments"):
            slot_pairs.append(("Prior treatments", ", ".join(profile["prior_treatments"])))
        if profile.get("location"):
            slot_pairs.append(("Location", str(profile["location"])))

        slot_html = "".join(
            f'<span class="tm-profile-slot"><strong>{label}</strong>{value}</span>'
            for label, value in slot_pairs
        )
        st.markdown(
            f"""
            <div class="tm-profile-card">
                <div class="tm-profile-label">🧠 What I understood from your query</div>
                <div>{slot_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Status banners
    if data.get("used_fallback"):
        reason = data.get("error") or "AI agent path failed"
        st.warning(
            "⚠️ **AI agent unavailable — showing basic hybrid search results.** "
            "No eligibility verdicts, no smart filtering. "
            f"Reason: `{reason}`"
        )
    elif data.get("error"):
        st.warning(f"⚠️ Partial result: {data['error']}")

    if not results:
        st.info(
            "**No matching trials found.** Try broadening your description, "
            "or include more specifics (age, biomarker, prior treatment)."
        )
    else:
        for i, r in enumerate(results):
            _render_card(r, i, n)

else:
    st.markdown(
        """
        <div class="tm-empty-state">
            <div class="tm-empty-icon">🩺</div>
            <div style="font-size: 1.05rem; color: rgba(255,255,255,0.75); margin-bottom: 0.3rem;">
                Search 140,000+ oncology trials with AI assistance
            </div>
            <div style="font-size: 0.88rem;">
                Type your situation above or pick an example to get started.
                The AI parses your query, applies smart filters, and explains
                why each trial might match.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="tm-disclaimer">
        <strong>⚠️ Research tool only — not medical advice.</strong>
        Results are AI-generated and may be incomplete or inaccurate. Always consult
        your healthcare provider or a clinical trial navigator before making medical
        decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    """Entry point for `trialmine-ui`."""
    # Streamlit runs the file directly; nothing else to do.


if __name__ == "__main__":
    main()
