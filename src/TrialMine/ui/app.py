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
    "Pancreatic cancer new treatments",
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
    layout="wide",
)


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
        # Header row: title + match score
        head_l, head_r = st.columns([4, 1])
        with head_l:
            title = r.get("title") or "(untitled)"
            url = r.get("url")
            if url:
                st.markdown(f"#### [{title}]({url})")
            else:
                st.markdown(f"#### {title}")
            st.caption(f"`{r.get('nct_id', '')}` &nbsp; · &nbsp; rank #{idx + 1}")

        with head_r:
            pct = _match_score(idx, total)
            st.progress(pct, text=f"Match {pct:.0%}")

        # Status + phase badges
        badges: list[str] = []
        if r.get("status"):
            badges.append(_status_pill(r["status"]))
        if r.get("phase"):
            badges.append(_phase_pill(r["phase"]))
        if badges:
            st.markdown(" &nbsp;|&nbsp; ".join(badges))

        # Condition chips
        chips = _condition_chips(r.get("conditions") or [])
        if chips:
            st.markdown(chips)

        # Warnings (skip "fallback path" — already shown as a top-level banner)
        for w in r.get("warnings") or []:
            if w == "fallback path":
                continue
            st.warning(f"⚠️ {w}")

        # Expandables
        if r.get("explanation"):
            with st.expander("🤖 Why this matches"):
                st.write(r["explanation"])

        if r.get("eligibility") is not None:
            with st.expander("📋 Eligibility details"):
                _render_eligibility(r["eligibility"])

        with st.expander("📄 Full details"):
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
    st.markdown("## ⚙️ Settings")

    use_agent = st.toggle(
        "🤖 Use AI Agent",
        value=True,
        help=(
            "When **on**, an LLM extracts patient details from natural language and "
            "the system applies smart filters + per-trial eligibility checking. "
            "When **off**, runs plain BM25 + semantic hybrid search."
        ),
    )

    if use_agent:
        st.caption("AI agent infers status / phase filters from your query.")
        status_filter = "Any"
        phase_filter = "Any"
    else:
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

    top_k = st.slider("Max results", min_value=5, max_value=50, value=20, step=5)

    # Stats — only after a successful search
    if st.session_state.results:
        data = st.session_state.results
        st.markdown("---")
        st.markdown("## 📊 Stats")

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
            with st.expander("🔍 Agent reasoning"):
                _render_agent_trace(trace)


# ── Main: header ─────────────────────────────────────────────────────────────

st.markdown("# 🔬 TrialMine — Find Clinical Trials With AI")
st.markdown("##### Describe your situation in plain language")


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
        placeholder="Example: I'm 55 with stage 3 lung cancer who has tried chemo...",
        label_visibility="collapsed",
        key="search_input",
    )
    st.form_submit_button(
        "🔍 Search",
        type="primary",
        on_click=_trigger_form,
    )

st.markdown("**Try an example:**")
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
    st.markdown(
        f"### {n} trial{'s' if n != 1 else ''} found"
        f" &nbsp; *for «{qtext}»*"
        f" &nbsp; · &nbsp; {elapsed_s:.1f} s"
    )

    # Patient profile summary (agent path only)
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
        with st.container(border=True):
            st.markdown("**🧠 Here's what I understood from your query:**")
            chips: list[str] = []
            if profile.get("condition"):
                chips.append(f"**Condition:** {profile['condition']}")
            if profile.get("condition_stage"):
                chips.append(f"**Stage:** {profile['condition_stage']}")
            if profile.get("age") is not None:
                chips.append(f"**Age:** {profile['age']}")
            if profile.get("sex"):
                chips.append(f"**Sex:** {profile['sex']}")
            if profile.get("biomarkers"):
                chips.append(f"**Biomarkers:** {', '.join(profile['biomarkers'])}")
            if profile.get("prior_treatments"):
                chips.append(f"**Prior treatments:** {', '.join(profile['prior_treatments'])}")
            if profile.get("location"):
                chips.append(f"**Location:** {profile['location']}")
            st.markdown(" &nbsp; · &nbsp; ".join(chips))

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
    st.info(
        "👆 **Type your situation above** or click an example to get started. "
        "TrialMine searches 140K+ oncology trials and explains why each one might match."
    )


# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ **Research tool only — not medical advice.** Results are AI-generated "
    "and may be incomplete or inaccurate. Always consult your healthcare "
    "provider or a clinical trial navigator before making medical decisions."
)


def main() -> None:
    """Entry point for `trialmine-ui`."""
    # Streamlit runs the file directly; nothing else to do.


if __name__ == "__main__":
    main()
