"""Darukaa.Earth Biodiversity Intelligence System - Streamlit demo."""
from __future__ import annotations

import streamlit as st

from core.recommender import advise
from core.retriever import get_retriever
from core.schemas import SiteProfile
from core.slots import derive_conditions, extract_profile
from core.verifier import verify

st.set_page_config(page_title="Darukaa Biodiversity Intelligence", page_icon="🌿", layout="wide")

# ---------------- theme ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    :root {
        --forest: #17331f;
        --forest-light: #2c4a34;
        --sage: #4a7856;
        --sage-light: #eaf2ea;
        --paper: #fbfbf9;
    }

    .stApp { background-color: var(--paper); }

    /* header */
    .db-hero {
        padding: 2.2rem 0 1.2rem 0;
        border-bottom: 1px solid #e2e5e0;
        margin-bottom: 1.5rem;
    }
    .db-eyebrow {
        color: var(--sage);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .db-title {
        color: var(--forest);
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }
    .db-sub { color: #5c6b60; font-size: 0.98rem; max-width: 640px; }

    /* buttons */
    .stButton > button {
        background-color: var(--forest) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 600 !important;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover { background-color: var(--forest-light) !important; }

    /* sidebar */
    section[data-testid="stSidebar"] { background-color: var(--sage-light); border-right: 1px solid #dbe6dc; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--forest); }

    /* chat bubbles */
    div[data-testid="stChatMessage"] {
        background-color: white;
        border: 1px solid #e6e9e5;
        border-radius: 12px;
        padding: 0.4rem 0.8rem;
    }

    /* form container */
    .db-card {
        background: white;
        border: 1px solid #e6e9e5;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }

    h3 { color: var(--forest) !important; }
    .stAlert { border-radius: 10px; }

    div[data-testid="stChatMessageContent"] * {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }
    div[data-testid="stChatMessageContent"] h3 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

if "profile" not in st.session_state:
    st.session_state.profile = SiteProfile()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_debug" not in st.session_state:
    st.session_state.last_debug = None


def render_response(resp) -> str:
    if resp.mode == "clarifying":
        lines = [resp.summary, ""]
        for q in resp.clarifying_questions:
            lines.append(f"- {q}")
        return "\n".join(lines)

    lines = []
    if resp.summary:
        lines += [f"**Assistant:** {resp.summary}", ""]
    if resp.diagnosis:
        lines += [f"**Diagnosis.** {resp.diagnosis}", ""]

    for i, r in enumerate(resp.recommendations, 1):
        flag = "" if r.grounded else "  *(grounding flagged)*"
        lines.append(f"### {i}. {r.action}{flag}")
        lines.append(r.scientific_reasoning)
        if r.expected_change:
            lines.append(f"\n**Expected change:** {r.expected_change}")
        if r.metrics_improved:
            lines.append(f"**Metrics improved:** {', '.join(r.metrics_improved)}")
        lines.append(f"**Time horizon:** {r.time_horizon.value}  |  **Confidence:** {r.confidence:.2f}")
        if r.evidence:
            lines.append("**Evidence:** " + ", ".join(f"[{e.card_id}] {e.source}" for e in r.evidence))
        if not r.grounded and r.grounding_note:
            lines.append(f"\n> Verifier note: {r.grounding_note}")
        lines.append("")
    return "\n".join(lines)


def run_pipeline(force: bool = False) -> None:
    profile = st.session_state.profile
    conditions = derive_conditions(profile)
    st.session_state.last_debug = {
        "profile": profile.model_dump(exclude_none=True),
        "conditions": conditions,
        "retrieved": get_retriever().search(profile.to_context_string()),
    }
    resp = verify(advise(profile, force_answer=force))
    st.session_state.messages.append({"role": "assistant", "content": render_response(resp)})


def run_chat_turn(user_text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.profile = extract_profile(user_text, st.session_state.profile)
    run_pipeline()

# ---------------- hero ----------------
st.markdown("""
<div class="db-hero">
    <div class="db-eyebrow">Built on evidence, grounded in reasoning</div>
    <div class="db-title">Biodiversity Intelligence System</div>
    <div class="db-sub">Describe your land below. Every recommendation is traced to a curated
    evidence base and audited for scientific grounding before it reaches you.</div>
</div>
""", unsafe_allow_html=True)

# ---------------- quick-entry form (fast path) ----------------
if not st.session_state.messages:
    st.markdown('<div class="db-card">', unsafe_allow_html=True)
    st.markdown("##### Quick site setup")
    c1, c2 = st.columns(2)
    with c1:
        soc = st.number_input("Soil organic carbon (%)", 0.0, 25.0, 0.5, 0.1)
        land_use = st.selectbox("Land use", ["monoculture", "polyculture", "grassland", "agroforestry", "fallow"])
    with c2:
        rainfall = st.selectbox("Rainfall pattern", ["low", "moderate", "high", "erratic"])
        region = st.selectbox("Region type", ["arid", "semi_arid", "subhumid", "humid", "coastal"])
    crop = st.text_input("Main crop (optional)", placeholder="e.g. wheat")

    if st.button("Get recommendations →"):
        st.session_state.profile = SiteProfile(
            soil_organic_carbon_pct=soc, land_use=land_use,
            rainfall_pattern=rainfall, region_type=region,
            crop=crop or None,
        )
        st.session_state.messages.append({
            "role": "user",
            "content": f"SOC {soc}%, {land_use} land, {rainfall} rainfall, {region.replace('_',' ')} region"
                       + (f", growing {crop}" if crop else ""),
        })
        try:
            with st.spinner("Retrieving evidence and tracing causal pathways..."):
                run_pipeline(force=True)
        except Exception as exc:
            st.error(f"Something went wrong generating the recommendation: {exc}")
            st.session_state.messages.pop()  # remove the user message so form reappears
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    hint_col, reset_col = st.columns([5, 1])
    with hint_col:
        st.caption("Prefer to just describe it in your own words? Type below instead.")
    with reset_col:
        if st.button("Start over", use_container_width=True):
            st.session_state.profile = SiteProfile()
            st.session_state.messages = []
            st.session_state.last_debug = None
            st.rerun()

# ---------------- chat ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Describe your land, or ask a follow-up..."):
    with st.spinner("Retrieving evidence and tracing causal pathways..."):
        run_chat_turn(prompt)
    st.rerun()

if st.session_state.last_debug:
    with st.expander("🔍 Pipeline transparency - how this answer was produced"):
        d = st.session_state.last_debug
        st.markdown("**Detected conditions:** " + (", ".join(d["conditions"]) or "none"))
        st.markdown("**Retrieved evidence:**")
        for c in d["retrieved"]:
            st.markdown(f"`{c['id']}` **{c['title']}** — score `{c['retrieval_score']}` via `{c['retrieved_by']}`")