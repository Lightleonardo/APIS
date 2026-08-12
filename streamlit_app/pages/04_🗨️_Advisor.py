import streamlit as st
from streamlit_app.utils.backend_adapter import run_analysis_with_advice
from streamlit_app.utils.session_state import SessionState
from streamlit_app.utils.formatters import tone_badge
from streamlit_app.components.forms import render_sidebar_form
from backend.advisor import run_advisor, AdvisorResult
from backend.orchestrator import pipeline_to_advisor_input
import time


SessionState.init()

# First, try to get existing pipeline from session state
pipeline = SessionState.get_pipeline()
advice = SessionState.get_advice()
advisor_result = st.session_state.get("advisor_result")  # Store AdvisorResult metadata

# Only show form if no pipeline exists yet
if not pipeline:
    student = render_sidebar_form()
    if student:
        with st.spinner("Analyzing..."):
            pipeline, advice = run_analysis_with_advice(student)
            SessionState.set_results(pipeline, advice)
            st.rerun()
else:
    # Show a compact status in sidebar instead of full form
    with st.sidebar:
        st.success(f"✅ Analysis loaded: {pipeline.student_name}")
        st.caption(f"CGPA: {pipeline.current_cgpa:.2f} | Health: {pipeline.academic_health_score}/100")
        if st.button("🔄 Run New Analysis", use_container_width=True):
            SessionState.clear()
            st.rerun()

st.header("🗨️ AI Academic Advisor")


# Tone selector
col1, col2 = st.columns([3, 1])
with col1:
    tone = st.selectbox(
        "Advisor Tone",
        ["encouraging", "direct", "analytical"],
        index=0,
        key="advisor_tone"
    )
with col2:
    if st.button("🔄 Regenerate", use_container_width=True):
        if pipeline:
            with st.spinner("Getting fresh advice..."):
                advisor_input = pipeline_to_advisor_input(pipeline)
                advisor_input.tone = tone
                result: AdvisorResult = run_advisor(advisor_input)
                SessionState.set_results(pipeline, result.response)
                st.session_state.advisor_result = result
                st.rerun()
        else:
            st.warning("No analysis data available. Run analysis from Dashboard first.")

# Status messages based on advisor result source
if advisor_result:
    if advisor_result.source == "cache":
        st.info("🔁 No change — loaded from cache")
    elif advisor_result.source == "mock_rate_limited":
        if advisor_result.rate_limit_reset_at:
            wait_seconds = max(0, int(advisor_result.rate_limit_reset_at - time.time()))
            st.warning(f"⏳ Rate limited. Try again in {wait_seconds}s (showing offline advisor)")
        else:
            st.warning("⏳ Rate limited — showing offline advisor")
    elif advisor_result.source == "mock_empty":
        st.warning("⚠️ AI returned empty response — showing offline advisor")
    elif advisor_result.source == "mock_error":
        st.error("❌ AI service error — showing offline advisor")
    # "llm" source shows no special message

if advice:
    st.markdown(f"**Tone:** {tone_badge(tone)}", unsafe_allow_html=True)
    st.markdown(f"> {advice}")
else:
    if pipeline:
        with st.spinner("Generating initial advice..."):
            advisor_input = pipeline_to_advisor_input(pipeline)
            advisor_input.tone = tone
            result: AdvisorResult = run_advisor(advisor_input)
            SessionState.set_results(pipeline, result.response)
            st.session_state.advisor_result = result
            st.rerun()
    else:
        st.info("👈 Run analysis from the **Dashboard** first to get AI advice.")

if advice and st.button("📋 Copy Advice"):
    st.code(advice, language=None)
    st.toast("Copied to clipboard!")