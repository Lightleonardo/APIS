from streamlit_app.utils.guards import require_pipeline_result
import streamlit as st
from streamlit_app.components.forms import render_sidebar_form
from streamlit_app.utils.backend_adapter import run_analysis_with_advice
from streamlit_app.utils.session_state import SessionState
from streamlit_app.utils.formatters import tone_badge


SessionState.init()

student = render_sidebar_form()

if student:
    with st.spinner("Analyzing..."):
        pipeline, advice = run_analysis_with_advice(student)
        SessionState.set_results(pipeline, advice)

st.header("🤖 AI Academic Advisor")

require_pipeline_result()

pipeline = SessionState.get_pipeline()
advice = SessionState.get_advice()

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
        if student:
            with st.spinner("Getting fresh advice..."):
                pipeline, advice = run_analysis_with_advice(student)
                SessionState.set_results(pipeline, advice)
                st.rerun()
        else:
            st.warning("Please re-submit the form in the sidebar to regenerate advice.")

if advice:
    st.markdown(f"**Tone:** {tone_badge(tone)}", unsafe_allow_html=True)
    st.markdown(f"> {advice}")
else:
    st.info("No advice generated yet — click 'Run Analysis' in the sidebar.")

if advice and st.button("📋 Copy Advice"):
    st.code(advice, language=None)
    st.toast("Copied to clipboard!")