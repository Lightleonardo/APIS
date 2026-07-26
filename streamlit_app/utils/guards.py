# guard to prevent error when non input is provided
import streamlit as st


def require_student_input():
    """Call at the top of any page that needs a completed StudentInput in session state."""
    if not st.session_state.get("form_data"):
        st.warning("Please complete your academic information first.")
        st.stop()


def require_pipeline_result():
    """Call at the top of any page that needs a computed PipelineResult in session state."""
    if st.session_state.get("pipeline_result") is None:
        st.warning("Run your analysis first — fill out the sidebar form and click 'Run Analysis'.")
        st.stop()