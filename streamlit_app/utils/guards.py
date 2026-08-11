# guard to prevent error when non input is provided
import streamlit as st
from streamlit_app.components.forms import render_sidebar_form
from streamlit_app.utils.backend_adapter import run_analysis

def require_student_input():
    """Call at the top of any page that needs a completed StudentInput in session state."""
    if not st.session_state.get(render_sidebar_form()):
        st.warning("Please complete your academic information first.")
        st.stop()


def require_pipeline_result():
    """Call at the top of any page that needs a computed PipelineResult in session state."""
    if st.session_state.get(run_analysis()) is None:
        st.warning("Run your analysis first — fill out the sidebar form and click 'Run Analysis'.")
        st.stop()