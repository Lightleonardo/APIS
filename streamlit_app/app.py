import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # streamlit_app/ -> APIS/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from streamlit_app.config.streamlit_config import configure_page
from streamlit_app.utils.session_state import SessionState

configure_page()
SessionState.init()

st.title("🧠 Academic Performance Intelligence System")
st.caption("Plan. Predict. Succeed.")

# Defination section
with st.expander("ℹ️ About APIS", expanded=False):
    st.markdown("""
    ### What is APIS?
    APIS tells you exactly where you stand academically, what's realistically still possible, and what you need to do about it. Backed by real math and machine learning, explained in plain language.
    """)
#Notice section
with st.expander("ℹ️ Important Notices", expanded=False):
    st.markdown("""
     ### Data Privacy Notice
    This application processes data entirely within your current session. We do not store, share, or use your data elsewhere. **Refreshing or closing this page will permanently erase all entered data.**

    ### 4 Crucial Things to Know
    1. **Data Loss Vulnerability** — Reloading, closing the tab, or a browser crash will instantly wipe your progress.
    2. **Scope of Analysis** — This tool provides analytical insights only. It is not an official school grading or placement system.
    3. **Accuracy of Results** — Results are provided "as-is" based on your inputs and the underlying models. We are not liable for unexpected calculations or decisions made based on this tool.
    4. **No Guarantee of Outcomes** — Projections are estimates, not guarantees. Consult your academic advisor for official guidance.
    """)
    
st.sidebar.title("Navigation")
st.sidebar.markdown("""
1. **📊 Dashboard** — Trajectory & metrics
2. **📈 Planner** — Semester targets
3. **🔮 What-If** — Simulate scenarios
4. **🗨️ Advisor** — AI guidance
5. **⚙️ Settings** — Preferences
""")

# Show current analysis status
pipeline = st.session_state.get("pipeline_result")
if pipeline:
    st.sidebar.success(f"✅ Analysis: {pipeline.student_name}")
    st.sidebar.metric("CGPA", f"{pipeline.current_cgpa:.2f}" if pipeline.current_cgpa else "—")
    st.sidebar.metric("Health", f"{pipeline.academic_health_score}/100")
else:
    st.sidebar.info("No analysis run yet")