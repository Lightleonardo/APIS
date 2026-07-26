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

st.title("📊 Academic Performance Intelligence System")
st.caption("Plan. Predict. Succeed.")

st.sidebar.title("Navigation")
st.sidebar.markdown("""
1. **📊 Dashboard** — Trajectory & metrics
2. **📈 Planner** — Semester targets
3. **🔮 What-If** — Simulate scenarios
4. **🤖 Advisor** — AI guidance
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