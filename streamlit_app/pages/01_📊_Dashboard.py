import streamlit as st
from streamlit_app.utils.backend_adapter import run_analysis
from streamlit_app.utils.session_state import SessionState
from streamlit_app.components.forms import render_sidebar_form
from streamlit_app.components.charts import (
    render_trajectory_chart, render_metric_cards, render_feature_tables
)
from streamlit_app.utils.formatters import trend_label


SessionState.init()

# Sidebar form
student = render_sidebar_form()

if student:
    with st.spinner("Analyzing..."):
        pipeline = run_analysis(student)
        SessionState.set_results(pipeline)

pipeline = SessionState.get_pipeline()

if pipeline:
    st.header(f"📊 {pipeline.student_name} — Academic Trajectory")

    # Metric cards
    render_metric_cards(pipeline)

    st.divider()

    # Trajectory chart
    st.subheader("Cumulative CGPA Trajectory")
    render_trajectory_chart(pipeline)

    # Summary info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Current Classification:** {pipeline.current_classification or '—'}")
        st.info(f"**GPA Trend:** {trend_label(pipeline.gpa_trend)}")
        st.info(f"**Consistency Index:** {pipeline.consistency_index}/25")
    with col2:
        req = pipeline.feasibility.required_average_gpa
        st.info(f"**Required Avg GPA:** {req:.2f}" if req else "**Required Avg GPA:** N/A (final semester)")
        st.info(f"**Max Achievable CGPA:** {pipeline.feasibility.max_achievable_cgpa:.2f} ({pipeline.feasibility.realistic_classification})")
        st.info(f"**Confidence:** {pipeline.feasibility.confidence:.0%}")

    st.divider()
    render_feature_tables(pipeline)

else:
    st.info("👈 Fill in your semester records and click **Run Analysis** to begin.")