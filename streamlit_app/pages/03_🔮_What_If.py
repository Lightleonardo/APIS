import streamlit as st
from streamlit_app.components.charts import render_what_if_chart
from streamlit_app.utils.formatters import fmt_cgpa
from backend.graphs import what_if_simulator
from backend.grading_rules import classify_cgpa
import plotly.graph_objects as go




st.header("🔮 What-If Simulator")

pipeline = st.session_state.get("pipeline_result")
if not pipeline:
    st.warning("Run analysis first from the Dashboard.")
    st.stop()

st.subheader("🔮 What-If Simulator")
st.caption("Adjust future semester GPAs to see impact on final CGPA and classification.")

n_remaining = pipeline.semesters_remaining
if n_remaining == 0:
    st.info("No remaining semesters to simulate.")
    st.stop()

# Initialize from session state or use plan targets
if (
    "what_if_gpas" not in st.session_state or st.session_state.what_if_gpas is None or len(st.session_state.what_if_gpas) != n_remaining):
    st.session_state.what_if_gpas = [p.target_gpa for p in pipeline.semester_plan]

gpas = []
cols = st.columns(min(n_remaining, 4))
for i in range(n_remaining):
    sem_num = pipeline.semesters_completed + i + 1
    with cols[i % 4]:
        gpa = st.slider(
            f"Semester {sem_num}",
            0.0, 5.0,
            st.session_state.what_if_gpas[i],
            0.01,
            format="%.2f",
            key=f"whatif_{i}"
        )
        gpas.append(gpa)

st.session_state.what_if_gpas = gpas

if st.button("📊 Simulate", type="primary"):
    st.rerun()

# Show simulated result
if gpas:
    render_what_if_chart(pipeline, gpas)

    # Quick summary
    fig_dict = what_if_simulator(pipeline, gpas)
    fig = go.Figure(fig_dict)
    final_cgpa = fig.data[-1].y[-1] if fig.data else pipeline.current_cgpa
    final_class = classify_cgpa(final_cgpa)
    change = final_cgpa - (pipeline.current_cgpa or 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated Final CGPA", fmt_cgpa(final_cgpa))
    col2.metric("Projected Class", final_class)
    col3.metric("Change vs Current", f"{change:+.2f}")