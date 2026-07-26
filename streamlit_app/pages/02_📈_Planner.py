import streamlit as st
from streamlit_app.components.charts import render_semester_planner, render_metric_cards
import pandas as pd


st.header("📈 Semester Planner")

pipeline = st.session_state.get("pipeline_result")
if not pipeline:
    st.warning("Run analysis first from the Dashboard.")
    st.stop()

render_metric_cards(pipeline)

st.divider()
st.subheader("Actual vs Target GPA per Semester")
render_semester_planner(pipeline)

st.divider()
st.subheader("Semester Targets")
if pipeline.semester_plan:
    df = pd.DataFrame([
        {
            "Semester": p.semester_number,
            "Target GPA": f"{p.target_gpa:.2f}",
            "Projected Cum. CGPA": f"{p.cumulative_cgpa_if_met:.2f}",
        }
        for p in pipeline.semester_plan
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.info("No remaining semesters — this is your final semester.")

st.caption(f"Feasibility: {pipeline.feasibility.message}")