import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from backend.graphs import trajectory_chart, semester_planner_chart, what_if_simulator
from backend.schemas import PipelineResult


def render_trajectory_chart(pipeline: PipelineResult):
    fig_dict = trajectory_chart(pipeline)
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)


def render_semester_planner(pipeline: PipelineResult):
    fig_dict = semester_planner_chart(pipeline)
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)


def render_what_if_chart(pipeline: PipelineResult, what_if_gpas: list[float]):
    fig_dict = what_if_simulator(pipeline, what_if_gpas)
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)


def render_metric_cards(pipeline: PipelineResult):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current CGPA", f"{pipeline.current_cgpa:.2f}" if pipeline.current_cgpa else "—")
    with col2:
        st.metric("Predicted Final CGPA", f"{pipeline.predicted_final_cgpa:.2f}")
    with col3:
        st.metric("Academic Health", f"{pipeline.academic_health_score}/100")
    with col4:
        st.metric("Goal Feasible", "✅ Yes" if pipeline.feasibility.goal_achievable else "❌ No")


def render_feature_tables(pipeline: PipelineResult):
    with st.expander("🔍 Top Predictive Features"):
        tabs = st.tabs(["Next GPA", "Final CGPA", "Graduation Class", "Academic Risk"])
        feature_sets = [
            ("top_features_next_gpa", tabs[0]),
            ("top_features_final_cgpa", tabs[1]),
            ("top_features_graduation_class", tabs[2]),
            ("top_features_academic_risk", tabs[3]),
        ]
        for attr, tab in feature_sets:
            with tab:
                feats = getattr(pipeline, attr, [])
                if feats:
                    df = pd.DataFrame([{"Feature": f.feature, "Importance": f.importance} for f in feats])
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.info("No feature importance available")