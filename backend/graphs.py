from typing import List
from backend.schemas import PipelineResult
import plotly.graph_objects as go


def trajectory_chart(pipeline: PipelineResult) -> dict:
    fig = go.Figure()

    # Historical trajectory
    semesters = [p.semester_number for p in pipeline.semester_history]
    cgpas = [p.cumulative_cgpa for p in pipeline.semester_history]
    fig.add_trace(go.Scatter(
        x=semesters, y=cgpas, mode='lines+markers',
        name='Actual CGPA', line=dict(color='blue')
    ))

    # Predicted next semester
    if pipeline.predicted_next_gpa is not None and pipeline.semesters_remaining > 0:
        next_sem = pipeline.semesters_completed + 1
        next_credits = estimate_credits_for_semester(pipeline.current_level)
        pred_cum_cgpa = (
            (pipeline.current_cgpa * pipeline.total_credits + pipeline.predicted_next_gpa * next_credits) /
            (pipeline.total_credits + next_credits)
        ) if pipeline.current_cgpa is not None else pipeline.predicted_next_gpa

        fig.add_trace(go.Scatter(
            x=[next_sem], y=[pred_cum_cgpa], mode='markers',
            name='Predicted Next', marker=dict(color='orange', size=10, symbol='diamond')
        ))

    # Goal line
    fig.add_hline(
        y=pipeline.target_cgpa_resolved, line_dash='dash', line_color='green',
        annotation_text=f"Target: {pipeline.target_cgpa_resolved:.2f}"
    )

    # First Class threshold
    fig.add_hline(
        y=4.50, line_dash='dot', line_color='gray',
        annotation_text="First Class threshold"
    )

    fig.update_layout(
        title="Academic Trajectory",
        xaxis_title="Semester",
        yaxis_title="Cumulative CGPA",
        yaxis_range=[0, 5.0],
        template="plotly_white",
        showlegend=True,
    )

    return fig.to_dict()


def semester_planner_chart(pipeline: PipelineResult) -> dict:
    fig = go.Figure()

    # Historical GPAs
    sem_hist = [p.semester_number for p in pipeline.semester_history]
    gpa_hist = [p.gpa for p in pipeline.semester_history]
    fig.add_trace(go.Bar(x=sem_hist, y=gpa_hist, name='Actual GPA', marker_color='blue'))

    # Target GPAs
    sem_targets = [p.semester_number for p in pipeline.semester_plan]
    gpa_targets = [p.target_gpa for p in pipeline.semester_plan]
    fig.add_trace(go.Bar(x=sem_targets, y=gpa_targets, name='Target GPA', marker_color='green', opacity=0.7))

    fig.update_layout(
        title="Semester GPA Plan",
        xaxis_title="Semester",
        yaxis_title="GPA",
        yaxis_range=[0, 5.0],
        barmode='group',
        template="plotly_white",
    )
    return fig.to_dict()


def what_if_simulator(pipeline: PipelineResult, what_if_gpas: List[float]) -> dict:
    """What-if: replace future semester GPAs with user values, recompute trajectory."""
    history = pipeline.semester_history.copy()
    current_cgpa = pipeline.current_cgpa or 0.0
    current_credits = pipeline.total_credits

    for i, what_if_gpa in enumerate(what_if_gpas):
        sem_num = pipeline.semesters_completed + i + 1
        next_credits = estimate_credits_for_semester(pipeline.current_level)
        current_credits += next_credits
        current_cgpa = (
            (current_cgpa * (current_credits - next_credits) + what_if_gpa * next_credits) /
            current_credits
        )
        history.append(type(pipeline.semester_history[0])(
            semester_number=sem_num,
            gpa=what_if_gpa,
            cumulative_cgpa=round(current_cgpa, 2),
            credits=next_credits,
            academic_session=f"Projected {sem_num}"
        ))

    from copy import deepcopy
    mod_pipeline = deepcopy(pipeline)
    mod_pipeline.semester_history = history
    mod_pipeline.semesters_remaining = 0
    mod_pipeline.predicted_next_gpa = None

    return trajectory_chart(mod_pipeline)


# Import at bottom to avoid circular
from backend.grading_rules import estimate_credits_for_semester