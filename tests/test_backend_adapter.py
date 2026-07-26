import pytest
from unittest.mock import Mock, patch
from streamlit_app.utils.backend_adapter import run_analysis, run_analysis_with_advice
from backend.schemas import StudentInput, SemesterRecord, PipelineResult, FeasibilityResult, SemesterTarget, SemesterHistoryPoint
from backend.grading_rules import CLASS_MIN_CGPA


def make_pipeline_result() -> PipelineResult:
    return PipelineResult(
        student_name="Test", university="U", faculty="F", department="D", course="C",
        programme_duration_years=4, current_level=100,
        target_graduation_class="First Class", target_cgpa=4.5,
        current_cgpa=3.5, total_credits=20, semesters_completed=1, semesters_remaining=7,
        current_classification="Second Class Upper", gpa_trend="Stable", consistency_index=25, academic_health_score=75,
        target_cgpa_resolved=4.5, feasibility=FeasibilityResult(
            goal_achievable=True, max_achievable_cgpa=4.5, required_average_gpa=4.7,
            realistic_classification="First Class", confidence=0.8, message="ok"
        ),
        best_possible_classification="First Class", semester_plan=[],
        predicted_next_gpa=4.0, predicted_final_cgpa=4.2,
        predicted_graduation_class="First Class", predicted_academic_risk="Low",
        top_features_next_gpa=[], top_features_final_cgpa=[],
        top_features_graduation_class=[], top_features_academic_risk=[],
        semester_history=[SemesterHistoryPoint(semester_number=1, gpa=3.5, cumulative_cgpa=3.5, credits=20, academic_session="2023/2024")],
    )


def test_run_analysis_calls_orchestrator():
    student = StudentInput(
        student_name="Test", university="U", faculty="F", department="D", course="C",
        programme_duration_years=4, current_level=100,
        semester_records=[SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2023/2024")],
        target_graduation_class="First Class",
    )
    with patch('streamlit_app.utils.backend_adapter.run_pipeline') as mock_run:
        mock_run.return_value = make_pipeline_result()
        result = run_analysis(student)
        mock_run.assert_called_once_with(student)
        assert result is mock_run.return_value


def test_run_analysis_with_advice_calls_full_pipeline():
    student = StudentInput(
        student_name="Test", university="U", faculty="F", department="D", course="C",
        programme_duration_years=4, current_level=100,
        semester_records=[SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2023/2024")],
        target_graduation_class="First Class",
    )
    pipeline_result = make_pipeline_result()
    with patch('streamlit_app.utils.backend_adapter.run_full_pipeline_with_advice') as mock_run:
        mock_run.return_value = (pipeline_result, "advice text")
        pipeline, advice = run_analysis_with_advice(student)
        mock_run.assert_called_once_with(student)
        assert pipeline is pipeline_result
        assert advice == "advice text"