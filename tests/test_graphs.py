import pytest
from backend.graphs import trajectory_chart, semester_planner_chart
from backend.schemas import PipelineResult, SemesterHistoryPoint, SemesterTarget, FeasibilityResult, FeatureImportance, ImprovementTrend


def make_mock_pipeline():
    return PipelineResult(
        student_name="Test", university="U", faculty="F", department="D", course="C",
        programme_duration_years=5, current_level=300,
        target_graduation_class="First Class", target_cgpa=4.5,
        current_cgpa=3.9, total_credits=80, semesters_completed=4, semesters_remaining=6,
        current_classification="Second Class Upper", gpa_trend=ImprovementTrend.STABLE, consistency_index=20, academic_health_score=75,
        target_cgpa_resolved=4.5, feasibility=FeasibilityResult(
            goal_achievable=True, max_achievable_cgpa=4.6, required_average_gpa=4.7,
            realistic_classification="First Class", confidence=0.8, message="ok"
        ),
        best_possible_classification="First Class", semester_plan=[
            SemesterTarget(semester_number=5, target_gpa=4.7, cumulative_cgpa_if_met=4.1),
            SemesterTarget(semester_number=6, target_gpa=4.7, cumulative_cgpa_if_met=4.2),
        ],
        predicted_next_gpa=4.2, predicted_final_cgpa=4.4,
        predicted_graduation_class="First Class", predicted_academic_risk="Low",
        top_features_next_gpa=[], top_features_final_cgpa=[],
        top_features_graduation_class=[], top_features_academic_risk=[],
        semester_history=[
            SemesterHistoryPoint(semester_number=1, gpa=3.5, cumulative_cgpa=3.5, credits=20, academic_session="2021/2022"),
            SemesterHistoryPoint(semester_number=2, gpa=3.8, cumulative_cgpa=3.65, credits=20, academic_session="2021/2022"),
            SemesterHistoryPoint(semester_number=3, gpa=4.0, cumulative_cgpa=3.77, credits=18, academic_session="2022/2023"),
            SemesterHistoryPoint(semester_number=4, gpa=4.2, cumulative_cgpa=3.9, credits=18, academic_session="2022/2023"),
        ],
    )


class TestTrajectoryChart:
    def test_returns_dict_with_expected_keys(self):
        pipeline = make_mock_pipeline()
        fig_dict = trajectory_chart(pipeline)
        assert isinstance(fig_dict, dict)
        assert "data" in fig_dict
        assert "layout" in fig_dict

    def test_has_historical_trace(self):
        pipeline = make_mock_pipeline()
        fig_dict = trajectory_chart(pipeline)
        trace_names = [t.get("name", "") for t in fig_dict["data"]]
        assert "Actual CGPA" in trace_names

    def test_has_goal_line(self):
        pipeline = make_mock_pipeline()
        fig_dict = trajectory_chart(pipeline)
        shapes = fig_dict.get("layout", {}).get("shapes", [])
        assert any(s.get("type") == "line" and s.get("line", {}).get("dash") == "dash" for s in shapes)


class TestSemesterPlannerChart:
    def test_returns_dict(self):
        pipeline = make_mock_pipeline()
        fig_dict = semester_planner_chart(pipeline)
        assert isinstance(fig_dict, dict)
        assert "data" in fig_dict

    def test_has_both_traces(self):
        pipeline = make_mock_pipeline()
        fig_dict = semester_planner_chart(pipeline)
        trace_names = [t.get("name", "") for t in fig_dict["data"]]
        assert "Actual GPA" in trace_names
        assert "Target GPA" in trace_names