import pytest
from backend.mock_advisor import mock_advisor
from backend.schemas import AdvisorInput, SemesterTarget, FeatureImportance, ImprovementTrend


def test_mock_advisor_returns_string():
    advisor_in = AdvisorInput(
        student_name="Test", course="Physics", current_cgpa=4.0,
        target_graduation_class="First Class", target_cgpa=4.5,
        remaining_semesters=4, required_average_gpa=4.7,
        predicted_final_cgpa=4.3, predicted_graduation_class="First Class",
        academic_risk="Low", goal_feasible=True,
        best_possible_classification="First Class",
        academic_health_score=85, gpa_trend=ImprovementTrend.STABLE, consistency_index=20,
        semester_plan=[], top_features_final_cgpa=[], top_features_graduation_class=[],
        top_features_academic_risk=[],
    )
    result = mock_advisor(advisor_in)
    assert isinstance(result, str)
    assert "Test" in result
    assert "4.0" in result
    assert "4.3" in result
    assert "[MOCK ADVISOR" in result


def test_mock_advisor_validates_required_fields():
    with pytest.raises(AssertionError):
        mock_advisor(AdvisorInput(
            student_name="", course="C", current_cgpa=None,
            target_graduation_class=None, target_cgpa=None,
            remaining_semesters=0, required_average_gpa=None,
            predicted_final_cgpa=0, predicted_graduation_class="Pass",
            academic_risk="Low", goal_feasible=False,
            best_possible_classification="Pass",
            academic_health_score=0, gpa_trend=ImprovementTrend.INSUFFICIENT_DATA, consistency_index=0,
            semester_plan=[], top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[],
        ))