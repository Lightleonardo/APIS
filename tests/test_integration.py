import pytest
from backend.orchestrator import run_pipeline, run_full_pipeline_with_advice
from backend.schemas import StudentInput, SemesterRecord, FEATURE_COLUMNS
from unittest.mock import Mock
import pytest


class MockModel:
    def __init__(self, val):
        self.val = val
    def predict(self, X):
        return [self.val]


def test_e2e_pipeline_with_mock_models(monkeypatch):
    """Test full pipeline with mocked ML models."""
    from backend.predictor import get_models
    from backend.schemas import FEATURE_COLUMNS

    mock_models = {
        'next_gpa': {'model': MockModel(4.2), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': []},
        'final_cgpa': {'model': MockModel(4.1), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': []},
        'graduation_class': {'model': MockModel("First Class"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': [], 'label_encoder': None},
        'academic_risk': {'model': MockModel("Low"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': [], 'label_encoder': None},
    }

    def mock_get_models():
        return mock_models

    monkeypatch.setattr('backend.predictor.get_models', mock_get_models)

    student = StudentInput(
        student_name="Integration Test",
        university="Test Uni", faculty="Science", department="Physics", course="Physics",
        programme_duration_years=5, current_level=200,
        semester_records=[
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2021/2022"),
            SemesterRecord(semester_number=2, gpa=3.8, credits=20, academic_session="2021/2022"),
            SemesterRecord(semester_number=3, gpa=4.0, credits=18, academic_session="2022/2023"),
            SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="2022/2023"),
        ],
        target_graduation_class="First Class",
    )

    result = run_pipeline(student)

    assert result.student_name == "Integration Test"
    assert result.current_cgpa is not None
    assert result.predicted_next_gpa == 4.2
    assert result.predicted_final_cgpa == 4.1
    assert result.predicted_graduation_class == "First Class"
    assert result.predicted_academic_risk == "Low"
    assert result.feasibility is not None
    assert len(result.semester_plan) == 6
    assert len(result.semester_history) == 4


def test_full_pipeline_with_advice_mocked(monkeypatch):
    """Test full pipeline + advisor with mocked ML models and advisor."""
    from backend.predictor import get_models
    from backend.schemas import FEATURE_COLUMNS

    mock_models = {
        'next_gpa': {'model': MockModel(4.2), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': []},
        'final_cgpa': {'model': MockModel(4.1), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': []},
        'graduation_class': {'model': MockModel("First Class"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': [], 'label_encoder': None},
        'academic_risk': {'model': MockModel("Low"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': [], 'label_encoder': None},
    }

    def mock_get_models():
        return mock_models

    monkeypatch.setattr('backend.predictor.get_models', mock_get_models)

    # Also mock the advisor to return a fixed response
    # run_advisor is imported inside run_full_pipeline_with_advice from backend.advisor
    with pytest.MonkeyPatch().context() as m:
        m.setattr('backend.advisor.run_advisor', lambda x: "Mocked advice response")

        student = StudentInput(
            student_name="Integration Test",
            university="U", faculty="F", department="D", course="C",
            programme_duration_years=4, current_level=100,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2022/2023"),
                SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="2022/2023"),
            ],
            target_graduation_class="First Class",
        )

        pipeline_result, advice = run_full_pipeline_with_advice(student)

        assert pipeline_result is not None
        assert pipeline_result.student_name == "Integration Test"
        assert advice == "Mocked advice response"


@pytest.mark.skipif(
    not pytest.importorskip("os").getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; set to run live integration test"
)
def test_full_pipeline_with_real_llm():
    import os
    from backend.orchestrator import run_full_pipeline_with_advice
    from backend.schemas import StudentInput, SemesterRecord

    student = StudentInput(
        student_name="Live Test Student",
        university="Test Uni", faculty="Science", department="CS", course="Computer Science",
        programme_duration_years=4, current_level=200,
        semester_records=[
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2022/2023"),
            SemesterRecord(semester_number=2, gpa=3.8, credits=20, academic_session="2022/2023"),
        ],
        target_graduation_class="First Class",
    )

    pipeline_result, advice = run_full_pipeline_with_advice(student)

    assert pipeline_result is not None
    assert isinstance(advice, str)
    assert len(advice) > 0
    assert len(advice) <= 500
    assert "Live Test Student" in advice or "[MOCK ADVISOR" in advice