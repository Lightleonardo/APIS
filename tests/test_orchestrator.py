import pytest
from backend.orchestrator import run_pipeline, build_semester_history
from backend.schemas import StudentInput, SemesterRecord, PipelineResult
from unittest.mock import Mock
from backend.schemas import FEATURE_COLUMNS

class MockModel:
    def __init__(self, val):
        self.val = val
    def predict(self, X):
        return [self.val]

def test_full_pipeline_returns_pipeline_result(monkeypatch):
    student = StudentInput(
        student_name="Test Student",
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

    mock_models = {
        'next_gpa': {'model': MockModel(4.2), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}},
        'final_cgpa': {'model': MockModel(4.1), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}},
        'graduation_class': {'model': MockModel("First Class"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'label_encoder': None},
        'academic_risk': {'model': MockModel("Low"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'label_encoder': None},
    }

    def mock_get_models():
        return mock_models

    monkeypatch.setattr('backend.predictor.get_models', mock_get_models)

    result = run_pipeline(student)
    assert isinstance(result, PipelineResult)
    assert result.student_name == "Test Student"
    assert result.current_cgpa is not None
    assert result.feasibility is not None
    assert result.predicted_final_cgpa is not None
    assert len(result.semester_history) == 4


class TestBuildSemesterHistory:
    def test_reconstructs_cumulative_cgpa(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=4, current_level=100,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.0, credits=20, academic_session="2022/2023"),
                SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="2022/2023"),
            ],
        )
        from backend.calculator import run_calculator
        calc = run_calculator(student)
        history = build_semester_history(student, calc)
        assert len(history) == 2
        # Sem 1: 3.0 * 20 / 20 = 3.0
        assert history[0].cumulative_cgpa == 3.0
        # Sem 2: (3*20 + 4*20) / 40 = 3.5
        assert history[1].cumulative_cgpa == 3.5