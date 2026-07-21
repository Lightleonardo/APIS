import pytest
import pandas as pd
from unittest.mock import Mock
from backend.predictor import (
    load_all_models, get_models, build_feature_vector, run_predictor, _decode_label
)
from backend.schemas import StudentInput, SemesterRecord, CalculatorOutput, PredictorOutput, FEATURE_COLUMNS, ImprovementTrend


class MockModel:
    def __init__(self, return_value):
        self.return_value = return_value
    def predict(self, X):
        return [self.return_value]


class TestBuildFeatureVector:
    def test_vector_shape_and_columns(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=200,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="x"),
                SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="x"),
                SemesterRecord(semester_number=3, gpa=4.2, credits=18, academic_session="x"),
                SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="x"),
            ],
        )
        calc = CalculatorOutput(
            current_cgpa=3.94, total_credits=76, semesters_completed=4,
            semesters_remaining=6, current_classification="Second Class Upper",
            gpa_trend=ImprovementTrend.STABLE, consistency_index=0, academic_health_score=0,
            gpa_trend_slope=0.1, gpa_volatility=0.25,
            recent_gpa_avg_3=4.1, credits_velocity=19.0,
        )
        X = build_feature_vector(student, calc)
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == FEATURE_COLUMNS
        assert X.shape == (1, len(FEATURE_COLUMNS))


class TestDecodeLabel:
    def test_string_label_passthrough(self):
        artifact = {'model': MockModel("First Class"), 'label_encoder': None}
        result = _decode_label(artifact, None)
        assert result == "First Class"

    def test_encoded_label_with_encoder(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["Fail", "Pass", "Third Class", "Second Class Lower", "Second Class Upper", "First Class"])
        artifact = {'model': MockModel(0), 'label_encoder': le}  # 0 = "Fail" after fit
        result = _decode_label(artifact, None)
        assert result == "Fail"


class TestRunPredictorWithMocks:
    def test_predictor_output_structure(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=200,
            semester_records=[
                SemesterRecord(semester_number=i, gpa=4.0, credits=20, academic_session="x") for i in range(1,5)
            ],
        )
        calc = CalculatorOutput(
            current_cgpa=4.0, total_credits=80, semesters_completed=4,
            semesters_remaining=6, current_classification="First Class",
            gpa_trend=ImprovementTrend.STABLE, consistency_index=25, academic_health_score=90,
            gpa_trend_slope=0.0, gpa_volatility=0.0,
            recent_gpa_avg_3=4.0, credits_velocity=20.0,
        )
        mock_models = {
            'next_gpa': {'model': MockModel(4.2), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}},
            'final_cgpa': {'model': MockModel(4.1), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}},
            'graduation_class': {'model': MockModel("First Class"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'label_encoder': None},
            'academic_risk': {'model': MockModel("Low"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'label_encoder': None},
        }
        out = run_predictor(student, calc, models=mock_models)
        assert isinstance(out, PredictorOutput)
        assert out.predicted_next_gpa == 4.2
        assert out.predicted_final_cgpa == 4.1
        assert out.predicted_graduation_class == "First Class"
        assert out.predicted_academic_risk == "Low"