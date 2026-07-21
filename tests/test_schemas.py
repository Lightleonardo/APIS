import pytest
from pydantic import ValidationError
from backend.schemas import (
    StudentInput, SemesterRecord, CalculatorOutput, PlannerOutput,
    FeasibilityResult, SemesterTarget, PredictorOutput, PipelineResult,
    AdvisorInput, ImprovementTrend, DatasetRow, FeatureImportance,
    FEATURE_COLUMNS, EXCLUDED_FROM_FEATURES, GRADUATION_CLASSES,
)


class TestSemesterRecord:
    def test_valid_record(self):
        r = SemesterRecord(semester_number=1, gpa=4.5, credits=18, academic_session="2023/2024")
        assert r.gpa == 4.5

    def test_gpa_bounds(self):
        with pytest.raises(ValidationError):
            SemesterRecord(semester_number=1, gpa=5.1, credits=18, academic_session="2023/2024")
        with pytest.raises(ValidationError):
            SemesterRecord(semester_number=1, gpa=-0.1, credits=18, academic_session="2023/2024")


class TestStudentInput:
    def test_valid_minimal(self):
        s = StudentInput(
            student_name="Test Student",
            university="Test Uni",
            faculty="Science",
            department="Physics",
            course="Physics",
            gpa_scale=5.0,
            programme_duration_years=5,
            current_level=100,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=2, gpa=4.2, credits=20, academic_session="2021/2022"),
            ],
        )
        assert s.current_level == 100

    def test_level_consistency_validation(self):
        with pytest.raises(ValidationError):
            StudentInput(
                student_name="Test",
                university="U", faculty="F", department="D", course="C",
                programme_duration_years=5, current_level=300,
                semester_records=[SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="2021/2022")],
            )


class TestCalculatorOutput:
    def test_all_fields_present(self):
        out = CalculatorOutput(
            current_cgpa=4.2, total_credits=40, semesters_completed=2,
            semesters_remaining=8, current_classification="First Class",
            gpa_trend=ImprovementTrend.STABLE, consistency_index=25,
            academic_health_score=85,
            gpa_trend_slope=0.1, gpa_volatility=0.2,
            recent_gpa_avg_3=4.1, credits_velocity=20.0,
        )
        assert out.current_cgpa == 4.2


class TestDatasetRow:
    def test_feature_columns_excludes_targets(self):
        targets = {"next_semester_gpa", "final_cgpa", "graduation_class", "academic_risk"}
        for t in targets:
            assert t not in FEATURE_COLUMNS
        assert "student_id" not in FEATURE_COLUMNS

    def test_6_classes_in_graduation(self):
        assert len(GRADUATION_CLASSES) == 6
        assert "Fail" in GRADUATION_CLASSES