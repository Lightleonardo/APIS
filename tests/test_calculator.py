import pytest
from backend.calculator import (
    calculate_cgpa, calculate_gpa_trend, calculate_consistency_index,
    calculate_academic_health_score, run_calculator
)
from backend.schemas import StudentInput, SemesterRecord, CalculatorOutput, ImprovementTrend


class TestCalculateCGPA:
    def test_empty_returns_none(self):
        assert calculate_cgpa([]) is None

    def test_single_semester(self):
        records = [SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="2023/2024")]
        assert calculate_cgpa(records) == 4.0

    def test_weighted_multiple(self):
        records = [
            SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="2023/2024"),
            SemesterRecord(semester_number=2, gpa=3.0, credits=18, academic_session="2023/2024"),
        ]
        # (4*20 + 3*18) / 38 = (80 + 54) / 38 = 134/38 = 3.526...
        assert calculate_cgpa(records) == pytest.approx(3.53, abs=0.01)


class TestCalculateGPATrend:
    def test_insufficient_data(self):
        assert calculate_gpa_trend([SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")]) \
            == ImprovementTrend.INSUFFICIENT_DATA

    def test_improving(self):
        records = [
            SemesterRecord(semester_number=1, gpa=3.0, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="x"),
        ]
        # slope = 1.0 > 0.1
        assert calculate_gpa_trend(records) == ImprovementTrend.IMPROVING

    def test_declining(self):
        records = [
            SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=3.0, credits=20, academic_session="x"),
        ]
        # slope = -1.0 < -0.1
        assert calculate_gpa_trend(records) == ImprovementTrend.DECLINING

    def test_stable(self):
        records = [
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=3.55, credits=20, academic_session="x"),
        ]
        # slope = 0.05, not > 0.1
        assert calculate_gpa_trend(records) == ImprovementTrend.STABLE


class TestCalculateConsistencyIndex:
    def test_insufficient_data(self):
        assert calculate_consistency_index([SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")]) == 25

    def test_high_consistency(self):
        records = [SemesterRecord(semester_number=i, gpa=4.0, credits=20, academic_session="x") for i in range(1, 5)]
        assert calculate_consistency_index(records) == 25

    def test_medium_consistency(self):
        # std dev = 0.577 (between 0.3 and 0.6)
        records = [
            SemesterRecord(semester_number=1, gpa=3.0, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="x"),
            SemesterRecord(semester_number=3, gpa=3.0, credits=20, academic_session="x"),
        ]
        assert calculate_consistency_index(records) == 15


class TestCalculateAcademicHealthScore:
    def test_full_score_components(self):
        score = calculate_academic_health_score(
            current_cgpa=5.0,
            trend=ImprovementTrend.IMPROVING,
            consistency_index=25,
            target_cgpa=4.5,
        )
        assert score == 100

    def test_no_goal(self):
        score = calculate_academic_health_score(
            current_cgpa=4.0, trend=ImprovementTrend.STABLE,
            consistency_index=15, target_cgpa=None
        )
        # CGPA: 4/5*30=24, Trend: 15, Consistency: 15, No goal: 10 = 64
        assert score == 64

    def test_insufficient_data_trend_neutral(self):
        score = calculate_academic_health_score(
            current_cgpa=3.0, trend=ImprovementTrend.INSUFFICIENT_DATA,
            consistency_index=25, target_cgpa=None
        )
        assert score == int(3/5*30) + 15 + 25 + 10  # 18+15+25+10=68


class TestRunCalculator:
    def test_full_pipeline(self):
        student = StudentInput(
            student_name="Test",
            university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=200,  # 4 semesters -> level 200 for 5yr
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=3, gpa=4.2, credits=18, academic_session="2022/2023"),
                SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="2022/2023"),
            ],
            target_graduation_class="First Class",
        )
        out = run_calculator(student)
        assert isinstance(out, CalculatorOutput)
        assert out.current_cgpa == pytest.approx(3.94, abs=0.02)
        assert out.semesters_completed == 4
        assert out.semesters_remaining == 6
        assert out.gpa_trend in [ImprovementTrend.IMPROVING, ImprovementTrend.STABLE]
        assert 0 <= out.academic_health_score <= 100
        assert out.gpa_trend_slope != 0.0  # has trend