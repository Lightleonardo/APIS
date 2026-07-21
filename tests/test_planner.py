import pytest
from backend.planner import (
    resolve_target_cgpa, compute_feasibility, compute_semester_plan, run_planner
)
from backend.schemas import (
    StudentInput, SemesterRecord, CalculatorOutput, FeasibilityResult,
    SemesterTarget, ImprovementTrend
)
from backend.grading_rules import CLASS_MIN_CGPA


class TestResolveTargetCGPA:
    def test_explicit_cgpa_target(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=100,
            semester_records=[SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")],
            target_cgpa=4.75,
        )
        assert resolve_target_cgpa(student) == 4.75

    def test_class_target_first_class(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=100,
            semester_records=[SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")],
            target_graduation_class="First Class",
        )
        assert resolve_target_cgpa(student) == CLASS_MIN_CGPA["First Class"]

    def test_default_first_class(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=100,
            semester_records=[SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")],
        )
        assert resolve_target_cgpa(student) == CLASS_MIN_CGPA["First Class"]


def make_calc(cgpa, credits, completed, remaining, classification, trend=ImprovementTrend.STABLE):
    return CalculatorOutput(
        current_cgpa=cgpa, total_credits=credits, semesters_completed=completed,
        semesters_remaining=remaining, current_classification=classification,
        gpa_trend=trend, consistency_index=0, academic_health_score=0,
        gpa_trend_slope=0, gpa_volatility=0, recent_gpa_avg_3=0, credits_velocity=0,
    )


class TestComputeFeasibility:
    def test_achievable_goal(self):
        calc = make_calc(4.0, 40, 2, 8, "Second Class Upper")
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=100,
            semester_records=[SemesterRecord(semester_number=i, gpa=4.0, credits=20, academic_session="x") for i in range(1,3)],
            target_graduation_class="First Class",
        )
        feasibility = compute_feasibility(calc, student, 4.50)
        assert feasibility.goal_achievable is True
        assert feasibility.required_average_gpa is not None
        assert feasibility.required_average_gpa <= 5.0
        assert feasibility.confidence > 0

    def test_unachievable_goal(self):
        calc = make_calc(2.0, 40, 2, 2, "Pass")
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=100,
            semester_records=[SemesterRecord(semester_number=i, gpa=2.0, credits=20, academic_session="x") for i in range(1,3)],
            target_graduation_class="First Class",
        )
        feasibility = compute_feasibility(calc, student, 4.50)
        assert feasibility.goal_achievable is False
        assert feasibility.required_average_gpa is not None
        assert feasibility.required_average_gpa > 5.0 or feasibility.max_achievable_cgpa < 4.50
        assert "NOT achievable" in feasibility.message

    def test_final_semester_none_required(self):
        calc = make_calc(4.2, 120, 10, 0, "First Class")
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=500,
            semester_records=[SemesterRecord(semester_number=i, gpa=4.0, credits=15, academic_session="x") for i in range(1,11)],
            target_graduation_class="First Class",
        )
        feasibility = compute_feasibility(calc, student, 4.50)
        assert feasibility.required_average_gpa is None
        assert feasibility.goal_achievable == (calc.current_cgpa >= 4.50)


class TestComputeSemesterPlan:
    def test_plan_returns_correct_length(self):
        calc = make_calc(3.5, 40, 2, 4, "Second Class Upper")
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=100,
            semester_records=[SemesterRecord(semester_number=i, gpa=3.5, credits=20, academic_session="x") for i in range(1,3)],
            target_cgpa=4.0,
        )
        feasibility = FeasibilityResult(
            goal_achievable=True, max_achievable_cgpa=4.2,
            required_average_gpa=4.25, realistic_classification="First Class",
            confidence=0.8, message="ok",
        )
        plan = compute_semester_plan(calc, student, 4.0, feasibility)
        assert len(plan) == 4
        for p in plan:
            assert p.target_gpa == 4.25
            assert p.cumulative_cgpa_if_met >= 3.5

    def test_plan_uses_target_cgpa_when_required_is_none(self):
        calc = make_calc(4.5, 120, 10, 0, "First Class")
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=500,
            semester_records=[SemesterRecord(semester_number=i, gpa=4.0, credits=15, academic_session="x") for i in range(1,11)],
            target_cgpa=4.5,
        )
        feasibility = FeasibilityResult(
            goal_achievable=True, max_achievable_cgpa=4.5,
            required_average_gpa=None, realistic_classification="First Class",
            confidence=1.0, message="final semester",
        )
        plan = compute_semester_plan(calc, student, 4.5, feasibility)
        assert plan == []


class TestRunPlanner:
    def test_integration(self):
        student = StudentInput(
            student_name="Test",
            university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=200,  # 4 semesters -> level 200 for 5yr
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=2, gpa=3.8, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=3, gpa=4.0, credits=18, academic_session="2022/2023"),
                SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="2022/2023"),
            ],
            target_graduation_class="First Class",
        )
        from backend.calculator import run_calculator
        calc = run_calculator(student)
        planner_out = run_planner(student, calc)
        assert planner_out.target_cgpa == 4.50
        assert isinstance(planner_out.feasibility, FeasibilityResult)
        assert len(planner_out.semester_plan) == 6