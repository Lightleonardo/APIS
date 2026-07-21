from typing import List, Tuple
from backend.schemas import StudentInput, PipelineResult, SemesterHistoryPoint, AdvisorInput
from backend.calculator import run_calculator
from backend.planner import run_planner
from backend.predictor import run_predictor
from backend.grading_rules import estimate_credits_for_semester


def build_semester_history(
    student_input: StudentInput,
    calculator_out,
) -> List[SemesterHistoryPoint]:
    history = []
    running_points = 0.0
    running_credits = 0
    for r in student_input.semester_records:
        running_points += r.gpa * r.credits
        running_credits += r.credits
        cum_cgpa = running_points / running_credits if running_credits > 0 else 0.0
        history.append(SemesterHistoryPoint(
            semester_number=r.semester_number,
            gpa=r.gpa,
            cumulative_cgpa=round(cum_cgpa, 2),
            credits=r.credits,
            academic_session=r.academic_session,
        ))
    return history


def pipeline_to_advisor_input(pipeline: PipelineResult) -> AdvisorInput:
    return AdvisorInput(
        student_name=pipeline.student_name,
        course=pipeline.course,
        current_cgpa=pipeline.current_cgpa,
        target_graduation_class=pipeline.target_graduation_class,
        target_cgpa=pipeline.target_cgpa,
        remaining_semesters=pipeline.semesters_remaining,
        required_average_gpa=pipeline.feasibility.required_average_gpa,
        predicted_final_cgpa=pipeline.predicted_final_cgpa,
        predicted_graduation_class=pipeline.predicted_graduation_class,
        academic_risk=pipeline.predicted_academic_risk,
        goal_feasible=pipeline.feasibility.goal_achievable,
        best_possible_classification=pipeline.best_possible_classification,
        academic_health_score=pipeline.academic_health_score,
        gpa_trend=pipeline.gpa_trend,
        consistency_index=pipeline.consistency_index,
        semester_plan=pipeline.semester_plan,
        top_features_final_cgpa=pipeline.top_features_final_cgpa,
        top_features_graduation_class=pipeline.top_features_graduation_class,
        top_features_academic_risk=pipeline.top_features_academic_risk,
    )


def run_pipeline(student_input: StudentInput) -> PipelineResult:
    calculator_out = run_calculator(student_input)
    planner_out = run_planner(student_input, calculator_out)
    predictor_out = run_predictor(student_input, calculator_out)
    history = build_semester_history(student_input, calculator_out)

    return PipelineResult(
        student_name=student_input.student_name,
        university=student_input.university,
        faculty=student_input.faculty,
        department=student_input.department,
        course=student_input.course,
        programme_duration_years=student_input.programme_duration_years,
        current_level=student_input.current_level,
        target_graduation_class=student_input.target_graduation_class,
        target_cgpa=student_input.target_cgpa,
        current_cgpa=calculator_out.current_cgpa,
        total_credits=calculator_out.total_credits,
        semesters_completed=calculator_out.semesters_completed,
        semesters_remaining=calculator_out.semesters_remaining,
        current_classification=calculator_out.current_classification,
        gpa_trend=calculator_out.gpa_trend,
        consistency_index=calculator_out.consistency_index,
        academic_health_score=calculator_out.academic_health_score,
        target_cgpa_resolved=planner_out.target_cgpa,
        feasibility=planner_out.feasibility,
        best_possible_classification=planner_out.best_possible_classification,
        semester_plan=planner_out.semester_plan,
        predicted_next_gpa=predictor_out.predicted_next_gpa,
        predicted_final_cgpa=predictor_out.predicted_final_cgpa,
        predicted_graduation_class=predictor_out.predicted_graduation_class,
        predicted_academic_risk=predictor_out.predicted_academic_risk,
        top_features_next_gpa=predictor_out.top_features_next_gpa,
        top_features_final_cgpa=predictor_out.top_features_final_cgpa,
        top_features_graduation_class=predictor_out.top_features_graduation_class,
        top_features_academic_risk=predictor_out.top_features_academic_risk,
        semester_history=history,
    )


def run_full_pipeline_with_advice(student_input: StudentInput) -> Tuple[PipelineResult, str]:
    """
    Runs full pipeline + AI advisor.
    Returns (PipelineResult, advisor_response_string).
    """
    from backend.advisor import run_advisor
    pipeline_result = run_pipeline(student_input)
    advisor_input = pipeline_to_advisor_input(pipeline_result)
    advice = run_advisor(advisor_input)
    return pipeline_result, advice