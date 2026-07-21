from typing import List, Optional
from backend.schemas import (
    StudentInput, CalculatorOutput, PlannerOutput, FeasibilityResult, SemesterTarget
)
from backend.grading_rules import (
    classify_cgpa, CLASS_MIN_CGPA, estimate_remaining_credits, estimate_credits_for_semester
)


def resolve_target_cgpa(student_input: StudentInput) -> float:
    if student_input.target_cgpa is not None:
        return student_input.target_cgpa
    class_target = student_input.target_graduation_class or "First Class"
    return CLASS_MIN_CGPA.get(class_target, 4.50)


def compute_feasibility_confidence(required_avg: float, historical_gpas: List[float]) -> float:
    if not historical_gpas:
        return 0.5
    avg_historical = sum(historical_gpas) / len(historical_gpas)
    distance = abs(required_avg - avg_historical)
    normalized = distance / 5.0
    confidence = 1.0 - normalized
    return max(0.0, min(1.0, confidence))


def compute_feasibility(
    calculator_out: CalculatorOutput,
    student_input: StudentInput,
    target_cgpa: float,
) -> FeasibilityResult:
    current_cgpa = calculator_out.current_cgpa
    current_credits = calculator_out.total_credits
    semesters_remaining = calculator_out.semesters_remaining

    if semesters_remaining == 0:
        final_cgpa = current_cgpa or 0.0
        return FeasibilityResult(
            goal_achievable=(final_cgpa >= target_cgpa),
            max_achievable_cgpa=final_cgpa,
            required_average_gpa=None,
            realistic_classification=classify_cgpa(final_cgpa),
            confidence=1.0 if final_cgpa >= target_cgpa else 0.0,
            message="Final semester — no remaining semesters to improve."
        )

    c_remaining = estimate_remaining_credits(
        student_input.current_level,
        semesters_remaining,
    )

    max_achievable = (
        (current_cgpa * current_credits + 5.0 * c_remaining) /
        (current_credits + c_remaining)
    ) if current_cgpa is not None else 5.0

    if current_cgpa is not None:
        required_raw = (
            (target_cgpa * (current_credits + c_remaining) - current_cgpa * current_credits) /
            c_remaining
        )
        required = max(0.0, min(required_raw, 5.0))
    else:
        required = target_cgpa

    goal_achievable = required_raw <= 5.0 if current_cgpa is not None else (target_cgpa <= 5.0)
    realistic_classification = classify_cgpa(max_achievable)

    historical_gpas = [r.gpa for r in student_input.semester_records]
    confidence = compute_feasibility_confidence(required, historical_gpas)

    if goal_achievable:
        message = f"Goal achievable. Required average: {required:.2f} per semester."
    else:
        message = (
            f"Goal NOT achievable. Even with perfect 5.0 GPAs, "
            f"max CGPA = {max_achievable:.2f} ({realistic_classification})."
        )

    return FeasibilityResult(
        goal_achievable=goal_achievable,
        max_achievable_cgpa=round(max_achievable, 2),
        required_average_gpa=round(required, 2) if required is not None else None,
        realistic_classification=realistic_classification,
        confidence=round(confidence, 2),
        message=message,
    )


def compute_semester_plan(
    calculator_out: CalculatorOutput,
    student_input: StudentInput,
    target_cgpa: float,
    feasibility: FeasibilityResult,
) -> List[SemesterTarget]:
    if calculator_out.semesters_remaining == 0:
        return []

    required_avg = (
        feasibility.required_average_gpa
        if feasibility.required_average_gpa is not None
        else target_cgpa
    )
    current_cgpa = calculator_out.current_cgpa or 0.0
    current_credits = calculator_out.total_credits
    c_per_sem = estimate_credits_for_semester(student_input.current_level)

    plan = []
    running_cgpa = current_cgpa
    running_credits = current_credits

    for i in range(calculator_out.semesters_remaining):
        sem_num = calculator_out.semesters_completed + i + 1
        target_gpa = required_avg

        running_credits += c_per_sem
        running_cgpa = (
            (running_cgpa * (running_credits - c_per_sem) + target_gpa * c_per_sem) /
            running_credits
        )

        plan.append(SemesterTarget(
            semester_number=sem_num,
            target_gpa=round(target_gpa, 2),
            cumulative_cgpa_if_met=round(running_cgpa, 2),
        ))

    return plan


def run_planner(
    student_input: StudentInput,
    calculator_out: CalculatorOutput,
) -> PlannerOutput:
    target_cgpa = resolve_target_cgpa(student_input)
    feasibility = compute_feasibility(calculator_out, student_input, target_cgpa)
    best_possible = classify_cgpa(feasibility.max_achievable_cgpa)
    semester_plan = compute_semester_plan(calculator_out, student_input, target_cgpa, feasibility)

    return PlannerOutput(
        target_cgpa=target_cgpa,
        feasibility=feasibility,
        best_possible_classification=best_possible,
        semester_plan=semester_plan,
    )