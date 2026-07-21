import statistics
from typing import List, Optional
from backend.schemas import (
    StudentInput, SemesterRecord, CalculatorOutput, ImprovementTrend
)
from backend.grading_rules import classify_cgpa
from backend.trajectory_features import compute_trajectory_features


def calculate_cgpa(records: List[SemesterRecord]) -> Optional[float]:
    if not records:
        return None
    total_points = sum(r.gpa * r.credits for r in records)
    total_credits = sum(r.credits for r in records)
    return round(total_points / total_credits, 2)


def calculate_gpa_trend(records: List[SemesterRecord]) -> ImprovementTrend:
    if len(records) < 2:
        return ImprovementTrend.INSUFFICIENT_DATA
    gpas = [r.gpa for r in records]
    n = len(gpas)
    x = list(range(1, n + 1))
    x_mean = sum(x) / n
    y_mean = sum(gpas) / n
    numerator = sum((x[i] - x_mean) * (gpas[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0.0
    if slope > 0.1:
        return ImprovementTrend.IMPROVING
    elif slope < -0.1:
        return ImprovementTrend.DECLINING
    else:
        return ImprovementTrend.STABLE


def calculate_consistency_index(records: List[SemesterRecord]) -> int:
    if len(records) < 2:
        return 25
    gpas = [r.gpa for r in records]
    std_dev = statistics.stdev(gpas)
    if std_dev <= 0.3:
        return 25
    elif std_dev <= 0.6:
        return 15
    else:
        return 5


def calculate_academic_health_score(
    current_cgpa: Optional[float],
    trend: ImprovementTrend,
    consistency_index: int,
    target_cgpa: Optional[float],
) -> int:
    score = 0
    if current_cgpa is not None:
        score += int((current_cgpa / 5.0) * 30)
    trend_scores = {
        ImprovementTrend.IMPROVING: 25,
        ImprovementTrend.STABLE: 15,
        ImprovementTrend.DECLINING: 5,
        ImprovementTrend.INSUFFICIENT_DATA: 15,
    }
    score += trend_scores[trend]
    score += consistency_index
    if target_cgpa is not None and current_cgpa is not None:
        progress = min(1.0, current_cgpa / target_cgpa)
        score += int(progress * 20)
    else:
        score += 10
    return min(100, max(0, score))


def run_calculator(student_input: StudentInput) -> CalculatorOutput:
    records = student_input.semester_records
    current_cgpa = calculate_cgpa(records)
    total_credits = sum(r.credits for r in records)
    semesters_completed = len(records)
    total_semesters = student_input.programme_duration_years * 2
    semesters_remaining = total_semesters - semesters_completed

    trend = calculate_gpa_trend(records)
    consistency = calculate_consistency_index(records)
    health_score = calculate_academic_health_score(
        current_cgpa=current_cgpa,
        trend=trend,
        consistency_index=consistency,
        target_cgpa=student_input.target_cgpa,
    )
    classification = classify_cgpa(current_cgpa) if current_cgpa is not None else None

    features = compute_trajectory_features(
        gpas=[r.gpa for r in records],
        total_credits=total_credits,
    )

    return CalculatorOutput(
        current_cgpa=current_cgpa,
        total_credits=total_credits,
        semesters_completed=semesters_completed,
        semesters_remaining=semesters_remaining,
        current_classification=classification,
        gpa_trend=trend,
        consistency_index=consistency,
        academic_health_score=health_score,
        gpa_trend_slope=features["gpa_trend_slope"],
        gpa_volatility=features["gpa_volatility"],
        recent_gpa_avg_3=features["recent_gpa_avg_3"],
        credits_velocity=features["credits_velocity"],
    )