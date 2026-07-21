import random
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from scipy.stats import truncnorm
from backend.trajectory_features import compute_trajectory_features


@dataclass
class TrajectoryProfile:
    base_ability: float      # Mean GPA this student gravitates toward (2.0–4.8)
    volatility: float        # Std dev of semester-to-semester noise (0.15–0.6)
    trend: float             # Slope per semester (-0.15 to +0.15)
    shock_probability: float # Chance of one bad semester (0.05–0.15)
    shock_magnitude: float   # GPA drop if shock occurs (0.5–1.5)


def sample_base_ability() -> float:
    """Mixture of truncated normals, all bounded to [2.0, 4.8]."""
    component = random.choices(
        population=['avg', 'high', 'low'],
        weights=[0.60, 0.30, 0.10],
        k=1
    )[0]

    if component == 'avg':
        a, b = (2.0 - 3.5) / 0.5, (4.8 - 3.5) / 0.5
        return float(truncnorm.rvs(a, b, loc=3.5, scale=0.5))
    elif component == 'high':
        a, b = (2.0 - 4.2) / 0.3, (4.8 - 4.2) / 0.3
        return float(truncnorm.rvs(a, b, loc=4.2, scale=0.3))
    else:  # 'low'
        a, b = (2.0 - 2.0) / 0.4, (4.8 - 2.0) / 0.4
        return float(truncnorm.rvs(a, b, loc=2.0, scale=0.4))


def sample_trajectory_profile() -> TrajectoryProfile:
    return TrajectoryProfile(
        base_ability=sample_base_ability(),
        volatility=float(np.random.lognormal(mean=-1.5, sigma=0.3)),
        trend=float(np.random.normal(0.0, 0.05)),
        shock_probability=float(np.random.beta(2, 20)),
        shock_magnitude=float(np.random.uniform(0.5, 1.5)),
    )


def apply_trajectory_noise(skeleton: List[Dict[str, Any]], profile: TrajectoryProfile) -> List[Dict[str, Any]]:
    result = []
    gpas_so_far = []
    credits_so_far = 0

    for i, row in enumerate(skeleton):
        sem_num = row["semester_number"]

        # Generate GPA for this semester
        raw_gpa = profile.base_ability + profile.trend * sem_num + np.random.normal(0, profile.volatility)
        if random.random() < profile.shock_probability:
            raw_gpa -= profile.shock_magnitude
        semester_gpa = max(0.0, min(5.0, raw_gpa))

        gpas_so_far.append(semester_gpa)
        credits_so_far += row["semester_credits"]

        # Compute engineered features
        features = compute_trajectory_features(gpas_so_far, credits_so_far)

        # Build complete row
        complete_row = row.copy()
        complete_row.update({
            "semester_gpa": round(semester_gpa, 2),
            "cumulative_cgpa": round(
                sum(g * row["semester_credits"] for g, row in zip(gpas_so_far, skeleton[:i+1])) / credits_so_far, 2
            ),
            "cumulative_credits": credits_so_far,
            "semesters_completed": sem_num,
            "semesters_remaining": len(skeleton) - sem_num,
            "is_final_semester": (sem_num == len(skeleton)),
            "gpa_trend_slope": features["gpa_trend_slope"],
            "gpa_volatility": features["gpa_volatility"],
            "recent_gpa_avg_3": features["recent_gpa_avg_3"],
            "credits_velocity": features["credits_velocity"],
        })
        result.append(complete_row)

    # Compute target labels from full trajectory
    final_cgpa = result[-1]["cumulative_cgpa"]
    from backend.grading_rules import classify_cgpa
    graduation_class = classify_cgpa(final_cgpa)

    for i, row in enumerate(result):
        if i < len(result) - 1:
            row["next_semester_gpa"] = result[i + 1]["semester_gpa"]
        else:
            row["next_semester_gpa"] = None
        row["final_cgpa"] = final_cgpa
        row["graduation_class"] = graduation_class

        # Academic risk label (heuristic)
        cum_cgpa = row["cumulative_cgpa"]
        latest_gpa = row["semester_gpa"]
        if cum_cgpa < 2.0 or latest_gpa < 1.5:
            row["academic_risk"] = "High"
        elif cum_cgpa < 3.0 or latest_gpa < 2.5:
            row["academic_risk"] = "Medium"
        else:
            row["academic_risk"] = "Low"

    return result