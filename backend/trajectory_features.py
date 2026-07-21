import statistics
from typing import List, Dict


def compute_trajectory_features(gpas: List[float], total_credits: int) -> Dict[str, float]:
    """
    Single source of truth for the 4 engineered features.
    gpas must be ONLY semesters completed so far (chronological, no future data).
    """
    n = len(gpas)

    gpa_trend_slope = 0.0
    if n >= 2:
        x_mean = (n + 1) / 2  # mean of 1..n
        y_mean = sum(gpas) / n
        num = sum((i + 1 - x_mean) * (gpas[i] - y_mean) for i in range(n))
        den = sum((i + 1 - x_mean) ** 2 for i in range(n))
        gpa_trend_slope = num / den if den != 0 else 0.0

    gpa_volatility = statistics.stdev(gpas) if n >= 2 else 0.0
    recent_gpa_avg_3 = sum(gpas[-3:]) / min(3, n) if n > 0 else 0.0
    credits_velocity = total_credits / n if n > 0 else 0.0

    return {
        "gpa_trend_slope": gpa_trend_slope,
        "gpa_volatility": gpa_volatility,
        "recent_gpa_avg_3": recent_gpa_avg_3,
        "credits_velocity": credits_velocity,
    }