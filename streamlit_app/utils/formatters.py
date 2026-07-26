from backend.schemas import ImprovementTrend


def fmt_gpa(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "—"


def fmt_cgpa(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "—"


def fmt_pct(val: float) -> str:
    return f"{val:.1f}%"


def tone_badge(tone: str) -> str:
    return f'<span class="tone-badge tone-{tone}">{tone.capitalize()}</span>'


def trend_label(trend: ImprovementTrend) -> str:
    return trend.value if hasattr(trend, 'value') else str(trend)