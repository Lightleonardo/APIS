import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from backend.schemas import AdvisorInput
from backend.mock_advisor import mock_advisor
from backend.llm_client import get_llm_client, call_llm
from backend.advisor.rate_limiter import advisor_rate_limiter


_PROMPT_TEMPLATE = None

# Simple in-memory response cache: prompt_hash -> response
_RESPONSE_CACHE: dict[str, str] = {}
_CACHE_MAX_SIZE = 128


@dataclass
class AdvisorResult:
    """Result from advisor with metadata for UI feedback."""
    response: str
    source: str  # "llm", "cache", "mock_rate_limited", "mock_error", "mock_empty"
    rate_limit_reset_at: float | None = None  # Unix timestamp when rate limit resets


def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "advisor_prompt.txt"
        _PROMPT_TEMPLATE = prompt_path.read_text(encoding="utf-8")
    return _PROMPT_TEMPLATE


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _get_cached_response(prompt: str) -> str | None:
    key = _cache_key(prompt)
    return _RESPONSE_CACHE.get(key)


def _set_cached_response(prompt: str, response: str) -> None:
    if len(_RESPONSE_CACHE) >= _CACHE_MAX_SIZE:
        # Remove oldest entry (simple FIFO)
        oldest = next(iter(_RESPONSE_CACHE))
        del _RESPONSE_CACHE[oldest]
    key = _cache_key(prompt)
    _RESPONSE_CACHE[key] = response


def _format_semester_plan(plan) -> str:
    if not plan:
        return "No remaining semesters (final semester)."
    parts = []
    for p in plan:
        parts.append(f"Sem {p.semester_number}: {p.target_gpa:.2f} \u2192 cum {p.cumulative_cgpa_if_met:.2f}")
    return "; ".join(parts)


def _format_top_features(features) -> str:
    if not features:
        return "N/A"
    return ", ".join(f"{f.feature} ({f.importance:.2f})" for f in features)


def _format_field(value, none_text="Not specified") -> str:
    if value is None:
        return none_text
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_prompt(advisor_input: AdvisorInput) -> str:
    template = _load_prompt_template()

    return template.format(
        student_name=advisor_input.student_name,
        course=advisor_input.course,
        current_cgpa=_format_field(advisor_input.current_cgpa, "Not available"),
        target_graduation_class=_format_field(advisor_input.target_graduation_class),
        target_cgpa=_format_field(advisor_input.target_cgpa),
        remaining_semesters=advisor_input.remaining_semesters,
        required_average_gpa=_format_field(advisor_input.required_average_gpa, "N/A (final semester)"),
        predicted_final_cgpa=advisor_input.predicted_final_cgpa,
        predicted_graduation_class=advisor_input.predicted_graduation_class,
        academic_risk=advisor_input.academic_risk,
        goal_feasible=advisor_input.goal_feasible,
        best_possible_classification=advisor_input.best_possible_classification,
        academic_health_score=advisor_input.academic_health_score,
        gpa_trend=advisor_input.gpa_trend.value if hasattr(advisor_input.gpa_trend, 'value') else str(advisor_input.gpa_trend),
        consistency_index=advisor_input.consistency_index,
        semester_plan_formatted=_format_semester_plan(advisor_input.semester_plan),
        top_features_final_cgpa_formatted=_format_top_features(advisor_input.top_features_final_cgpa),
        top_features_graduation_class_formatted=_format_top_features(advisor_input.top_features_graduation_class),
        top_features_academic_risk_formatted=_format_top_features(advisor_input.top_features_academic_risk),
        tone=advisor_input.tone,
    )


def run_advisor(advisor_input: AdvisorInput) -> AdvisorResult:
    try:
        prompt = build_prompt(advisor_input)

        # Check cache first
        cached = _get_cached_response(prompt)
        if cached is not None:
            print("[ADVISOR] Cache hit")
            return AdvisorResult(response=cached, source="cache")

        # Check rate limit
        if not advisor_rate_limiter.allow_request():
            # Calculate when rate limit resets (oldest request + window)
            reset_at = None
            if advisor_rate_limiter._timestamps:
                reset_at = advisor_rate_limiter._timestamps[0] + advisor_rate_limiter.window_seconds
            print("[ADVISOR] Rate limited → mock")
            return AdvisorResult(
                response=mock_advisor(advisor_input),
                source="mock_rate_limited",
                rate_limit_reset_at=reset_at
            )

        client = get_llm_client()
        response = call_llm(client, prompt).strip()
        print(f"[ADVISOR] LLM raw response ({len(response)} chars): {response[:100]}...")

        if not response:
            print("[ADVISOR] Empty response → mock")
            return AdvisorResult(
                response=mock_advisor(advisor_input),
                source="mock_empty"
            )

        # Cache successful response
        _set_cached_response(prompt, response)
        print("[ADVISOR] LLM response OK")
        return AdvisorResult(response=response, source="llm")

    except Exception as e:
        print(f"[ADVISOR] Exception: {type(e).__name__}: {e} → mock")
        return AdvisorResult(
            response=mock_advisor(advisor_input),
            source="mock_error"
        )