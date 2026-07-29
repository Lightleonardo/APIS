from backend.schemas import AdvisorInput, PipelineResult
from backend.orchestrator import pipeline_to_advisor_input, run_pipeline


def mock_advisor(advisor_input: AdvisorInput) -> str:
    """Validates AdvisorInput contract with a deterministic mock response."""
    assert advisor_input.student_name
    assert advisor_input.predicted_final_cgpa >= 0.0
    assert advisor_input.academic_risk in ["Low", "Medium", "High"]
    assert advisor_input.goal_feasible in [True, False]
    assert isinstance(advisor_input.semester_plan, list)

    return (
        f"Hello {advisor_input.student_name}! "
        f"Your current CGPA is {advisor_input.current_cgpa or 'N/A'}. "
        f"Predicted final CGPA: {advisor_input.predicted_final_cgpa:.2f}. "
        f"Goal feasible: {advisor_input.goal_feasible}. "
        f"[MOCK ADVISOR — AI client is offline]"
    )