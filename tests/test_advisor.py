import pytest
from backend.advisor import (
    build_prompt, extract_numbers, numeric_echo_check, run_advisor
)
from backend.schemas import AdvisorInput, SemesterTarget, FeatureImportance, ImprovementTrend


def make_advisor_input(**overrides):
    defaults = {
        "student_name": "Test Student",
        "course": "Computer Science",
        "current_cgpa": 3.8,
        "target_graduation_class": "First Class",
        "target_cgpa": 4.5,
        "remaining_semesters": 4,
        "required_average_gpa": 4.7,
        "predicted_final_cgpa": 4.3,
        "predicted_graduation_class": "Second Class Upper",
        "academic_risk": "Low",
        "goal_feasible": True,
        "best_possible_classification": "First Class",
        "academic_health_score": 78,
        "gpa_trend": ImprovementTrend.IMPROVING,
        "consistency_index": 20,
        "semester_plan": [
            SemesterTarget(semester_number=5, target_gpa=4.7, cumulative_cgpa_if_met=4.0),
            SemesterTarget(semester_number=6, target_gpa=4.7, cumulative_cgpa_if_met=4.1),
        ],
        "top_features_final_cgpa": [
            FeatureImportance(feature="cumulative_cgpa", importance=0.42),
            FeatureImportance(feature="gpa_trend_slope", importance=0.18),
        ],
        "top_features_graduation_class": [
            FeatureImportance(feature="cumulative_cgpa", importance=0.38),
        ],
        "top_features_academic_risk": [
            FeatureImportance(feature="cumulative_cgpa", importance=0.45),
        ],
        "tone": "encouraging",
    }
    defaults.update(overrides)
    return AdvisorInput(**defaults)


class TestBuildPrompt:
    def test_renders_all_fields(self):
        advisor_in = make_advisor_input()
        prompt = build_prompt(advisor_in)

        assert "Test Student" in prompt
        assert "Computer Science" in prompt
        assert "3.8" in prompt
        assert "First Class" in prompt
        assert "4.7" in prompt
        assert "4.3" in prompt
        assert "Second Class Upper" in prompt
        assert "Low" in prompt
        assert "True" in prompt
        assert "78" in prompt
        assert "Improving" in prompt
        assert "20" in prompt
        assert "Sem 5: 4.70" in prompt
        assert "cumulative_cgpa (0.42)" in prompt
        assert "encouraging" in prompt

    def test_handles_none_values(self):
        advisor_in = make_advisor_input(
            current_cgpa=None,
            target_graduation_class=None,
            target_cgpa=None,
            required_average_gpa=None,
        )
        prompt = build_prompt(advisor_in)

        assert "Not available" in prompt or "N/A" in prompt or "None" in prompt


class TestExtractNumbers:
    def test_extracts_floats_and_ints(self):
        text = "Your CGPA is 3.8 and you need 4.7 for 4 semesters. Score: 78"
        nums = extract_numbers(text)
        assert 3.8 in nums
        assert 4.7 in nums
        assert 4.0 in nums
        assert 78.0 in nums

    def test_no_numbers_returns_empty(self):
        assert extract_numbers("No numbers here") == []


class TestNumericEchoCheck:
    def test_passes_for_correct_numbers(self):
        advisor_in = make_advisor_input()
        response = "Your CGPA is 3.8. You need 4.7 average. Predicted: 4.3. Health: 78."
        assert numeric_echo_check(response, advisor_in) is True

    def test_fails_on_hallucinated_number(self):
        advisor_in = make_advisor_input(current_cgpa=3.8)
        response = "Your CGPA is 4.99. Great job!"  # 4.99 not in expected
        assert numeric_echo_check(response, advisor_in) is False

    def test_tolerance_allows_small_diff(self):
        advisor_in = make_advisor_input(predicted_final_cgpa=4.30)
        response = "Predicted final CGPA: 4.31"  # Within 0.02
        assert numeric_echo_check(response, advisor_in) is True

    def test_fails_outside_tolerance(self):
        advisor_in = make_advisor_input(predicted_final_cgpa=4.30)
        response = "Predicted final CGPA: 4.35"  # Outside 0.02
        assert numeric_echo_check(response, advisor_in) is False


class TestRunAdvisorWithMock:
    def test_returns_mock_on_exception(self, monkeypatch):
        def failing_client():
            raise ConnectionError("API down")
        monkeypatch.setattr("backend.advisor.advisor.get_llm_client", failing_client)

        advisor_in = make_advisor_input()
        result = run_advisor(advisor_in)

        assert "[MOCK ADVISOR" in result
        assert "Test Student" in result

    def test_returns_mock_on_echo_check_fail(self, monkeypatch):
        class BadClient:
            def generate(self, prompt):
                return "Your CGPA is 99.9 and you will get 100%"
        monkeypatch.setattr("backend.advisor.advisor.get_llm_client", lambda: BadClient())

        advisor_in = make_advisor_input()
        result = run_advisor(advisor_in)

        assert "[MOCK ADVISOR" in result