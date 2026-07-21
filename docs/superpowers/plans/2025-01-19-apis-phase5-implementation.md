# APIS Phase 5 Implementation Plan

> **For agentic workers:** Use subagent-driven development (recommended) or executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement AI Academic Advisor (Phase 5) — LLM integration with Gemini 1.5 Flash, fact-grounding guardrails, and fallback to mock advisor.

**Prerequisites:** Phases 1–4 complete. `AdvisorInput` contract frozen in `backend/schemas.py`. `pipeline_to_advisor_input()` exists in `orchestrator.py`. All 117 tests passing.

**Tech Stack Additions:** `google-generativeai`, `python-dotenv`

---

## File Structure Map (Phase 5 Additions)

| File | Responsibility |
|------|----------------|
| `backend/llm_client.py` | LLMClient abstraction + GeminiClient implementation |
| `backend/advisor.py` | `run_advisor()`, prompt building, numeric echo-check, fallback |
| `backend/config.py` | +LLM settings (API key, model, generation config) |
| `backend/orchestrator.py` | +`run_full_pipeline_with_advice()` |
| `prompts/advisor_prompt.txt` | Single prompt template with tone variable |
| `.env.example` | Template for `GEMINI_API_KEY` |
| `tests/test_advisor.py` | Unit tests for advisor logic |
| `tests/test_integration.py` | +Integration test for full pipeline + advisor |
| `requirements.txt` | +`google-generativeai`, `python-dotenv` |

---

## Phase 5 Implementation Tasks

---

### Task 1: Update Configuration & Dependencies

**Files:**
- `backend/config.py` (modify)
- `requirements.txt` (modify)
- `.env.example` (create)

**Step 1: Write failing test for config**

```python
# tests/test_config.py (ADD to existing)
def test_llm_settings():
    from backend.config import settings
    assert hasattr(settings, 'GEMINI_API_KEY')
    assert hasattr(settings, 'LLM_MODEL')
    assert settings.LLM_MODEL == "gemini-1.5-flash"
    assert settings.LLM_TEMPERATURE == 0.3
    assert settings.LLM_MAX_TOKENS == 200
```

Run: `pytest tests/test_config.py::test_llm_settings -v` → Expected: FAIL

**Step 2: Implement config additions**

```python
# backend/config.py (add to Settings class)
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 200
    LLM_TOP_P: float = 0.9
```

```text
# requirements.txt (add)
google-generativeai>=0.3.0
python-dotenv>=1.0
```

```text
# .env.example
GEMINI_API_KEY=your_api_key_here
```

**Step 3: Run test** → Expected: PASS

**Step 4: Install deps** `pip install -r requirements.txt`

**Step 5: Commit**
```bash
git add backend/config.py requirements.txt .env.example tests/test_config.py
git commit -m "feat: add LLM configuration settings and dependencies"
```

---

### Task 2: LLM Client Abstraction

**Files:**
- `backend/llm_client.py` (create)
- `tests/test_llm_client.py` (create)

**Step 1: Write failing tests**

```python
# tests/test_llm_client.py
import pytest
from unittest.mock import Mock, patch
from backend.llm_client import LLMClient, GeminiClient, get_llm_client, call_llm

class TestLLMClientAbstraction:
    def test_abstract_base_class(self):
        assert issubclass(GeminiClient, LLMClient)
    
    def test_get_llm_client_returns_gemini(self):
        client = get_llm_client()
        assert isinstance(client, GeminiClient)

class TestGeminiClient:
    @patch('backend.llm_client.genai')
    def test_generate_returns_text(self, mock_genai):
        mock_model = Mock()
        mock_model.generate_content.return_value.text = "Test response"
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        result = client.generate("Test prompt")
        
        assert result == "Test response"
        mock_model.generate_content.assert_called_once()
    
    @patch('backend.llm_client.genai')
    def test_generation_config_applied(self, mock_genai):
        mock_model = Mock()
        mock_model.generate_content.return_value.text = "Response"
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        client.generate("Prompt")
        
        # Verify generation config passed
        call_kwargs = mock_model.generate_content.call_args.kwargs
        assert 'generation_config' in call_kwargs
        config = call_kwargs['generation_config']
        assert config['temperature'] == 0.3
        assert config['max_output_tokens'] == 200
    
    @patch('backend.llm_client.genai')
    def test_safety_settings_applied(self, mock_genai):
        mock_model = Mock()
        mock_model.generate_content.return_value.text = "Response"
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        client.generate("Prompt")
        
        call_kwargs = mock_model.generate_content.call_args.kwargs
        assert 'safety_settings' in call_kwargs
        # Should have HARM_BLOCK_MEDIUM_AND_ABOVE for all categories

class TestCallLLM:
    def test_call_llm_delegates_to_client(self):
        mock_client = Mock(spec=LLMClient)
        mock_client.generate.return_value = "Delegated response"
        
        result = call_llm(mock_client, "Test prompt")
        
        assert result == "Delegated response"
        mock_client.generate.assert_called_once_with("Test prompt")
```

Run: `pytest tests/test_llm_client.py -v` → Expected: FAIL

**Step 2: Implement**

```python
# backend/llm_client.py
from abc import ABC, abstractmethod
from typing import Any
import google.generativeai as genai
from backend.config import settings

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class GeminiClient(LLMClient):
    def __init__(self, api_key: str | None = None):
        api_key = api_key or settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name=settings.LLM_MODEL,
            generation_config={
                "temperature": settings.LLM_TEMPERATURE,
                "max_output_tokens": settings.LLM_MAX_TOKENS,
                "top_p": settings.LLM_TOP_P,
            },
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
            }
        )
    
    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text or ""

def get_llm_client() -> LLMClient:
    return GeminiClient()

def call_llm(client: LLMClient, prompt: str) -> str:
    return client.generate(prompt)
```

**Step 3: Run tests** → Expected: PASS

**Step 4: Commit**
```bash
git add backend/llm_client.py tests/test_llm_client.py
git commit -m "feat: add LLM client abstraction with Gemini implementation"
```

---

### Task 3: Prompt Template

**Files:**
- `prompts/advisor_prompt.txt` (create)

**Step 1: Create prompt file**

```text
# prompts/advisor_prompt.txt
You are an AI Academic Advisor for university students. Your role is to provide clear, 
personalized academic guidance based on the student's data below.

STUDENT PROFILE:
- Name: {student_name}
- Course: {course}
- Current CGPA: {current_cgpa}
- Target Graduation Class: {target_graduation_class}
- Target CGPA: {target_cgpa}
- Remaining Semesters: {remaining_semesters}
- Required Average GPA per Semester: {required_average_gpa}
- Predicted Final CGPA: {predicted_final_cgpa}
- Predicted Graduation Class: {predicted_graduation_class}
- Academic Risk Level: {academic_risk}
- Goal Feasible: {goal_feasible}
- Best Possible Classification: {best_possible_classification}
- Academic Health Score: {academic_health_score}/100
- GPA Trend: {gpa_trend}
- Consistency Index: {consistency_index}/25

SEMESTER-BY-SEMESTER PLAN:
{semester_plan_formatted}

TOP PREDICTIVE FACTORS:
- Final CGPA: {top_features_final_cgpa_formatted}
- Graduation Class: {top_features_graduation_class_formatted}
- Academic Risk: {top_features_academic_risk_formatted}

TONE: {tone}

INSTRUCTIONS:
- Write in {tone} tone.
- Keep response to approximately 150 words.
- NEVER invent numbers. Only reference values explicitly provided above.
- If goal is not feasible, be honest but constructive.
- Mention the semester plan briefly (e.g., "You need ~4.7 GPA each of the next 4 semesters").
- Do not use markdown formatting. Plain text only.
```

**Step 2: Commit**
```bash
git add prompts/advisor_prompt.txt
git commit -m "feat: add advisor prompt template"
```

---

### Task 4: Advisor Core Logic

**Files:**
- `backend/advisor.py` (create)
- `tests/test_advisor.py` (create)

**Step 1: Write failing tests**

```python
# tests/test_advisor.py
import pytest
from backend.advisor import (
    build_prompt, extract_numbers, numeric_echo_check, run_advisor
)
from backend.schemas import AdvisorInput, SemesterTarget, FeatureImportance

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
        "gpa_trend": "Improving",
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
        # Force get_llm_client to raise
        def failing_client():
            raise ConnectionError("API down")
        monkeypatch.setattr("backend.advisor.get_llm_client", failing_client)
        
        advisor_in = make_advisor_input()
        result = run_advisor(advisor_in)
        
        assert "[MOCK ADVISOR" in result
        assert "Test Student" in result
    
    def test_returns_mock_on_echo_check_fail(self, monkeypatch):
        # Mock client that returns hallucinated response
        class BadClient:
            def generate(self, prompt):
                return "Your CGPA is 99.9 and you will get 100%"
        monkeypatch.setattr("backend.advisor.get_llm_client", lambda: BadClient())
        
        advisor_in = make_advisor_input()
        result = run_advisor(advisor_in)
        
        assert "[MOCK ADVISOR" in result
```

Run: `pytest tests/test_advisor.py -v` → Expected: FAIL

**Step 2: Implement**

```python
# backend/advisor.py
import re
from pathlib import Path
from backend.schemas import AdvisorInput
from backend.mock_advisor import mock_advisor
from backend.llm_client import get_llm_client, call_llm
from backend.config import settings

_PROMPT_TEMPLATE = None

def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        prompt_path = Path(__file__).parent.parent / "prompts" / "advisor_prompt.txt"
        _PROMPT_TEMPLATE = prompt_path.read_text(encoding="utf-8")
    return _PROMPT_TEMPLATE

def _format_semester_plan(plan) -> str:
    if not plan:
        return "No remaining semesters (final semester)."
    parts = []
    for p in plan:
        parts.append(f"Sem {p.semester_number}: {p.target_gpa:.2f} → cum {p.cumulative_cgpa_if_met:.2f}")
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

def extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r'\b\d+\.?\d*\b', text)]

def numeric_echo_check(response: str, advisor_input: AdvisorInput) -> bool:
    found = extract_numbers(response)
    expected = [
        advisor_input.current_cgpa,
        advisor_input.target_cgpa,
        advisor_input.required_average_gpa,
        advisor_input.predicted_final_cgpa,
        advisor_input.predicted_next_gpa,
        advisor_input.academic_health_score,
        advisor_input.consistency_index,
        advisor_input.remaining_semesters,
    ]
    expected = [x for x in expected if x is not None]
    
    for f in found:
        if not any(abs(f - e) <= 0.02 for e in expected):
            return False
    return True

def run_advisor(advisor_input: AdvisorInput) -> str:
    try:
        client = get_llm_client()
        prompt = build_prompt(advisor_input)
        response = call_llm(client, prompt).strip()
        
        if not response:
            raise ValueError("Empty response from LLM")
        if len(response) > 500:
            raise ValueError("Response too long")
        
        if not numeric_echo_check(response, advisor_input):
            raise ValueError("Numeric echo-check failed: hallucinated numbers detected")
        
        return response
        
    except Exception:
        return mock_advisor(advisor_input)
```

**Step 3: Run tests** → Expected: PASS

**Step 4: Commit**
```bash
git add backend/advisor.py tests/test_advisor.py
git commit -m "feat: add advisor core logic with prompt, echo-check, and fallback"
```

---

### Task 5: Orchestrator Integration

**Files:**
- `backend/orchestrator.py` (modify)
- `tests/test_integration.py` (modify)

**Step 1: Write failing test**

```python
# tests/test_integration.py (ADD to existing)
from unittest.mock import patch, Mock
from backend.orchestrator import run_full_pipeline_with_advice
from backend.schemas import StudentInput, SemesterRecord

def test_full_pipeline_with_advice_mocked():
    student = StudentInput(
        student_name="Integration Test",
        university="U", faculty="F", department="D", course="C",
        programme_duration_years=4, current_level=200,
        semester_records=[
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2022/2023"),
            SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="2022/2023"),
        ],
        target_graduation_class="First Class",
    )
    
    with patch('backend.orchestrator.run_advisor') as mock_advisor:
        mock_advisor.return_value = "Mocked advice response"
        
        pipeline_result, advice = run_full_pipeline_with_advice(student)
        
        assert pipeline_result is not None
        assert pipeline_result.student_name == "Integration Test"
        assert advice == "Mocked advice response"
        mock_advisor.assert_called_once()
```

Run: `pytest tests/test_integration.py::test_full_pipeline_with_advice_mocked -v` → Expected: FAIL

**Step 2: Implement**

```python
# backend/orchestrator.py (ADD to existing file)
from backend.advisor import run_advisor

def run_full_pipeline_with_advice(student_input: StudentInput) -> tuple[PipelineResult, str]:
    pipeline_result = run_pipeline(student_input)
    advisor_input = pipeline_to_advisor_input(pipeline_result)
    advice = run_advisor(advisor_input)
    return pipeline_result, advice
```

**Step 3: Run test** → Expected: PASS

**Step 4: Commit**
```bash
git add backend/orchestrator.py tests/test_integration.py
git commit -m "feat: add full pipeline with advisor integration"
```

---

### Task 6: End-to-End Integration Test (Real API Optional)

**Files:**
- `tests/test_integration.py` (add)

**Step 1: Write test (skipped by default, runs with `--run-e2e`)**

```python
# tests/test_integration.py (ADD)
import pytest
import os
from backend.orchestrator import run_full_pipeline_with_advice
from backend.schemas import StudentInput, SemesterRecord

@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; set to run live integration test"
)
def test_full_pipeline_with_real_llm():
    student = StudentInput(
        student_name="Live Test Student",
        university="Test Uni", faculty="Science", department="CS", course="Computer Science",
        programme_duration_years=4, current_level=200,
        semester_records=[
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2022/2023"),
            SemesterRecord(semester_number=2, gpa=3.8, credits=20, academic_session="2022/2023"),
        ],
        target_graduation_class="First Class",
    )
    
    pipeline_result, advice = run_full_pipeline_with_advice(student)
    
    assert pipeline_result is not None
    assert isinstance(advice, str)
    assert len(advice) > 0
    assert len(advice) <= 500
    assert "Live Test Student" in advice or "[MOCK ADVISOR" in advice
```

Run: `pytest tests/test_integration.py::test_full_pipeline_with_real_llm -v --run-e2e` (manual)

---

### Task 7: Verify Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

**Expected:** All 117+ existing tests + new Phase 5 tests pass.

**Step 2: Check for regressions**

```bash
pytest tests/test_calculator.py tests/test_planner.py tests/test_predictor.py tests/test_orchestrator.py -v
```

---

### Task 8: Documentation Updates

**Files:**
- `README.md` (create or update)

**Add to README:**
```markdown
## Phase 5: AI Academic Advisor

### Setup
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Usage
```python
from backend.orchestrator import run_full_pipeline_with_advice
from backend.schemas import StudentInput, SemesterRecord

student = StudentInput(...)
pipeline_result, advice = run_full_pipeline_with_advice(student)
print(advice)
```

### Configuration
- `GEMINI_API_KEY`: Get from Google AI Studio
- Model: `gemini-1.5-flash` (free tier)
- Fallback: Automatic to mock advisor on any failure
```

---

## Phase 5 Commit Summary

| Commit | Files |
|--------|-------|
| `feat: add LLM configuration settings and dependencies` | `config.py`, `requirements.txt`, `.env.example`, `tests/test_config.py` |
| `feat: add LLM client abstraction with Gemini implementation` | `llm_client.py`, `tests/test_llm_client.py` |
| `feat: add advisor prompt template` | `prompts/advisor_prompt.txt` |
| `feat: add advisor core logic with prompt, echo-check, and fallback` | `advisor.py`, `tests/test_advisor.py` |
| `feat: add full pipeline with advisor integration` | `orchestrator.py`, `tests/test_integration.py` |
| `docs: update README for Phase 5` | `README.md` |

---

## Execution Options

**Option 1: Subagent-Driven (Recommended)**
- Dispatch one subagent per task above
- Review each task completion before next

**Option 2: Inline with Checkpoints**
- Execute tasks sequentially in this session
- Run full test suite after Tasks 1-2, 3-4, 5-7

**Option 3: Batch Execute**
- Implement all tasks, then run full test suite once

**Which approach?**