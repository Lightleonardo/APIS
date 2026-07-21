# Academic Performance Intelligence System (APIS) — Phase 5 Design Specification

**Version:** 1.0  
**Date:** 2026-07-19  
**Status:** Approved for Implementation  
**Prerequisite:** Phases 1–4 Complete (Frozen `AdvisorInput` contract in `backend/schemas.py`)

---

## 1. Phase 5 Scope

**AI Academic Advisor** — LLM-powered natural language explanation layer that consumes the frozen `AdvisorInput` contract and produces student-facing guidance.

| In Scope | Out of Scope (Phase 6+) |
|----------|------------------------|
| LLM provider integration (Gemini 1.5 Flash) | Streaming / progressive display |
| Prompt engineering with tone control | Multi-turn conversation / chat history |
| Fact-grounding guardrails (numeric echo-check) | User accounts / persistence |
| Deterministic what-if narration (calculator-only math) | Structured JSON output schema |
| Fallback to `mock_advisor` on any failure | Fine-tuning / custom model training |
| Content safety (Gemini built-in + prompt constraints) | Multi-language support beyond English |

---

## 2. Architecture

```
PipelineResult (Phase 4 output)
        │
        ▼
pipeline_to_advisor_input()  ──►  AdvisorInput (frozen Pydantic contract)
        │
        ▼
LLM Advisor Function (single entry point)
        │
        ├──► Gemini 1.5 Flash API (free tier)
        │       │
        │       ├── Prompt: single template + tone variable
        │       ├── Output: free-form text (~150 words, max_tokens cap)
        │       ├── Guardrail 1: Gemini safety settings (HARM_BLOCK_MEDIUM_AND_ABOVE)
        │       ├── Guardrail 2: Numeric echo-check (extract numbers, diff vs AdvisorInput)
        │       ├── Guardrail 3: Prompt constraint "never invent numbers"
        │       └── Guardrail 4: On ANY failure → fallback to mock_advisor()
        │
        └──► mock_advisor() (deterministic fallback, Phase 4)
```

**Key Principle:** The LLM **never computes**. All math (CGPA projections, feasibility, what-if scenarios) is computed deterministically by `calculator.py`/`planner.py` in Phase 4. The LLM only *narrates* the pre-computed numbers.

---

## 3. Design Decisions (Frozen)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Model** | Gemini 1.5 Flash (free tier) | Zero cost for beta; swappable via single function |
| **Model Swappability** | `get_llm_client()` returns client; `call_llm(client, prompt)` — swap provider by changing client init | Single function boundary |
| **Structured Output** | None — free-form text with `max_output_tokens=200` | Simplicity; avoids JSON parsing failures |
| **Prompt Structure** | Single prompt template, `tone` injected as variable | Minimal prompt engineering surface |
| **Tone Options** | `encouraging` \| `direct` \| `analytical` (matches `AdvisorInput.tone`) | Matches frozen contract |
| **Output Length** | ~150 words target, enforced by `max_output_tokens=200` | Concise, student-friendly |
| **What-If Narration** | `calculator.py`/`planner.py` compute; LLM only narrates | Deterministic math; no hallucinated numbers |
| **Streaming** | Disabled — full response returned | Simplicity; MVP scope |
| **Safety: Content Filtering** | Gemini built-in `safety_settings` (HARM_BLOCK_MEDIUM_AND_ABOVE) | Zero-config baseline |
| **Safety: Fact Grounding** | Numeric echo-check: extract all floats from LLM response, assert each ≈ corresponding `AdvisorInput` field (±0.02) | Catches hallucinated numbers |
| **Safety: Prompt Constraint** | "Never invent numbers. Only reference values provided in the context." | Defense in depth |
| **Fallback** | On ANY failure (API error, timeout, echo-check fail, malformed response) → `mock_advisor()` | Guaranteed working response |

---

## 4. LLM Client Abstraction

```python
# backend/llm_client.py (NEW FILE)
from abc import ABC, abstractmethod
from backend.schemas import AdvisorInput

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Returns raw text response."""
        pass

def get_llm_client() -> LLMClient:
    """Single swap point. Returns GeminiClient for Phase 5."""
    from backend.llm_client import GeminiClient
    return GeminiClient()

def call_llm(client: LLMClient, prompt: str) -> str:
    """Single call boundary. Swappable for other providers."""
    return client.generate(prompt)
```

**GeminiClient Implementation:**
- Uses `google-generativeai` SDK
- Model: `gemini-1.5-flash`
- Safety settings: `HARM_BLOCK_MEDIUM_AND_ABOVE` for all categories
- Generation config: `temperature=0.3`, `max_output_tokens=200`, `top_p=0.9`

---

## 5. Prompt Template

**File:** `prompts/advisor_prompt.txt`

```
You are an AI Academic Advisor for university students. Your role is to provide clear, 
personalized academic guidance based on the student's data below.

STUDENT PROFILE:
- Name: {student_name}
- Course: {course}
- Current CGPA: {current_cgpa or "Not available"}
- Target Graduation Class: {target_graduation_class or "Not specified"}
- Target CGPA: {target_cgpa or "Not specified"}
- Remaining Semesters: {remaining_semesters}
- Required Average GPA per Semester: {required_average_gpa or "N/A (final semester)"}
- Predicted Final CGPA: {predicted_final_cgpa:.2f}
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

**Formatting Helpers (in `backend/advisor.py`):**
- `semester_plan_formatted`: "Sem 5: 4.70 → cum 4.10; Sem 6: 4.70 → cum 4.20; ..."
- `top_features_*_formatted`: "cumulative_cgpa (0.42), gpa_trend_slope (0.18), ..."

---

## 6. Advisor Function (Single Entry Point)

```python
# backend/advisor.py (NEW FILE)
from backend.schemas import AdvisorInput
from backend.mock_advisor import mock_advisor
from backend.llm_client import get_llm_client, call_llm
import re

def build_prompt(advisor_input: AdvisorInput) -> str:
    """Renders the prompt template with AdvisorInput values."""
    # ... formatting logic ...
    return rendered_prompt

def extract_numbers(text: str) -> list[float]:
    """Extracts all float-like numbers from text."""
    return [float(m) for m in re.findall(r'\b\d+\.?\d*\b', text)]

def numeric_echo_check(response: str, advisor_input: AdvisorInput) -> bool:
    """
    Validates that numbers in LLM response match AdvisorInput (within tolerance).
    Returns True if check passes, False otherwise.
    """
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
            return False  # Hallucinated number detected
    return True

def run_advisor(advisor_input: AdvisorInput) -> str:
    """
    Main entry point. Returns advisor response string.
    Falls back to mock_advisor on ANY failure.
    """
    try:
        client = get_llm_client()
        prompt = build_prompt(advisor_input)
        response = call_llm(client, prompt).strip()
        
        # Guardrail: numeric echo-check
        if not numeric_echo_check(response, advisor_input):
            raise ValueError("Numeric echo-check failed: hallucinated numbers detected")
        
        # Guardrail: non-empty, reasonable length
        if not response or len(response) > 500:
            raise ValueError("Response empty or too long")
        
        return response
        
    except Exception:
        # Fallback on ANY error: API failure, timeout, echo-check, malformed response
        return mock_advisor(advisor_input)
```

---

## 7. Integration with Phase 4 Orchestrator

**Existing:** `orchestrator.py::pipeline_to_advisor_input()` already converts `PipelineResult → AdvisorInput`.

**New:** `orchestrator.py` gets a new public function:

```python
# backend/orchestrator.py (ADDITION)
from backend.advisor import run_advisor

def run_full_pipeline_with_advice(student_input: StudentInput) -> tuple[PipelineResult, str]:
    """
    Runs full pipeline + AI advisor.
    Returns (PipelineResult, advisor_response_string).
    """
    pipeline_result = run_pipeline(student_input)
    advisor_input = pipeline_to_advisor_input(pipeline_result)
    advice = run_advisor(advisor_input)
    return pipeline_result, advice
```

---

## 8. Configuration

**File:** `backend/config.py` (ADDITIONS)

```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Phase 5: LLM
    GEMINI_API_KEY: str = ""  # From .env
    LLM_MODEL: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 200
    LLM_TOP_P: float = 0.9
    
    class Config:
        env_file = ".env"
```

**.env.example (NEW FILE):**
```
GEMINI_API_KEY=your_api_key_here
```

---

## 9. Testing Strategy

| Test | Description |
|------|-------------|
| `test_advisor_prompt_rendering` | Prompt template renders all `AdvisorInput` fields correctly |
| `test_numeric_echo_check_pass` | Valid response with correct numbers passes |
| `test_numeric_echo_check_fail` | Hallucinated number (e.g., 4.99 when CGPA is 3.2) fails |
| `test_run_advisor_mock_fallback` | Mock LLM client → returns mocked response |
| `test_run_advisor_fallback_on_error` | Simulated API error → falls back to `mock_advisor` |
| `test_run_advisor_fallback_on_echo_fail` | Response with wrong numbers → falls back |
| `test_integration_pipeline_advisor` | Full pipeline + advisor with mocked models |

---

## 10. File Structure (Phase 5 Additions)

```
APIS/
├── backend/
│   ├── __init__.py
│   ├── advisor.py           # NEW: run_advisor(), build_prompt(), echo-check
│   ├── llm_client.py        # NEW: LLMClient abstraction + GeminiClient
│   ├── config.py            # UPDATED: LLM settings
│   ├── orchestrator.py      # UPDATED: run_full_pipeline_with_advice()
│   └── mock_advisor.py      # EXISTING: fallback
├── prompts/
│   └── advisor_prompt.txt   # NEW: Prompt template
├── tests/
│   ├── test_advisor.py      # NEW: Advisor unit tests
│   └── test_integration.py  # UPDATED: Full pipeline + advisor
├── .env.example             # NEW: Template for API key
└── requirements.txt         # UPDATED: google-generativeai
```

---

## 11. Dependencies (requirements.txt additions)

```
google-generativeai>=0.3.0
python-dotenv>=1.0
```

---

## 12. Acceptance Criteria

- [ ] `run_advisor(AdvisorInput)` returns string ≤500 chars
- [ ] Numeric echo-check catches hallucinated numbers (unit test)
- [ ] Fallback to `mock_advisor` triggers on: API error, timeout, echo-check fail, empty response
- [ ] Three tones (`encouraging`, `direct`, `analytical`) produce visibly different output
- [ ] `run_full_pipeline_with_advice()` returns `(PipelineResult, str)` end-to-end
- [ ] All Phase 4 tests still pass (no regression)
- [ ] New tests in `tests/test_advisor.py` pass
- [ ] Integration test in `tests/test_integration.py` passes with mocked LLM

---

## 13. Deferred to Phase 6+

- Streaming response (Server-Sent Events / WebSocket)
- Structured JSON output schema for frontend parsing
- Multi-turn conversation with context window
- User feedback loop (thumbs up/down on advice)
- Fine-tuned model on APIS advice corpus
- Multi-language support (prompt template localization)

---

## 14. Approval

This design has been reviewed and approved for Phase 5 implementation.

**Next Step:** Create implementation plan using `writing-plans` skill, then execute.