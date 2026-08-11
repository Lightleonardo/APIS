import pytest
from backend.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.MODEL_DIR == "models"
    assert s.NEXT_GPA_MODEL == "next_gpa.pkl"
    assert s.FINAL_CGPA_MODEL == "final_cgpa.pkl"
    assert s.GRADUATION_CLASS_MODEL == "graduation_class.pkl"
    assert s.ACADEMIC_RISK_MODEL == "academic_risk.pkl"


def test_llm_settings():
    s = Settings()
    assert hasattr(s, 'GEMINI_API_KEY')
    assert hasattr(s, 'LLM_MODEL')
    assert s.LLM_MODEL == "gemini 3.1 Flash Lite"
    assert s.LLM_TEMPERATURE == 0.3
    assert s.LLM_MAX_TOKENS == 200
    assert s.LLM_TOP_P == 0.9