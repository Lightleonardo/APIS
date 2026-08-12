from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    MODEL_DIR: str = "models"
    NEXT_GPA_MODEL: str = "next_gpa.pkl"
    FINAL_CGPA_MODEL: str = "final_cgpa.pkl"
    GRADUATION_CLASS_MODEL: str = "graduation_class.pkl"
    ACADEMIC_RISK_MODEL: str = "academic_risk.pkl"

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 200
    LLM_TOP_P: float = 0.9

    class Config:
        env_file = ".env"


settings = Settings()