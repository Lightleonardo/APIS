from abc import ABC, abstractmethod
from google import genai
from google.genai import types  # 1. Added import for types
from backend.config import settings


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GeminiClient(LLMClient):
    def __init__(self):
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured in .env file")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = settings.LLM_MODEL

    def generate(self, prompt: str) -> str:
        # Wrapping settings dictionary into types.GenerateContentConfig
        config = types.GenerateContentConfig(
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_TOKENS,
            top_p=settings.LLM_TOP_P,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,  # Pass the explicit typed config object
        )
        return response.text or ""


def get_llm_client() -> LLMClient:
    return GeminiClient()


def call_llm(client: LLMClient, prompt: str) -> str:
    return client.generate(prompt)
