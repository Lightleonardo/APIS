from abc import ABC, abstractmethod
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