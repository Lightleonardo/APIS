import pytest
from unittest.mock import Mock, patch
from backend.llm_client import LLMClient, GeminiClient, get_llm_client, call_llm
from backend.config import settings


class TestLLMClientAbstraction:
    def test_abstract_base_class(self):
        assert issubclass(GeminiClient, LLMClient)

    def test_get_llm_client_returns_gemini(self):
        # Temporarily set API key for this test
        original_key = settings.GEMINI_API_KEY
        settings.GEMINI_API_KEY = "test_key_for_testing"
        try:
            client = get_llm_client()
            assert isinstance(client, GeminiClient)
        finally:
            settings.GEMINI_API_KEY = original_key


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
    def test_generation_config_applied_during_init(self, mock_genai):
        mock_model = Mock()
        mock_model.generate_content.return_value.text = "Response"
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(api_key="test_key")
        # generation_config is passed to GenerativeModel constructor
        mock_genai.GenerativeModel.assert_called_once()
        call_kwargs = mock_genai.GenerativeModel.call_args.kwargs
        assert 'generation_config' in call_kwargs
        config = call_kwargs['generation_config']
        assert config['temperature'] == 0.3
        assert config['max_output_tokens'] == 200

    @patch('backend.llm_client.genai')
    def test_safety_settings_applied_during_init(self, mock_genai):
        mock_model = Mock()
        mock_model.generate_content.return_value.text = "Response"
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(api_key="test_key")
        # safety_settings is passed to GenerativeModel constructor
        call_kwargs = mock_genai.GenerativeModel.call_args.kwargs
        assert 'safety_settings' in call_kwargs
        safety = call_kwargs['safety_settings']
        assert safety["HARM_CATEGORY_HARASSMENT"] == "BLOCK_MEDIUM_AND_ABOVE"
        assert safety["HARM_CATEGORY_HATE_SPEECH"] == "BLOCK_MEDIUM_AND_ABOVE"
        assert safety["HARM_CATEGORY_SEXUALLY_EXPLICIT"] == "BLOCK_MEDIUM_AND_ABOVE"
        assert safety["HARM_CATEGORY_DANGEROUS_CONTENT"] == "BLOCK_MEDIUM_AND_ABOVE"


class TestCallLLM:
    def test_call_llm_delegates_to_client(self):
        mock_client = Mock(spec=LLMClient)
        mock_client.generate.return_value = "Delegated response"

        result = call_llm(mock_client, "Test prompt")

        assert result == "Delegated response"
        mock_client.generate.assert_called_once_with("Test prompt")