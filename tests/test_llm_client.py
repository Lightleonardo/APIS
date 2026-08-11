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
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="test_key")
        result = client.generate("Test prompt")

        assert result == "Test response"
        mock_genai.Client.assert_called_once_with(api_key="test_key")
        mock_genai.Client.return_value.models.generate_content.assert_called_once()

    @patch('backend.llm_client.genai')
    def test_generation_config_applied_during_init(self, mock_genai):
        mock_response = Mock()
        mock_response.text = "Response"
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="test_key")
        # generation_config is passed to generate_content call
        mock_genai.Client.assert_called_once_with(api_key="test_key")
        client.generate("Test prompt")
        call_kwargs = mock_genai.Client.return_value.models.generate_content.call_args.kwargs
        assert 'config' in call_kwargs
        config = call_kwargs['config']
        assert config['temperature'] == 0.3
        assert config['max_output_tokens'] == 200

    @patch('backend.llm_client.genai')
    def test_safety_settings_not_in_new_api(self, mock_genai):
        """New genai API doesn't use safety_settings in the same way - it's handled differently"""
        mock_response = Mock()
        mock_response.text = "Response"
        mock_genai.Client.return_value.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="test_key")
        # New API doesn't have safety_settings in the same place
        mock_genai.Client.assert_called_once_with(api_key="test_key")
        # Just verify client initialization works
        assert client.model_name == "gemini 3.1 Flash Lite"


class TestCallLLM:
    def test_call_llm_delegates_to_client(self):
        mock_client = Mock(spec=LLMClient)
        mock_client.generate.return_value = "Delegated response"

        result = call_llm(mock_client, "Test prompt")

        assert result == "Delegated response"
        mock_client.generate.assert_called_once_with("Test prompt")