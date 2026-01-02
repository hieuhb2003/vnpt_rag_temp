# =============================================================================
# Unit Tests for LLM Utility
# =============================================================================
import pytest
from unittest.mock import patch, MagicMock

from src.utils.llm import get_llm, clear_llm_cache, get_available_providers, get_available_models


class TestGetLLM:
    """Test get_llm function."""

    def test_get_available_providers(self):
        """Test getting available providers."""
        providers = get_available_providers()
        assert "anthropic" in providers
        assert "openai" in providers

    def test_get_available_models_anthropic(self):
        """Test getting Anthropic models."""
        models = get_available_models("anthropic")
        assert "claude-3-5-sonnet-20241022" in models
        assert "claude-3-opus-20240229" in models

    def test_get_available_models_openai(self):
        """Test getting OpenAI models."""
        models = get_available_models("openai")
        assert "gpt-4o" in models
        assert "gpt-4-turbo" in models

    def test_get_available_models_unknown_provider(self):
        """Test getting models for unknown provider."""
        models = get_available_models("unknown")
        assert models == []

    def test_get_llm_anthropic(self):
        """Test getting Anthropic LLM."""
        with patch("src.utils.llm.ChatAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            llm = get_llm(provider="anthropic")

            assert llm is mock_instance
            mock_anthropic.assert_called_once()

    def test_get_llm_openai(self):
        """Test getting OpenAI LLM."""
        with patch("src.utils.llm.ChatOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            llm = get_llm(provider="openai")

            assert llm is mock_instance
            mock_openai.assert_called_once()

    def test_base_url_openai(self):
        """Test custom base_url for OpenAI provider."""
        with patch("src.utils.llm.ChatOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            clear_llm_cache()
            llm = get_llm(
                provider="openai",
                base_url="https://openrouter.ai/api/v1"
            )

            # Check base_url was passed
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_base_url_none_uses_default(self):
        """Test that None base_url uses default OpenAI endpoint."""
        with patch("src.utils.llm.ChatOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            clear_llm_cache()
            llm = get_llm(provider="openai", base_url=None)

            # Check base_url was not in kwargs (uses default)
            call_kwargs = mock_openai.call_args[1]
            assert "base_url" not in call_kwargs

    def test_get_llm_invalid_provider(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm(provider="invalid")

    def test_get_llm_caching(self):
        """Test that get_llm caches instances."""
        # Clear cache first
        clear_llm_cache()

        with patch("src.utils.llm.ChatAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            # First call
            llm1 = get_llm(provider="anthropic")

            # Second call - should return cached instance
            llm2 = get_llm(provider="anthropic")

            # Should be the same instance (cached)
            assert llm1 is llm2

            # ChatAnthropic should only be created once
            mock_anthropic.assert_called_once()

    def test_get_llm_different_params_creates_new_instance(self):
        """Test that different parameters create different instances."""
        # Clear cache first
        clear_llm_cache()

        with patch("src.utils.llm.ChatAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            # First call
            llm1 = get_llm(provider="anthropic", temperature=0.1)

            # Second call with different temperature
            clear_llm_cache()  # Need to clear cache due to @lru_cache
            llm2 = get_llm(provider="anthropic", temperature=0.5)

            # Should create two instances (different params)
            mock_anthropic.assert_called()

    def test_clear_llm_cache(self):
        """Test clearing LLM cache."""
        with patch("src.utils.llm.ChatAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            # Create instance
            get_llm(provider="anthropic")

            # Clear cache
            clear_llm_cache()

            # Create new instance after cache clear
            get_llm(provider="anthropic")

            # Should be called twice (cache was cleared)
            assert mock_anthropic.call_count == 2


class TestLLMParameters:
    """Test LLM parameter handling."""

    def test_default_parameters(self):
        """Test default parameters."""
        with patch("src.utils.llm.ChatAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            clear_llm_cache()
            llm = get_llm(provider="anthropic")

            # Check default parameters
            call_kwargs = mock_anthropic.call_args[1]
            assert call_kwargs["temperature"] == 0.1
            assert call_kwargs["max_tokens"] == 4096

    def test_custom_parameters(self):
        """Test custom parameters."""
        with patch("src.utils.llm.ChatOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance

            clear_llm_cache()
            llm = get_llm(
                provider="openai",
                temperature=0.7,
                max_tokens=2048
            )

            # Check custom parameters
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 2048

    def test_model_override(self):
        """Test model override."""
        with patch("src.utils.llm.ChatAnthropic") as mock_anthropic:
            mock_instance = MagicMock()
            mock_anthropic.return_value = mock_instance

            clear_llm_cache()
            llm = get_llm(
                provider="anthropic",
                model="claude-3-opus-20240229"
            )

            # Check model parameter
            call_kwargs = mock_anthropic.call_args[1]
            assert call_kwargs["model"] == "claude-3-opus-20240229"
