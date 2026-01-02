# =============================================================================
# LLM Utility - Multi-Provider LLM Factory
# =============================================================================
from functools import lru_cache
from typing import Optional, Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    base_url: Optional[str] = None,
    **kwargs: Any
) -> BaseChatModel:
    """
    Get configured LLM instance with caching.

    This function creates and caches LLM instances based on provider configuration.
    Uses @lru_cache to ensure only one instance per unique configuration is created.

    Args:
        provider: LLM provider ("anthropic" or "openai"). If None, uses settings.llm_provider
        model: Model name. If None, uses settings.llm_model
        temperature: Sampling temperature (0.0-1.0). Default 0.1 for deterministic responses
        max_tokens: Maximum tokens in response. Default 4096
        base_url: Custom base URL for OpenAI-compatible APIs (e.g., OpenRouter).
                  If None, uses settings.llm_base_url. Only for "openai" provider.
        **kwargs: Additional parameters passed to the LLM constructor

    Returns:
        BaseChatModel: Configured ChatAnthropic or ChatOpenAI instance

    Raises:
        ValueError: If provider is not supported

    Examples:
        >>> # Get default LLM from settings
        >>> llm = get_llm()
        >>>
        >>> # Use OpenRouter via environment variable
        >>> # Set LLM_BASE_URL=https://openrouter.ai/api/v1 in .env
        >>> llm = get_llm(provider="openai")
        >>>
        >>> # Override base_url programmatically
        >>> llm = get_llm(provider="openai", base_url="https://openrouter.ai/api/v1")
        >>>
        >>> # Override model and temperature
        >>> llm = get_llm(model="claude-3-opus-20240229", temperature=0.3)

    Note:
        When using base_url with "openai" provider, you can connect to OpenAI-compatible
        services like:
        - OpenRouter: https://openrouter.ai/api/v1
        - Azure OpenAI: https://your-resource.openai.azure.com/openai/deployments/your-deployment
        - Local LLMs: http://localhost:8000/v1

        The base_url can be set via:
        1. Environment variable: LLM_BASE_URL
        2. Function parameter: base_url="https://..."
    """
    settings = get_settings()
    provider = provider or settings.llm_provider
    model = model or settings.llm_model
    # Use parameter base_url if provided, otherwise use from settings
    base_url = base_url if base_url is not None else settings.llm_base_url or None

    logger.info(
        "Creating LLM instance",
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )

    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "openai":
        llm_kwargs = {
            "model": model,
            "api_key": settings.openai_api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Add base_url if provided (for OpenRouter, Azure, etc.)
        if base_url:
            llm_kwargs["base_url"] = base_url
            logger.info(f"Using custom base_url for OpenAI provider: {base_url}")
        llm_kwargs.update(kwargs)
        return ChatOpenAI(**llm_kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: anthropic, openai"
        )


def clear_llm_cache() -> None:
    """
    Clear the LLM instance cache.

    Use this when you need to force creation of a new LLM instance,
    for example when changing API keys or model configuration.

    Example:
        >>> clear_llm_cache()
        >>> llm = get_llm()  # Creates fresh instance
    """
    get_llm.cache_clear()
    logger.info("LLM cache cleared")


def get_available_providers() -> list[str]:
    """
    Get list of available LLM providers.

    Returns:
        List of supported provider names
    """
    return ["anthropic", "openai"]


def get_available_models(provider: str) -> list[str]:
    """
    Get list of available models for a provider.

    Args:
        provider: LLM provider name

    Returns:
        List of model names supported by the provider
    """
    models = {
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]
    }
    return models.get(provider, [])
