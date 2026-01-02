# =============================================================================
# Test LLM Configuration
# =============================================================================
import asyncio
import sys

from src.utils.llm import get_llm, clear_llm_cache, get_available_providers, get_available_models
from src.config.settings import get_settings
from src.storage import init_storage, close_storage

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_success(msg: str):
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg: str):
    print(f"{RED}❌ {msg}{RESET}")


def print_info(msg: str):
    print(f"{BLUE}ℹ️  {msg}{RESET}")


def print_warning(msg: str):
    print(f"{YELLOW}⚠️  {msg}{RESET}")


async def test_settings():
    """Test settings configuration."""
    print(f"\n{BOLD}=== Testing Settings ==={RESET}")

    settings = get_settings()

    print(f"\nLLM Settings:")
    print(f"  Provider: {settings.llm_provider}")
    print(f"  Model: {settings.llm_model}")
    print(f"  Base URL: {settings.llm_base_url or '(default)'}")
    print(f"  Anthropic API Key: {'✓ Set' if settings.anthropic_api_key else '✗ Not set'}")
    print(f"  OpenAI API Key: {'✓ Set' if settings.openai_api_key else '✗ Not set'}")

    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        print_warning("Using Anthropic provider but API key not set")
        return False

    if settings.llm_provider == "openai" and not settings.openai_api_key:
        print_warning("Using OpenAI provider but API key not set")
        if not settings.llm_base_url:
            print_error("OpenAI provider needs either API key or custom base_url")
            return False

    print_success("Settings configuration valid")
    return True


async def test_llm_creation():
    """Test LLM instance creation."""
    print(f"\n{BOLD}=== Testing LLM Creation ==={RESET}")

    try:
        # Clear cache to force fresh creation
        clear_llm_cache()

        # Create LLM instance
        llm = get_llm()
        print_success(f"LLM created: {type(llm).__name__}")

        # Check provider
        provider = get_settings().llm_provider
        print_info(f"Provider: {provider}")

        if provider == "anthropic":
            print_info(f"Model: {llm.model}")
        elif provider == "openai":
            print_info(f"Model: {llm.model_name}")
            if get_settings().llm_base_url:
                print_info(f"Base URL: {get_settings().llm_base_url}")

        return llm
    except Exception as e:
        print_error(f"Failed to create LLM: {e}")
        return None


async def test_llm_invoke(llm):
    """Test LLM invocation."""
    print(f"\n{BOLD}=== Testing LLM Invocation ==={RESET}")

    test_queries = [
        "Xin chào! Bạn là gì?",
        "What is 2+2?",
        "Say 'test' if you can understand this.",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}/{len(test_queries)}]")
        print(f"Query: {query}")

        try:
            response = await llm.ainvoke(query)

            # Get response content
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)

            # Truncate if too long
            if len(content) > 200:
                content = content[:200] + "..."

            print_success(f"Response: {content}")

        except Exception as e:
            print_error(f"Invocation failed: {e}")
            print_info(f"Error type: {type(e).__name__}")
            return False

    return True


async def test_available_options():
    """Test available providers and models."""
    print(f"\n{BOLD}=== Testing Available Options ==={RESET}")

    providers = get_available_providers()
    print(f"\nSupported providers: {', '.join(providers)}")

    print("\nAvailable models by provider:")
    for provider in providers:
        models = get_available_models(provider)
        print(f"\n  {provider}:")
        for model in models[:5]:  # Show first 5
            print(f"    - {model}")
        if len(models) > 5:
            print(f"    ... and {len(models) - 5} more")


async def main():
    """Run all LLM tests."""
    print(f"\n{BOLD}{'='*60}")
    print(f"{'LLM Configuration Test':^54}")
    print(f"{'='*60}{RESET}\n")

    # Initialize storage
    await init_storage()

    try:
        # Test 1: Settings
        if not await test_settings():
            print_error("\n❌ Settings validation failed. Please check your .env configuration.")
            sys.exit(1)

        # Test 2: LLM Creation
        llm = await test_llm_creation()
        if not llm:
            print_error("\n❌ Failed to create LLM instance.")
            sys.exit(1)

        # Test 3: LLM Invocation
        if not await test_llm_invoke(llm):
            print_warning("\n⚠️  LLM invocation had issues. Check your API keys and network connection.")

        # Test 4: Available Options
        await test_available_options()

        # Summary
        print(f"\n{BOLD}{'='*60}")
        print(f"{'Test Summary':^54}")
        print(f"{'='*60}{RESET}")
        print_success("All tests completed!")
        print_info("\nTo use the LLM in your code:")
        print("```python")
        print("from src.utils.llm import get_llm")
        print("")
        print("# Get default LLM from settings")
        print("llm = get_llm()")
        print("")
        print("# Use LLM")
        print("response = await llm.ainvoke('Your question here')")
        print("print(response.content)")
        print("```")

    finally:
        await close_storage()
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
