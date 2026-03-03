import os
import asyncio
import logging
from utils.config_loader import load_dotenv, ApiConfig, load_config
from utils.llm_client import MarketVerifier

# Setup minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_env_and_llm():
    # 1. Test .env loading
    print("--- 1. Testing .env loading ---")
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL")

    print(f"OPENROUTER_API_KEY: {'[FOUND]' if api_key else '[NOT FOUND]'}")
    if api_key:
        print(f"Key starts with: {api_key[:10]}...")
    print(f"OPENROUTER_MODEL: {model}")

    # 2. Test config loader
    print("\n--- 2. Testing config loader ---")
    try:
        # Create a dummy config if one doesn't exist just for testing
        if not os.path.exists("config.yaml"):
            with open("config.yaml", "w") as f:
                f.write("api:\n  polymarket_rest_url: 'https://clob.polymarket.com'")

        config = load_config("config.yaml")
        print(
            f"Config openrouter_api_key: {'[FOUND]' if config.api.openrouter_api_key else '[NOT FOUND]'}"
        )
        print(f"Config openrouter_model: {config.api.openrouter_model}")
    except Exception as e:
        print(f"Config error: {e}")

    # 3. Test LLM client
    print("\n--- 3. Testing LLM client ---")
    if not api_key:
        print("Skipping LLM test: No API key found in environment")
        return

    verifier = MarketVerifier(
        api_key=api_key, model=model or "google/gemini-2.0-flash-001"
    )

    # Test with a simple pair
    poly = "Will Bitcoin exceed $100k by March 31?"
    kalshi = "BTC above 100k March 31"

    print(f"Verifying: '{poly}' <-> '{kalshi}'")
    try:
        # Clear cache for this test to force an API call
        key = verifier._get_cache_key("test_poly", "test_kalshi")
        if key in verifier._cache:
            del verifier._cache[key]

        result = await verifier.verify(poly, kalshi, "test_poly", "test_kalshi")
        print(f"Verification result: {result}")
    except Exception as e:
        print(f"LLM Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_env_and_llm())
