"""
LLM Client for Market Verification
====================================

Uses OpenRouter (openrouter.ai) for LLM-based verification of
semantic equivalence between Polymarket and Kalshi market pairs.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMVerificationError(Exception):
    """LLM verification error."""

    pass


class MarketVerifier:
    """Verifies market pair equivalence using LLM via OpenRouter."""

    SYSTEM_PROMPT = """You are a market matching assistant. Your task is to determine if two prediction markets refer to the SAME underlying event.

Rules:
1. YES if the markets are about the SAME event with the SAME outcome criteria.
2. NO if they are about DIFFERENT events, even if similar teams are involved.
3. Markets about a game result (Who will win?) are DIFFERENT from markets about a point spread (Will they win by 10+ points?).
4. Answer with your reasoning first, but your FINAL line MUST be exactly "RESULT: YES" or "RESULT: NO".
5. If the title lists MULTIPLE games or outcomes, it is a multi-outcome market and usually DOES NOT match a single-game market.

Example:
Polymarket: "Lakers vs Celtics"
Kalshi: "Lakers win by 5+, Celtics win by 3+"
Result: NO (because Kalshi is a spread/multi-market)"""

    USER_TEMPLATE = """Polymarket: "{poly}"
Kalshi: "{kalshi}"

Do these markets refer to the same event? Provide your reasoning and end with RESULT: YES or RESULT: NO."""

    DEFAULT_MODEL = "google/gemini-2.0-flash-001"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_path: str = "logs/verified_matches.json",
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.cache_path = cache_path
        self.model = model or os.environ.get("OPENROUTER_MODEL") or self.DEFAULT_MODEL
        self._cache = self._load_cache()
        self._client = None

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load LLM cache: {e}")
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _get_cache_key(self, poly_id: str, kalshi_ticker: str) -> str:
        return f"{poly_id}|{kalshi_ticker}"

    def is_cached(self, poly_id: str, kalshi_ticker: str) -> bool:
        key = self._get_cache_key(poly_id, kalshi_ticker)
        return key in self._cache

    def get_cached_result(self, poly_id: str, kalshi_ticker: str) -> Optional[bool]:
        key = self._get_cache_key(poly_id, kalshi_ticker)
        if key in self._cache:
            return self._cache[key].get("verified")
        return None

    async def verify(
        self, poly_question: str, kalshi_title: str, poly_id: str, kalshi_ticker: str
    ) -> Optional[bool]:
        """Verify if two markets refer to the same event."""
        cache_key = self._get_cache_key(poly_id, kalshi_ticker)

        if cache_key in self._cache:
            result = self._cache[cache_key].get("verified")
            logger.debug(f"Cache hit for {poly_id[:8]} <-> {kalshi_ticker}: {result}")
            return result

        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set, skipping LLM verification")
            return None

        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.warning("openai package not installed, skipping LLM verification")
            return None

        if not self._client:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        user_prompt = self.USER_TEMPLATE.format(poly=poly_question, kalshi=kalshi_title)

        try:
            # For Minimax and other reasoning-heavy models on OpenRouter,
            # we need to be very explicit and increase token limit to let them "reason"
            # even if we ignore it

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,  # Increased for reasoning
                temperature=0,
            )

            # Debug the response structure if needed
            msg = response.choices[0].message
            content = msg.content

            # OpenRouter often places reasoning in 'reasoning' or 'reasoning_content'
            # and may keep 'content' null. We check all possibilities.
            raw_msg = msg.model_dump()

            reasoning = raw_msg.get("reasoning") or raw_msg.get("reasoning_content")

            if content is None:
                if reasoning:
                    logger.info(f"Using reasoning as content for {self.model}")
                    content = reasoning
                else:
                    # Last resort: check the actual choice object for hidden fields
                    choice_dict = response.choices[0].model_dump()
                    if "message" in choice_dict:
                        content = (
                            choice_dict["message"].get("content")
                            or choice_dict["message"].get("reasoning")
                            or choice_dict["message"].get("reasoning_content")
                        )

            if content is None:
                logger.error(f"LLM returned empty content. Response: {response}")
                return None

            # Clean up content - some models start with reasoning and then say YES/NO
            # We look for the words YES or NO specifically
            content_upper = content.upper()

            # For reasoning-heavy models, the answer is often at the VERY end
            # We look for YES or NO as whole words
            import re

            # Try to find the last occurrence of RESULT: YES or RESULT: NO
            matches = list(re.finditer(r"RESULT:\s*(YES|NO)", content_upper))
            if matches:
                last_match = matches[-1].group(1)
                verified = last_match == "YES"
            else:
                # Fallback to general YES/NO as standalone words at the very end
                matches = list(re.finditer(r"\b(YES|NO)\b", content_upper))
                if matches:
                    last_match = matches[-1].group(1)
                    verified = last_match == "YES"
                else:
                    logger.warning(
                        f"Could not find YES or NO in LLM response: {content}"
                    )
                    return None

            self._cache[cache_key] = {
                "verified": verified,
                "poly_question": poly_question,
                "kalshi_title": kalshi_title,
                "llm_response": content[:100],  # Store first 100 chars
            }
            self._save_cache()

            logger.info(
                f"LLM verified: '{poly_question[:30]}' <-> '{kalshi_title[:30]}': {'YES' if verified else 'NO'}"
            )
            return verified
            self._save_cache()

            logger.info(
                f"LLM verified: '{poly_question[:30]}' <-> '{kalshi_title[:30]}': {answer}"
            )
            return verified

        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return None

    def get_stats(self) -> dict:
        """Get cache statistics."""
        verified = sum(1 for v in self._cache.values() if v.get("verified"))
        rejected = sum(1 for v in self._cache.values() if not v.get("verified"))
        return {
            "total": len(self._cache),
            "verified": verified,
            "rejected": rejected,
        }
