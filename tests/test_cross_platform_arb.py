"""
Tests for the Cross-Platform Arbitrage Engine (MarketMatcher).
"""

import pytest

from core.cross_platform_arb import MarketMatcher


@pytest.fixture
def matcher() -> MarketMatcher:
    return MarketMatcher(min_similarity=0.0)  # No threshold — test scores only


class TestDatesCompatible:
    """Tests for dates_match()."""

    def test_same_date_string_matches(self, matcher: MarketMatcher):
        assert matcher.dates_match("2025-12-31", "2025-12-31") is True

    def test_different_date_strings_do_not_match(self, matcher: MarketMatcher):
        assert matcher.dates_match("2025-12-24", "2025-12-31") is False

    def test_none_first_returns_true(self, matcher: MarketMatcher):
        """Cannot check when first date is missing — allow by default."""
        assert matcher.dates_match(None, "2025-12-31") is True

    def test_none_second_returns_true(self, matcher: MarketMatcher):
        """Cannot check when second date is missing — allow by default."""
        assert matcher.dates_match("2025-12-31", None) is True

    def test_both_none_returns_true(self, matcher: MarketMatcher):
        """When neither platform provides dates, allow the match."""
        assert matcher.dates_match(None, None) is True


class TestNumericPenalty:
    """Tests for numeric-conflict rejection in _fast_similarity."""

    def test_different_strike_prices_lower_score(self, matcher: MarketMatcher):
        """'Bitcoin above $90k' vs '$100k' should score 0.0 due to numeric conflict."""
        pd = matcher._precompute_market_data("Bitcoin above $90k by December 2025")
        kd = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        score_different = matcher._fast_similarity(pd, kd)

        pd_same = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        kd_same = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        score_same = matcher._fast_similarity(pd_same, kd_same)

        assert score_different < score_same

    def test_numeric_conflict_returns_zero(self, matcher: MarketMatcher):
        """Conflicting numbers with no shared numeric entities → score of 0.0."""
        # No shared date year — only the strike prices differ
        pd = matcher._precompute_market_data("Will Bitcoin hit $90k?")
        kd = matcher._precompute_market_data("Will Bitcoin hit $100k?")
        assert matcher._fast_similarity(pd, kd) == 0.0

    def test_same_numbers_no_penalty(self, matcher: MarketMatcher):
        """Identical strike prices do not trigger numeric conflict."""
        pd = matcher._precompute_market_data("S&P 500 above 5000 points by June 2025")
        kd = matcher._precompute_market_data("S&P 500 above 5000 by June 2025")
        assert matcher._fast_similarity(pd, kd) > 0.0

    def test_fast_similarity_penalises_different_numbers(self, matcher: MarketMatcher):
        """Numeric conflict causes lower score in _fast_similarity (the live matching path)."""
        pd_different = matcher._precompute_market_data("Bitcoin above $90k by December 2025")
        kd_different = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        pd_same = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        kd_same = matcher._precompute_market_data("Bitcoin above $100k by December 2025")

        score_different = matcher._fast_similarity(pd_different, kd_different)
        score_same = matcher._fast_similarity(pd_same, kd_same)

        assert score_different < score_same

    def test_precomputed_data_includes_numeric_entities(self, matcher: MarketMatcher):
        """_precompute_market_data exposes numbers via 'entities' key (normalized)."""
        data = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        assert "entities" in data
        numeric_entities = {n for n in data["entities"] if any(c.isdigit() for c in n)}
        assert len(numeric_entities) > 0
