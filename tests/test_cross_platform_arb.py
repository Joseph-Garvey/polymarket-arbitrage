"""
Tests for the Cross-Platform Arbitrage Engine (MarketMatcher).
"""

import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

from core.cross_platform_arb import MarketMatcher


@pytest.fixture
def matcher() -> MarketMatcher:
    return MarketMatcher(min_similarity=0.0)  # No threshold — test scores only


def _market(end_date=None):
    """Minimal poly-market stub with an optional end_date."""
    return SimpleNamespace(end_date=end_date)


def _kalshi(close_time=None):
    """Minimal kalshi-market stub with an optional close_time."""
    return SimpleNamespace(close_time=close_time)


class TestDatesCompatible:
    """Tests for _dates_compatible()."""

    def test_same_date_is_compatible(self, matcher: MarketMatcher):
        base = datetime(2025, 12, 31)
        assert matcher._dates_compatible(_market(base), _kalshi(base)) is True

    def test_exactly_7_days_apart_is_compatible(self, matcher: MarketMatcher):
        poly = datetime(2025, 12, 24)
        kalshi = datetime(2025, 12, 31)
        assert matcher._dates_compatible(_market(poly), _kalshi(kalshi)) is True

    def test_8_days_apart_is_incompatible(self, matcher: MarketMatcher):
        poly = datetime(2025, 12, 23)
        kalshi = datetime(2025, 12, 31)
        assert matcher._dates_compatible(_market(poly), _kalshi(kalshi)) is False

    def test_none_poly_date_returns_true(self, matcher: MarketMatcher):
        """Cannot check when poly end_date is missing — allow by default."""
        assert matcher._dates_compatible(_market(None), _kalshi(datetime(2025, 12, 31))) is True

    def test_none_kalshi_date_returns_true(self, matcher: MarketMatcher):
        """Cannot check when kalshi close_time is missing — allow by default."""
        assert matcher._dates_compatible(_market(datetime(2025, 12, 31)), _kalshi(None)) is True

    def test_both_none_returns_true(self, matcher: MarketMatcher):
        """When neither platform provides dates, allow the match."""
        assert matcher._dates_compatible(_market(None), _kalshi(None)) is True


class TestNumericPenalty:
    """Tests for numeric-threshold penalty in calculate_similarity and _fast_similarity."""

    def test_different_strike_prices_lower_score(self, matcher: MarketMatcher):
        """'Bitcoin above $90k' vs '$100k' should score lower than matching strikes."""
        score_different = matcher.calculate_similarity(
            "Bitcoin above $90k by December 2025",
            "Bitcoin above $100k by December 2025",
        )
        score_same = matcher.calculate_similarity(
            "Bitcoin above $100k by December 2025",
            "Bitcoin above $100k by December 2025",
        )
        assert score_different < score_same

    def test_numeric_penalty_is_50_percent(self, matcher: MarketMatcher):
        """Penalty halves the combined score when numbers differ."""
        score_different = matcher.calculate_similarity(
            "Bitcoin above $90k by December 2025",
            "Bitcoin above $100k by December 2025",
        )
        score_same = matcher.calculate_similarity(
            "Bitcoin above $100k by December 2025",
            "Bitcoin above $100k by December 2025",
        )
        # With penalty: score_different ≈ score_same * 0.5
        assert score_different == pytest.approx(score_same * 0.5, rel=0.15)

    def test_same_numbers_no_penalty(self, matcher: MarketMatcher):
        """Identical strike prices do not trigger the penalty."""
        score = matcher.calculate_similarity(
            "Will the S&P 500 exceed 5000 points by June?",
            "S&P 500 above 5000 by June 2025",
        )
        # Penalty not applied — score should be substantial
        assert score > 0.3

    def test_fast_similarity_penalises_different_numbers(self, matcher: MarketMatcher):
        """Numeric penalty also applies in _fast_similarity (the live matching path)."""
        pd_different = matcher._precompute_market_data("Bitcoin above $90k by December 2025")
        kd_different = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        pd_same      = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        kd_same      = matcher._precompute_market_data("Bitcoin above $100k by December 2025")

        score_different = matcher._fast_similarity(pd_different, kd_different)
        score_same      = matcher._fast_similarity(pd_same, kd_same)

        assert score_different < score_same

    def test_precomputed_data_includes_numbers_key(self, matcher: MarketMatcher):
        """_precompute_market_data must expose a 'numbers' key for penalty to work."""
        data = matcher._precompute_market_data("Bitcoin above $100k by December 2025")
        assert "numbers" in data
        assert "100k" in data["numbers"] or any("100" in n for n in data["numbers"])
