"""
Multi-leg arbitrage regression and unit tests.
"""
import pytest
from polymarket_client.models import (
    Market, MarketState, OrderBook, OrderBookSide, PriceLevel, TokenOrderBook, TokenType, OpportunityType,
)
from core.arb_engine import ArbEngine, ArbConfig


@pytest.fixture
def arb_config() -> ArbConfig:
    """Default arbitrage configuration for tests (no fees)."""
    return ArbConfig(
        min_edge=0.01,
        bundle_arb_enabled=True,
        min_spread=0.05,
        mm_enabled=True,
        tick_size=0.01,
        default_order_size=50.0,
        maker_fee_bps=0,
        taker_fee_bps=0,
        gas_cost_per_order=0,
    )


def _create_order_book_ml(
    market_id: str, yes_bid: float, yes_ask: float, no_bid: float, no_ask: float, size: float = 100.0
) -> OrderBook:
    return OrderBook(
        market_id=market_id,
        yes=TokenOrderBook(
            token_type=TokenType.YES,
            bids=OrderBookSide(levels=[PriceLevel(price=yes_bid, size=size)]),
            asks=OrderBookSide(levels=[PriceLevel(price=yes_ask, size=size)]),
        ),
        no=TokenOrderBook(
            token_type=TokenType.NO,
            bids=OrderBookSide(levels=[PriceLevel(price=no_bid, size=size)]),
            asks=OrderBookSide(levels=[PriceLevel(price=no_ask, size=size)]),
        ),
    )


class TestMultilegDoubleSignalBug:
    """Regression tests for the double-signal bug in 2-market NegRisk groups."""

    def _bundle_profitable(self, market_id: str, group_id: str) -> MarketState:
        ob = _create_order_book_ml(market_id, yes_bid=0.43, yes_ask=0.45, no_bid=0.48, no_ask=0.50)
        return MarketState(
            market=Market(market_id=market_id, condition_id=market_id, question="Test",
                          active=True, volume_24h=50000.0, group_id=group_id),
            order_book=ob,
        )

    def test_no_double_signal_when_bundle_fires(self, arb_config: ArbConfig):
        """BUNDLE_LONG must not also emit MULTILEG_LONG for the same legs."""
        engine = ArbEngine(arb_config)
        group_id = "neg_risk_group_1"
        state_a = self._bundle_profitable("market_a", group_id)
        state_b = self._bundle_profitable("market_b", group_id)
        engine.analyze(state_b, bankroll=1000.0)
        signals = engine.analyze(state_a, bankroll=1000.0)
        bundle_signals = [s for s in signals if s.opportunity and s.opportunity.opportunity_type
                          in (OpportunityType.BUNDLE_LONG, OpportunityType.BUNDLE_SHORT)]
        multileg_signals = [s for s in signals if s.opportunity and
                            s.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG]
        assert len(bundle_signals) == 1
        assert len(multileg_signals) == 0

    def test_multileg_fires_when_no_bundle_signal(self, arb_config: ArbConfig):
        """When no bundle signal fires, multileg check runs normally."""
        engine = ArbEngine(arb_config)
        group_id = "neg_risk_group_2"

        def _ml_only(mid: str) -> MarketState:
            ob = _create_order_book_ml(mid, yes_bid=0.37, yes_ask=0.40,
                                       no_bid=0.58, no_ask=0.62, size=200.0)
            return MarketState(
                market=Market(market_id=mid, condition_id=mid, question="Test",
                              active=True, volume_24h=50000.0, group_id=group_id),
                order_book=ob,
            )

        engine.analyze(_ml_only("market_c"), bankroll=1000.0)
        signals = engine.analyze(_ml_only("market_d"), bankroll=1000.0)
        ml = [s for s in signals if s.opportunity and
              s.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG]
        assert len(ml) == 1


class TestMultilegResolvedMarketEviction:
    """Regression: KeyError when a market resolves mid-analysis."""

    def _make_state(self, market_id: str, group_id: str, *, resolved: bool = False) -> MarketState:
        ob = _create_order_book_ml(market_id, yes_bid=0.37, yes_ask=0.40,
                                   no_bid=0.58, no_ask=0.62, size=200.0)
        return MarketState(
            market=Market(market_id=market_id, condition_id=market_id, question="Test",
                          active=not resolved, volume_24h=50000.0,
                          group_id=group_id, resolved=resolved),
            order_book=ob,
        )

    def test_no_keyerror_when_market_resolves(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "eviction_test_group"
        engine.analyze(self._make_state("evict_a", group_id), bankroll=1000.0)
        engine.analyze(self._make_state("evict_b", group_id), bankroll=1000.0)
        assert group_id in engine._group_states
        try:
            signals = engine.analyze(
                self._make_state("evict_a", group_id, resolved=True), bankroll=1000.0
            )
        except KeyError as exc:
            pytest.fail(f"KeyError raised when market resolved: {exc}")
        assert group_id not in engine._group_states
        assert not any(s.opportunity and s.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG
                       for s in signals)


class TestMultilegReEntryGuard:
    """Tests for the _active_opportunities re-entry guard."""

    def _ml_state(self, market_id: str, group_id: str) -> MarketState:
        ob = _create_order_book_ml(market_id, yes_bid=0.37, yes_ask=0.40,
                                   no_bid=0.58, no_ask=0.62, size=200.0)
        return MarketState(
            market=Market(market_id=market_id, condition_id=market_id, question="Test",
                          active=True, volume_24h=50000.0, group_id=group_id),
            order_book=ob,
        )

    def test_guard_suppresses_second_signal(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "guard_test_group"
        state_a = self._ml_state("guard_market_a", group_id)
        state_b = self._ml_state("guard_market_b", group_id)
        engine.analyze(state_a, bankroll=1000.0)
        signals1 = engine.analyze(state_b, bankroll=1000.0)
        ml1 = [s for s in signals1 if s.opportunity and
               s.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG]
        assert len(ml1) == 1
        signals2 = engine.analyze(state_a, bankroll=1000.0)
        ml2 = [s for s in signals2 if s.opportunity and
               s.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG]
        assert len(ml2) == 0

    def test_guard_suppresses_different_member_market(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "guard_test_group_2"
        state_c = self._ml_state("guard_market_c", group_id)
        state_d = self._ml_state("guard_market_d", group_id)
        engine.analyze(state_c, bankroll=1000.0)
        engine.analyze(state_d, bankroll=1000.0)
        signals2 = engine.analyze(state_c, bankroll=1000.0)
        ml2 = [s for s in signals2 if s.opportunity and
               s.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG]
        assert len(ml2) == 0

    def test_guard_releases_after_expiry(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "guard_test_group_3"
        state_e = self._ml_state("guard_market_e", group_id)
        state_f = self._ml_state("guard_market_f", group_id)
        engine.analyze(state_e, bankroll=1000.0)
        engine.analyze(state_f, bankroll=1000.0)
        multileg_key = f"{group_id}_multileg_long"
        del engine._active_opportunities[multileg_key]
        signals2 = engine.analyze(state_e, bankroll=1000.0)
        ml2 = [s for s in signals2 if s.opportunity and
               s.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG]
        assert len(ml2) == 1

    def test_expiry_check_evicts_when_prices_move(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "guard_test_group_4"
        state_g = self._ml_state("guard_market_g", group_id)
        state_h = self._ml_state("guard_market_h", group_id)
        engine.analyze(state_g, bankroll=1000.0)
        engine.analyze(state_h, bankroll=1000.0)
        multileg_key = f"{group_id}_multileg_long"
        assert multileg_key in engine._active_opportunities

        def _no_edge(mid: str) -> MarketState:
            ob = _create_order_book_ml(mid, yes_bid=0.50, yes_ask=0.52,
                                       no_bid=0.46, no_ask=0.48, size=200.0)
            return MarketState(
                market=Market(market_id=mid, condition_id=mid, question="Test",
                              active=True, volume_24h=50000.0, group_id=group_id),
                order_book=ob,
            )

        no_g = _no_edge("guard_market_g")
        no_h = _no_edge("guard_market_h")
        engine._group_states[group_id]["guard_market_g"] = no_g
        engine._group_states[group_id]["guard_market_h"] = no_h
        engine.analyze(no_g, bankroll=1000.0)
        assert multileg_key not in engine._active_opportunities


# ---------------------------------------------------------------------------
# Helpers and tests for TestCheckMultilegArbitrage
# ---------------------------------------------------------------------------

def _make_orderbook_ml(
    market_id: str, yes_ask: float, yes_ask_size: float = 100.0,
    yes_bid: float = None, no_ask: float = None, no_bid: float = None,
) -> OrderBook:
    _yes_bid = yes_bid if yes_bid is not None else yes_ask - 0.02
    _no_ask = no_ask if no_ask is not None else 1.0 - yes_ask + 0.02
    _no_bid = no_bid if no_bid is not None else 1.0 - yes_ask
    return OrderBook(
        market_id=market_id,
        yes=TokenOrderBook(
            token_type=TokenType.YES,
            bids=OrderBookSide(levels=[PriceLevel(price=_yes_bid, size=yes_ask_size)]),
            asks=OrderBookSide(levels=[PriceLevel(price=yes_ask, size=yes_ask_size)]),
        ),
        no=TokenOrderBook(
            token_type=TokenType.NO,
            bids=OrderBookSide(levels=[PriceLevel(price=_no_bid, size=yes_ask_size)]),
            asks=OrderBookSide(levels=[PriceLevel(price=_no_ask, size=yes_ask_size)]),
        ),
    )


def _make_market_state_ml(
    market_id: str, group_id: str, group_size: int,
    yes_ask: float, yes_ask_size: float = 100.0,
) -> MarketState:
    market = Market(
        market_id=market_id, condition_id=market_id,
        question=f"Question for {market_id}?", group_id=group_id, group_size=group_size,
    )
    return MarketState(market=market, order_book=_make_orderbook_ml(market_id, yes_ask, yes_ask_size))


@pytest.fixture
def fee_config() -> ArbConfig:
    return ArbConfig(
        min_edge=0.01, bundle_arb_enabled=True, mm_enabled=False,
        taker_fee_bps=150, gas_cost_per_order=0.02, min_order_size=5.0,
    )


class TestCheckMultilegArbitrage:
    """Unit tests for ArbEngine._check_multileg_arbitrage."""

    def test_2leg_detection(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "ml_2leg_group"
        states = {
            "ml_2leg_a": _make_market_state_ml("ml_2leg_a", group_id, 2, yes_ask=0.425),
            "ml_2leg_b": _make_market_state_ml("ml_2leg_b", group_id, 2, yes_ask=0.425),
        }
        signal = engine._check_multileg_arbitrage(group_id, states, bankroll=1000.0)
        assert signal is not None
        assert signal.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG
        assert signal.opportunity.edge > 0

    def test_3leg_detection(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "ml_3leg_group"
        yes_ask_per_leg = 0.88 / 3
        states = {
            f"ml_3leg_{i}": _make_market_state_ml(f"ml_3leg_{i}", group_id, 3, yes_ask=yes_ask_per_leg)
            for i in range(3)
        }
        signal = engine._check_multileg_arbitrage(group_id, states, bankroll=1000.0)
        assert signal is not None
        assert len(signal.orders) == 3

    def test_no_signal_when_sum_above_1(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        states = {
            "ml_no_a": _make_market_state_ml("ml_no_a", "g", 2, yes_ask=0.51),
            "ml_no_b": _make_market_state_ml("ml_no_b", "g", 2, yes_ask=0.51),
        }
        assert engine._check_multileg_arbitrage("g", states, bankroll=1000.0) is None

    def test_no_signal_when_net_edge_below_min(self, fee_config: ArbConfig):
        engine = ArbEngine(fee_config)
        states = {
            "ml_below_a": _make_market_state_ml("ml_below_a", "g", 2, yes_ask=0.47),
            "ml_below_b": _make_market_state_ml("ml_below_b", "g", 2, yes_ask=0.47),
        }
        assert engine._check_multileg_arbitrage("g", states, bankroll=1000.0) is None

    def test_fee_calculation_is_correct(self, fee_config: ArbConfig):
        engine = ArbEngine(fee_config)
        group_id = "ml_fee_calc_group"
        yes_ask_a, yes_ask_b = 0.40, 0.35
        total_ask = yes_ask_a + yes_ask_b
        states = {
            "ml_fee_a": _make_market_state_ml("ml_fee_a", group_id, 2, yes_ask=yes_ask_a),
            "ml_fee_b": _make_market_state_ml("ml_fee_b", group_id, 2, yes_ask=yes_ask_b),
        }
        signal = engine._check_multileg_arbitrage(group_id, states, bankroll=1000.0)
        assert signal is not None
        expected = (
            1.0 - total_ask
            - (fee_config.taker_fee_bps / 10000) * total_ask
            - fee_config.gas_cost_per_order * 2
        )
        assert abs(signal.opportunity.edge - expected) < 1e-9

    def test_partial_group_suppression(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "ml_partial_group"
        state_a = _make_market_state_ml("ml_partial_a", group_id, group_size=5, yes_ask=0.40)
        state_b = _make_market_state_ml("ml_partial_b", group_id, group_size=5, yes_ask=0.40)
        engine.analyze(state_a, bankroll=1000.0)
        signals = engine.analyze(state_b, bankroll=1000.0)
        assert not any(s.opportunity and s.opportunity.opportunity_type == OpportunityType.MULTILEG_LONG
                       for s in signals)

    def test_liquidity_gate(self, arb_config: ArbConfig):
        config = ArbConfig(min_edge=0.01, bundle_arb_enabled=True, mm_enabled=False,
                           taker_fee_bps=0, gas_cost_per_order=0, min_order_size=5.0)
        engine = ArbEngine(config)
        group_id = "ml_liquidity_group"
        states = {
            "ml_liq_a": _make_market_state_ml("ml_liq_a", group_id, 2, yes_ask=0.40, yes_ask_size=3.0),
            "ml_liq_b": _make_market_state_ml("ml_liq_b", group_id, 2, yes_ask=0.40, yes_ask_size=3.0),
        }
        assert engine._check_multileg_arbitrage(group_id, states, bankroll=1000.0) is None

    def test_re_entry_suppression(self, arb_config: ArbConfig):
        engine = ArbEngine(arb_config)
        group_id = "ml_reentry_group"
        states = {
            "ml_re_a": _make_market_state_ml("ml_re_a", group_id, 2, yes_ask=0.40),
            "ml_re_b": _make_market_state_ml("ml_re_b", group_id, 2, yes_ask=0.40),
        }
        signal1 = engine._check_multileg_arbitrage(group_id, states, bankroll=1000.0)
        assert signal1 is not None
        multileg_key = f"{group_id}_multileg_long"
        assert multileg_key in engine._active_opportunities
        # Simulate the analyze() guard
        signal2 = None if multileg_key in engine._active_opportunities else (
            engine._check_multileg_arbitrage(group_id, states, bankroll=1000.0)
        )
        assert signal2 is None
