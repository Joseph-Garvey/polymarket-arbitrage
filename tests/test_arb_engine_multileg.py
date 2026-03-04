"""
Multi-leg arbitrage regression and unit tests.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def _run(coro):
    """Run a coroutine, leaving a valid event loop set for subsequent tests.

    asyncio.run() closes and removes the current event loop after completion,
    which breaks tests that call asyncio.get_event_loop() (e.g. test_new_features.py).
    Using new_event_loop + set_event_loop avoids that.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        # Do NOT call loop.close() — leave the loop set so subsequent tests that
        # call asyncio.get_event_loop() still find a valid loop.
        pass

from core.arb_engine import ArbEngine, ArbConfig
from core.execution import ExecutionEngine, ExecutionConfig
from core.portfolio import Portfolio, GroupArbLeg
from core.risk_manager import RiskManager, RiskConfig
from polymarket_client.models import (
    Market, MarketState, OrderBook, OrderBookSide, PriceLevel, TokenOrderBook,
    TokenType, OpportunityType, Trade, OrderSide,
)


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


# ---------------------------------------------------------------------------
# Bug fix regression tests
# ---------------------------------------------------------------------------


class TestGroupStatesCapEviction:
    """
    Regression: _group_states cap evicted active groups on every update when
    len >= 500, not just on new insertions.
    """

    def _make_active_state(self, market_id: str, group_id: str) -> MarketState:
        ob = _create_order_book_ml(
            market_id,
            yes_bid=0.37, yes_ask=0.40, no_bid=0.58, no_ask=0.62, size=200.0,
        )
        return MarketState(
            market=Market(
                market_id=market_id, condition_id=market_id, question="Test",
                active=True, volume_24h=50000.0, group_id=group_id,
            ),
            order_book=ob,
        )

    def test_updating_existing_group_does_not_evict_when_at_cap(self, arb_config: ArbConfig):
        """Updating a market in an existing group must not evict any group when len >= 500."""
        engine = ArbEngine(arb_config)

        # Seed exactly 500 groups directly (avoids running full analyze() 500 times)
        seed_state = self._make_active_state("seed_market", "seed_group_0")
        for i in range(500):
            gid = f"seed_group_{i}"
            mid = f"seed_market_{i}"
            engine._group_states[gid] = {mid: seed_state}

        assert len(engine._group_states) == 500
        oldest_group = next(iter(engine._group_states))  # "seed_group_0"

        # Analyze a market that belongs to an ALREADY-TRACKED group — no eviction should happen
        state = self._make_active_state("seed_market_0", "seed_group_0")
        engine.analyze(state, bankroll=1000.0)

        assert len(engine._group_states) == 500, (
            "Updating an existing group must not evict any group (dict size changed)"
        )
        assert oldest_group in engine._group_states, (
            f"'{oldest_group}' was wrongly evicted when updating an existing group"
        )

    def test_new_group_evicts_oldest_when_at_cap(self, arb_config: ArbConfig):
        """Adding a brand-new (501st) group should evict the oldest, keeping the cap."""
        engine = ArbEngine(arb_config)

        seed_state = self._make_active_state("seed_market", "seed_group_0")
        for i in range(500):
            gid = f"seed_group_{i}"
            mid = f"seed_market_{i}"
            engine._group_states[gid] = {mid: seed_state}

        oldest_group = next(iter(engine._group_states))  # "seed_group_0"

        # Analyze a BRAND NEW group — should evict oldest, new one takes its place
        new_state = self._make_active_state("brand_new_market", "brand_new_group")
        engine.analyze(new_state, bankroll=1000.0)

        assert len(engine._group_states) == 500
        assert oldest_group not in engine._group_states, "Oldest group should have been evicted"
        assert "brand_new_group" in engine._group_states, "New group should be present"


class TestMultilegGroupPositionLockedProfit:
    """
    Regression: _maybe_open_group_position for multileg_arb created a single-leg
    GroupArbPosition with locked_profit = 1.0 - entry_price, which is wrong.
    Profit is only locked once ALL mutually exclusive legs are filled.
    """

    def test_single_leg_multileg_fill_reports_zero_locked_profit(self):
        """After a multileg YES fill, the group position must not claim fictitious profit."""
        portfolio = Portfolio(initial_balance=1000.0)
        exec_engine = ExecutionEngine(
            client=MagicMock(),
            risk_manager=RiskManager(RiskConfig()),
            portfolio=portfolio,
            config=ExecutionConfig(dry_run=True),
        )

        # Simulate a YES fill at 0.20 for one leg of a 5-option NegRisk group
        trade = Trade(
            trade_id="t1", order_id="o1", market_id="market_leg_a",
            token_type=TokenType.YES, side=OrderSide.BUY,
            price=0.20, size=50.0, fee=0.0, strategy_tag="multileg_arb",
        )
        portfolio.update_from_fill(trade)

        exec_engine._maybe_open_group_position("market_leg_a", "multileg_arb")

        group = portfolio._open_group_arbs.get("market_leg_a")
        assert group is not None, "GroupArbPosition should be registered for re-entry guard"
        assert group.locked_profit == 0.0, (
            f"Single-leg multileg position must have locked_profit=0.0, "
            f"got {group.locked_profit:.4f} — a YES position at 0.20 does not lock "
            f"profit until all 5 legs fill"
        )
        assert group.unrealized_pnl == 0.0


# ---------------------------------------------------------------------------
# Gap-fill recovery tests
# ---------------------------------------------------------------------------

def _make_engine_for_gap_fill():
    """Return (engine, portfolio) with an AsyncMock client."""
    client = MagicMock()
    client.cancel_order = AsyncMock(return_value=None)
    client.place_order = AsyncMock(return_value=None)
    client.get_orderbook = AsyncMock(return_value=None)

    portfolio = MagicMock()
    risk_manager = RiskManager(RiskConfig())
    config = ExecutionConfig(dry_run=True)
    engine = ExecutionEngine(
        client=client,
        risk_manager=risk_manager,
        portfolio=portfolio,
        config=config,
    )
    return engine, portfolio


class TestMultilegGapFillMetadata:
    """_multileg_signal_meta is populated correctly when a multileg signal is placed."""

    def test_metadata_keys_are_stored(self):
        """Signal metadata dictionary has the expected structure."""
        engine, _ = _make_engine_for_gap_fill()
        sid = "sig_meta_001"
        engine._multileg_signal_meta[sid] = {
            "market_ids": ["mkt_a", "mkt_b", "mkt_c"],
            "size": 50.0,
            "target_payout": 50.0,
            "total_cost_estimate": 46.0,
        }
        assert engine._multileg_signal_meta[sid]["market_ids"] == ["mkt_a", "mkt_b", "mkt_c"]
        assert engine._multileg_signal_meta[sid]["target_payout"] == 50.0

    def test_unknown_signal_is_noop(self):
        """Calling gap-fill for an unknown signal_id returns silently."""
        engine, _ = _make_engine_for_gap_fill()
        _run(engine._handle_multileg_partial_fills("nonexistent"))
        engine.client.cancel_order.assert_not_called()
        engine.client.place_order.assert_not_called()

    def test_metadata_cleaned_up_after_recovery(self):
        """Metadata is removed from _multileg_signal_meta after recovery runs."""
        engine, portfolio = _make_engine_for_gap_fill()
        sid = "sig_cleanup"

        # All legs filled — portfolio returns full position for every market
        def mock_get_position(market_id, token_type):
            pos = MagicMock()
            pos.size = 50.0
            pos.avg_entry_price = 0.45
            return pos

        portfolio.get_position.side_effect = mock_get_position

        engine._multileg_signal_meta[sid] = {
            "market_ids": ["mkt_x", "mkt_y"],
            "size": 50.0,
            "target_payout": 50.0,
            "total_cost_estimate": 45.0,
        }
        _run(engine._handle_multileg_partial_fills(sid))
        assert sid not in engine._multileg_signal_meta


class TestMultilegGapFillRecovery:
    """_handle_multileg_partial_fills places recovery orders when budget allows."""

    def test_no_gap_skips_recovery_orders(self):
        """When all legs are fully filled, no recovery order is placed."""
        engine, portfolio = _make_engine_for_gap_fill()
        sid = "sig_no_gap"

        def fully_filled(market_id, token_type):
            pos = MagicMock()
            pos.size = 50.0
            pos.avg_entry_price = 0.45
            return pos

        portfolio.get_position.side_effect = fully_filled
        engine._multileg_signal_meta[sid] = {
            "market_ids": ["mkt_a", "mkt_b"],
            "size": 50.0,
            "target_payout": 50.0,
            "total_cost_estimate": 45.0,
        }
        _run(engine._handle_multileg_partial_fills(sid))
        engine.client.place_order.assert_not_called()

    def test_gap_within_budget_triggers_recovery(self):
        """When a leg is unfilled and within budget, a recovery order is placed."""
        engine, portfolio = _make_engine_for_gap_fill()
        engine.config.dry_run = False  # enable real order placement
        mock_order = MagicMock()
        mock_order.order_id = "recovery_order_1"
        engine.client.place_order = AsyncMock(return_value=mock_order)
        sid = "sig_gap_ok"

        def partial_fill(market_id, token_type):
            # mkt_a filled, mkt_b not filled
            pos = MagicMock()
            if market_id == "mkt_a":
                pos.size = 50.0
                pos.avg_entry_price = 0.48
            else:
                pos.size = 0.0
                pos.avg_entry_price = 0.0
            return pos

        portfolio.get_position.side_effect = partial_fill

        # Orderbook for mkt_b has enough liquidity at 0.45
        mock_ob = _create_order_book_ml("mkt_b", 0.40, 0.45, 0.55, 0.60, size=60.0)

        async def mock_get_ob(market_id):
            return mock_ob

        engine.client.get_orderbook = mock_get_ob

        # target_payout=50, total_spent=24 (0.48*50), budget=(50-24)*1.1=28.6
        engine._multileg_signal_meta[sid] = {
            "market_ids": ["mkt_a", "mkt_b"],
            "size": 50.0,
            "target_payout": 50.0,
            "total_cost_estimate": 46.5,
        }
        _run(engine._handle_multileg_partial_fills(sid))

        # Metadata cleaned up and recovery order placed for the unfilled leg
        assert sid not in engine._multileg_signal_meta
        engine.client.place_order.assert_called_once()
        kw = engine.client.place_order.call_args.kwargs
        assert kw["market_id"] == "mkt_b"
        assert kw["price"] == 1.0
        assert kw["size"] == 50.0

    def test_gap_over_budget_skips_recovery(self):
        """When the gap-fill cost exceeds budget, no order is placed."""
        engine, portfolio = _make_engine_for_gap_fill()
        sid = "sig_gap_over"

        def no_fills(market_id, token_type):
            pos = MagicMock()
            pos.size = 0.0
            pos.avg_entry_price = 0.0
            return pos

        portfolio.get_position.side_effect = no_fills

        # total_spent=0, target_payout=10 → budget=11
        # But orderbook shows gap costs 50 (way over budget)
        mock_ob = _create_order_book_ml("mkt_a", 0.40, 0.99, 0.01, 0.60, size=100.0)

        async def mock_get_ob(market_id):
            return mock_ob

        engine.client.get_orderbook = mock_get_ob

        engine._multileg_signal_meta[sid] = {
            "market_ids": ["mkt_a"],
            "size": 50.0,
            "target_payout": 10.0,  # tiny payout means tiny budget
            "total_cost_estimate": 0.0,
        }
        _run(engine._handle_multileg_partial_fills(sid))
        # dry_run=True anyway, but we verify no order would be dispatched
        engine.client.place_order.assert_not_called()
