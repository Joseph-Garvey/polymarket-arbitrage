"""
Tests for features added in the performance/correctness improvement commits:
- Kelly sizing (_kelly_size)
- Liquidity depth gate
- _validate_arb_still_live (staleness + direction)
- _place_orders_concurrent (partial failure rollback)
- register_bundle_signal / check_bundle_completion
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

import pytest

from polymarket_client.models import (
    Market,
    MarketState,
    Order,
    OrderBook,
    OrderBookSide,
    OrderSide,
    OrderStatus,
    OpportunityType,
    Opportunity,
    PriceLevel,
    Signal,
    TokenOrderBook,
    TokenType,
)
from core.arb_engine import ArbEngine, ArbConfig
from core.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_arb_config(**kwargs) -> ArbConfig:
    defaults = dict(
        min_edge=0.01,
        bundle_arb_enabled=True,
        min_spread=0.05,
        mm_enabled=False,
        taker_fee_bps=0,
        gas_cost_per_order=0,
        min_order_size=1.0,
        max_order_size=500.0,
        default_order_size=50.0,
    )
    defaults.update(kwargs)
    return ArbConfig(**defaults)


def make_order_book(
    market_id: str,
    yes_bid: float, yes_ask: float,
    no_bid: float, no_ask: float,
    size: float = 100.0,
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


def make_market_state(ob: OrderBook) -> MarketState:
    return MarketState(
        market=Market(
            market_id=ob.market_id,
            condition_id=ob.market_id,
            question="Test",
            active=True,
            volume_24h=50000.0,
        ),
        order_book=ob,
    )


def make_order(order_id: str, status: OrderStatus = OrderStatus.PENDING) -> Order:
    o = Order(
        order_id=order_id,
        market_id="test_market",
        token_type=TokenType.YES,
        side=OrderSide.BUY,
        price=0.45,
        size=10.0,
    )
    o.status = status
    return o


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

class TestKellySizing:
    def test_positive_edge_long(self):
        """Standard long case: price < 1, positive edge → sized above min."""
        cfg = make_arb_config(min_order_size=1.0, max_order_size=500.0)
        engine = ArbEngine(cfg)
        size = engine._kelly_size(edge=0.05, price=0.90, max_size=200.0, bankroll=1000.0)
        assert size > cfg.min_order_size

    def test_bundle_short_price_above_one(self):
        """Bundle short: total_bid > 1.0, must NOT fall back to min_order_size."""
        cfg = make_arb_config(min_order_size=1.0, max_order_size=500.0)
        engine = ArbEngine(cfg)
        # total_bid = 1.04 (typical short arb), edge = 0.03
        size = engine._kelly_size(edge=0.03, price=1.04, max_size=200.0, bankroll=1000.0)
        assert size > cfg.min_order_size, (
            "Kelly sizing must not return min_order_size for bundle short (price > 1)"
        )

    def test_zero_edge_returns_min(self):
        """Zero or negative edge falls back to min_order_size."""
        cfg = make_arb_config(min_order_size=2.0)
        engine = ArbEngine(cfg)
        assert engine._kelly_size(edge=0.0, price=0.90, max_size=200.0, bankroll=1000.0) == 2.0
        assert engine._kelly_size(edge=-0.01, price=0.90, max_size=200.0, bankroll=1000.0) == 2.0

    def test_capped_at_max_order_size(self):
        """Huge bankroll + large edge still cannot exceed max_order_size."""
        cfg = make_arb_config(min_order_size=1.0, max_order_size=50.0)
        engine = ArbEngine(cfg)
        size = engine._kelly_size(edge=0.20, price=0.80, max_size=10000.0, bankroll=100_000.0)
        assert size <= 50.0

    def test_capped_at_20pct_bankroll(self):
        """Kelly fraction is capped at 20% of bankroll internally."""
        cfg = make_arb_config(min_order_size=1.0, max_order_size=100_000.0)
        engine = ArbEngine(cfg)
        size = engine._kelly_size(edge=0.50, price=0.50, max_size=100_000.0, bankroll=1000.0)
        assert size <= 1000.0 * 0.20

    def test_capped_at_liquidity(self):
        """Result cannot exceed max_size (available liquidity)."""
        cfg = make_arb_config(min_order_size=1.0, max_order_size=500.0)
        engine = ArbEngine(cfg)
        size = engine._kelly_size(edge=0.10, price=0.80, max_size=5.0, bankroll=10_000.0)
        assert size <= 5.0


# ---------------------------------------------------------------------------
# Liquidity depth gate
# ---------------------------------------------------------------------------

class TestLiquidityGate:
    def test_signal_blocked_when_depth_too_low(self):
        """Signals are suppressed when either ask side has < min_order_size*2 depth."""
        # min_order_size=5, so gate requires >= 10 on each side
        cfg = make_arb_config(min_order_size=5.0, taker_fee_bps=0, gas_cost_per_order=0)
        engine = ArbEngine(cfg)
        # Large edge but only 3 shares available — below gate
        ob = make_order_book("mkt", yes_bid=0.40, yes_ask=0.42, no_bid=0.40, no_ask=0.42, size=3.0)
        state = make_market_state(ob)
        signals = engine.analyze(state, bankroll=1000.0)
        bundle = [s for s in signals if s.opportunity and s.opportunity.is_bundle_arb]
        assert len(bundle) == 0

    def test_signal_allowed_when_depth_sufficient(self):
        """Signals are emitted when both ask sides have >= min_order_size*2 depth."""
        cfg = make_arb_config(min_order_size=5.0, taker_fee_bps=0, gas_cost_per_order=0)
        engine = ArbEngine(cfg)
        # Edge: total_ask = 0.84 → gross edge 0.16, depth = 100
        ob = make_order_book("mkt", yes_bid=0.40, yes_ask=0.42, no_bid=0.40, no_ask=0.42, size=100.0)
        state = make_market_state(ob)
        signals = engine.analyze(state, bankroll=1000.0)
        bundle = [s for s in signals if s.opportunity and s.opportunity.is_bundle_arb]
        assert len(bundle) == 1


# ---------------------------------------------------------------------------
# Bundle completion tracking (Portfolio)
# ---------------------------------------------------------------------------

class TestBundleCompletion:
    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_balance=1000.0)

    def test_pending_when_no_fills(self, portfolio):
        orders = [make_order("o1"), make_order("o2")]
        portfolio.register_bundle_signal("sig1", orders)
        assert portfolio.check_bundle_completion("sig1") == "pending"

    def test_partial_when_one_leg_filled(self, portfolio):
        o1 = make_order("o1", OrderStatus.FILLED)
        o2 = make_order("o2", OrderStatus.PENDING)
        portfolio.register_bundle_signal("sig1", [o1, o2])
        assert portfolio.check_bundle_completion("sig1") == "partial"

    def test_complete_when_all_legs_filled(self, portfolio):
        o1 = make_order("o1", OrderStatus.FILLED)
        o2 = make_order("o2", OrderStatus.FILLED)
        portfolio.register_bundle_signal("sig1", [o1, o2])
        assert portfolio.check_bundle_completion("sig1") == "complete"

    def test_complete_prunes_entry(self, portfolio):
        """Completed signals should be removed from _bundle_signals to avoid growth."""
        o1 = make_order("o1", OrderStatus.FILLED)
        o2 = make_order("o2", OrderStatus.FILLED)
        portfolio.register_bundle_signal("sig1", [o1, o2])
        portfolio.check_bundle_completion("sig1")
        # Second call: entry pruned, returns "pending"
        assert portfolio.check_bundle_completion("sig1") == "pending"

    def test_unknown_signal_returns_pending(self, portfolio):
        assert portfolio.check_bundle_completion("nonexistent") == "pending"


# ---------------------------------------------------------------------------
# _validate_arb_still_live
# ---------------------------------------------------------------------------

class TestValidateArbStillLive:
    """Verify staleness check handles both bundle long and bundle short directions."""

    def _make_execution_engine(self, mock_book: OrderBook):
        """Build a minimal ExecutionEngine with a mocked client."""
        from core.execution import ExecutionEngine, ExecutionConfig
        from core.risk_manager import RiskManager, RiskConfig

        client = MagicMock()
        client.get_orderbook = AsyncMock(return_value=mock_book)

        risk_mgr = RiskManager(RiskConfig())
        portfolio = Portfolio(initial_balance=1000.0)
        return ExecutionEngine(
            client=client,
            risk_manager=risk_mgr,
            portfolio=portfolio,
            config=ExecutionConfig(dry_run=True),
        )

    def _make_signal(self, opp_type: OpportunityType, age_ms: float = 0) -> Signal:
        detected = datetime.utcnow() - timedelta(milliseconds=age_ms)
        opp = Opportunity(
            opportunity_id="test_opp",
            opportunity_type=opp_type,
            market_id="test_market",
            edge=0.04,
            detected_at=detected,
        )
        return Signal(
            signal_id="sig_test",
            action="place_orders",
            market_id="test_market",
            opportunity=opp,
            orders=[],
        )

    def test_bundle_long_still_live(self):
        """Bundle long accepted when total_ask gap >= 0.02."""
        # total_ask = 0.90 → gap = 0.10 ≥ 0.02
        ob = make_order_book("test_market", 0.42, 0.45, 0.42, 0.45)
        engine = self._make_execution_engine(ob)
        signal = self._make_signal(OpportunityType.BUNDLE_LONG)
        result = asyncio.get_event_loop().run_until_complete(
            engine._validate_arb_still_live(signal)
        )
        assert result is True

    def test_bundle_long_closed(self):
        """Bundle long rejected when total_ask gap < 0.02."""
        # total_ask = 0.99 → gap = 0.01 < 0.02
        ob = make_order_book("test_market", 0.49, 0.50, 0.48, 0.49)
        engine = self._make_execution_engine(ob)
        signal = self._make_signal(OpportunityType.BUNDLE_LONG)
        result = asyncio.get_event_loop().run_until_complete(
            engine._validate_arb_still_live(signal)
        )
        assert result is False

    def test_bundle_short_still_live(self):
        """Bundle short accepted when total_bid gap >= 0.02."""
        # total_bid = 1.06 → gap = 0.06 ≥ 0.02
        ob = make_order_book("test_market", 0.55, 0.57, 0.51, 0.53)
        engine = self._make_execution_engine(ob)
        signal = self._make_signal(OpportunityType.BUNDLE_SHORT)
        result = asyncio.get_event_loop().run_until_complete(
            engine._validate_arb_still_live(signal)
        )
        assert result is True

    def test_bundle_short_closed(self):
        """Bundle short rejected when total_bid gap < 0.02 (using BID, not ASK)."""
        # total_bid = 1.01 → gap = 0.01 < 0.02
        ob = make_order_book("test_market", 0.51, 0.53, 0.50, 0.52)
        engine = self._make_execution_engine(ob)
        signal = self._make_signal(OpportunityType.BUNDLE_SHORT)
        result = asyncio.get_event_loop().run_until_complete(
            engine._validate_arb_still_live(signal)
        )
        assert result is False

    def test_stale_signal_rejected(self):
        """Signals older than 5 s are rejected without fetching the book."""
        ob = make_order_book("test_market", 0.42, 0.45, 0.42, 0.45)
        engine = self._make_execution_engine(ob)
        signal = self._make_signal(OpportunityType.BUNDLE_LONG, age_ms=6000)
        result = asyncio.get_event_loop().run_until_complete(
            engine._validate_arb_still_live(signal)
        )
        assert result is False
        # Book should NOT have been fetched for an expired signal
        engine.client.get_orderbook.assert_not_called()


# ---------------------------------------------------------------------------
# _place_orders_concurrent — partial failure rollback
# ---------------------------------------------------------------------------

class TestPlaceOrdersConcurrent:
    """Verify that a partial placement failure triggers cancellation of placed legs."""

    def _make_engine_with_mock_client(self):
        from core.execution import ExecutionEngine, ExecutionConfig
        from core.risk_manager import RiskManager, RiskConfig

        client = MagicMock()
        risk_mgr = MagicMock()
        risk_mgr.check_order.return_value = True
        portfolio = Portfolio(initial_balance=1000.0)
        cfg = ExecutionConfig(dry_run=True, enable_slippage_check=False)
        engine = ExecutionEngine(
            client=client, risk_manager=risk_mgr, portfolio=portfolio, config=cfg
        )
        return engine, client

    def test_all_legs_placed_registers_bundle_signal(self):
        """When all legs succeed, register_bundle_signal is called."""
        engine, client = self._make_engine_with_mock_client()

        order_a = make_order("ord_a", OrderStatus.OPEN)
        order_b = make_order("ord_b", OrderStatus.OPEN)
        client.place_order = AsyncMock(side_effect=[order_a, order_b])

        signal = Signal(
            signal_id="sig_xyz",
            action="place_orders",
            market_id="test_market",
            opportunity=MagicMock(is_bundle_arb=True),
            orders=[
                {"token_type": TokenType.YES, "side": OrderSide.BUY,
                 "price": 0.45, "size": 5.0, "strategy_tag": "bundle_arb"},
                {"token_type": TokenType.NO, "side": OrderSide.BUY,
                 "price": 0.45, "size": 5.0, "strategy_tag": "bundle_arb"},
            ],
        )
        asyncio.get_event_loop().run_until_complete(
            engine._place_orders_concurrent(signal)
        )

        assert "sig_xyz" in engine.portfolio._bundle_signals
        assert engine._order_signal_map.get("ord_a") == "sig_xyz"
        assert engine._order_signal_map.get("ord_b") == "sig_xyz"

    def test_partial_failure_cancels_placed_leg(self):
        """When one leg fails, the placed leg is cancelled."""
        engine, client = self._make_engine_with_mock_client()

        placed_order = make_order("ord_a", OrderStatus.OPEN)
        client.place_order = AsyncMock(
            side_effect=[placed_order, Exception("rejected")]
        )
        client.cancel_order = AsyncMock()

        signal = Signal(
            signal_id="sig_fail",
            action="place_orders",
            market_id="test_market",
            opportunity=MagicMock(is_bundle_arb=True),
            orders=[
                {"token_type": TokenType.YES, "side": OrderSide.BUY,
                 "price": 0.45, "size": 5.0, "strategy_tag": "bundle_arb"},
                {"token_type": TokenType.NO, "side": OrderSide.BUY,
                 "price": 0.45, "size": 5.0, "strategy_tag": "bundle_arb"},
            ],
        )
        asyncio.get_event_loop().run_until_complete(
            engine._place_orders_concurrent(signal)
        )

        # The placed leg must have been cancelled
        client.cancel_order.assert_called_once_with("ord_a")
        # No bundle signal registered for a failed placement
        assert "sig_fail" not in engine.portfolio._bundle_signals
