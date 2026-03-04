"""
Tests for the Risk Manager
"""

import pytest

from polymarket_client.models import Order, OrderSide, OrderStatus, TokenType, Trade
from core.risk_manager import RiskManager, RiskConfig


@pytest.fixture
def risk_config() -> RiskConfig:
    """Default risk configuration for tests."""
    return RiskConfig(
        max_position_per_market=200.0,
        max_global_exposure=1000.0,
        max_daily_loss=100.0,
        max_drawdown_pct=0.10,
        trade_only_high_volume=True,
        min_24h_volume=10000.0,
        whitelist=[],
        blacklist=["blocked_market"],
        kill_switch_enabled=True,
    )


@pytest.fixture
def risk_manager(risk_config: RiskConfig) -> RiskManager:
    """Create risk manager for tests."""
    rm = RiskManager(risk_config)
    # Set some market volumes
    rm.set_market_volumes({
        "test_market": 50000.0,
        "low_volume_market": 1000.0,
    })
    return rm


def create_order(
    market_id: str = "test_market",
    side: OrderSide = OrderSide.BUY,
    price: float = 0.50,
    size: float = 100.0,
) -> Order:
    """Helper to create test orders."""
    return Order(
        order_id="test_order",
        market_id=market_id,
        token_type=TokenType.YES,
        side=side,
        price=price,
        size=size,
        status=OrderStatus.PENDING,
    )


class TestOrderValidation:
    """Tests for order validation."""
    
    def test_valid_order_passes(self, risk_manager: RiskManager):
        """Test that valid orders pass risk checks."""
        order = create_order(size=100.0, price=0.50)  # $50 notional
        assert risk_manager.check_order(order) is True
    
    def test_reject_blacklisted_market(self, risk_manager: RiskManager):
        """Test rejection of blacklisted markets."""
        order = create_order(market_id="blocked_market")
        assert risk_manager.check_order(order) is False
    
    def test_reject_low_volume_market(self, risk_manager: RiskManager):
        """Test rejection of low volume markets."""
        order = create_order(market_id="low_volume_market")
        assert risk_manager.check_order(order) is False
    
    def test_reject_exceeds_market_limit(self, risk_manager: RiskManager):
        """Test rejection when exceeding per-market limit."""
        # Add existing position
        risk_manager.update_position("test_market", TokenType.YES, 350, 0.50)
        
        # Try to add more - would exceed $200 limit
        order = create_order(size=100.0, price=0.50)  # Additional $50
        assert risk_manager.check_order(order) is False
    
    def test_reject_exceeds_global_limit(self, risk_manager: RiskManager):
        """Test rejection when exceeding global limit."""
        # Add positions to reach near limit
        risk_manager.update_position("market_1", TokenType.YES, 800, 1.0)  # $800
        risk_manager.update_position("market_2", TokenType.YES, 150, 1.0)  # $150
        
        # Try to add more - would exceed $1000 global limit
        order = create_order(size=200.0, price=0.50)  # Additional $100
        assert risk_manager.check_order(order) is False


class TestKillSwitch:
    """Tests for kill switch functionality."""
    
    def test_kill_switch_on_daily_loss(self, risk_manager: RiskManager):
        """Test kill switch triggers on daily loss limit."""
        # Simulate loss exceeding limit
        risk_manager.update_pnl(-150.0, 0.0)  # $150 loss > $100 limit
        
        order = create_order()
        assert risk_manager.check_order(order) is False
        assert risk_manager.state.kill_switch_triggered is True
    
    def test_kill_switch_on_drawdown(self, risk_manager: RiskManager):
        """Test kill switch triggers on drawdown limit."""
        # Simulate profit then loss
        risk_manager.update_pnl(1000.0, 0.0)  # Peak at $1000
        risk_manager.update_pnl(800.0, 0.0)   # Now at $800 = 20% drawdown
        
        order = create_order()
        assert risk_manager.check_order(order) is False
        assert risk_manager.state.kill_switch_triggered is True
    
    def test_kill_switch_reset(self, risk_manager: RiskManager):
        """Test kill switch can be reset."""
        risk_manager.update_pnl(-150.0, 0.0)
        assert risk_manager.state.kill_switch_triggered is True
        
        risk_manager.reset_kill_switch()
        assert risk_manager.state.kill_switch_triggered is False


class TestExposureTracking:
    """Tests for exposure tracking."""
    
    def test_market_exposure_tracking(self, risk_manager: RiskManager):
        """Test per-market exposure tracking."""
        risk_manager.update_position("market_1", TokenType.YES, 100, 0.50)
        
        assert risk_manager.get_market_exposure("market_1") == 50.0
        assert risk_manager.get_market_exposure("market_2") == 0.0
    
    def test_global_exposure_tracking(self, risk_manager: RiskManager):
        """Test global exposure tracking."""
        risk_manager.update_position("market_1", TokenType.YES, 100, 0.50)
        risk_manager.update_position("market_2", TokenType.NO, 200, 0.25)
        
        assert risk_manager.state.global_exposure == 100.0  # 50 + 50
    
    def test_available_exposure(self, risk_manager: RiskManager):
        """Test available exposure calculation."""
        risk_manager.update_position("test_market", TokenType.YES, 100, 1.0)
        
        available = risk_manager.get_available_exposure("test_market")
        assert available == 100.0  # 200 limit - 100 used


class TestRiskSummary:
    """Tests for risk summary functionality."""
    
    def test_summary_structure(self, risk_manager: RiskManager):
        """Test risk summary contains expected fields."""
        summary = risk_manager.get_summary()
        
        expected_keys = [
            "global_exposure",
            "max_global_exposure",
            "utilization_pct",
            "daily_pnl",
            "max_daily_loss",
            "kill_switch_triggered",
            "within_limits",
        ]
        
        for key in expected_keys:
            assert key in summary
    
    def test_within_limits_check(self, risk_manager: RiskManager):
        """Test within_global_limits check."""
        assert risk_manager.within_global_limits() is True
        
        # Trigger kill switch
        risk_manager.update_pnl(-150.0, 0.0)
        assert risk_manager.within_global_limits() is False


class TestHedgedOrderExemptions:
    """Hedged orders (bundle_arb / multileg_arb) bypass exposure limits."""

    def _make_hedged_order(
        self,
        market_id: str = "test_market",
        price: float = 0.55,
        size: float = 100.0,
        strategy_tag: str = "bundle_arb",
        token_type: TokenType = TokenType.YES,
    ) -> Order:
        return Order(
            order_id="hedged_order",
            market_id=market_id,
            token_type=token_type,
            side=OrderSide.BUY,
            price=price,
            size=size,
            status=OrderStatus.PENDING,
            strategy_tag=strategy_tag,
        )

    def test_bundle_arb_yes_leg_passes_when_at_market_limit(
        self, risk_manager: RiskManager
    ):
        """YES leg of a bundle arb is allowed even when market is at its limit."""
        # Push market exposure to exactly the limit via an unhedged fill
        risk_manager.update_position("test_market", TokenType.YES, 400, 0.50)
        assert risk_manager._market_exposure["test_market"] == pytest.approx(200.0)

        yes_order = self._make_hedged_order(strategy_tag="bundle_arb")
        assert risk_manager.check_order(yes_order) is True

    def test_bundle_arb_no_leg_passes_when_yes_leg_filled(
        self, risk_manager: RiskManager
    ):
        """NO leg of a bundle arb is allowed even after YES leg consumed half the budget.

        Without the hedged-order exemption the NO leg would be rejected because
        YES (0.55 * 100 = $55) + NO (0.45 * 100 = $45) = $100 notional, which
        pushes the per-market projection above zero for a low-limit config, and
        for larger sizes can exceed max_position_per_market.
        """
        # Simulate YES leg fill adding to per-market exposure
        risk_manager.update_position("test_market", TokenType.YES, 100, 0.55)
        assert risk_manager._market_exposure["test_market"] == pytest.approx(55.0)

        no_order = self._make_hedged_order(
            price=0.45, token_type=TokenType.NO, strategy_tag="bundle_arb"
        )
        assert risk_manager.check_order(no_order) is True

    def test_bundle_arb_not_blocked_by_global_limit(self, risk_manager: RiskManager):
        """Global limit does not block hedged orders."""
        # Exhaust global budget via unhedged fills
        risk_manager.state.global_exposure = risk_manager.config.max_global_exposure

        order = self._make_hedged_order(strategy_tag="bundle_arb")
        assert risk_manager.check_order(order) is True

    def test_multileg_arb_not_blocked_by_global_limit(
        self, risk_manager: RiskManager
    ):
        """multileg_arb orders are also exempt from global exposure limit."""
        risk_manager.state.global_exposure = risk_manager.config.max_global_exposure

        order = self._make_hedged_order(strategy_tag="multileg_arb")
        assert risk_manager.check_order(order) is True

    def test_unhedged_order_still_blocked_by_market_limit(
        self, risk_manager: RiskManager
    ):
        """Non-hedged orders are still subject to per-market limits."""
        risk_manager.update_position("test_market", TokenType.YES, 400, 0.50)

        order = create_order(size=100.0, price=0.50)  # no strategy_tag → unhedged
        assert risk_manager.check_order(order) is False


class TestBlacklistManagement:
    """Tests for blacklist management."""
    
    def test_add_to_blacklist(self, risk_manager: RiskManager):
        """Test adding market to blacklist."""
        risk_manager.add_to_blacklist("new_blocked_market")
        
        order = create_order(market_id="new_blocked_market")
        assert risk_manager.check_order(order) is False
    
    def test_remove_from_blacklist(self, risk_manager: RiskManager):
        """Test removing market from blacklist."""
        risk_manager.remove_from_blacklist("blocked_market")
        risk_manager.set_market_volumes({"blocked_market": 50000.0})
        
        order = create_order(market_id="blocked_market")
        assert risk_manager.check_order(order) is True

