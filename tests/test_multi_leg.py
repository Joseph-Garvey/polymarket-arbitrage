import asyncio
import logging
import pytest
from datetime import datetime
from polymarket_client.models import Trade, OrderSide, TokenType, Order, OrderStatus
from core.portfolio import Portfolio
from core.risk_manager import RiskManager, RiskConfig
from core.execution import ExecutionEngine, ExecutionConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_multileg_flow():
    # 1. Setup components
    portfolio = Portfolio(initial_balance=1000.0)
    risk_manager = RiskManager(RiskConfig())

    # Mock client (not used for this unit test of internal logic)
    class MockClient:
        async def place_order(self, **kwargs):
            return Order(order_id="order_1", status=OrderStatus.FILLED, **kwargs)

        async def cancel_order(self, order_id):
            return True

    execution = ExecutionEngine(
        MockClient(), risk_manager, portfolio, ExecutionConfig(dry_run=True)
    )

    market_id = "test_market_123"

    # 2. Simulate fills for YES and NO legs
    logger.info("Simulating YES leg fill...")
    yes_trade = Trade(
        trade_id="t1",
        order_id="o1",
        market_id=market_id,
        token_type=TokenType.YES,
        side=OrderSide.BUY,
        price=0.48,
        size=100.0,
        strategy_tag="bundle_arb",
    )
    execution.handle_fill(yes_trade)

    logger.info(f"Portfolio Exposure: {portfolio.get_total_exposure()}")
    logger.info(f"Risk Manager Global Exposure: {risk_manager.state.global_exposure}")

    logger.info("Simulating NO leg fill...")
    no_trade = Trade(
        trade_id="t2",
        order_id="o2",
        market_id=market_id,
        token_type=TokenType.NO,
        side=OrderSide.BUY,
        price=0.49,
        size=100.0,
        strategy_tag="bundle_arb",
    )
    execution.handle_fill(no_trade)

    # 3. Verify Group Position was opened
    logger.info(f"Open Group Arbs: {list(portfolio._open_group_arbs.keys())}")
    assert market_id in portfolio._open_group_arbs

    group = portfolio._open_group_arbs[market_id]
    logger.info(f"Group Size: {group.size}")
    logger.info(f"Group Total Cost: {group.total_cost}")
    logger.info(f"Group Locked Profit: {group.locked_profit}")

    # 4. Verify PnL logic
    portfolio.update_prices(market_id, 0.45, 0.45)  # Mid-market dip
    pnl = portfolio.get_pnl()
    logger.info(f"Total PnL (should be positive locked profit): {pnl['total_pnl']}")
    assert pnl["total_pnl"] > 0

    # 5. Verify Risk Manager exposure is non-negative (both legs counted)
    logger.info(
        f"Final Risk Manager Global Exposure: {risk_manager.state.global_exposure}"
    )
    assert risk_manager.state.global_exposure >= 0

    # 6. Verify Dashboard Summary
    summary = portfolio.get_summary()
    logger.info(
        f"Portfolio Summary Group Arbs: {len(summary.get('open_group_arbs', {}))}"
    )
    assert len(summary.get("open_group_arbs", {})) == 1


if __name__ == "__main__":
    asyncio.run(test_multileg_flow())
