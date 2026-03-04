"""
Portfolio Module
=================

Tracks inventory, positions, and PnL across all markets.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from polymarket_client.models import (
    Order,
    OrderSide,
    OrderStatus,
    Position,
    TokenType,
    Trade,
)


logger = logging.getLogger(__name__)


@dataclass
class GroupArbLeg:
    """Single leg of a grouped arbitrage position."""

    market_id: str
    token_type: TokenType
    entry_price: float
    size: float


@dataclass
class GroupArbPosition:
    """
    Tracks a multi-leg arbitrage group as a single unit.

    Supports both 2-leg (YES+NO) and N-leg (Categorical) arbs.
    """

    group_id: str
    legs: list[GroupArbLeg]
    size: float
    total_cost: float  # Sum of entry_prices (< 1.0 for a profitable long)
    locked_profit: float  # 1.0 - total_cost for fully-filled arbs; 0.0 for partial multileg placeholders
    opened_at: datetime
    status: str = "open"  # open, resolving, closed

    @property
    def unrealized_pnl(self) -> float:
        """Profit is locked at entry — return it unconditionally."""
        return self.locked_profit * self.size


@dataclass
class PortfolioPosition:
    """Extended position tracking with PnL."""

    market_id: str
    token_type: TokenType
    size: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    cost_basis: float = 0.0

    # Trade history
    total_bought: float = 0.0
    total_sold: float = 0.0
    trade_count: int = 0

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self.size == 0:
            return 0.0
        return self.size * (current_price - self.avg_entry_price)

    def total_pnl(self, current_price: float) -> float:
        """Calculate total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl(current_price)

    @property
    def notional(self) -> float:
        """Current position notional value."""
        return abs(self.size) * self.avg_entry_price


@dataclass
class PortfolioStats:
    """Portfolio-level statistics."""

    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_fees_paid: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_volume: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def win_rate(self) -> float:
        """
        Individual-fill win rate.

        NOTE: For bundle arb this will appear near 0% because each leg is
        "sold below cost" individually until resolution.  Use Portfolio.arb_win_rate
        instead for strategy-level performance measurement.
        """
        if self.winning_trades + self.losing_trades == 0:
            return 0.0
        return self.winning_trades / (self.winning_trades + self.losing_trades)


class Portfolio:
    """
    Portfolio and inventory tracking.

    Maintains positions per market/token and calculates PnL.
    """

    def __init__(self, initial_balance: float = 0.0):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance

        # Positions: market_id -> token_type -> PortfolioPosition
        self._positions: dict[str, dict[TokenType, PortfolioPosition]] = {}

        # Group arb tracking (open and closed)
        self._open_group_arbs: dict[str, GroupArbPosition] = {}
        self._closed_group_arbs: list[GroupArbPosition] = []

        # Trade history
        self._trades: list[Trade] = []

        # Stats
        self.stats = PortfolioStats()

        # Current prices for unrealized PnL calculation
        self._current_prices: dict[str, dict[TokenType, float]] = {}

        # Bundle signal tracking: signal_id -> list of placed Orders
        self._bundle_signals: dict[str, list[Order]] = {}

        logger.info(f"Portfolio initialized with balance: {initial_balance}")

    def update_from_fill(self, trade: Trade) -> None:
        """Update portfolio from a trade fill."""
        market_id = trade.market_id
        token_type = trade.token_type

        # Ensure position exists
        if market_id not in self._positions:
            self._positions[market_id] = {}

        if token_type not in self._positions[market_id]:
            self._positions[market_id][token_type] = PortfolioPosition(
                market_id=market_id,
                token_type=token_type,
            )

        position = self._positions[market_id][token_type]

        # Process based on side
        if trade.side == OrderSide.BUY:
            self._process_buy(position, trade)
        else:
            self._process_sell(position, trade)

        # Update trade count
        position.trade_count += 1

        # Update cash (simplified)
        if trade.side == OrderSide.BUY:
            self.cash_balance -= trade.net_cost
        else:
            self.cash_balance += trade.notional - trade.fee

        # Track trade
        self._trades.append(trade)
        self.stats.total_trades += 1
        self.stats.total_fees_paid += trade.fee
        self.stats.total_volume += trade.notional

        logger.debug(
            f"Portfolio updated: {market_id}/{token_type.value} | "
            f"size={position.size:.4f} @ avg={position.avg_entry_price:.4f}"
        )

    def _process_buy(self, position: PortfolioPosition, trade: Trade) -> None:
        """Process a buy trade."""
        new_size = position.size + trade.size

        if position.size >= 0:
            # Adding to long position
            total_cost = (position.avg_entry_price * position.size) + (
                trade.price * trade.size
            )
            position.avg_entry_price = total_cost / new_size if new_size > 0 else 0
            position.cost_basis += trade.net_cost
        else:
            # Covering short position
            if trade.size <= abs(position.size):
                # Partial cover
                realized = (position.avg_entry_price - trade.price) * trade.size
                position.realized_pnl += realized
                self.stats.total_realized_pnl += realized

                if realized > 0:
                    self.stats.winning_trades += 1
                else:
                    self.stats.losing_trades += 1
            else:
                # Full cover + go long
                short_size = abs(position.size)
                realized = (position.avg_entry_price - trade.price) * short_size
                position.realized_pnl += realized
                self.stats.total_realized_pnl += realized

                # New long portion
                long_size = trade.size - short_size
                position.avg_entry_price = trade.price
                position.cost_basis = long_size * trade.price

                if realized > 0:
                    self.stats.winning_trades += 1
                else:
                    self.stats.losing_trades += 1

        position.size = new_size
        position.total_bought += trade.size

    def _process_sell(self, position: PortfolioPosition, trade: Trade) -> None:
        """Process a sell trade."""
        new_size = position.size - trade.size

        if position.size > 0:
            # Reducing long position
            if trade.size <= position.size:
                # Partial sell
                realized = (trade.price - position.avg_entry_price) * trade.size
                position.realized_pnl += realized
                self.stats.total_realized_pnl += realized

                if realized > 0:
                    self.stats.winning_trades += 1
                else:
                    self.stats.losing_trades += 1
            else:
                # Full sell + go short
                long_size = position.size
                realized = (trade.price - position.avg_entry_price) * long_size
                position.realized_pnl += realized
                self.stats.total_realized_pnl += realized

                # New short portion
                short_size = trade.size - long_size
                position.avg_entry_price = trade.price
                position.cost_basis = short_size * trade.price

                if realized > 0:
                    self.stats.winning_trades += 1
                else:
                    self.stats.losing_trades += 1
        else:
            # Adding to short position
            total_value = (position.avg_entry_price * abs(position.size)) + (
                trade.price * trade.size
            )
            new_short_size = abs(new_size)
            position.avg_entry_price = (
                total_value / new_short_size if new_short_size > 0 else 0
            )
            position.cost_basis += trade.notional

        position.size = new_size
        position.total_sold += trade.size

    def update_prices(self, market_id: str, yes_price: float, no_price: float) -> None:
        """Update current prices for unrealized PnL calculation."""
        if market_id not in self._current_prices:
            self._current_prices[market_id] = {}

        self._current_prices[market_id][TokenType.YES] = yes_price
        self._current_prices[market_id][TokenType.NO] = no_price

        # Recalculate unrealized PnL
        self._recalculate_unrealized_pnl()

    def _recalculate_unrealized_pnl(self) -> None:
        """Recalculate total unrealized PnL.

        For individual legs of a bundle arb, mid-market unrealised PnL will look
        negative until resolution (both legs cost ~$0.48 and both mark-to-market
        below that until one resolves to $1).  Open arb pairs instead contribute
        their locked_profit directly, which is the correct economic value.
        """
        total = 0.0

        # Markets that belong to an open group arb — skip their individual legs
        arb_market_ids = set()
        for group in self._open_group_arbs.values():
            for leg in group.legs:
                arb_market_ids.add((leg.market_id, leg.token_type))

        for market_id, tokens in self._positions.items():
            if market_id not in self._current_prices:
                continue
            for token_type, position in tokens.items():
                if (market_id, token_type) in arb_market_ids:
                    continue  # Handled below via locked_profit
                if token_type in self._current_prices[market_id]:
                    current_price = self._current_prices[market_id][token_type]
                    total += position.unrealized_pnl(current_price)

        # Add locked profit from open group arbs
        for group in self._open_group_arbs.values():
            total += group.unrealized_pnl

        self.stats.total_unrealized_pnl = total

    def get_position(
        self, market_id: str, token_type: TokenType
    ) -> Optional[PortfolioPosition]:
        """Get a specific position."""
        if market_id not in self._positions:
            return None
        return self._positions[market_id].get(token_type)

    def get_exposure(self, market_id: str) -> dict:
        """Get exposure breakdown for a market."""
        if market_id not in self._positions:
            return {
                "yes_size": 0.0,
                "no_size": 0.0,
                "yes_notional": 0.0,
                "no_notional": 0.0,
                "total_notional": 0.0,
                "net_position": 0.0,
            }

        yes_pos = self._positions[market_id].get(TokenType.YES)
        no_pos = self._positions[market_id].get(TokenType.NO)

        yes_size = yes_pos.size if yes_pos else 0.0
        no_size = no_pos.size if no_pos else 0.0
        yes_notional = yes_pos.notional if yes_pos else 0.0
        no_notional = no_pos.notional if no_pos else 0.0

        return {
            "yes_size": yes_size,
            "no_size": no_size,
            "yes_notional": yes_notional,
            "no_notional": no_notional,
            "total_notional": yes_notional + no_notional,
            "net_position": yes_size - no_size,
        }

    def get_total_exposure(self) -> float:
        """Get total notional exposure across all markets."""
        total = 0.0
        for tokens in self._positions.values():
            for position in tokens.values():
                total += position.notional
        return total

    def get_pnl(self) -> dict:
        """Get PnL breakdown."""
        return {
            "realized_pnl": self.stats.total_realized_pnl,
            "unrealized_pnl": self.stats.total_unrealized_pnl,
            "total_pnl": self.stats.total_pnl,
            "fees_paid": self.stats.total_fees_paid,
            "net_pnl": self.stats.total_pnl - self.stats.total_fees_paid,
        }

    @property
    def arb_win_rate(self) -> float:
        """Percentage of completed group arbs that resolved profitably."""
        if not self._closed_group_arbs:
            return 0.0
        winners = sum(1 for p in self._closed_group_arbs if p.locked_profit > 0)
        return winners / len(self._closed_group_arbs)

    def open_group_position(
        self,
        group_id: str,
        legs: list[GroupArbLeg],
        size: float,
        locked_profit: Optional[float] = None,
        fees_paid: float = 0.0,
    ) -> GroupArbPosition:
        """
        Register a new multi-leg arb group as fully executed.

        If `locked_profit` is explicitly provided, it is used verbatim (e.g. for partial
        multileg registration used only as a re-entry guard, where profit is
        not yet locked because not all legs have filled).  When omitted it is
        computed as ``1.0 - total_cost - fees_paid``.
        """
        total_cost = sum(leg.entry_price for leg in legs)
        if locked_profit is None:
            locked_profit = 1.0 - total_cost - fees_paid
            if locked_profit <= 0:
                logger.warning(
                    f"Group arb opened with locked_profit={locked_profit:.4f} (not profitable): "
                    f"{group_id} | total_cost={total_cost:.4f} | fees={fees_paid:.4f}"
                )

        group = GroupArbPosition(
            group_id=group_id,
            legs=legs,
            size=size,
            total_cost=total_cost,
            locked_profit=locked_profit,
            opened_at=datetime.utcnow(),
        )
        self._open_group_arbs[group_id] = group
        logger.info(
            f"Group arb opened: {group_id} | cost={total_cost:.4f} | "
            f"locked_profit={locked_profit:.4f} | size={size} | legs={len(legs)}"
        )
        return group

    def close_group_position(self, group_id: str) -> Optional[GroupArbPosition]:
        """Mark an open group arb as closed (e.g. on market resolution)."""
        group = self._open_group_arbs.pop(group_id, None)
        if group:
            group.status = "closed"
            self._closed_group_arbs.append(group)
            logger.info(
                f"Group arb closed: {group_id} | locked_profit={group.locked_profit:.4f}"
            )
        return group

    def get_summary(self) -> dict:
        """Get portfolio summary."""
        return {
            "initial_balance": self.initial_balance,
            "cash_balance": self.cash_balance,
            "total_exposure": self.get_total_exposure(),
            "pnl": self.get_pnl(),
            "total_trades": self.stats.total_trades,
            "win_rate": self.stats.win_rate,
            "arb_win_rate": self.arb_win_rate,
            "total_volume": self.stats.total_volume,
            "positions_count": sum(len(tokens) for tokens in self._positions.values()),
            "markets_traded": len(self._positions),
            "open_group_arbs": {
                gid: {
                    "group_id": group.group_id,
                    "legs": [
                        {
                            "market_id": leg.market_id,
                            "token_type": leg.token_type.value,
                            "entry_price": leg.entry_price,
                            "size": leg.size,
                        }
                        for leg in group.legs
                    ],
                    "size": group.size,
                    "total_cost": group.total_cost,
                    "locked_profit": group.locked_profit,
                    "status": group.status,
                }
                for gid, group in self._open_group_arbs.items()
            },
            "positions": {
                mid: {
                    tt.value: {
                        "size": pos.size,
                        "avg_entry_price": pos.avg_entry_price,
                        "unrealized_pnl": pos.unrealized_pnl(
                            self._current_prices.get(mid, {}).get(
                                tt, pos.avg_entry_price
                            )
                        ),
                    }
                    for tt, pos in tokens.items()
                }
                for mid, tokens in self._positions.items()
            },
        }

    def get_all_positions(self) -> dict[str, dict[TokenType, PortfolioPosition]]:
        """Get all positions."""
        return self._positions.copy()

    def get_recent_trades(self, limit: int = 50) -> list[Trade]:
        """Get recent trades."""
        return self._trades[-limit:]

    def register_bundle_signal(self, signal_id: str, orders: list[Order]) -> None:
        """Register the placed orders for a bundle arb signal for completion tracking."""
        self._bundle_signals[signal_id] = list(orders)

    def check_bundle_completion(self, signal_id: str) -> str:
        """
        Check whether all legs of a bundle arb signal have filled.

        Returns 'complete', 'partial', or 'pending'.
        A 'partial' result means at least one leg filled but not all —
        the caller should trigger an unwind to close directional exposure.
        """
        orders = self._bundle_signals.get(signal_id)
        if not orders:
            return "pending"

        expected_legs = len(orders)
        filled_legs = sum(1 for o in orders if o.status == OrderStatus.FILLED)

        if filled_legs == expected_legs:
            del self._bundle_signals[signal_id]  # Prune completed entries
            return "complete"
        if filled_legs > 0:
            return "partial"
        return "pending"

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self._positions = {}
        self._trades = []
        self._open_group_arbs = {}
        self._closed_group_arbs = []
        self.cash_balance = self.initial_balance

        self.stats = PortfolioStats()
        self._current_prices = {}
        self._bundle_signals = {}
        logger.info("Portfolio reset")
