"""
Arbitrage Engine Module
========================

Detects trading opportunities including:
1. Bundle mispricing (YES + NO != 1.0)
2. Market-making spread capture
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from polymarket_client.models import (
    MarketState,
    Opportunity,
    OpportunityType,
    OrderBook,
    OrderSide,
    Signal,
    TokenType,
)

# Avoid circular import — Portfolio is imported lazily inside ArbEngine.__init__
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.portfolio import Portfolio


logger = logging.getLogger(__name__)


@dataclass
class ArbConfig:
    """Configuration for the arbitrage engine."""

    # Bundle arbitrage
    min_edge: float = 0.01  # Minimum edge required (1%)
    bundle_arb_enabled: bool = True

    # Market-making
    min_spread: float = 0.05  # Minimum spread to MM (5c)
    mm_enabled: bool = True
    tick_size: float = 0.01

    # Sizing
    default_order_size: float = 50.0
    min_order_size: float = 5.0
    max_order_size: float = 200.0

    # Signal expiry
    signal_expiry_seconds: float = 5.0

    # Fees (in basis points - 100 bps = 1%)
    # Polymarket: ~0% maker, ~1.5% taker
    maker_fee_bps: float = 0  # Limit orders adding liquidity
    taker_fee_bps: float = 150  # Taking liquidity (1.5%)
    gas_cost_per_order: float = 0.02  # ~$0.02 on Polygon


@dataclass
class OpportunityTiming:
    """Tracks timing of a specific opportunity."""

    opportunity_id: str
    market_id: str
    opportunity_type: str
    detected_at: datetime
    edge: float
    expired_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    was_executed: bool = False

    def mark_expired(self, executed: bool = False) -> None:
        """Mark this opportunity as expired."""
        self.expired_at = datetime.utcnow()
        self.duration_ms = (self.expired_at - self.detected_at).total_seconds() * 1000
        self.was_executed = executed


@dataclass
class ArbStats:
    """Statistics for the arbitrage engine."""

    bundle_opportunities_detected: int = 0
    mm_opportunities_detected: int = 0
    signals_generated: int = 0
    last_opportunity_time: Optional[datetime] = None

    # Opportunity duration tracking
    # NOTE: these durations measure how long an arb persisted in the order book
    # AFTER our scanner detected it — NOT our execution latency.  They are a
    # competition signal: if >80% close in <100 ms, sophisticated bots are
    # filling them before us and we need WebSocket feeds.  Compare
    # avg_opportunity_duration_ms vs ExecutionStats.avg_execution_latency_ms.
    total_opportunities_tracked: int = 0
    avg_opportunity_duration_ms: float = 0.0
    min_opportunity_duration_ms: float = float("inf")
    max_opportunity_duration_ms: float = 0.0
    opportunities_under_100ms: int = 0
    opportunities_under_500ms: int = 0
    opportunities_under_1s: int = 0
    opportunities_over_1s: int = 0


class ArbEngine:
    """
    Arbitrage and market-making opportunity detection engine.

    Analyzes market states from the DataFeed and generates trading
    signals for the ExecutionEngine.
    """

    def __init__(self, config: ArbConfig, portfolio: "Portfolio | None" = None):
        self.config = config
        self._portfolio = portfolio
        self.stats = ArbStats()

        # Track recent opportunities to avoid duplicates
        self._recent_opportunities: dict[str, Opportunity] = {}
        # Cooldown used exclusively for market-making signals (not bundle arb;
        # bundle arb re-entry is gated by open-position check via self._portfolio).
        self._opportunity_cooldown: dict[str, datetime] = {}

        # Track active opportunities for duration measurement
        self._active_opportunities: dict[str, OpportunityTiming] = {}
        self._opportunity_history: list[OpportunityTiming] = []

        # Track market states by group_id for multi-leg arbs
        self._group_states: dict[str, dict[str, MarketState]] = {}

        logger.info(
            f"ArbEngine initialized with min_edge={config.min_edge}, min_spread={config.min_spread}"
        )

    def analyze(
        self, market_state: MarketState, bankroll: float = 100.0
    ) -> list[Signal]:
        """
        Analyze a market state and generate trading signals.

        Returns a list of signals (may be empty if no opportunities).
        bankroll is the current available cash, used for Kelly position sizing.
        """
        signals: list[Signal] = []

        order_book = market_state.order_book
        market_id = market_state.market.market_id
        group_id = market_state.market.group_id

        # Update group states for multi-leg arbs
        if group_id:
            if group_id not in self._group_states:
                self._group_states[group_id] = {}
            self._group_states[group_id][market_id] = market_state

            # Evict resolved/closed markets from group tracking to prevent memory leak
            if market_state.market.resolved or market_state.market.closed:
                self._group_states.pop(group_id, None)
            # Cap total group count at 500: evict the oldest entry (dicts are
            # insertion-ordered in Python 3.7+, so the first key is oldest)
            elif len(self._group_states) >= 500:
                oldest_key = next(iter(self._group_states))
                del self._group_states[oldest_key]

        # Check if previously tracked opportunities have expired
        self._check_expired_opportunities(market_id, order_book)

        # Check for bundle arbitrage
        if self.config.bundle_arb_enabled:
            bundle_signal = self._check_bundle_arbitrage(
                market_id, order_book, bankroll
            )
            if bundle_signal:
                signals.append(bundle_signal)

            # Check for multi-leg arbitrage if this market belongs to a group.
            # Skip if bundle_signal was already generated for this market — a 2-market
            # NegRisk group would otherwise emit both BUNDLE_LONG and MULTILEG_LONG for
            # the same legs, doubling the exposure.
            # Also skip if we are already tracking an active multileg opportunity for
            # this group — without this guard every WebSocket tick to any member market
            # would call _check_multileg_arbitrage(), generating hundreds of signals per
            # minute and creating signal-queue back-pressure.
            # Also skip if group_size is set and we have not yet seen all expected legs
            # — a partial group view means we cannot confirm the arbitrage is risk-free.
            if group_id and group_id in self._group_states and len(self._group_states[group_id]) > 1 and bundle_signal is None:
                group_states_now = self._group_states[group_id]
                expected_legs = market_state.market.group_size
                if expected_legs > 0 and len(group_states_now) < expected_legs:
                    pass  # Partial group — wait until all legs are observed
                else:
                    multileg_key = f"{group_id}_multileg_long"
                    if multileg_key not in self._active_opportunities:
                        multileg_signal = self._check_multileg_arbitrage(
                            group_id, group_states_now, bankroll
                        )
                        if multileg_signal:
                            signals.append(multileg_signal)

        # Check for market-making opportunities
        if self.config.mm_enabled:
            mm_signals = self._check_market_making(market_id, order_book)
            signals.extend(mm_signals)

        return signals

    def _check_expired_opportunities(
        self, market_id: str, order_book: OrderBook
    ) -> None:
        """Check if any tracked opportunities have expired (prices moved away)."""
        now = datetime.utcnow()
        expired_keys = []

        # Determine which group_ids this market_id belongs to, so we can also
        # evaluate multileg opportunities whose timing.market_id is the group_id.
        groups_for_market: set[str] = {
            gid
            for gid, members in self._group_states.items()
            if market_id in members
        }

        for key, timing in self._active_opportunities.items():
            is_multileg = "multileg_long" in timing.opportunity_type
            # For bundle entries: only check when the triggering market matches.
            # For multileg entries: check when any member market of that group fires.
            if not is_multileg and timing.market_id != market_id:
                continue
            if is_multileg and timing.market_id not in groups_for_market:
                continue

            # Check if opportunity still exists
            still_valid = False

            if "bundle_long" in timing.opportunity_type:
                # Check if total ask is still < 1 - min_edge
                if order_book.best_ask_yes and order_book.best_ask_no:
                    total_ask = order_book.best_ask_yes + order_book.best_ask_no
                    if (
                        1.0 - total_ask >= self.config.min_edge * 0.5
                    ):  # Use lower threshold
                        still_valid = True

            elif "bundle_short" in timing.opportunity_type:
                # Check if total bid is still > 1 + min_edge
                if order_book.best_bid_yes and order_book.best_bid_no:
                    total_bid = order_book.best_bid_yes + order_book.best_bid_no
                    if total_bid - 1.0 >= self.config.min_edge * 0.5:
                        still_valid = True

            elif is_multileg:
                # For multileg, timing.market_id IS the group_id.
                # The opportunity is still live if the sum of YES asks across all
                # legs in the group is still < 1.0 (using a relaxed threshold,
                # consistent with the bundle checks above).
                group_id_ml = timing.market_id
                if group_id_ml in self._group_states:
                    total_yes_ask = 0.0
                    all_valid = True
                    for state in self._group_states[group_id_ml].values():
                        ask_yes = state.order_book.best_ask_yes
                        if ask_yes is None:
                            all_valid = False
                            break
                        total_yes_ask += ask_yes
                    if all_valid and 1.0 - total_yes_ask >= self.config.min_edge * 0.5:
                        still_valid = True

            # Also expire if too old (10 seconds max)
            age_seconds = (now - timing.detected_at).total_seconds()
            if age_seconds > 10:
                still_valid = False

            if not still_valid:
                timing.mark_expired(executed=False)
                self._record_opportunity_duration(timing)
                expired_keys.append(key)

        for key in expired_keys:
            del self._active_opportunities[key]

    def _record_opportunity_duration(self, timing: OpportunityTiming) -> None:
        """Record the duration of an expired opportunity and update stats."""
        if timing.duration_ms is None:
            return

        self._opportunity_history.append(timing)

        # Keep only last 1000 opportunities
        if len(self._opportunity_history) > 1000:
            self._opportunity_history = self._opportunity_history[-500:]

        # Update stats
        self.stats.total_opportunities_tracked += 1

        # Update min/max
        if timing.duration_ms < self.stats.min_opportunity_duration_ms:
            self.stats.min_opportunity_duration_ms = timing.duration_ms
        if timing.duration_ms > self.stats.max_opportunity_duration_ms:
            self.stats.max_opportunity_duration_ms = timing.duration_ms

        # Update running average
        n = self.stats.total_opportunities_tracked
        old_avg = self.stats.avg_opportunity_duration_ms
        self.stats.avg_opportunity_duration_ms = (
            old_avg + (timing.duration_ms - old_avg) / n
        )

        # Update duration buckets
        if timing.duration_ms < 100:
            self.stats.opportunities_under_100ms += 1
        elif timing.duration_ms < 500:
            self.stats.opportunities_under_500ms += 1
        elif timing.duration_ms < 1000:
            self.stats.opportunities_under_1s += 1
        else:
            self.stats.opportunities_over_1s += 1

        logger.info(
            f"Opportunity EXPIRED: {timing.opportunity_type} | "
            f"duration={timing.duration_ms:.0f}ms | edge={timing.edge:.4f} | "
            f"market={timing.market_id}"
        )

    def _start_tracking_opportunity(self, opportunity: Opportunity) -> None:
        """Start tracking an opportunity for duration measurement."""
        key = f"{opportunity.market_id}_{opportunity.opportunity_type.value}"

        # Don't double-track
        if key in self._active_opportunities:
            return

        timing = OpportunityTiming(
            opportunity_id=opportunity.opportunity_id,
            market_id=opportunity.market_id,
            opportunity_type=opportunity.opportunity_type.value,
            detected_at=datetime.utcnow(),
            edge=opportunity.edge,
        )
        self._active_opportunities[key] = timing

    def mark_opportunity_executed(self, market_id: str, opportunity_type: str) -> None:
        """Mark an opportunity as executed (for accurate tracking)."""
        key = f"{market_id}_{opportunity_type}"
        if key in self._active_opportunities:
            timing = self._active_opportunities[key]
            timing.mark_expired(executed=True)
            self._record_opportunity_duration(timing)
            del self._active_opportunities[key]

    def get_timing_stats(self) -> dict:
        """Get opportunity timing statistics for dashboard."""
        recent_history = (
            self._opportunity_history[-100:] if self._opportunity_history else []
        )

        return {
            "total_tracked": self.stats.total_opportunities_tracked,
            "avg_duration_ms": round(self.stats.avg_opportunity_duration_ms, 1),
            "min_duration_ms": round(self.stats.min_opportunity_duration_ms, 1)
            if self.stats.min_opportunity_duration_ms != float("inf")
            else None,
            "max_duration_ms": round(self.stats.max_opportunity_duration_ms, 1),
            "under_100ms": self.stats.opportunities_under_100ms,
            "under_500ms": self.stats.opportunities_under_500ms,
            "under_1s": self.stats.opportunities_under_1s,
            "over_1s": self.stats.opportunities_over_1s,
            "active_opportunities": len(self._active_opportunities),
            "recent_durations": [
                {
                    "type": t.opportunity_type,
                    "duration_ms": round(t.duration_ms, 1) if t.duration_ms else 0,
                    "edge": round(t.edge, 4),
                    "executed": t.was_executed,
                    "time": t.detected_at.isoformat(),
                }
                for t in recent_history[-20:]
            ],
        }

    def _kelly_size(
        self,
        edge: float,
        price: float,
        max_size: float,
        bankroll: float,
        fraction: float = 0.25,
    ) -> float:
        """
        Kelly criterion position sizing (quarter-Kelly by default).

        For binary arb, p(win) ≈ 1 but execution risk means we apply a fraction.
        f* ≈ edge / price, then scale by fraction and bankroll.
        Capped at 20% of bankroll per trade.
        """
        if edge <= 0 or price <= 0:
            return self.config.min_order_size

        kelly_fraction = (edge / price) * fraction
        kelly_fraction = max(0.0, min(kelly_fraction, 0.20))

        suggested = bankroll * kelly_fraction
        return max(
            self.config.min_order_size,
            min(suggested, max_size, self.config.max_order_size),
        )

    def _check_bundle_arbitrage(
        self, market_id: str, order_book: OrderBook, bankroll: float = 100.0
    ) -> Optional[Signal]:
        """
        Check for bundle mispricing opportunities.

        Bundle Long: Buy YES + NO when total_ask < 1 - min_edge - fees
        Bundle Short: Sell YES + NO when total_bid > 1 + min_edge + fees

        Fees are factored in to ensure net profitability!
        """
        # Get prices
        best_ask_yes = order_book.best_ask_yes
        best_ask_no = order_book.best_ask_no
        best_bid_yes = order_book.best_bid_yes
        best_bid_no = order_book.best_bid_no

        # Need all prices to evaluate
        if None in (best_ask_yes, best_ask_no, best_bid_yes, best_bid_no):
            return None

        total_ask = best_ask_yes + best_ask_no
        total_bid = best_bid_yes + best_bid_no

        # Calculate total fees for 2 orders (buy YES + buy NO, or sell both)
        # Fee is percentage of notional, applied to each leg
        taker_fee_pct = self.config.taker_fee_bps / 10000  # Convert bps to decimal
        gas_cost = self.config.gas_cost_per_order * 2  # 2 orders

        # For bundle long: we buy both, pay fees on each
        # Fee cost = taker_fee_pct * (ask_yes + ask_no) = taker_fee_pct * total_ask
        fee_cost_long = taker_fee_pct * total_ask

        # For bundle short: we sell both, pay fees on each
        fee_cost_short = taker_fee_pct * total_bid

        opportunity: Optional[Opportunity] = None

        # Check for bundle long opportunity (buy both for < $1)
        # Must be profitable AFTER fees: 1.0 - total_ask - fees > min_edge
        gross_edge_long = 1.0 - total_ask
        net_edge_long = gross_edge_long - fee_cost_long - gas_cost

        if net_edge_long >= self.config.min_edge:
            edge = net_edge_long  # Use NET edge (after fees)

            # Calculate max size based on liquidity
            yes_ask_size = order_book.yes.best_ask_size or 0
            no_ask_size = order_book.no.best_ask_size or 0
            max_size = min(yes_ask_size, no_ask_size)

            # Liquidity gate: both legs must have enough depth to execute
            min_executable = self.config.min_order_size * 2
            if yes_ask_size < min_executable or no_ask_size < min_executable:
                logger.debug(
                    f"Skipping {market_id} bundle long: insufficient liquidity "
                    f"(YES={yes_ask_size:.2f}, NO={no_ask_size:.2f})"
                )
                return None

            suggested_size = self._kelly_size(
                edge=net_edge_long,
                price=total_ask,
                max_size=max_size,
                bankroll=bankroll,
            )

            opportunity = Opportunity(
                opportunity_id=f"bundle_long_{uuid.uuid4().hex[:8]}",
                opportunity_type=OpportunityType.BUNDLE_LONG,
                market_id=market_id,
                edge=edge,
                best_bid_yes=best_bid_yes,
                best_ask_yes=best_ask_yes,
                best_bid_no=best_bid_no,
                best_ask_no=best_ask_no,
                suggested_size=suggested_size,
                max_size=max_size,
                expires_at=datetime.utcnow()
                + timedelta(seconds=self.config.signal_expiry_seconds),
            )

            self.stats.bundle_opportunities_detected += 1
            logger.info(
                f"Bundle LONG opportunity: {market_id} | "
                f"total_ask={total_ask:.4f} | gross={gross_edge_long:.4f} | "
                f"fees={fee_cost_long:.4f} | NET edge={edge:.4f} | size={suggested_size:.2f}"
            )

        # Check for bundle short opportunity (sell both for > $1)
        # Must be profitable AFTER fees: total_bid - 1.0 - fees > min_edge
        gross_edge_short = total_bid - 1.0
        net_edge_short = gross_edge_short - fee_cost_short - gas_cost

        if opportunity is None and net_edge_short >= self.config.min_edge:
            edge = net_edge_short  # Use NET edge (after fees)

            # Calculate max size based on liquidity
            yes_bid_size = order_book.yes.best_bid_size or 0
            no_bid_size = order_book.no.best_bid_size or 0
            max_size = min(yes_bid_size, no_bid_size)

            # Liquidity gate: both legs must have enough depth to execute
            min_executable = self.config.min_order_size * 2
            if yes_bid_size < min_executable or no_bid_size < min_executable:
                logger.debug(
                    f"Skipping {market_id} bundle short: insufficient liquidity "
                    f"(YES={yes_bid_size:.2f}, NO={no_bid_size:.2f})"
                )
                return None

            suggested_size = self._kelly_size(
                edge=net_edge_short,
                price=total_bid,
                max_size=max_size,
                bankroll=bankroll,
            )

            opportunity = Opportunity(
                opportunity_id=f"bundle_short_{uuid.uuid4().hex[:8]}",
                opportunity_type=OpportunityType.BUNDLE_SHORT,
                market_id=market_id,
                edge=edge,
                best_bid_yes=best_bid_yes,
                best_ask_yes=best_ask_yes,
                best_bid_no=best_bid_no,
                best_ask_no=best_ask_no,
                suggested_size=suggested_size,
                max_size=max_size,
                expires_at=datetime.utcnow()
                + timedelta(seconds=self.config.signal_expiry_seconds),
            )

            self.stats.bundle_opportunities_detected += 1
            logger.info(
                f"Bundle SHORT opportunity: {market_id} | "
                f"total_bid={total_bid:.4f} | gross={gross_edge_short:.4f} | "
                f"fees={fee_cost_short:.4f} | NET edge={edge:.4f} | size={suggested_size:.2f}"
            )

        if not opportunity:
            return None

        # Only suppress re-entry if we already have an open position in this market
        # (the old 2-second blanket cooldown was blocking valid repeating opportunities).
        if self._portfolio is not None:
            has_open_position = (
                self._portfolio.get_exposure(market_id)["total_notional"] > 0
                or any(
                    leg.market_id == market_id
                    for g in self._portfolio._open_group_arbs.values()
                    for leg in g.legs
                )
            )
            if has_open_position:
                return None

        self._recent_opportunities[opportunity.opportunity_id] = opportunity
        self.stats.last_opportunity_time = datetime.utcnow()

        # Start tracking for duration measurement
        self._start_tracking_opportunity(opportunity)

        # Generate signal
        return self._create_bundle_signal(opportunity)

    def _create_bundle_signal(self, opportunity: Opportunity) -> Signal:
        """Create a trading signal for a bundle arbitrage opportunity."""
        orders = []

        if opportunity.opportunity_type == OpportunityType.BUNDLE_LONG:
            # Buy both YES and NO at ask prices
            orders = [
                {
                    "token_type": TokenType.YES,
                    "side": OrderSide.BUY,
                    "price": opportunity.best_ask_yes,
                    "size": opportunity.suggested_size,
                    "strategy_tag": "bundle_arb",
                },
                {
                    "token_type": TokenType.NO,
                    "side": OrderSide.BUY,
                    "price": opportunity.best_ask_no,
                    "size": opportunity.suggested_size,
                    "strategy_tag": "bundle_arb",
                },
            ]
        else:
            # Sell both YES and NO at bid prices
            orders = [
                {
                    "token_type": TokenType.YES,
                    "side": OrderSide.SELL,
                    "price": opportunity.best_bid_yes,
                    "size": opportunity.suggested_size,
                    "strategy_tag": "bundle_arb",
                },
                {
                    "token_type": TokenType.NO,
                    "side": OrderSide.SELL,
                    "price": opportunity.best_bid_no,
                    "size": opportunity.suggested_size,
                    "strategy_tag": "bundle_arb",
                },
            ]

        signal = Signal(
            signal_id=f"sig_{uuid.uuid4().hex[:12]}",
            action="place_orders",
            market_id=opportunity.market_id,
            opportunity=opportunity,
            orders=orders,
            priority=10,  # High priority for arb
        )

        self.stats.signals_generated += 1
        return signal

    def _check_multileg_arbitrage(
        self,
        group_id: str,
        group_states: dict[str, MarketState],
        bankroll: float = 100.0,
    ) -> Optional[Signal]:
        """
        Check for multi-leg arbitrage in mutually exclusive markets.
        If the sum of best YES asks across all options is < 1.0 (after fees), we can buy them all.
        """
        if not group_states:
            return None

        # Gather YES asks
        total_yes_ask = 0.0
        max_possible_size = float("inf")
        market_legs = []

        for m_id, state in group_states.items():
            best_ask_yes = state.order_book.best_ask_yes
            ask_yes_size = state.order_book.yes.best_ask_size

            # Need valid asks on ALL legs to guarantee an arbitrage
            if best_ask_yes is None or ask_yes_size is None or ask_yes_size <= 0:
                return None

            total_yes_ask += best_ask_yes
            max_possible_size = min(max_possible_size, ask_yes_size)
            market_legs.append(
                {"market_id": m_id, "ask_yes": best_ask_yes, "ask_size": ask_yes_size}
            )

        # Must have at least 2 legs
        num_legs = len(market_legs)
        if num_legs < 2:
            return None

        # Calculate fees
        taker_fee_pct = self.config.taker_fee_bps / 10000
        gas_cost = self.config.gas_cost_per_order * num_legs
        fee_cost = taker_fee_pct * total_yes_ask

        gross_edge = 1.0 - total_yes_ask
        net_edge = gross_edge - fee_cost - gas_cost

        if net_edge >= self.config.min_edge:
            # Check liquidity gate: max_possible_size is the per-leg minimum depth,
            # and min_order_size is also per-leg, so the comparison is direct.
            if max_possible_size < self.config.min_order_size:
                logger.debug(
                    f"Skipping group {group_id} multileg long: insufficient liquidity "
                    f"(max size: {max_possible_size:.2f})"
                )
                return None

            suggested_size = self._kelly_size(
                edge=net_edge,
                price=total_yes_ask,
                max_size=max_possible_size,
                bankroll=bankroll,
            )

            # Avoid re-entry if we already have exposure
            if self._portfolio is not None:
                for m_id in group_states:
                    has_open_position = (
                        self._portfolio.get_exposure(m_id)["total_notional"] > 0
                        or any(
                            leg.market_id == m_id
                            for g in self._portfolio._open_group_arbs.values()
                            for leg in g.legs
                        )
                    )
                    if has_open_position:
                        return None

            opportunity = Opportunity(
                opportunity_id=f"multileg_long_{uuid.uuid4().hex[:8]}",
                opportunity_type=OpportunityType.MULTILEG_LONG,
                market_id=group_id,  # Use group_id as the aggregate market ID
                edge=net_edge,
                suggested_size=suggested_size,
                max_size=max_possible_size,
                expires_at=datetime.utcnow()
                + timedelta(seconds=self.config.signal_expiry_seconds),
            )

            # We use group_id for tracking in active ops
            self._start_tracking_opportunity(opportunity)
            self.stats.bundle_opportunities_detected += 1
            logger.info(
                f"MULTILEG LONG opportunity: {group_id} ({num_legs} legs) | "
                f"total_yes_ask={total_yes_ask:.4f} | gross={gross_edge:.4f} | "
                f"fees={fee_cost:.4f} | NET edge={net_edge:.4f} | size={suggested_size:.2f}"
            )

            # Create signal with all leg orders
            orders = []
            for leg in market_legs:
                orders.append(
                    {
                        "market_id": leg[
                            "market_id"
                        ],  # Per-leg market_id — signal.market_id is the aggregate group_id
                        "token_type": TokenType.YES,
                        "side": OrderSide.BUY,
                        "price": leg["ask_yes"],
                        "size": suggested_size,
                        "strategy_tag": "multileg_arb",
                    }
                )

            signal = Signal(
                signal_id=f"sig_{uuid.uuid4().hex[:12]}",
                action="place_orders",
                market_id=group_id,
                opportunity=opportunity,
                orders=orders,
                priority=10,
            )
            self.stats.signals_generated += 1
            return signal

        return None

    def _check_market_making(
        self, market_id: str, order_book: OrderBook
    ) -> list[Signal]:
        """
        Check for market-making opportunities on YES and NO tokens.

        Place limit orders inside the spread to capture the bid-ask spread.
        """
        signals = []

        # Check YES token
        yes_signal = self._check_mm_token(market_id, order_book.yes, TokenType.YES)
        if yes_signal:
            signals.append(yes_signal)

        # Check NO token
        no_signal = self._check_mm_token(market_id, order_book.no, TokenType.NO)
        if no_signal:
            signals.append(no_signal)

        return signals

    def _check_mm_token(
        self, market_id: str, token_book, token_type: TokenType
    ) -> Optional[Signal]:
        """Check market-making opportunity for a single token."""
        best_bid = token_book.best_bid
        best_ask = token_book.best_ask
        spread = token_book.spread

        if spread is None or best_bid is None or best_ask is None:
            return None

        # Check if spread is wide enough
        if spread < self.config.min_spread:
            return None

        # Check cooldown
        cooldown_key = f"mm_{market_id}_{token_type.value}"
        if cooldown_key in self._opportunity_cooldown:
            if datetime.utcnow() < self._opportunity_cooldown[cooldown_key]:
                return None

        self._opportunity_cooldown[cooldown_key] = datetime.utcnow() + timedelta(
            seconds=5
        )

        # Calculate our prices (inside the spread)
        our_bid = best_bid + self.config.tick_size
        our_ask = best_ask - self.config.tick_size

        # Make sure we still have positive edge
        if our_ask <= our_bid:
            return None

        our_spread = our_ask - our_bid
        if our_spread < self.config.tick_size * 2:
            return None

        # Calculate size
        order_size = self.config.default_order_size / ((our_bid + our_ask) / 2)
        order_size = min(order_size, self.config.max_order_size)
        order_size = max(order_size, self.config.min_order_size)

        # Create opportunity for logging
        opportunity = Opportunity(
            opportunity_id=f"mm_{token_type.value}_{uuid.uuid4().hex[:8]}",
            opportunity_type=OpportunityType.MM_BID
            if token_type == TokenType.YES
            else OpportunityType.MM_ASK,
            market_id=market_id,
            edge=our_spread / 2,  # Expected edge per side
            suggested_size=order_size,
            max_size=order_size * 2,
        )

        self.stats.mm_opportunities_detected += 1
        self.stats.last_opportunity_time = datetime.utcnow()

        logger.info(
            f"MM opportunity: {market_id}/{token_type.value} | "
            f"spread={spread:.4f} | our_spread={our_spread:.4f} | size={order_size:.2f}"
        )

        # Generate signal with both bid and ask orders
        orders = [
            {
                "token_type": token_type,
                "side": OrderSide.BUY,
                "price": our_bid,
                "size": order_size,
                "strategy_tag": "market_making",
            },
            {
                "token_type": token_type,
                "side": OrderSide.SELL,
                "price": our_ask,
                "size": order_size,
                "strategy_tag": "market_making",
            },
        ]

        signal = Signal(
            signal_id=f"sig_{uuid.uuid4().hex[:12]}",
            action="place_orders",
            market_id=market_id,
            opportunity=opportunity,
            orders=orders,
            priority=5,  # Lower priority than arb
        )

        self.stats.signals_generated += 1
        return signal

    def get_recent_opportunities(
        self, max_age_seconds: float = 60.0
    ) -> list[Opportunity]:
        """Get recently detected opportunities."""
        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        return [
            opp
            for opp in self._recent_opportunities.values()
            if opp.detected_at > cutoff
        ]

    def clear_expired_opportunities(self) -> int:
        """Remove expired opportunities from cache."""
        now = datetime.utcnow()
        expired = [
            opp_id
            for opp_id, opp in self._recent_opportunities.items()
            if opp.expires_at and opp.expires_at < now
        ]
        for opp_id in expired:
            del self._recent_opportunities[opp_id]
        return len(expired)

    def get_stats(self) -> ArbStats:
        """Get engine statistics."""
        return self.stats
