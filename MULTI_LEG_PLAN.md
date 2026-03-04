# Multi-Leg Arbitrage Implementation Plan

This document outlines the remaining steps to fully integrate N-leg (categorical) arbitrage into the Polymarket bot's core execution and portfolio tracking.

## 1. Portfolio Model Updates
The current `ArbPairPosition` is hardcoded for two legs (YES + NO). To support multi-leg groups (e.g., a "Winner" market with 5 candidates), we need a generic container.

- [x] **Refactor `ArbPairPosition` to `GroupArbPosition`**:
    - Change `yes_entry` and `no_entry` to a `legs: List[GroupArbLeg]` structure.
    - Store `market_id` (group/event ID), `total_cost` (sum of all leg prices), and `size`.
    - Update `locked_profit` calculation: `(1.0 - total_cost) * size`.
- [x] **Update `Portfolio` tracking**:
    - Modify `open_arb_pair` to `open_group_position`.
    - `unrealized_pnl` for group positions returns `locked_profit * size` unconditionally (no directional mark-to-market), resolving the "directional loss" visual bug during active trades.

Both items fully implemented in `core/portfolio.py`.

## 2. Execution Logic for Multi-Leg
The execution engine needs to recognize when a multi-leg "NegRisk" group has been fully filled to move it from individual positions to a locked-profit group.

- [x] **Implement `negrisk_executor.py`**:
    - [x] Create a standalone high-frequency executor for NegRisk groups.
    - [x] Simultaneous multi-leg limit order placement with real-time slippage/ROI verification.
    - [x] Dry-run mode for safe verification.
- [x] **Implement Gap-Fill Recovery**:
    - [x] Added `_handle_partial_fills` to `negrisk_executor.py`.
    - [x] Logic to cancel open orders and identify "Gaps" (missing legs).
    - [x] Profitability-aware "market buy" (high-limit order) to close exposures.
- [x] **Integration with Core `ExecutionEngine`**:
    - [x] Implement `_maybe_open_group_position` in `ExecutionEngine` — bundle path opens when YES+NO both filled; multileg path opens on first YES fill with `locked_profit=0.0` (re-entry guard only, not PnL). Implemented in `core/execution.py`.
    - [x] Port gap-fill recovery into `ExecutionEngine` as `_handle_multileg_partial_fills()`. Triggered automatically from `_monitor_order_timeouts()` when a multileg order times out. Cancels remaining legs, identifies unfilled gaps via portfolio positions, and places aggressive limit orders (price=1.0) if the gap-fill cost is within 110% of remaining profit budget.

## 3. Cross-Platform Matching Enhancements
- [x] **LLM 'Synthetic Equivalence'**:
    - [x] Updated `baseline_matcher.py` prompt to support NBA spreads vs. winner markets.
    - [x] Added specific logic for Team A YES == Team B NO in binary matches.

## 4. Dashboard UI Enhancements
The UI currently has placeholders for positions but doesn't render them dynamically.

- [x] **Implement `updatePortfolio()` Javascript** (was `updatePositions()` in plan):
    - `dashboard/server.py:1797` iterates `state.portfolio.positions` and `open_group_arbs`.
    - Renders group arbs first, then individual positions, deduplicating any position leg that belongs to a group.
- [x] **Visual Grouping**:
    - Group arb rows render with a `[GROUP]` badge, event ID (truncated), `N legs | Locked: $X` subtitle, and PnL column showing locked profit.

## 4. Risk Management Adjustments
- [x] **Group Exposure**:
    - `RiskManager.check_order()` now skips both per-market and global exposure limits for hedged orders (`bundle_arb` / `multileg_arb`).
    - For bundle_arb, the YES and NO legs on the same market would otherwise double-count notional even though net risk is only `locked_profit`. For multileg_arb, global exposure is committed simultaneously across all legs so a second check here would double-reject.
    - Per-market and global exposure are still *tracked* via `update_from_fill()`/`update_position()` for dashboard visibility; only the order-rejection gate is bypassed.

## 5. Validation & Testing
- [x] **Simulated Multi-Leg Test**:
    - `tests/test_arb_engine_multileg.py` covers 2-leg and 3-leg detection, partial group suppression, fill routing, locked_profit correctness, and `_group_states` cap — 31 tests total.
    - `TestCheckMultilegArbitrage::test_3leg_detection` and `TestMultilegGroupPositionLockedProfit` cover the concurrent-orders and locked-profit scenarios from this plan.
