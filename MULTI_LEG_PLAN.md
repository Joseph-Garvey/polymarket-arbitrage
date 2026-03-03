# Multi-Leg Arbitrage Implementation Plan

This document outlines the remaining steps to fully integrate N-leg (categorical) arbitrage into the Polymarket bot's core execution and portfolio tracking.

## 1. Portfolio Model Updates
The current `ArbPairPosition` is hardcoded for two legs (YES + NO). To support multi-leg groups (e.g., a "Winner" market with 5 candidates), we need a generic container.

- [ ] **Refactor `ArbPairPosition` to `GroupArbPosition`**:
    - Change `yes_entry` and `no_entry` to a `legs: List[Dict]` structure.
    - Store `market_id` (group/event ID), `total_cost` (sum of all leg prices), and `size`.
    - Update `locked_profit` calculation: `(1.0 - total_cost) * size`.
- [ ] **Update `Portfolio` tracking**:
    - Modify `open_arb_pair` to `open_group_position`.
    - Ensure `unrealized_pnl` for these groups returns the `locked_profit` to prevent the "directional loss" visual bug during active trades.

## 2. Execution Logic for Multi-Leg
The execution engine needs to recognize when a multi-leg "NegRisk" group has been fully filled to move it from individual positions to a locked-profit group.

- [ ] **Implement `_maybe_open_multileg_group`**:
    - Track fills by `group_id`.
    - Once all expected legs (based on `market.group_size`) have a position, aggregate them into a `GroupArbPosition`.
- [ ] **Partial Fill Handling**:
    - If one leg of a 5-leg arb fails to fill, the bot currently leaves 4 directional positions open.
    - Add a "Cleanup" task to either retry the missing leg at a worse price (to lock profit) or market-sell the existing legs to neutralize risk.

## 3. Dashboard UI Enhancements
The UI currently has placeholders for positions but doesn't render them dynamically.

- [ ] **Implement `updatePositions()` Javascript**:
    - Add a function to `dashboard/server.py` to iterate through `state.portfolio.positions`.
    - Differentiate between "Open Directional Positions" and "Locked Group Arbitrages".
- [ ] **Visual Grouping**:
    - Display multi-leg positions as a single row in the "Positions" table, showing the Event Name and the total net edge.

## 4. Risk Management Adjustments
- [ ] **Group Exposure**:
    - Update `RiskManager` to treat a fully hedged multi-leg group as "Zero Net Exposure" (or only count the `total_cost` as exposure) rather than summing the notional of every individual leg.
    - This prevents the bot from hitting "Max Global Exposure" limits while holding low-risk arbitrage bundles.

## 5. Validation & Testing
- [ ] **Simulated Multi-Leg Test**:
    - Create a test script that feeds a mock 3-leg `NegRisk` opportunity to the engine.
    - Verify that `ExecutionEngine` places 3 concurrent orders.
    - Verify that `Portfolio` correctly calculates the locked profit once all 3 are marked as filled.
