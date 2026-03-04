# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run bot with live dashboard (http://localhost:8000)
python run_with_dashboard.py

# Run bot only (no dashboard)
python main.py
python main.py -v                          # verbose logging
python main.py --config config.live.yaml  # custom config
python main.py --live                      # live trading mode
python main.py --backtest                  # backtest simulation

# Run all tests (currently 99 pass, 1 skipped: test_multi_leg.py async test needs pytest-asyncio config)
pytest tests/ -v

# Run a single test file
pytest tests/test_arb_engine.py -v

# With coverage
pytest tests/ --cov=core --cov=polymarket_client
```

## Architecture

The bot is an async Python system that cross-references Polymarket and Kalshi prediction markets to detect pricing discrepancies.

**Data flow:**
1. `core/data_feed.py` (`DataFeed`) — polls both APIs for market data and fires `on_update` callbacks
2. `core/arb_engine.py` (`ArbEngine`) — analyzes single-platform opportunities (bundle arb and market making)
3. `core/cross_platform_arb.py` (`CrossPlatformArbEngine`) — pairs Polymarket and Kalshi markets by text similarity (+ optional LLM verification via OpenRouter), then detects cross-platform price gaps
4. `core/execution.py` (`ExecutionEngine`) — receives `Signal` objects, validates against risk limits, and places orders via the API client
5. `core/risk_manager.py` (`RiskManager`) — enforces per-market and global limits; has a kill switch
6. `core/portfolio.py` (`Portfolio`) — tracks positions and PnL; uses `GroupArbPosition` to model multi-leg arb bundles as a single locked-profit unit

**Key data models** (in `polymarket_client/models.py`):
- `Market`, `OrderBook`, `MarketState` — Polymarket market data
- `Opportunity`, `Signal` — detected opportunity → execution intent
- `GroupArbPosition` / `GroupArbLeg` — N-leg arbitrage tracker (in `core/portfolio.py`)

**Client modules:**
- `polymarket_client/api.py` — Polymarket REST + WebSocket + Gamma API
- `kalshi_client/api.py` — Kalshi REST API (no auth needed for public market data)

**Dashboard:** FastAPI server (`dashboard/server.py`) with embedded single-file HTML/JS served from Python. `dashboard/integration.py` bridges bot state to the web UI.

**LLM market matching:** `utils/llm_client.py` (`MarketVerifier`) calls OpenRouter to verify if two market questions refer to the same underlying event. Requires `OPENROUTER_API_KEY` env var. Text similarity matching runs first (`min_match_similarity: 0.82` in config); LLM verification is a second pass on close matches.

## Configuration

All runtime behavior is controlled by `config.yaml`. Key decisions:

- `mode.data_mode`: `"real"` (live APIs) or `"simulation"` (generated fake data for demos)
- `mode.trading_mode`: `"dry_run"` or `"live"` — overridable via `--live` flag
- `mode.cross_platform_enabled` / `mode.kalshi_enabled`: toggle cross-platform detection
- `trading.min_edge`: minimum net profit after fees to act on a signal (currently 2%)
- Credentials go in `config.yaml` (`api_key`, `private_key`) or env vars (`POLYMARKET_API_KEY`, `POLYMARKET_PRIVATE_KEY`)

## Active Development Context

`MULTI_LEG_PLAN.md` tracks N-leg (NegRisk categorical) arbitrage work. Current status:

**Done:**
- `GroupArbPosition` refactor and `open_group_position` API (including explicit `locked_profit` override)
- `_maybe_open_group_position` routing in `ExecutionEngine` for both `bundle_arb` and `multileg_arb`
- `_group_states` memory management (500-group cap, resolved-market eviction, new-insertion-only trigger)
- `RiskManager.check_order()` bypasses per-market and global limits for hedged orders (`bundle_arb`/`multileg_arb`)
- `ExecutionEngine._handle_multileg_partial_fills()` — gap-fill recovery wired into timeout monitor
- Dashboard `updatePortfolio()` renders group arb rows with `[GROUP]` badge, leg count, and locked profit
- 110 tests pass (1 skipped: async fixture config)

Several root-level `inspect_*.py`, `verify_*.py`, `negrisk_*.py`, and `test_markets*.py` files are ad-hoc debugging/exploration scripts, not part of the production codebase.
