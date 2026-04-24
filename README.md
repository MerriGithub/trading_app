# Trading Monitor

A Streamlit dashboard for monitoring and trading index spread strategies across 12 global equity indices. Built around a volatility-targeting framework.
---

## Overview

The app tracks long/short spread trades across a basket of indices (FTSE, CAC, DAX, MIB, IBEX, STOXX50, SMI, HSI, ASX, NDX, SPX, DJI). It sizes positions so that a 1 standard deviation price move produces a target P&L, scales exposure based on rolling volatility, and provides a full suite of analysis tools from signal detection through to trade journalling.

---

## Features

### Dashboard
- Live price table with daily returns, rolling volatility, and position scaling for all instruments
- Account overview: starting capital, total equity, unrealised P&L, daily funding cost, and open spread costs
- Signal scanner: trend-velocity-ratio, crossing signals, contraction betas, and pre-trade stake/margin estimates for each instrument, sorted by distance from rolling mean
- Multi-timeframe range signals: shows where each instrument sits within its daily, weekly, and monthly price range

### Analysis
- Normalised price and rolling volatility charts
- Cumulative spread return with OLS linear trend overlay
- Velocity and acceleration of the spread (rate-of-change)
- Crossing signal chart: distance from rolling mean in standard deviations
- Pair statistics: rolling 1/2/3/5-day return distributions with empirical vs Gaussian tail probabilities

### Stake Calculator
- Volatility-targeted position sizing: `stake = target_exposure / (price × daily_vol × point_size) × scaling`
- Editable price inputs for what-if scenarios
- P&L scenario slider: estimates outcome for a symmetric long/short move

### Portfolio
- Pairwise return correlation heatmap
- Volatility and scaling bar chart by instrument
- Rolling N-day spread return chart
- P&L attribution by instrument and region (EU / US / Asia) for open positions

### Search Engine
- Enumerates all long/short instrument combinations for a given leg-count range
- Scores each pair on 5 metrics: ReturnSD (Sharpe), TrendVolRatio, ReturnTopology (skewness), FitDataMinMaxSD (price range), LastSD (distance from mean)
- Metrics are z-score normalised before ranking to avoid scale bias
- Configurable metric filters and history window
- Results can be loaded directly into the analysis tabs or saved to a named portfolio

### Live / Intraday Monitor
- Intraday spread return from yesterday's close, plotted against ±1 SD historical daily range
- Per-instrument intraday volatility vs historical vol with a high-vol flag

### Trade Journal
- Open multi-leg spread trades with per-leg buy/sell instrument, entry price, and stake
- Mark-to-market unrealised P&L with a warning when instrument prices are unavailable
- Partial close: close any percentage of a leg at a specified exit price
- Full trade history with realised P&L and win/loss count

---

## Architecture

| File | Responsibility |
|------|---------------|
| `config.py` | Instrument mappings, spreads, point sizes, algorithm parameters |
| `data.py` | Yahoo Finance data fetching, file-based cache, intraday prices |
| `calculations.py` | Returns, volatility, scaling, portfolio returns, trend analysis, crossing signals |
| `stake_calc.py` | Position sizing and P&L scenario calculation |
| `account.py` | Account state persistence, daily funding cost, spread cost calculation |
| `journal.py` | Trade persistence: open, partial close, full close, live P&L |
| `saved.py` | Named portfolio persistence |
| `search.py` | Vectorised portfolio search engine (numpy) |
| `app.py` | Streamlit UI — all seven tabs |

---

## Setup

```bash
pip install streamlit pandas numpy scipy plotly yfinance
streamlit run app.py
```

Price data is fetched from Yahoo Finance on first run and cached locally in `cache/prices.csv`. Subsequent loads use the cache and only fetch missing days incrementally.

---

## Configuration

Key parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_daily_vol` | 1% | Daily vol target for position scaling |
| `vol_calc_days` | 262 | Rolling window for volatility (1 trading year) |
| `xing_tolerance_sd` | 2.0 | SD threshold for crossing signals |
| `roc_days` | 12 | Rate-of-change look-back period |
| `margin_rate` | 10% | Broker margin rate for hypothetical estimates |

Account settings (starting capital, long/short financing rates, margin) are editable in the Dashboard tab and persisted to `account.json`.
