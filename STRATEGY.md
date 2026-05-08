# Spread Trading Platform
## Project Reference & Development Guide

*Last updated: 2026-05-08*
*Purpose: Give any new Claude session or developer a complete picture of where this project stands, what has been proven, what remains open, and what to build next.*

---

## 1. What This Project Is

A Python/Streamlit application for trading long/short spreads across global financial instruments using CFDs (Contracts for Difference). Originally a 20-year-old spreadsheet system for equity index spreads, now rebuilt with modern analytics, multi-asset support, and a formal research programme (Q1–Q11) that has fundamentally reshaped the strategy.

The core idea: buy one basket of instruments, short another, profit from the difference in performance. A mean-reversion "crossing signal" fires when the cumulative spread deviates more than ±2 standard deviations from its rolling 1-year mean.

### Application Structure (9 tabs)

| Tab | Purpose | Key files |
|-----|---------|-----------|
| 1 — Dashboard | Live spread chart, signal status, P&L | app.py, calculations.py |
| 2 — Analysis | Deep metrics, velocity, acceleration | app.py, calculations.py |
| 3 — Stake Calculator | Position sizing per instrument | app.py, stake_calc.py |
| 4 — Portfolio | Multi-basket portfolio view | app.py |
| 5 — Search | Exhaustive basket enumeration + scoring | app.py, search.py, scoring.py |
| 6 — Live | Real-time signal scanner | app.py |
| 7 — Journal | Trade log and notes | app.py |
| 8 — Backtest | Formal crossing signal backtest with costs | app.py, backtest.py, numba_core.py |
| 9 — Walk-Forward | OOS validation of scoring systems | app.py, walkforward.py |

### Key Technical Components

| File | Role |
|------|------|
| `numba_core.py` | JIT-compiled rolling stats, trade detection, batch backtest (with pure-Python fallback) |
| `scoring.py` | Centralised pair scoring: composite, cost_rank, contrarian modes |
| `walkforward.py` | Walk-forward validation engine — IS scoring vs OOS performance |
| `search.py` | Exhaustive NvM basket combination search |
| `backtest.py` | Full crossing signal backtest with cost model |
| `calculations.py` | Vol scaling, spread construction, signal metrics |
| `asset_configs.py` | Instrument definitions per asset class |
| `config.py` | Global parameters (vol window, SD thresholds, etc.) |
| `data.py` | Price data loaders |

---

## 2. Research Programme — What Has Been Proven

Eleven formal research papers (Q1–Q11) have been completed. The findings are definitive on several questions and have already changed the strategy. Every claim below is backed by exhaustive backtests across 27 years of data.

### 2.1 The Crossing Signal Works (but costs kill equities)

**Q1 — Equity index spreads** (280,913 trades across 18,480 3v3 baskets):
- Gross win rate: **72.8%** — statistically significant (p < 0.001)
- Gross expectancy: **+0.36%** per trade
- Net expectancy at 4.88% CFD financing: **−1.26%** per trade
- Average holding: **135 days** — financing accumulates to ~1.48% per trade
- Break-even financing rate: **~0.5% p.a.**
- The signal has genuine predictive power. The cost structure makes equity CFD implementation unprofitable.

### 2.2 The Signal Is Universal Across Asset Classes

The ~72–73% gross win rate appears independently in every asset class tested — this is strong evidence of a genuine statistical feature, not a fluke.

| Asset Class | Paper | GWR | Gross Exp | Net Exp (4.88%) | BE Rate | Verdict |
|-------------|-------|-----|-----------|-----------------|---------|---------|
| Equity indices | Q1 | 72.8% | +0.36% | −1.26% | ~0.5% | Signal valid; costs kill |
| FX pairs (PvP) | Q5 | 72.6% | +0.49% | −1.47% | ~0.5% | Better vehicle than equity |
| Commodities | Q6 | 72.6% | +4.83% | **+3.40%** | ~3.0% | **Best standalone class** |
| Fixed income | Q7 | 70.7% | +0.39% | −1.06% | ~1.8% | Best diversifier (−0.26 corr) |
| Cross-asset (1v1) | Q8 | 70.4% | varies | 109 net+ pairs | 5.9–8.0%+ | **Viable with partial exits** |

### 2.3 Commodities Are the Best Standalone Asset Class

Q6 found commodity spreads generate **+3.40% net per trade** even at 4.88% CFD financing — the only single asset class with positive net expectancy at retail rates. Key drivers:
- Gross returns per trade are 13× higher than equities (+4.83% vs +0.36%) because commodity spread dislocations are larger
- Performance improves monotonically with higher SD thresholds (3.0 SD → +7.97% net)
- Profitable in every regime tested, including COVID/inflation (2020–2023)
- Natural Gas pairs dominate the top performers
- Under futures implementation (near-zero carry), expectancy rises to ~+5.0% per trade

### 2.4 Partial Reversion Exits Are Transformative

Q8 found that exiting at 1.0 SD (capturing ~50% of theoretical reversion) instead of waiting for full reversion to 0.0 SD:
- Reduces average holding from **213 days → 77 days** (median: 52 days)
- Increases net-positive pairs from **21 → 109** (at 4.88% financing)
- Increases annual capital turnover from **2.7× → 6.3×**
- Improves gross win rate by 3.4 percentage points (counterintuitively — avoids round-trips)

This is the single highest-impact improvement identified in the entire research programme.

### 2.5 Cross-Asset Pairs Are the Strongest Configuration

Q8's top cross-asset pairs at 1.0 SD exit:

| Pair | Type | Net Exp | Median Hold | Trades | BE Rate |
|------|------|---------|-------------|--------|---------|
| Platinum vs AUD/USD | Cmd vs FX | +1.11% | 26d | 36 | >8.0% |
| CAC vs GBP/JPY | Eq vs FX | +1.21% | 34d | 49 | ~6.7% |
| STOXX50 vs Wheat | Eq vs Cmd | +1.56% | 38d | 39 | ~6.5% |
| Platinum vs US HY | Cmd vs FI | +1.54% | 36d | 33 | ~5.9% |
| FTSE vs GBP/JPY | Eq vs FX | +0.94% | 33d | 53 | ~6.7% |

Multi-leg baskets (2v2, 3v3) fail at CFD rates — doubled financing costs overwhelm the edge. Portfolio diversification should come from running multiple 1v1 pairs simultaneously.

### 2.6 In-Sample Scoring Is Pure Data Mining

Q11's walk-forward analysis (14 windows, 1,516 pair-window observations) proved:
- Composite score vs OOS gross return: **ρ = −0.037, p = 0.15** (no predictive power)
- FitDataMinMaxSD: **ρ = −0.129, p < 0.001** (significant *negative* predictor — removed from scoring)
- IS win rate → OOS win rate: ρ = −0.024 (does not persist)
- IS expectancy → OOS gross: ρ = +0.010 (does not persist)
- All quintiles have similar OOS gross returns (~2.6%/trade, ~83% GWR)
- The gross-to-net gap (3.1 ppt) is **13× larger** than the Q1-vs-Q5 selection gap (0.24 ppt)

The crossing signal itself is robust regardless of which pair it is applied to. Pair selection adds no value beyond random selection.

### 2.7 Current Scoring Modes

Three modes are implemented (in `scoring.py`), reflecting the Q11 findings:
- **Composite** (default): `z(|LastSD|) + z(TrendVolRatio) + z(WinRate) + z(Expectancy)` — improved but not validated as having positive OOS predictive power
- **Cost-Based**: Ranks by lowest estimated trading cost — since all pairs have similar gross returns, cheapest-to-trade maximises net expectancy
- **Contrarian**: Inverts the composite — exploits weak negative IS→OOS correlation

---

## 3. Current State of the Codebase

### 3.1 What Is Complete and Working

- [x] Core signal framework (crossing signal, TVR, velocity, acceleration)
- [x] Vol scaling and spread construction across 4 asset classes
- [x] Exhaustive search engine with asymmetric leg counts (e.g. 2v5)
- [x] Numba-accelerated batch backtest with pure-Python fallback
- [x] Centralised scoring module (3 modes: composite, cost_rank, contrarian)
- [x] Walk-forward validation engine (tab 9)
- [x] Full cost model (bid-ask + financing, using account.json rates)
- [x] Tabs 1–9 all functional
- [x] 28/28 numba parity tests passing
- [x] Price data: equities (prices.csv), FX (fx_prices.csv), commodities (commodity_prices.csv), fixed income (fi_prices.csv)

### 3.2 Recent Fixes (2026-05-08)

- Tab5 subheader now dynamically reflects selected scoring mode
- AvgHolding columns in tab5 and tab8 now display as "15d" not "15.234"
- STRATEGY.md Section 8.3 updated to reference Backtest and Walk-Forward tabs

### 3.3 Known Environment Issues

- Correct Python environment: `C:\Users\gordo\AppData\Local\Python\bin\python.exe` — the system default (`C:\Python314\python.exe`) lacks pandas
- `test_numba_parity.py` requires `PYTHONIOENCODING=utf-8` on Windows to handle ✓ characters in cp1252

---

## 4. What Remains Open — Prioritised

### Priority 1 — Implement partial reversion exits (HIGHEST IMPACT)

This is the single most valuable change identified across Q1–Q8. Currently the system only supports exit at 0.0 SD (full reversion). Adding configurable exit thresholds (0.5, 1.0, 1.5 SD) would:
- Make equity, FX, and fixed income spreads viable at CFD rates for the first time
- Reduce average holding periods by 60%+ (directly reducing the financing drag that kills profitability)
- Increase the universe of net-positive pairs from ~21 to ~109+ for cross-asset

**What needs to happen:**
1.-- `config.py` / UI sidebar — add `exit_sd` parameter (default 0.0, range 0.0–2.0)--
2.--`numba_core.py` — the `exit_sd` parameter already exists in `backtest_spread()` and `detect_trades()` — verify it works correctly at non-zero values--
3.-- `search.py` — pass `exit_sd` through to the batch backtest--
4.-- `backtest.py` — same--
5.-- `app.py` tabs 5, 8, 9 — add exit_sd slider/input--
6. **Re-run Q1 equity, Q5 FX, Q7 FI backtests at exit_sd = 0.5 and 1.0** to quantify the aggregate impact. Q8 already has this data for cross-asset pairs.

**Verification:** Run backtest tab with equity indices at exit_sd=1.0. Average holding should drop from ~135 to ~60–80 days. Net expectancy should improve materially (may become positive for some pairs).

### Priority 2 — Cross-asset search and trading

The search engine currently operates within a single asset class. Q8 proved that the best pairs are cross-asset (Platinum vs AUD/USD, CAC vs GBP/JPY, etc.). The system needs:
1. A cross-asset search mode that combines instruments from multiple price CSVs
2. A unified price DataFrame with aligned dates across asset classes
3. Date alignment logic (different asset classes have different trading calendars)
4. Update the Search and Backtest tabs to support cross-asset pair selection

### Priority 3 — Commodity-focused implementation

Q6 showed commodities are the best standalone asset class. Specific next steps:
1. Test 3.0 SD entry threshold for commodity pairs (Q6 showed +7.97% net — best of any threshold)
2. Implement commodity futures cost model (near-zero carry vs 4.88% CFD) as a selectable option
3. Identify the optimal commodity pair portfolio (likely 10–15 pairs running simultaneously)
4. Natural Gas pairs dominate — assess concentration risk

### Priority 4 — Paper trading the Q8 recommended pairs

The five pairs recommended in Q8 Section 10.2 should be tracked live:
1. Platinum vs AUD/USD, CAC vs GBP/JPY, STOXX50 vs Wheat, Platinum vs US HY, FTSE vs GBP/JPY
2. All using 1.0 SD partial reversion exit
3. Track for minimum 6 months to generate out-of-sample data
4. Compare actual performance to backtest expectations

### Priority 5 — Portfolio construction framework

Running multiple 1v1 pairs simultaneously requires:
1. Cross-pair correlation matrix (to avoid concentration)
2. Position sizing across pairs (capital allocation)
3. Aggregate portfolio risk metrics (max drawdown, VaR)
4. Q8 found equity–bond correlation = −0.26 and equity–gold = +0.003 — genuine diversification

### Priority 6 — Regime detection / filters

Q8 showed STOXX50 vs NatGas lost −5.32% during COVID+ (European energy crisis). Potential filters:
1. Volatility-of-volatility filter for NatGas pairs during extreme energy market conditions
2. Correlation regime detection (Q9 — not yet researched)
3. Credit cycle positioning (from Q7 fixed income findings)

---

## 5. What Should Be Re-Tested Now

The backtest and walk-forward tabs now exist. Several Q1–Q8 findings should be validated or extended using these tools:

### 5.1 Partial reversion exits across all asset classes

**Test:** Run backtest tab for each asset class (equity, FX, commodity, FI) at exit_sd = 0.0, 0.5, and 1.0. Compare net expectancy, holding period, and win rate.

**Expected result:** Matches Q8 findings — 1.0 SD exit dramatically improves net results across all classes.

**Why re-test:** The backtest tab may use slightly different logic than the research scripts. Confirming parity ensures the tool can be trusted for live decisions.

### 5.2 Walk-forward validation of cost_rank and contrarian scoring

**Test:** Run walk-forward tab (tab 9) for equity indices with each of the three scoring modes.

**Expected results:**
- Composite: ρ ≈ −0.037 (reproduces Q11 — no predictive power)
- Cost_rank: may show different pattern (untested in Q11)
- Contrarian: may show weak positive correlation (exploits negative IS→OOS relationship)

**Why test:** Q11 only formally tested the composite mode. Cost_rank and contrarian were proposed as alternatives but not walk-forward validated.

### 5.3 Walk-forward on commodity pairs

**Test:** Run walk-forward tab with commodity instruments.

**Expected result:** Unknown — Q11 only tested equity indices. Commodities have fundamentally different return distributions (larger dislocations, supply-driven) so the scoring→OOS relationship may differ.

**Why test:** If scoring has predictive power for commodities (unlike equities), it would change the basket selection approach entirely for the best-performing asset class.

### 5.4 Walk-forward on cross-asset pairs

**Test:** Once cross-asset search is implemented (Priority 2), run walk-forward across cross-asset pairs.

**Why test:** Cross-asset pairs have the highest net expectancy. Understanding whether scoring predicts OOS performance for these pairs is critical for the portfolio construction step.

### 5.5 Sensitivity grids for key parameters

**Test:** Use backtest tab sensitivity grid for:
- SD entry threshold: 1.0, 1.5, 2.0, 2.5, 3.0 (Q6 showed commodities improve monotonically)
- Vol window: 130, 196, 262, 393, 524 days
- Exit SD: 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5

**Why test:** Q1 tested SD sensitivity for equities. Q6 tested it for commodities. A unified sensitivity grid across all asset classes would identify the optimal universal parameters (or confirm that each class needs different settings).

---

## 6. Original Research Questions — Status Tracker

| # | Question | Status | Finding | Paper |
|---|----------|--------|---------|-------|
| Q1 | Does the crossing signal have positive expectancy after costs? | ✅ ANSWERED | +0.36% gross, −1.26% net (equity) | Q1 |
| Q2 | How sensitive is performance to the 2 SD threshold? | ⚠️ PARTIAL | Tested for equity (Q1) and commodity (Q6). No unified cross-asset grid. | Q1, Q6 |
| Q3 | Typical holding period vs financing interaction? | ✅ ANSWERED | 135d equity, 121d commodity. Financing = 91% of costs. | Q1, Q6 |
| Q4 | Performance by market regime? | ✅ ANSWERED | Signal works in all regimes. GFC best, low-vol weakest for equities. | Q1, Q6, Q8 |
| Q5 | FX pairs — better mean reversion? | ✅ ANSWERED | Similar GWR (72.6%), still negative net at CFD rates. PvP preferred over baskets. | Q5 |
| Q6 | Commodity spreads? | ✅ ANSWERED | Best standalone class. +3.40% net at CFD rates. NatGas dominant. | Q6 |
| Q7 | Fixed income — lower correlation alternative? | ✅ ANSWERED | 70.7% GWR, −0.26 corr vs equity. Best diversifier. HY vs IG pairs viable. | Q7 |
| Q8 | Cross-asset baskets? | ✅ ANSWERED | 1v1 pairs with 1.0 SD exit = 109 net+ pairs. Multi-leg baskets fail. | Q8 |
| Q9 | Correlation regime detection? | ❌ NOT STARTED | — | — |
| Q10 | Correlation-adjusted weighting? | ❌ NOT STARTED | — | — |
| Q11 | Does IS scoring predict OOS? | ✅ ANSWERED | No (ρ = −0.037). Pure data mining. Cost-based ranking recommended. | Q11 |

---

## 7. Future Enhancements — Development Backlog

### 7.1 Near-Term (next 1–3 Claude Code sessions)

1. **Add exit_sd parameter to all tabs** — the numba_core already supports it; just needs UI wiring
2. **Re-run equity backtest at exit_sd = 1.0** — confirm Q8 findings in the Backtest tab
3. **Walk-forward validate cost_rank and contrarian modes** — extend Q11 analysis
4. **Walk-forward on commodities** — test whether scoring works differently for the best asset class

### 7.2 Medium-Term (next month)

5. **Cross-asset price alignment** — unify the four price CSVs into a single aligned DataFrame
6. **Cross-asset search mode** — let the search engine combine instruments from different asset classes
7. **Futures cost model toggle** — add a "CFD vs Futures" option that switches the financing rate assumption (critical for commodities and FX)
8. **Portfolio tab upgrade** — multi-pair portfolio with correlation matrix and aggregate risk
9. **Paper trading framework** — track the Q8 recommended pairs in real time

### 7.3 Longer-Term

10. **Regime detection (Q9)** — correlation regime filter to reduce exposure when all assets move together
11. **Correlation-adjusted weighting (Q10)** — test whether non-equal weighting improves Sharpe
12. **Alternative signal research** — the crossing signal is proven but the search/scoring layer adds no value; explore fundamentally different selection criteria (momentum, carry, macro factors)
13. **Real-time alerting** — push notifications when crossing signals fire on monitored pairs
14. **Drawdown control** — maximum drawdown rules, trailing stops, position reduction on further divergence

---

## 8. Key Metrics and Parameters

### 8.1 Default Parameters

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| Target daily vol | 1.0% | config.py | Standard risk-parity target |
| Vol window | 262 days | config.py | 1 trading year |
| Crossing threshold | ±2.0 SD | config.py | Entry signal. Q6 suggests 3.0 SD for commodities |
| Exit threshold | 0.0 SD | config.py | **Should be configurable — Priority 1** |
| ROC look-back | 12 days | config.py | Velocity calculation |
| Linear fit points | 10 | config.py | TVR calculation |
| Long financing | 4.88% p.a. | account.json | Retail CFD rate |
| Short rebate | 0.88% p.a. | account.json | |
| Target exposure | £500 per 1 SD | UI sidebar | Position sizing |
| Margin rate | 10% | config.py | Estimated |

### 8.2 Break-Even Financing Rates by Asset Class

| Asset Class | BE Rate | Implication |
|-------------|---------|-------------|
| Equity indices (3v3, 0.0 exit) | ~0.5% | Unprofitable at any retail rate |
| FX pairs (PvP, 0.0 exit) | ~0.5% | Unprofitable at retail; viable via forwards |
| Commodities (PvP, 0.0 exit) | ~3.0% | Profitable at 4.88% CFD |
| Fixed income (PvP, 0.0 exit) | ~1.8% | Unprofitable at retail; HY/IG pairs viable |
| Cross-asset (PvP, 1.0 exit) | 5.9–8.0%+ | **Profitable at any retail rate** |

### 8.3 Instrument Universe

**Equities (12):** FTSE, CAC, DAX, MIB, IBEX, STOXX50, SMI, HSI, ASX, NDX, SPX, DJI

**FX (12):** EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD, EURGBP, EURJPY, GBPJPY, EURCHF, AUDNZD

**Commodities (12):** WTI, Brent, NatGas, Gold, Silver, Platinum, Copper, Palladium, Wheat, Corn, Soybeans, Coffee (Sugar in CSV but 14 columns)

**Fixed Income (15):** SHY, IEI, IEF, TLT, LQD, HYG, TIP, BWX, EMB, IGLT, IEAG, IBTM, UST10Y, UST30Y, UST5Y

---

## 9. Guiding Principles for Future Development

1. **Costs dominate everything.** The gross-to-net gap (3.1 ppt) is 13× larger than any scoring improvement (0.24 ppt). Every feature should be evaluated through the lens of cost reduction: shorter holds, cheaper instruments, or lower financing rates.

2. **The crossing signal works — don't change it.** ~72–73% gross win rate across 4 asset classes, 27 years, and hundreds of thousands of trades. The signal is proven. Improvements come from cost structure, exit timing, and pair selection (by cost, not by in-sample performance).

3. **Walk-forward validate everything.** Q11 proved the old scoring was pure data mining. Any new metric, parameter, or scoring approach must pass the walk-forward tab before deployment.

4. **1v1 pairs, not multi-leg baskets.** Multi-leg baskets fail because each additional leg adds financing cost. Diversification should come from running multiple independent 1v1 pairs in a portfolio.

5. **Partial exits are the key lever.** Moving from 0.0 to 1.0 SD exit converts many unprofitable configurations to profitable ones by cutting holding periods in half.

6. **Commodities and cross-asset pairs first.** These are the only configurations with positive net expectancy at retail CFD rates. Equity-only 3v3 baskets — the original strategy — are the least viable configuration in the research.

---

## 10. File Inventory

### Code Files
| File | Lines (approx) | Last major change |
|------|-----------------|-------------------|
| app.py | ~1,850 | 2026-05-08 (Q11 bug fixes) |
| numba_core.py | ~500 | Pre-Q11 (stable) |
| search.py | ~400 | Q11 (scoring_mode param) |
| backtest.py | ~470 | Q11 (scoring_mode param) |
| scoring.py | ~120 | Q11 (new file) |
| walkforward.py | ~300 | Q11 (new file) |
| calculations.py | ~300 | Stable |
| stake_calc.py | ~100 | Stable |
| asset_configs.py | ~100 | Stable |
| config.py | ~50 | Stable |
| data.py | ~80 | Stable |

### Research Papers
| File | Topic | Key finding |
|------|-------|-------------|
| Q1_Crossing_Signal_Research.docx | Signal expectancy | +0.36% gross, −1.26% net |
| Q5_FX_Research_Paper.docx | FX pairs | 72.6% GWR, negative net |
| Q6_Commodity_Research_Paper.docx | Commodities | **+3.40% net at CFD rates** |
| Q7_Fixed_Income_Research_Paper.docx | Fixed income | Best diversifier (−0.26 corr) |
| Q8_Cross_Asset_Research_Paper.docx | Cross-asset | 109 net+ pairs at 1.0 SD exit |
| Q11_InSample_Scoring_Research_Paper.docx | Scoring validation | Pure data mining (ρ = −0.037) |

### Data Files
| File | Rows | Columns | Period |
|------|------|---------|--------|
| prices.csv | 7,062 | 13 | 1999–2026 (equity indices) |
| fx_prices.csv | 7,114 | 13 | 1999–2026 (FX pairs) |
| commodity_prices.csv | 6,881 | 14 | 1999–2026 (commodities) |
| fi_prices.csv | 6,970 | 16 | 1999–2026 (fixed income ETFs) |

### Prompts and Documentation
| File | Purpose |
|------|---------|
| STRATEGY.md | Strategy description and research findings |
| CLAUDE_CODE_PROMPT.md | Original build prompt (numba, asymmetric search, backtest tab) |
| CLAUDE_CODE_PROMPT_Q11.md | Q11 implementation prompt (scoring, walk-forward) |
| CLAUDE_CODE_PROMPT_Q11_GAPS.md | Gap closure prompt (missing files, bugs) |
| 2026-05-08_Q11-Gap-Closure-Bugs-Fixed.md | Session log confirming Q11 fixes applied |
| PROJECT_REFERENCE.md | This file |