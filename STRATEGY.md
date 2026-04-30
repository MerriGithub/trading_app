# Strategy Brief — Global Index Spread Trading

This document describes the trading strategy implemented in this application. It is written for analytical purposes — to enable deep research, back-testing review, and product comparison — without requiring familiarity with the code.

---

## 1. Background

This strategy was originally developed approximately 20 years ago as a spreadsheet-based system for trading long/short spreads across global equity indices using CFDs (Spreadbetting / Contract for Differences). The current application is a Python/Streamlit rebuild of that original framework, preserving the core logic while adding modern tooling for analysis, signal scanning, and trade management.

The original system was developed empirically over years of live trading. It has not been formally back-tested with statistical rigour since rebuilding. A key open question is whether the historical performance reflects genuine edge or was a product of a specific market regime (2000s–2020s bull market with periodic crises).

---

## 2. Instrument Universe

12 global equity indices traded as cash CFDs:

| Code | Index | Region |
|------|-------|--------|
| FTSE | FTSE 100 | Europe |
| CAC  | CAC 40 | Europe |
| DAX  | DAX 40 | Europe |
| MIB  | FTSE MIB | Europe |
| IBEX | IBEX 35 | Europe |
| STOXX50 | Euro Stoxx 50 | Europe |
| SMI  | Swiss Market Index | Europe |
| HSI  | Hang Seng | Asia |
| ASX  | ASX 200 | Asia |
| NDX  | NASDAQ 100 | US |
| SPX  | S&P 500 | US |
| DJI  | Dow Jones | US |

**Why indices?** Global equity indices share a common driver (global risk appetite) but diverge based on regional factors — sector composition, currency, interest rates, political risk. This creates persistent relative value opportunities without the idiosyncratic single-stock risk of pairs trading individual equities.

---

## 3. Strategy Type

This is a **relative value / spread strategy**: simultaneously long one basket of indices and short another. It is not a directional bet on whether markets go up or down — it profits from the *difference* in performance between the two baskets.

The strategy sits conceptually between:
- **Statistical arbitrage** (mean reversion of correlated assets)
- **Cross-sectional momentum** (finding which baskets are trending relative to each other)

In practice the signals used suggest a blend of both, which creates a tension that has not been formally resolved (see Section 6).

---

## 4. Core Mathematics

### 4.1 Daily Returns
Simple percentage returns: `r(t) = (P(t) - P(t-1)) / P(t-1)`

### 4.2 Rolling Volatility
Rolling 1-year (262 trading day) standard deviation of daily returns. Not annualised — used as a daily figure throughout.

### 4.3 Volatility Scaling
Each instrument is scaled so that its effective daily volatility equals a 1% target:

```
scaling(i) = min(1.0, target_daily_vol / rolling_daily_vol(i))
```

If an instrument's daily vol is 1.5%, scaling = 1%/1.5% = 67%. If vol is 0.8%, scaling would be 1.25% but is capped at 1.0 (no leverage). This is a standard risk-parity / volatility-targeting approach — high-vol instruments get smaller positions.

### 4.4 Spread Return
The daily spread return is the difference between the average vol-scaled return of the long basket and the short basket:

```
spread_return(t) = mean(r_long(t) × scaling_long(t)) - mean(r_short(t) × scaling_short(t))
```

Each side is equally weighted across its chosen instruments. The cumulative spread is `cumprod(1 + spread_return)`.

### 4.5 Position Sizing
Stakes are sized so that a 1 standard deviation price move produces a target P&L (default £500):

```
stake = (target_exposure / (price × daily_vol × point_size)) × scaling
```

Example: FTSE at 8,000, daily vol 1%, target £500, scaling 87%  
→ `500 / (8000 × 0.01 × 1.0) × 0.87 = 5.44 contracts`

### 4.6 Normal Trading Range (NTR)
A fixed index-points range representing typical daily movement for each instrument (e.g., FTSE = 50pts, DAX = 40pts). Used as a contextual reference in the signal scanner but not in position sizing.

---

## 5. Signal Framework

### 5.1 Crossing Signal (Primary Entry Signal)
Measures how far the cumulative spread return is from its 1-year rolling mean, expressed in standard deviations:

```
distance_SD = (cumulative_spread - rolling_mean) / rolling_std
```

A signal fires when `|distance_SD| > 2.0`. This is a **mean-reversion signal**: when the spread is more than 2 SDs above its mean, the short is winning by too much — fade it (or reverse). When more than 2 SDs below, the long is underperforming — add to it.

**Assumption**: The spread will revert toward its rolling mean. This holds if the two baskets are genuinely co-integrated or share a stable long-run relationship.

### 5.2 Trend-Velocity-Ratio (TVR)
Measures the strength of the trend in the cumulative spread relative to its noise:

```
TVR = |OLS slope of cumulative spread over last N points| / daily_vol
```

Higher TVR = stronger trend relative to volatility. Used in the signal scanner and as a search engine ranking metric. **This creates a tension with the crossing signal**: a high TVR spread is trending strongly (momentum), but the crossing signal is looking for mean reversion. A spread that is extended AND strongly trending might be entering a regime break rather than a reversion opportunity.

### 5.3 Velocity and Acceleration
- **Velocity**: 12-day rate-of-change of the cumulative spread
- **Acceleration**: Day-on-day change in velocity

Used to identify whether a trend is gaining or losing momentum. Not currently used as a hard entry/exit signal — displayed for manual interpretation.

### 5.4 Contraction Betas
Beta of each instrument vs the equal-weight average of all instruments. Values >1 mean the instrument amplifies market moves (high beta); <1 means it dampens them. Used to assess whether a basket is systematically more or less reactive to broad market moves — relevant for sizing and risk management.

---

## 6. Portfolio Construction — Search Engine

The search engine enumerates all possible long/short basket combinations for a given leg count (typically 3–4 instruments per side) and scores each on 5 metrics:

| Metric | Description | Direction |
|--------|-------------|-----------|
| **ReturnSD** | Annualised Sharpe ratio of the spread | Higher better |
| **TrendVolRatio** | Trend slope / daily vol | Higher better |
| **ReturnTopology** | Skewness of daily returns | Less negative better |
| **FitDataMinMaxSD** | Cumulative price range / trailing vol | Higher better |
| **LastSD** | Current distance from rolling mean in SDs | Higher absolute value better |

The composite score (post-Q11) is: `z(|LastSD|) + z(TrendVolRatio) + z(WinRate) + z(Expectancy)`  
(each metric z-score normalised before combining to avoid scale bias)

FitDataMinMaxSD was **removed** from the composite formula after Q11 walk-forward analysis showed it is a significant negative predictor of out-of-sample return (ρ = −0.129, p < 0.001). It is still computed and displayed for information.

Three ranking modes are available (selectable in the Search and Backtest tabs):
- **Composite** (default): z(|LastSD|) + z(TrendVolRatio) + z(WinRate) + z(Expectancy). Improved but not validated as having positive OOS predictive power.
- **Cost-Based**: Ranks by lowest estimated trading cost. Since all pairs have similar gross returns (~83% GWR, ~2.6% per trade), selecting the cheapest pairs to hold maximises net expectancy.
- **Contrarian**: Inverts the composite score. Exploits the weak negative IS→OOS rank correlation found in Q11.

**Known limitation**: Scoring ~500K combinations over the same historical window introduces look-ahead / data-mining bias. Use the Walk-Forward tab to validate any scoring approach before trading.

---

## 7. Cost Structure

### 7.1 Bid/Ask Spreads (Entry Cost)
Fixed spreads in index points per instrument:

| Instrument | Spread (pts) | At current price | Approx % cost |
|------------|-------------|-----------------|---------------|
| FTSE | 3 | ~8,000 | 0.038% |
| CAC | 4 | ~8,000 | 0.050% |
| DAX | 4 | ~18,000 | 0.022% |
| MIB | 50 | ~37,000 | 0.135% |
| IBEX | 6 | ~11,000 | 0.055% |
| STOXX50 | 4 | ~5,000 | 0.080% |
| SMI | 5 | ~12,000 | 0.042% |
| HSI | 10 | ~20,000 | 0.050% |
| ASX | 4 | ~8,000 | 0.050% |
| NDX | 2 | ~20,000 | 0.010% |
| SPX | 0.5 | ~5,500 | 0.009% |
| DJI | 5 | ~40,000 | 0.013% |

### 7.2 Daily Financing Cost
Long positions are charged an annual financing rate (default 4.88%). Short positions receive a rebate (default 0.88%). Daily cost = notional × rate / 365.

At a £500 target exposure per instrument and a typical 3v3 basket:
- 3 long notional positions ≈ £15,000–£25,000
- Daily financing drag ≈ £2–£4/day (≈ £500–£1,000/year)

This is a significant cost for a strategy that may hold positions for weeks. **Financing cost has not been systematically modelled against expected P&L in the current framework.**

---

## 8. Known Limitations and Open Questions

### 8.1 Correlation Stability
Global equity indices are highly correlated in crises (correlations converge toward 1.0 during sell-offs) and more differentiated in calm markets. A long/short spread strategy that relies on stable relative performance will suffer when all indices move together. **The framework has no regime detection and does not adjust for correlation breakdown.**

### 8.2 Mean Reversion vs Momentum Tension
The crossing signal assumes mean reversion. TVR and velocity reward trending behaviour. These are contradictory — a spread cannot reliably both trend and mean-revert. The current framework presents both signals simultaneously but provides no rule for resolving the conflict.

### 8.3 No Formal Back-Test
The application shows historical cumulative spread returns and Sharpe ratios, but these are **in-sample** — the same data was used to select the instrument combination. There is no walk-forward test, no out-of-sample period, and no transaction cost adjustment applied to the historical P&L.

### 8.4 Parameter Sensitivity
Key parameters (2 SD threshold, 262-day window, 1% vol target, 12-day ROC) were set empirically. No sensitivity analysis has been done to understand how performance varies across parameter choices.

### 8.5 Execution Assumptions
The framework assumes fills at mid-price plus the fixed spread. In practice, large moves may gap through entry levels, and CFD providers may widen spreads in volatile conditions.

### 8.6 No Drawdown Control
There is no maximum drawdown rule, no stop-loss, and no position reduction mechanism if a spread moves further against the trade. The position sizing is fixed at entry based on vol at that time.

### 8.7 Dividends and Index Adjustments
The strategy uses price-only index data (no total return). For long holding periods, dividend effects accumulate. The framework does not account for this.

### 8.8 Search Engine Scoring (Q11 Findings)
A 14-window walk-forward analysis across 1,516 pair-window observations (Q11) found:

- Composite score vs OOS gross return: Spearman ρ = **−0.037**, p = 0.15 (no predictive power)
- FitDataMinMaxSD vs OOS gross return: ρ = **−0.129**, p < 0.001 (negative predictor — removes worse pairs)
- IS win rate vs OOS win rate: ρ = −0.024 (does not persist)
- IS expectancy vs OOS gross: ρ = +0.010 (does not persist)
- All quintiles have similar OOS gross returns (~2.6%/trade); costs dominate net returns
- The gross-to-net gap (~3.1 ppt) is 13× larger than the Q1-vs-Q5 selection gap (~0.24 ppt)

The crossing signal itself is robust (~83% gross win rate across all quintiles out-of-sample). The weakness is exclusively in the pair selection layer. See `Q11_InSample_Scoring_Research_Paper.docx` for full analysis.

---

## 9. Research Questions

The following questions are the primary focus for deeper analysis:

**Back-testing**
1. What is the historical win rate and payoff ratio of the crossing signal (entry at ±2 SD, exit at mean reversion)? Does it have statistically significant positive expectancy after costs?
2. How sensitive is performance to the 2 SD threshold? Does 1.5 SD or 2.5 SD materially change results?
3. What is the typical holding period after a crossing signal fires? How does this interact with financing costs?
4. How does performance split across market regimes (2000–2008 bull, 2008–2009 crisis, 2010–2020 QE era, 2020–present)?

**Product Expansion**
5. Would FX pairs offer better mean-reversion properties than equity index spreads? FX has stronger theoretical mean reversion (purchasing power parity, carry dynamics) and much lower transaction costs.
6. Would commodity spreads (e.g., WTI/Brent, gold/silver) work within the same framework?
7. Could fixed income instruments (e.g., sovereign bond futures) provide a lower-correlation alternative?
8. Is there value in mixing asset classes within a basket (e.g., long equity + short bonds)?

**Framework Improvements**
9. Should the strategy explicitly detect correlation regimes and reduce exposure when all indices are moving together?
10. Is equal weighting within each basket optimal, or would correlation-adjusted weighting improve Sharpe?
11. Does the search engine's in-sample scoring predict out-of-sample performance, or is it pure data mining?

---

## 10. Appendix — Parameter Reference

| Parameter | Value | Location |
|-----------|-------|----------|
| Target daily vol | 1.0% | `config.py: PARAMS['target_daily_vol']` |
| Vol calculation window | 262 days | `config.py: PARAMS['vol_calc_days']` |
| Crossing signal threshold | ±2.0 SD | `config.py: PARAMS['xing_tolerance_sd']` |
| ROC look-back (velocity) | 12 days | `config.py: PARAMS['roc_days']` |
| Linear fit points | 10 | `config.py: PARAMS['linear_fit_points']` |
| Long financing rate | 4.88% p.a. | `account.json: long_rate` |
| Short rebate rate | 0.88% p.a. | `account.json: short_rate` |
| Default target exposure | £500 per 1 SD | UI sidebar |
| Margin rate (estimate) | 10% | `config.py: PARAMS['margin_rate']` |
