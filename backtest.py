"""
backtest.py — Crossing signal backtest engine
==============================================

Provides the high-level API used by the Backtest tab in app.py and by
standalone research scripts (Q1–Q8).  Delegates computation to numba_core
for speed and uses asset_configs for cost models.

Key functions:
    load_asset_prices()     — load CSV for any asset class
    run_backtest()          — full backtest on a single spread
    run_exhaustive_search() — backtest all NvM basket combinations
    aggregate_trades()      — compute summary statistics from trade arrays
    regime_split()          — split trades by market regime
    sensitivity_grid()      — parameter sweep across SD/exit/financing
"""

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from numba_core import (
    backtest_spread, batch_backtest, detect_trades,
    rolling_mean_std, COL_ENTRY_IDX, COL_EXIT_IDX, COL_SIDE,
    COL_GROSS_RETURN, COL_HOLDING_DAYS,
    BR_N_TRADES, BR_GROSS_WR, BR_AVG_GROSS, BR_AVG_HOLDING,
    BR_AVG_WINNER, BR_AVG_LOSER, BR_PAYOFF_RATIO, BR_TOTAL_GROSS_PNL,
)
from search import _batch_scores
from scoring import apply_scoring, estimate_trade_cost

# Default parameters (matching config.py)
DEFAULT_VOL_WINDOW = 262
DEFAULT_XING_SD = 2.0
DEFAULT_EXIT_SD = 0.0    # full reversion; 0.5 = partial reversion
DEFAULT_TARGET_VOL = 0.01


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_asset_prices(
    filepath: str | Path,
    start_date: str = '1999-01-01',
    min_obs: int = 262,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load a price CSV with Date index and instrument columns.

    Handles prices.csv, fx_prices.csv, commodity_prices.csv, fi_prices.csv,
    or any user-uploaded CSV in the same format.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file.
    start_date : str
        Trim data before this date.
    min_obs : int
        Drop columns with fewer non-null observations.

    Returns
    -------
    prices : pd.DataFrame
        Clean price DataFrame with DatetimeIndex.
    instruments : list[str]
        List of valid instrument codes (column names).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Price file not found: {filepath}")

    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    df = df.ffill(limit=3)

    if start_date:
        df = df.loc[start_date:]

    # Drop instruments with insufficient data
    valid = [c for c in df.columns if df[c].notna().sum() >= min_obs]
    df = df[valid].dropna(how='all')

    return df, valid


def prepare_returns(
    prices: pd.DataFrame,
    instruments: list[str],
    vol_window: int = DEFAULT_VOL_WINDOW,
    target_vol: float = DEFAULT_TARGET_VOL,
    window_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Compute vol-scaled returns matrix from prices.

    Returns
    -------
    scaled : np.ndarray (T, N)
        Vol-normalised daily returns.
    day_ints : np.ndarray (T,) int64
        Integer day count per timestep.
    index : pd.DatetimeIndex
        Date index aligned to the output arrays.
    """
    rets = prices[instruments].pct_change().dropna(how='all')
    vols = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    scalings = (target_vol / vols.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)

    scaled_df = (rets * scalings).dropna(how='any')
    if window_days is not None:
        scaled_df = scaled_df.tail(window_days)

    scaled = scaled_df.values.astype(np.float64)
    index = scaled_df.index
    day_ints = ((index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).values.astype(np.int64)

    return scaled, day_ints, index


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE SPREAD BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(
    spread_returns: np.ndarray,
    day_ints: np.ndarray,
    vol_window: int = DEFAULT_VOL_WINDOW,
    xing_sd: float = DEFAULT_XING_SD,
    exit_sd: float = DEFAULT_EXIT_SD,
    spread_cost_pct: float = 0.001,
    financing_daily_pct: float = 0.0,
    n_legs: int = 2,
) -> dict:
    """
    Run crossing signal backtest on a single spread return series.

    Parameters
    ----------
    spread_returns : np.ndarray (T,)
        Daily vol-scaled spread returns.
    day_ints : np.ndarray (T,)
        Integer day count per timestep.
    vol_window : int
        Rolling window for crossing signal.
    xing_sd : float
        Entry threshold in standard deviations.
    exit_sd : float
        Exit threshold (0.0 = full reversion).
    spread_cost_pct : float
        Round-trip spread cost as fraction (applied once per trade).
    financing_daily_pct : float
        Daily financing cost as fraction (per leg, compounded over holding).
    n_legs : int
        Number of legs (for financing cost scaling).

    Returns
    -------
    dict with keys:
        trades_raw  — np.ndarray (n, 5)
        n_trades    — int
        cum         — np.ndarray (T,)
        dist_sd     — np.ndarray (T,)
        summary     — dict of aggregate statistics
    """
    trades, n_trades, cum, dist_sd = backtest_spread(
        spread_returns, vol_window, xing_sd, exit_sd, day_ints
    )

    summary = aggregate_trades(
        trades, n_trades, spread_cost_pct, financing_daily_pct, n_legs
    )

    return {
        'trades_raw': trades,
        'n_trades': n_trades,
        'cum': cum,
        'dist_sd': dist_sd,
        'summary': summary,
    }


def aggregate_trades(
    trades: np.ndarray,
    n_trades: int,
    spread_cost_pct: float = 0.0,
    financing_daily_pct: float = 0.0,
    n_legs: int = 2,
) -> dict:
    """
    Compute summary statistics from a trade array, including cost-adjusted
    net returns.

    The cost model:
        spread_cost = spread_cost_pct × n_legs × 2 (round trip, both legs)
        fin_cost    = financing_daily_pct × n_legs × holding_days
        net_return  = gross_return - spread_cost - fin_cost
    """
    if n_trades == 0:
        return {
            'n_trades': 0, 'gross_wr': 0.0, 'net_wr': 0.0,
            'avg_gross': 0.0, 'avg_net': 0.0,
            'avg_winner_gross': 0.0, 'avg_loser_gross': 0.0,
            'avg_winner_net': 0.0, 'avg_loser_net': 0.0,
            'payoff_gross': 0.0, 'payoff_net': 0.0,
            'avg_holding': 0.0, 'median_holding': 0.0,
            'avg_spread_cost': 0.0, 'avg_fin_cost': 0.0, 'avg_total_cost': 0.0,
        }

    t = trades[:n_trades]
    gross = t[:, COL_GROSS_RETURN]
    holdings = t[:, COL_HOLDING_DAYS]

    # Cost calculation
    per_trade_spread = spread_cost_pct * n_legs * 2
    fin_costs = financing_daily_pct * n_legs * holdings
    net = gross - per_trade_spread - fin_costs

    # Gross stats
    gross_wins = gross > 0
    gross_losses = ~gross_wins
    n_gw = gross_wins.sum()
    n_gl = gross_losses.sum()
    avg_gw = gross[gross_wins].mean() if n_gw > 0 else 0.0
    avg_gl = gross[gross_losses].mean() if n_gl > 0 else 0.0

    # Net stats
    net_wins = net > 0
    net_losses = ~net_wins
    n_nw = net_wins.sum()
    n_nl = net_losses.sum()
    avg_nw = net[net_wins].mean() if n_nw > 0 else 0.0
    avg_nl = net[net_losses].mean() if n_nl > 0 else 0.0

    return {
        'n_trades': int(n_trades),
        'gross_wr': float(n_gw / n_trades),
        'net_wr': float(n_nw / n_trades),
        'avg_gross': float(gross.mean()),
        'avg_net': float(net.mean()),
        'avg_winner_gross': float(avg_gw),
        'avg_loser_gross': float(avg_gl),
        'avg_winner_net': float(avg_nw),
        'avg_loser_net': float(avg_nl),
        'payoff_gross': float(abs(avg_gw / avg_gl)) if avg_gl != 0 else 0.0,
        'payoff_net': float(abs(avg_nw / avg_nl)) if avg_nl != 0 else 0.0,
        'avg_holding': float(holdings.mean()),
        'median_holding': float(np.median(holdings)),
        'avg_spread_cost': float(per_trade_spread),
        'avg_fin_cost': float(fin_costs.mean()),
        'avg_total_cost': float(per_trade_spread + fin_costs.mean()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXHAUSTIVE / SAMPLED SEARCH WITH BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def estimate_combinations(
    min_long: int, max_long: int,
    min_short: int, max_short: int,
    n: int = 12,
    # Backward compatibility
    min_legs: int | None = None, max_legs: int | None = None,
) -> int:
    """Total non-overlapping (long, short) pairs for given leg ranges."""
    if min_legs is not None:
        min_long = min_short = min_legs
    if max_legs is not None:
        max_long = max_short = max_legs

    total = 0
    for nl in range(min_long, max_long + 1):
        combos_l = list(combinations(range(n), nl))
        for ns in range(min_short, max_short + 1):
            combos_s = list(combinations(range(n), ns))
            for l in combos_l:
                for s in combos_s:
                    if not (set(l) & set(s)):
                        total += 1
    return total


def run_exhaustive_search(
    scaled: np.ndarray,
    day_ints: np.ndarray,
    instruments: list[str],
    display_names: dict[str, str] | None = None,
    min_long_legs: int = 3,
    max_long_legs: int = 3,
    min_short_legs: int = 3,
    max_short_legs: int = 3,
    vol_window: int = DEFAULT_VOL_WINDOW,
    xing_sd: float = DEFAULT_XING_SD,
    exit_sd: float = DEFAULT_EXIT_SD,
    spread_cost_pct: float = 0.001,
    financing_daily_pct: float = 0.0,
    top_n: int = 30,
    sample_n: int = 0,
    seed: int = 42,
    scoring_mode: str = 'composite',
    progress_cb=None,
    # Backward compatibility
    min_legs: int | None = None,
    max_legs: int | None = None,
) -> pd.DataFrame:
    """
    Enumerate and backtest all (long, short) basket combinations.

    Supports asymmetric baskets (different leg counts per side) and
    optional sampling for very large search spaces.

    Parameters
    ----------
    scaled : np.ndarray (T, N)
        Vol-normalised daily returns matrix.
    day_ints : np.ndarray (T,)
        Integer day counts.
    instruments : list[str]
        Instrument codes matching columns of scaled.
    display_names : dict
        Code → display name mapping.  Defaults to codes.
    min_long_legs, max_long_legs : int
        Leg count range for the long basket.
    min_short_legs, max_short_legs : int
        Leg count range for the short basket.
    vol_window, xing_sd, exit_sd : signal parameters
    spread_cost_pct, financing_daily_pct : cost parameters
    top_n : int
        Number of top results to return.
    sample_n : int
        If > 0, randomly sample this many combinations instead of exhaustive.
    seed : int
        Random seed for sampling.
    progress_cb : callable(float) | None

    Returns
    -------
    pd.DataFrame ranked by composite score with backtest metrics.
    """
    # Backward compatibility
    if min_legs is not None:
        min_long_legs = min_short_legs = min_legs
    if max_legs is not None:
        max_long_legs = max_short_legs = max_legs

    if display_names is None:
        display_names = {i: i for i in instruments}

    T, N = scaled.shape

    # Pre-compute equal-weight basket returns for each leg size
    leg_cache = {}
    for k in set(list(range(min_long_legs, max_long_legs + 1)) +
                 list(range(min_short_legs, max_short_legs + 1))):
        combos = list(combinations(range(N), k))
        # (T, M_k) matrix of equal-weight basket returns
        basket_rets = np.zeros((T, len(combos)), dtype=np.float64)
        for ci, combo in enumerate(combos):
            for idx in combo:
                basket_rets[:, ci] += scaled[:, idx]
            basket_rets[:, ci] /= k
        leg_cache[k] = (basket_rets, combos)

    # Enumerate all valid (non-overlapping) long/short pairs
    all_pairs = []
    for nl in range(min_long_legs, max_long_legs + 1):
        _, long_combos = leg_cache[nl]
        for ns in range(min_short_legs, max_short_legs + 1):
            _, short_combos = leg_cache[ns]
            for li, lc in enumerate(long_combos):
                for si, sc in enumerate(short_combos):
                    if not (set(lc) & set(sc)):
                        all_pairs.append((nl, ns, li, si, lc, sc))

    total = len(all_pairs)

    # Sample if requested
    if sample_n > 0 and sample_n < total:
        rng = np.random.RandomState(seed)
        indices = rng.choice(total, sample_n, replace=False)
        all_pairs = [all_pairs[i] for i in sorted(indices)]
        total = len(all_pairs)

    # Process in batches: group by (nl, ns, li) to reuse spread_mat broadcasting
    records = []
    done = 0

    for nl in range(min_long_legs, max_long_legs + 1):
        long_rets, long_combos = leg_cache[nl]

        for ns in range(min_short_legs, max_short_legs + 1):
            short_rets, short_combos = leg_cache[ns]
            n_legs_total = nl + ns

            for li, long_combo in enumerate(long_combos):
                # Build spread matrix: this long basket vs ALL short baskets
                lr = long_rets[:, li]
                spread_mat = lr[:, None] - short_rets  # (T, M_short)

                # Batch backtest and spread-shape metrics (same vectorised pass)
                batch_results  = batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints)
                shape_scores   = _batch_scores(spread_mat)

                # Extract results for valid (non-overlapping) combos only
                for si, short_combo in enumerate(short_combos):
                    # Check if this (nl, ns, li, si) pair is in our list
                    if set(long_combo) & set(short_combo):
                        continue

                    done += 1
                    br = batch_results[si]

                    n_trades = int(br[BR_N_TRADES])
                    if n_trades == 0:
                        continue

                    avg_holding = float(br[BR_AVG_HOLDING])
                    # User-rate net (shown in results, reflects their actual broker cost)
                    total_spread_cost = spread_cost_pct * n_legs_total * 2
                    avg_fin_cost = financing_daily_pct * n_legs_total * avg_holding
                    avg_net = float(br[BR_AVG_GROSS]) - total_spread_cost - avg_fin_cost
                    # Account-rate cost (used for cost_rank scoring mode)
                    est_cost = estimate_trade_cost(avg_holding, nl, ns, spread_cost_pct)

                    long_names = [display_names.get(instruments[i], instruments[i])
                                  for i in long_combo]
                    short_names = [display_names.get(instruments[i], instruments[i])
                                   for i in short_combo]

                    sc = {k: float(v[si]) for k, v in shape_scores.items()}
                    records.append({
                        'Long': ' | '.join(long_names),
                        'Short': ' | '.join(short_names),
                        'Config': f'{nl}v{ns}',
                        '_long_flags': {instruments[i]: 1 for i in long_combo},
                        '_short_flags': {instruments[i]: 1 for i in short_combo},
                        # Backtest metrics
                        'Trades':        n_trades,
                        'WinRate':       float(br[BR_GROSS_WR]),
                        'Expectancy':    float(br[BR_AVG_GROSS]),
                        'NetExpectancy': avg_net,
                        'EstCost':       est_cost,
                        'AvgHolding':    avg_holding,
                        'PayoffRatio':   float(br[BR_PAYOFF_RATIO]),
                        'AvgWinner':     float(br[BR_AVG_WINNER]),
                        'AvgLoser':      float(br[BR_AVG_LOSER]),
                        'SpreadCost':    total_spread_cost,
                        'FinCost':       avg_fin_cost,
                        **sc,  # ReturnSD, TrendVolRatio, ReturnTopology, FitDataMinMaxSD, LastSD
                    })

                if progress_cb and done % max(1, total // 50) == 0:
                    progress_cb(min(done / total, 1.0))

    if progress_cb:
        progress_cb(1.0)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = apply_scoring(df, scoring_mode)
    df = (df.sort_values('_score', ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    df.index += 1
    return df.drop(columns=['_score'])


# ═══════════════════════════════════════════════════════════════════════════
# REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

REGIMES = [
    ('Pre-GFC',    2000, 2007),
    ('GFC',        2008, 2009),
    ('Post-GFC',   2010, 2014),
    ('Low Vol',    2015, 2019),
    ('COVID+',     2020, 2022),
    ('Recent',     2023, 2026),
]


def regime_split(
    trades: np.ndarray,
    n_trades: int,
    index: pd.DatetimeIndex,
    spread_cost_pct: float = 0.0,
    financing_daily_pct: float = 0.0,
    n_legs: int = 2,
) -> list[dict]:
    """
    Split trades by market regime and compute per-regime statistics.

    Parameters
    ----------
    trades : np.ndarray (n, 5)
    n_trades : int
    index : pd.DatetimeIndex
        The date index of the original return series (for mapping indices to dates).
    spread_cost_pct, financing_daily_pct, n_legs : cost parameters

    Returns
    -------
    list of dicts, one per regime that has trades.
    """
    if n_trades == 0:
        return []

    t = trades[:n_trades]
    entry_indices = t[:, COL_ENTRY_IDX].astype(int)

    # Map array indices to years
    entry_years = np.array([index[i].year for i in entry_indices])

    results = []
    for name, y_start, y_end in REGIMES:
        mask = (entry_years >= y_start) & (entry_years <= y_end)
        regime_trades = t[mask]
        n_regime = mask.sum()
        if n_regime < 5:
            continue

        summary = aggregate_trades(
            regime_trades, n_regime,
            spread_cost_pct, financing_daily_pct, n_legs
        )
        summary['regime'] = name
        summary['year_start'] = y_start
        summary['year_end'] = y_end
        results.append(summary)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SENSITIVITY GRID
# ═══════════════════════════════════════════════════════════════════════════

def sensitivity_grid(
    spread_returns: np.ndarray,
    day_ints: np.ndarray,
    vol_window: int = DEFAULT_VOL_WINDOW,
    sd_thresholds: list[float] | None = None,
    exit_targets: list[float] | None = None,
    financing_rates: list[float] | None = None,
    spread_cost_pct: float = 0.001,
    n_legs: int = 2,
) -> pd.DataFrame:
    """
    Run a parameter sweep and return results as a DataFrame.

    Defaults:
        sd_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
        exit_targets = [0.0, 0.5, 1.0]
        financing_rates = [0.0, 0.5, 1.0, 1.8, 3.0, 4.88]  (annual %)
    """
    if sd_thresholds is None:
        sd_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    if exit_targets is None:
        exit_targets = [0.0, 0.5, 1.0]
    if financing_rates is None:
        financing_rates = [0.0, 0.5, 1.0, 1.8, 3.0, 4.88]

    rows = []
    for sd in sd_thresholds:
        for ex in exit_targets:
            trades, n_trades, _, _ = backtest_spread(
                spread_returns, vol_window, sd, ex, day_ints
            )
            for fin_ann in financing_rates:
                fin_daily = fin_ann / 100 / 365
                summary = aggregate_trades(
                    trades, n_trades, spread_cost_pct, fin_daily, n_legs
                )
                rows.append({
                    'SD_Threshold': sd,
                    'Exit_Target': ex,
                    'Financing_Ann_Pct': fin_ann,
                    **summary,
                })

    return pd.DataFrame(rows)


def find_breakeven_financing(
    trades: np.ndarray,
    n_trades: int,
    spread_cost_pct: float = 0.001,
    n_legs: int = 2,
    precision: float = 0.01,
) -> float:
    """
    Binary search for the annual financing rate where net expectancy = 0.

    Returns the rate as a percentage (e.g. 0.5 means 0.5% p.a.).
    Returns -1 if net expectancy is positive even at 10% financing,
    or if no trades exist.
    """
    if n_trades == 0:
        return -1.0

    lo, hi = 0.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        fin_daily = mid / 100 / 365
        summary = aggregate_trades(trades, n_trades, spread_cost_pct, fin_daily, n_legs)
        if summary['avg_net'] > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < precision:
            break

    return round((lo + hi) / 2, 2)
