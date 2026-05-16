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
from asset_configs import basket_spread_cost as _basket_spread_cost, FI_EXCLUDE as _FI_EXCLUDE

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


def load_cross_asset_prices(
    long_class: str,
    short_class: str,
    cache_dir,
    fi_exclude: frozenset = _FI_EXCLUDE,
    start_date: str = '1999-01-01',
    min_common_days: int = 500,
) -> tuple:
    """
    Load and date-align prices from two asset class CSVs for cross-asset search.

    Uses INNER JOIN on dates — only trading days present in BOTH datasets are kept.

    Returns
    -------
    prices : pd.DataFrame
        Unified price DataFrame (DatetimeIndex, all instruments as columns).
    long_instruments : list[str]
    short_instruments : list[str]
    asset_class_map : dict[str, str]
        {instrument_code: asset_class_key}
    """
    from asset_configs import ASSET_CLASSES, _KEY_ALIASES

    cache_dir = Path(cache_dir)

    def _load(cls: str) -> tuple:
        canonical = _KEY_ALIASES.get(cls, cls)
        cfg = ASSET_CLASSES[canonical]
        path = cache_dir / cfg['csv_file']
        if not path.exists():
            raise FileNotFoundError(
                f"Price file not found: {path}\n"
                f"Expected for asset class '{cls}'."
            )
        df, instruments = load_asset_prices(str(path), start_date=start_date)
        if canonical == 'fixed_income':
            instruments = [i for i in instruments if i not in fi_exclude]
            df = df[instruments]
        return df, instruments

    long_prices,  long_instr  = _load(long_class)
    short_prices, short_instr = _load(short_class)

    overlap = set(long_instr) & set(short_instr)
    if overlap:
        raise ValueError(
            f"Instrument code collision between '{long_class}' and '{short_class}': {overlap}"
        )

    common_idx = long_prices.index.intersection(short_prices.index)
    if len(common_idx) < min_common_days:
        raise ValueError(
            f"Only {len(common_idx)} common trading days between "
            f"'{long_class}' and '{short_class}' (minimum {min_common_days})."
        )

    prices = pd.concat([
        long_prices.loc[common_idx,  long_instr],
        short_prices.loc[common_idx, short_instr],
    ], axis=1)

    asset_class_map = (
        {i: long_class  for i in long_instr} |
        {i: short_class for i in short_instr}
    )

    return prices, long_instr, short_instr, asset_class_map


def prepare_returns_aligned(
    prices: pd.DataFrame,
    long_instruments: list,
    short_instruments: list,
    vol_window: int = DEFAULT_VOL_WINDOW,
    target_vol: float = DEFAULT_TARGET_VOL,
    window_days: int | None = None,
) -> tuple:
    """
    Vol-scale returns for a cross-asset unified price DataFrame.

    Returns separate scaled arrays for long and short instruments,
    plus a shared day_ints array aligned to the common date range.

    Returns
    -------
    long_scaled  : np.ndarray (T, N_long)
    short_scaled : np.ndarray (T, N_short)
    day_ints     : np.ndarray (T,) int64
    index        : pd.DatetimeIndex
    """
    all_instruments = long_instruments + short_instruments
    rets = prices[all_instruments].pct_change().dropna(how='all')
    vols = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    scalings = (target_vol / vols.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    scaled_df = (rets * scalings).dropna(how='any')

    if window_days is not None:
        scaled_df = scaled_df.tail(window_days)

    index    = scaled_df.index
    day_ints = ((index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).values.astype(np.int64)

    n_long       = len(long_instruments)
    long_scaled  = scaled_df[long_instruments].values.astype(np.float64)
    short_scaled = scaled_df[short_instruments].values.astype(np.float64)

    return long_scaled, short_scaled, day_ints, index


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
    long_instrument_subset: list[str] | None = None,
    short_instrument_subset: list[str] | None = None,
    min_long_legs: int = 3,
    max_long_legs: int = 3,
    min_short_legs: int = 3,
    max_short_legs: int = 3,
    vol_window: int = DEFAULT_VOL_WINDOW,
    xing_sd: float = DEFAULT_XING_SD,
    exit_sd: float = DEFAULT_EXIT_SD,
    spread_cost_pct: float = 0.001,
    spread_cost_lookup: dict[str, float] | None = None,
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
    long_instrument_subset : list[str] | None
        If provided, only these instruments are eligible for the long basket.
        Must be a subset of `instruments`. When None, all instruments are eligible.
        Used for cross-asset search (e.g. long_instrument_subset = equity instruments).
    short_instrument_subset : list[str] | None
        Same as long_instrument_subset but for the short basket. When both are None
        the function behaves identically to the current implementation (backward compat).
    min_long_legs, max_long_legs : int
        Leg count range for the long basket.
    min_short_legs, max_short_legs : int
        Leg count range for the short basket.
    vol_window, xing_sd, exit_sd : signal parameters
    spread_cost_pct : float
        Flat round-trip spread cost per instrument (used when spread_cost_lookup is None).
    spread_cost_lookup : dict[str, float] | None
        Per-instrument one-way spread cost as fraction (from asset_configs.get_spread_cost_lookup).
        When provided, overrides spread_cost_pct for cost calculation using actual instrument
        spreads. When None, falls back to the flat spread_cost_pct value.
        Example: {'UKX': 0.00035, 'CBK': 0.00049, ...}
    financing_daily_pct : float
        Daily financing cost as fraction.
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

    # Resolve instrument subsets to column index positions
    all_idx   = list(range(N))
    long_idx  = (
        [instruments.index(i) for i in long_instrument_subset  if i in instruments]
        if long_instrument_subset  is not None else all_idx
    )
    short_idx = (
        [instruments.index(i) for i in short_instrument_subset if i in instruments]
        if short_instrument_subset is not None else all_idx
    )

    # Pre-compute equal-weight basket returns for long and short sides separately
    def _build_leg_cache(idx_list, k_min, k_max):
        cache = {}
        for k in range(k_min, k_max + 1):
            combos = list(combinations(idx_list, k))
            basket_rets = np.zeros((T, len(combos)), dtype=np.float64)
            for ci, combo in enumerate(combos):
                for idx in combo:
                    basket_rets[:, ci] += scaled[:, idx]
                basket_rets[:, ci] /= k
            cache[k] = (basket_rets, combos)
        return cache

    leg_cache_long  = _build_leg_cache(long_idx,  min_long_legs,  max_long_legs)
    leg_cache_short = _build_leg_cache(short_idx, min_short_legs, max_short_legs)

    # Enumerate all valid (non-overlapping) long/short pairs
    all_pairs = []
    for nl in range(min_long_legs, max_long_legs + 1):
        _, long_combos = leg_cache_long[nl]
        for ns in range(min_short_legs, max_short_legs + 1):
            _, short_combos = leg_cache_short[ns]
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
        long_rets, long_combos = leg_cache_long[nl]

        for ns in range(min_short_legs, max_short_legs + 1):
            short_rets, short_combos = leg_cache_short[ns]
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
                    if spread_cost_lookup is not None:
                        total_spread_cost = _basket_spread_cost(
                            long_combo, short_combo, instruments, spread_cost_lookup
                        )
                    else:
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
