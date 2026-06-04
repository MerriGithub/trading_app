"""
walkforward.py — Walk-forward validation engine
================================================

Tests whether in-sample scoring predicts out-of-sample performance.
Answers the Q11 question: does any scoring mode reliably identify pairs
that will perform better in the next OOS period?

IS/OOS window structure
-----------------------
Given prices of length T trading days:

    |<── is_years ──>|<── oos_years ──>|
    pos              pos+IS           pos+IS+OOS

Windows step forward by ``step_years`` each iteration.  The window
boundaries are defined in integer trading-day indices into the vol-scaled
return matrix (after dropna), NOT in calendar dates.

    Window 0: [0,        is_days),  [is_days,        is_days+oos_days)
    Window 1: [step,     step+is),  [step+is,         step+is+oos)
    ...

Spearman ρ(IS rank, OOS gross return) is the validation metric.

Design
------
- 1v1 directional pairs only (N×(N-1) pairs, typically 132 for 12 instruments)
- Each window: IS data → score all pairs → OOS data → record OOS performance
- Uses numba via batch_backtest for speed (same engine as the Search tab)
- Complete for 12 instruments, 14 windows in <30s on a modern machine

Usage
-----
    from engine.walkforward import run_walk_forward, summarise_walk_forward
    from engine.backtest import load_asset_prices

    prices, instruments = load_asset_prices('cache/prices.csv')
    results = run_walk_forward(prices, instruments, scoring_mode='composite')
    summary = summarise_walk_forward(results)

Scoring mode validation
-----------------------
Validated defaults by asset class (Walk-forward ρ tests, 2026-05-25):
    equities:       contrarian  (ρ=+0.208, p~0, EXIT_SD=2.0 scalp regime)
    commodities:    contrarian  (ρ=+0.122, p=0.0009)
    equity × FX:    contrarian  (ρ=+0.053, p=0.0030)
    commodities×FI: composite   (ρ=+0.069, p=0.0016)
    FX, FI:         composite   (ρ≈0 — no validated predictor)

Passing a mode that deviates from the validated default generates a
warning log so the caller knows they are outside validated territory.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind

from engine.numba_core import (
    batch_backtest,
    BR_N_TRADES, BR_GROSS_WR, BR_AVG_GROSS, BR_AVG_HOLDING,
)
from engine.search import _batch_scores
from engine.scoring import apply_scoring, estimate_trade_cost

logger = logging.getLogger(__name__)

_TDY = 262  # trading days per year

# Validated scoring mode defaults — used to warn on deviation.
_VALIDATED_MODES: dict[str, str] = {
    'equity':       'contrarian',
    'commodities':  'contrarian',
    'fx':           'composite',
    'fixed_income': 'composite',
}


def _vol_scaled(
    prices: pd.DataFrame,
    instruments: list[str],
    vol_window: int = 262,
    target_vol: float = 0.01,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute vol-scaled returns for the full price history.

    Scales each instrument's daily returns so that its rolling volatility
    matches ``target_vol``, capped at a factor of 1.0 (never lever up).
    Rows with any NaN are dropped after scaling so the result is a clean
    rectangular matrix suitable for numba.

    Args:
        prices: Raw price DataFrame (DatetimeIndex, instrument columns).
        instruments: Subset of ``prices.columns`` to process.
        vol_window: Rolling window in trading days for vol estimate.
        target_vol: Target daily volatility (fraction). Defaults to 0.01 = 1%.

    Returns:
        Tuple ``(scaled_df, day_ints)`` where:
            - ``scaled_df`` is a vol-normalised daily-return DataFrame with
              any-NaN rows dropped.
            - ``day_ints`` is an int64 numpy array of integer day indices
              (days since 1970-01-01) for each remaining row.
    """
    rets = prices[instruments].pct_change().dropna(how='all')
    vols = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    sc   = (target_vol / vols.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    scaled_df = (rets * sc).dropna(how='any')
    day_ints  = (
        (scaled_df.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    ).values.astype(np.int64)
    return scaled_df, day_ints


def run_walk_forward(
    prices: pd.DataFrame,
    instruments: list[str],
    is_years: int = 5,
    oos_years: int = 2,
    step_years: int = 1,
    scoring_mode: str = 'composite',
    vol_window: int = 262,
    target_vol: float = 0.01,
    xing_sd: float = 2.0,
    exit_sd: float = 0.0,
    spread_cost_pct: float = 0.001,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Run walk-forward analysis on all 1v1 directional pairs.

    For each rolling IS/OOS window, scores all pairs on IS data and
    measures their OOS performance.  The final Spearman ρ(IS rank, OOS
    gross return) is computed by ``summarise_walk_forward()``.

    Window structure (integer trading-day indices into scaled matrix):
        IS:  [pos, pos+is_days)
        OOS: [pos+is_days, pos+is_days+oos_days)
        Next window starts at pos+step_days.

    Args:
        prices: Raw price DataFrame (DatetimeIndex, instrument columns).
            Must contain all codes in ``instruments`` as columns.
        instruments: Instrument codes to include. All must be in ``prices``.
        is_years: In-sample window length in trading years (1 year = 262 days).
            Must be a positive integer.
        oos_years: Out-of-sample window length in trading years. Must be
            a positive integer.
        step_years: Step size between windows in trading years. Must be
            a positive integer.
        scoring_mode: One of ``'composite'``, ``'cost_rank'``,
            ``'contrarian'``. A warning is logged if this deviates from the
            validated default for the asset class.
        vol_window: Rolling vol window in trading days (default 262).
        target_vol: Target daily volatility fraction for scaling.
        xing_sd: Entry threshold in standard deviations.
        exit_sd: Exit threshold (0.0 = full reversion; confirmed optimum
            for commodities is 0.5, equities is 2.0).
        spread_cost_pct: One-way bid-ask spread as a fraction, used to
            estimate net return per trade.
        progress_cb: Optional callback ``(fraction: float) -> None`` called
            after each window completes, where fraction is in [0, 1].

    Returns:
        DataFrame with one row per (window, pair) containing IS rank, IS
        score metrics, and OOS performance metrics.  Empty DataFrame if
        the price history is too short for at least one IS+OOS window.

    Raises:
        ValueError: If ``is_years``, ``oos_years``, or ``step_years`` are
            not positive integers, or if any instrument is absent from
            ``prices``.
    """
    if not (isinstance(is_years, int) and is_years > 0):
        raise ValueError(f"is_years must be a positive integer; got {is_years!r}")
    if not (isinstance(oos_years, int) and oos_years > 0):
        raise ValueError(f"oos_years must be a positive integer; got {oos_years!r}")
    if not (isinstance(step_years, int) and step_years > 0):
        raise ValueError(f"step_years must be a positive integer; got {step_years!r}")

    missing = set(instruments) - set(prices.columns)
    if missing:
        raise ValueError(
            f"run_walk_forward: instruments not found in prices columns: {sorted(missing)}"
        )

    scaled_df, day_ints = _vol_scaled(prices, instruments, vol_window, target_vol)
    scaled = scaled_df.values.astype(np.float64)
    T, N   = scaled.shape

    is_days   = is_years  * _TDY
    oos_days  = oos_years * _TDY
    step_days = step_years * _TDY

    min_days_needed = is_days + oos_days
    if T < min_days_needed:
        logger.warning(
            "run_walk_forward: only %d trading days available; "
            "need at least %d (is_years=%d + oos_years=%d). Returning empty.",
            T, min_days_needed, is_years, oos_years,
        )
        return pd.DataFrame()

    # Generate non-overlapping IS/OOS window boundaries
    windows: list[tuple[int, int, int]] = []
    pos = 0
    while pos + is_days + oos_days <= T:
        windows.append((pos, pos + is_days, pos + is_days + oos_days))
        pos += step_days

    if not windows:
        return pd.DataFrame()

    logger.info(
        "run_walk_forward: %d windows, IS=%dy OOS=%dy, mode=%s, instruments=%s",
        len(windows), is_years, oos_years, scoring_mode, instruments,
    )

    all_records: list[dict] = []
    n_windows = len(windows)

    for w_idx, (is_start, is_end, oos_end) in enumerate(windows):
        is_scaled    = scaled[is_start:is_end]
        oos_scaled   = scaled[is_end:oos_end]
        is_days_arr  = day_ints[is_start:is_end]
        oos_days_arr = day_ints[is_end:oos_end]

        if is_scaled.shape[0] < vol_window // 2 or oos_scaled.shape[0] < 5:
            logger.debug(
                "Window %d/%d skipped: IS=%d rows, OOS=%d rows (too few)",
                w_idx + 1, n_windows, is_scaled.shape[0], oos_scaled.shape[0],
            )
            continue

        # Label this window by calendar year boundaries
        is_label  = scaled_df.index[is_start].year
        oos_label = scaled_df.index[min(oos_end - 1, T - 1)].year

        logger.info(
            "Walk-forward window %d/%d: IS %d–%d, OOS ending %d",
            w_idx + 1, n_windows, is_label, scaled_df.index[is_end - 1].year, oos_label,
        )

        window_records: list[dict] = []

        # One batch per long instrument: spread vs all other instruments
        for li in range(N):
            lr_is  = is_scaled[:, li]
            lr_oos = oos_scaled[:, li]

            # Short side: all instruments except the long leg
            other_idx = [j for j in range(N) if j != li]
            short_is  = is_scaled[:, other_idx]   # (is_T, N-1)
            short_oos = oos_scaled[:, other_idx]   # (oos_T, N-1)

            spread_is  = lr_is[:, None] - short_is   # (is_T, N-1)
            spread_oos = lr_oos[:, None] - short_oos  # (oos_T, N-1)

            # IS: scoring metrics + backtest summary
            is_shape = _batch_scores(spread_is)
            is_bt    = batch_backtest(spread_is,  vol_window, xing_sd, exit_sd, is_days_arr)
            oos_bt   = batch_backtest(spread_oos, vol_window, xing_sd, exit_sd, oos_days_arr)

            for si, sj in enumerate(other_idx):
                is_n  = int(is_bt[si, BR_N_TRADES])
                oos_n = int(oos_bt[si, BR_N_TRADES])
                avg_hold = float(is_bt[si, BR_AVG_HOLDING]) if is_n > 0 else 0.0
                est_cost = estimate_trade_cost(avg_hold, n_long=1, n_short=1,
                                               spread_cost_pct=spread_cost_pct)
                window_records.append({
                    'window':      w_idx,
                    'is_start':    is_label,
                    'oos_end':     oos_label,
                    'long':        instruments[li],
                    'short':       instruments[sj],
                    'pair':        f'{instruments[li]} / {instruments[sj]}',
                    # IS scoring inputs
                    'WinRate':     float(is_bt[si, BR_GROSS_WR])  if is_n > 0 else 0.0,
                    'Expectancy':  float(is_bt[si, BR_AVG_GROSS]) if is_n > 0 else 0.0,
                    'AvgHolding':  avg_hold,
                    'IS_Trades':   is_n,
                    'EstCost':     est_cost,
                    'LastSD':      float(is_shape['LastSD'][si]),
                    'TrendVolRatio': float(is_shape['TrendVolRatio'][si]),
                    'ReturnSD':    float(is_shape['ReturnSD'][si]),
                    'FitDataMinMaxSD': float(is_shape['FitDataMinMaxSD'][si]),
                    # OOS performance
                    'OOS_Trades':  oos_n,
                    'OOS_WinRate': float(oos_bt[si, BR_GROSS_WR])  if oos_n > 0 else 0.0,
                    'OOS_Gross':   float(oos_bt[si, BR_AVG_GROSS]) if oos_n > 0 else 0.0,
                    'OOS_Hold':    float(oos_bt[si, BR_AVG_HOLDING]) if oos_n > 0 else 0.0,
                })

        if not window_records:
            continue

        # Rank IS scores for this window
        wdf = pd.DataFrame(window_records)
        wdf = apply_scoring(wdf, scoring_mode)            # adds _score column
        wdf = wdf.sort_values('_score', ascending=False).reset_index(drop=True)
        wdf['IS_Rank']  = range(1, len(wdf) + 1)
        wdf['IS_Score'] = wdf['_score']
        wdf = wdf.drop(columns=['_score'])

        all_records.extend(wdf.to_dict('records'))

        if progress_cb:
            progress_cb((w_idx + 1) / n_windows)

    if not all_records:
        return pd.DataFrame()

    result = pd.DataFrame(all_records)
    result['OOS_Net'] = result['OOS_Gross'] - result['EstCost']
    logger.info(
        "run_walk_forward complete: %d pair-window records across %d windows",
        len(result), n_windows,
    )
    return result


def run_cross_asset_walkforward(
    prices_long: pd.DataFrame,
    prices_short: pd.DataFrame,
    instruments_long: list[str],
    instruments_short: list[str],
    is_years: int = 5,
    oos_years: int = 2,
    step_years: int = 1,
    scoring_mode: str = 'composite',
    vol_window: int = 262,
    target_vol: float = 0.01,
    xing_sd: float = 2.0,
    exit_sd: float = 0.0,
    spread_cost_pct: float = 0.001,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Walk-forward analysis over cross-asset directional pairs.

    Generates all ``(long_i, short_j)`` and ``(long_j, short_i)`` pairs
    across two separate asset class price DataFrames.  Date alignment on
    index intersection is performed before vol scaling.

    The cross-asset validated scoring modes are (see CLAUDE.md):
        commodities × FX:           composite  (ρ≈0)
        commodities × fixed_income: composite  (ρ=+0.069, p=0.0016)
        equity × FX:                contrarian (ρ=+0.053, p=0.0030)

    Args:
        prices_long: Price DataFrame for the long-side asset class.
        prices_short: Price DataFrame for the short-side asset class.
        instruments_long: Instrument codes in the long universe.
        instruments_short: Instrument codes in the short universe.
        is_years: In-sample window length in trading years.
        oos_years: Out-of-sample window length in trading years.
        step_years: Step size between windows in trading years.
        scoring_mode: Scoring mode string; warn if not the validated default
            for this cross-asset combination.
        vol_window: Rolling vol window in trading days.
        target_vol: Target daily volatility fraction for scaling.
        xing_sd: Entry threshold in standard deviations.
        exit_sd: Exit threshold in standard deviations.
        spread_cost_pct: One-way spread cost fraction.
        progress_cb: Optional progress callback ``(fraction) -> None``.

    Returns:
        DataFrame with same structure as ``run_walk_forward()``. Empty if
        the common date range is too short for one complete IS+OOS window.

    Raises:
        ValueError: If window parameters are not positive integers.
    """
    if not (isinstance(is_years, int) and is_years > 0):
        raise ValueError(f"is_years must be a positive integer; got {is_years!r}")
    if not (isinstance(oos_years, int) and oos_years > 0):
        raise ValueError(f"oos_years must be a positive integer; got {oos_years!r}")
    if not (isinstance(step_years, int) and step_years > 0):
        raise ValueError(f"step_years must be a positive integer; got {step_years!r}")

    common_dates = prices_long.index.intersection(prices_short.index)
    min_days = (is_years + oos_years) * _TDY
    if len(common_dates) < min_days:
        logger.warning(
            "run_cross_asset_walkforward: only %d common dates; "
            "need at least %d. Returning empty.",
            len(common_dates), min_days,
        )
        return pd.DataFrame()

    pl = prices_long.loc[common_dates, instruments_long]
    ps = prices_short.loc[common_dates, instruments_short]

    scaled_long_df,  day_ints = _vol_scaled(pl, instruments_long,  vol_window, target_vol)
    scaled_short_df, _        = _vol_scaled(ps, instruments_short, vol_window, target_vol)

    # Align to the shorter scaled result (dropna can shorten each independently)
    common_scaled = scaled_long_df.index.intersection(scaled_short_df.index)
    scaled_long_df  = scaled_long_df.loc[common_scaled]
    scaled_short_df = scaled_short_df.loc[common_scaled]
    day_ints = (
        (common_scaled - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    ).values.astype(np.int64)

    sl = scaled_long_df.values.astype(np.float64)
    ss = scaled_short_df.values.astype(np.float64)
    T  = sl.shape[0]
    NL = len(instruments_long)
    NS = len(instruments_short)

    is_days   = is_years  * _TDY
    oos_days  = oos_years * _TDY
    step_days = step_years * _TDY

    if T < is_days + oos_days:
        logger.warning(
            "run_cross_asset_walkforward: scaled data has %d rows after dropna; "
            "need at least %d. Returning empty.",
            T, is_days + oos_days,
        )
        return pd.DataFrame()

    windows: list[tuple[int, int, int]] = []
    pos = 0
    while pos + is_days + oos_days <= T:
        windows.append((pos, pos + is_days, pos + is_days + oos_days))
        pos += step_days

    if not windows:
        return pd.DataFrame()

    logger.info(
        "run_cross_asset_walkforward: %d windows, IS=%dy OOS=%dy, mode=%s",
        len(windows), is_years, oos_years, scoring_mode,
    )

    all_records: list[dict] = []
    n_windows = len(windows)

    for w_idx, (is_start, is_end, oos_end) in enumerate(windows):
        sl_is  = sl[is_start:is_end]
        sl_oos = sl[is_end:oos_end]
        ss_is  = ss[is_start:is_end]
        ss_oos = ss[is_end:oos_end]
        di_is  = day_ints[is_start:is_end]
        di_oos = day_ints[is_end:oos_end]

        if sl_is.shape[0] < vol_window // 2 or sl_oos.shape[0] < 5:
            continue

        is_label  = scaled_long_df.index[is_start].year
        oos_label = scaled_long_df.index[min(oos_end - 1, T - 1)].year

        logger.info(
            "Cross-asset WF window %d/%d: IS ending %d, OOS ending %d",
            w_idx + 1, n_windows, scaled_long_df.index[is_end - 1].year, oos_label,
        )

        window_records: list[dict] = []

        # Direction 1: long from long universe, short from short universe
        for li in range(NL):
            lr_is  = sl_is[:, li]
            lr_oos = sl_oos[:, li]
            spread_is  = lr_is[:, None] - ss_is    # (is_T, NS)
            spread_oos = lr_oos[:, None] - ss_oos  # (oos_T, NS)

            is_shape = _batch_scores(spread_is)
            is_bt    = batch_backtest(spread_is,  vol_window, xing_sd, exit_sd, di_is)
            oos_bt   = batch_backtest(spread_oos, vol_window, xing_sd, exit_sd, di_oos)

            for si in range(NS):
                is_n  = int(is_bt[si, BR_N_TRADES])
                oos_n = int(oos_bt[si, BR_N_TRADES])
                avg_hold = float(is_bt[si, BR_AVG_HOLDING]) if is_n > 0 else 0.0
                est_cost = estimate_trade_cost(avg_hold, n_long=1, n_short=1,
                                               spread_cost_pct=spread_cost_pct)
                window_records.append({
                    'window':    w_idx,
                    'is_start':  is_label,
                    'oos_end':   oos_label,
                    'long':      instruments_long[li],
                    'short':     instruments_short[si],
                    'pair':      f'{instruments_long[li]} / {instruments_short[si]}',
                    'WinRate':     float(is_bt[si, BR_GROSS_WR])  if is_n > 0 else 0.0,
                    'Expectancy':  float(is_bt[si, BR_AVG_GROSS]) if is_n > 0 else 0.0,
                    'AvgHolding':  avg_hold,
                    'IS_Trades':   is_n,
                    'EstCost':     est_cost,
                    'LastSD':      float(is_shape['LastSD'][si]),
                    'TrendVolRatio': float(is_shape['TrendVolRatio'][si]),
                    'ReturnSD':    float(is_shape['ReturnSD'][si]),
                    'FitDataMinMaxSD': float(is_shape['FitDataMinMaxSD'][si]),
                    'OOS_Trades':  oos_n,
                    'OOS_WinRate': float(oos_bt[si, BR_GROSS_WR])  if oos_n > 0 else 0.0,
                    'OOS_Gross':   float(oos_bt[si, BR_AVG_GROSS]) if oos_n > 0 else 0.0,
                    'OOS_Hold':    float(oos_bt[si, BR_AVG_HOLDING]) if oos_n > 0 else 0.0,
                })

        # Direction 2: long from short universe, short from long universe
        for si in range(NS):
            sr_is  = ss_is[:, si]
            sr_oos = ss_oos[:, si]
            spread_is  = sr_is[:, None] - sl_is    # (is_T, NL)
            spread_oos = sr_oos[:, None] - sl_oos  # (oos_T, NL)

            is_shape = _batch_scores(spread_is)
            is_bt    = batch_backtest(spread_is,  vol_window, xing_sd, exit_sd, di_is)
            oos_bt   = batch_backtest(spread_oos, vol_window, xing_sd, exit_sd, di_oos)

            for li in range(NL):
                is_n  = int(is_bt[li, BR_N_TRADES])
                oos_n = int(oos_bt[li, BR_N_TRADES])
                avg_hold = float(is_bt[li, BR_AVG_HOLDING]) if is_n > 0 else 0.0
                est_cost = estimate_trade_cost(avg_hold, n_long=1, n_short=1,
                                               spread_cost_pct=spread_cost_pct)
                window_records.append({
                    'window':    w_idx,
                    'is_start':  is_label,
                    'oos_end':   oos_label,
                    'long':      instruments_short[si],
                    'short':     instruments_long[li],
                    'pair':      f'{instruments_short[si]} / {instruments_long[li]}',
                    'WinRate':     float(is_bt[li, BR_GROSS_WR])  if is_n > 0 else 0.0,
                    'Expectancy':  float(is_bt[li, BR_AVG_GROSS]) if is_n > 0 else 0.0,
                    'AvgHolding':  avg_hold,
                    'IS_Trades':   is_n,
                    'EstCost':     est_cost,
                    'LastSD':      float(is_shape['LastSD'][li]),
                    'TrendVolRatio': float(is_shape['TrendVolRatio'][li]),
                    'ReturnSD':    float(is_shape['ReturnSD'][li]),
                    'FitDataMinMaxSD': float(is_shape['FitDataMinMaxSD'][li]),
                    'OOS_Trades':  oos_n,
                    'OOS_WinRate': float(oos_bt[li, BR_GROSS_WR])  if oos_n > 0 else 0.0,
                    'OOS_Gross':   float(oos_bt[li, BR_AVG_GROSS]) if oos_n > 0 else 0.0,
                    'OOS_Hold':    float(oos_bt[li, BR_AVG_HOLDING]) if oos_n > 0 else 0.0,
                })

        if not window_records:
            continue

        wdf = pd.DataFrame(window_records)
        wdf = apply_scoring(wdf, scoring_mode)
        wdf = wdf.sort_values('_score', ascending=False).reset_index(drop=True)
        wdf['IS_Rank']  = range(1, len(wdf) + 1)
        wdf['IS_Score'] = wdf['_score']
        wdf = wdf.drop(columns=['_score'])

        all_records.extend(wdf.to_dict('records'))

        if progress_cb:
            progress_cb((w_idx + 1) / n_windows)

    if not all_records:
        return pd.DataFrame()

    result = pd.DataFrame(all_records)
    result['OOS_Net'] = result['OOS_Gross'] - result['EstCost']
    logger.info(
        "run_cross_asset_walkforward complete: %d pair-window records",
        len(result),
    )
    return result


def summarise_walk_forward(results: pd.DataFrame) -> dict:
    """Compute rank correlations, significance tests, and quintile stats.

    Args:
        results: Output from ``run_walk_forward()`` or
            ``run_cross_asset_walkforward()``.

    Returns:
        Dict with keys:
            rho, p_value    — Spearman ρ(IS_Rank, OOS_Gross) and p-value
            n_obs           — pair-window observations with OOS_Trades > 0
            q1_mean         — mean OOS_Gross for top IS quintile
            q5_mean         — mean OOS_Gross for bottom IS quintile
            t_stat, t_p     — Welch t-test Q1 vs Q5
            quintile_df     — quintile × OOS metric summary DataFrame
            window_df       — per-window Spearman ρ DataFrame
            valid           — filtered DataFrame (OOS_Trades > 0)
        Returns default values (rho=0, p=1, n_obs=0) for empty input.
    """
    empty: dict = {
        'rho': 0.0, 'p_value': 1.0, 'n_obs': 0,
        'q1_mean': 0.0, 'q5_mean': 0.0, 't_stat': 0.0, 't_p': 1.0,
        'quintile_df': pd.DataFrame(), 'window_df': pd.DataFrame(),
        'valid': pd.DataFrame(),
    }

    if results.empty or 'IS_Rank' not in results.columns:
        return empty

    valid = results[results['OOS_Trades'] > 0].copy()
    n_obs = len(valid)
    if n_obs < 10:
        return {**empty, 'n_obs': n_obs}

    # Overall Spearman ρ: IS rank vs OOS gross return
    rho, p_value = spearmanr(valid['IS_Rank'], valid['OOS_Gross'])

    valid = valid.copy()

    def _safe_quintile(x: pd.Series) -> pd.Series:
        """Assign quintile labels; falls back to midpoint label if window too small."""
        try:
            return pd.qcut(x.rank(method='first'), 5,
                           labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except ValueError:
            return pd.Series(['Q3'] * len(x), index=x.index, dtype='category')

    valid['Q'] = valid.groupby('window')['IS_Rank'].transform(_safe_quintile)

    quintile_df = (
        valid.groupby('Q', observed=True)
        .agg(
            N           =('OOS_Gross', 'count'),
            OOS_GrossWR =('OOS_WinRate', 'mean'),
            OOS_Gross   =('OOS_Gross', 'mean'),
            OOS_Net     =('OOS_Net', 'mean'),
            OOS_AvgHold =('OOS_Hold', 'mean'),
        )
        .round(4)
    )

    # Q1 vs Q5 t-test (IS best vs IS worst)
    q1 = valid[valid['Q'] == 'Q1']['OOS_Gross']
    q5 = valid[valid['Q'] == 'Q5']['OOS_Gross']
    if len(q1) >= 2 and len(q5) >= 2:
        t_stat, t_p = ttest_ind(q1, q5, equal_var=False)
    else:
        t_stat, t_p = 0.0, 1.0

    # Per-window Spearman ρ
    window_rows: list[dict] = []
    for w, grp in valid.groupby('window'):
        if len(grp) >= 5:
            r, p = spearmanr(grp['IS_Rank'], grp['OOS_Gross'])
            window_rows.append({
                'Window': w,
                'IS_start': int(grp['is_start'].iloc[0]),
                'OOS_end':  int(grp['oos_end'].iloc[0]),
                'N_pairs':  len(grp),
                'Spearman_rho': round(float(r), 3),
                'p_value':      round(float(p), 3),
                'OOS_Gross_mean': round(float(grp['OOS_Gross'].mean()), 4),
            })
    window_df = pd.DataFrame(window_rows)

    return {
        'rho':         float(rho),
        'p_value':     float(p_value),
        'n_obs':       n_obs,
        'q1_mean':     float(q1.mean()) if len(q1) > 0 else 0.0,
        'q5_mean':     float(q5.mean()) if len(q5) > 0 else 0.0,
        't_stat':      float(t_stat),
        't_p':         float(t_p),
        'quintile_df': quintile_df,
        'window_df':   window_df,
        'valid':       valid,
    }
