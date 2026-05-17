"""
walkforward.py — Walk-forward validation engine
================================================

Tests whether in-sample scoring predicts out-of-sample performance.
Answers the Q11 question: does any scoring mode reliably identify pairs
that will perform better in the next OOS period?

Design
------
- 1v1 directional pairs only (N×(N-1) pairs, typically 132 for 12 instruments)
- Each window: IS data → score all pairs → OOS data → record OOS performance
- Spearman ρ(IS rank, OOS gross return) is the validation metric
- Uses numba via batch_backtest for speed (same engine as the Search tab)
- Complete for 12 instruments, 14 windows in <30s on a modern machine

Usage
-----
    from walkforward import run_walk_forward, summarise_walk_forward
    from backtest import load_asset_prices

    prices, instruments = load_asset_prices('cache/prices.csv')
    results = run_walk_forward(prices, instruments, scoring_mode='composite')
    summary = summarise_walk_forward(results)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind

from numba_core import (
    batch_backtest,
    BR_N_TRADES, BR_GROSS_WR, BR_AVG_GROSS, BR_AVG_HOLDING,
)
from search import _batch_scores
from scoring import apply_scoring, estimate_trade_cost

_TDY = 262  # trading days per year


def _vol_scaled(
    prices: pd.DataFrame,
    instruments: list[str],
    vol_window: int = 262,
    target_vol: float = 0.01,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute vol-scaled returns for the full price history.

    Returns
    -------
    scaled_df : pd.DataFrame
        Vol-normalised daily returns, rows with any NaN dropped.
    day_ints : np.ndarray (T,) int64
        Integer day count per row (for holding period calculation).
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
    progress_cb=None,
) -> pd.DataFrame:
    """
    Run walk-forward analysis on all 1v1 directional pairs.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw price DataFrame (DatetimeIndex, instrument columns).
    instruments : list[str]
        Instrument codes to include (must be columns in prices).
    is_years, oos_years, step_years : int
        Window lengths in trading years (1 year = 262 days).
    scoring_mode : str
        One of 'composite', 'cost_rank', 'contrarian'.
    progress_cb : callable(float) | None
        Called with fraction complete (0.0–1.0) each window.

    Returns
    -------
    pd.DataFrame with one row per (window, pair) containing IS rank,
    IS score metrics, and OOS performance metrics.  Empty DataFrame if
    insufficient data for at least one window.
    """
    scaled_df, day_ints = _vol_scaled(prices, instruments, vol_window, target_vol)
    scaled = scaled_df.values.astype(np.float64)
    T, N   = scaled.shape

    is_days  = is_years  * _TDY
    oos_days = oos_years * _TDY
    step_days = step_years * _TDY

    # Generate non-overlapping IS/OOS window boundaries
    windows: list[tuple[int, int, int]] = []
    pos = 0
    while pos + is_days + oos_days <= T:
        windows.append((pos, pos + is_days, pos + is_days + oos_days))
        pos += step_days

    if not windows:
        return pd.DataFrame()

    all_records: list[dict] = []
    n_windows = len(windows)

    for w_idx, (is_start, is_end, oos_end) in enumerate(windows):
        is_scaled   = scaled[is_start:is_end]
        oos_scaled  = scaled[is_end:oos_end]
        is_days_arr = day_ints[is_start:is_end]
        oos_days_arr = day_ints[is_end:oos_end]

        if is_scaled.shape[0] < vol_window // 2 or oos_scaled.shape[0] < 5:
            continue

        # Label this window by calendar year boundaries
        is_label  = scaled_df.index[is_start].year
        oos_label = scaled_df.index[min(oos_end - 1, T - 1)].year

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
        wdf = apply_scoring(wdf, scoring_mode)            # adds _score
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
    return result


def summarise_walk_forward(results: pd.DataFrame) -> dict:
    """
    Compute rank correlations, significance tests, and quintile stats.

    Parameters
    ----------
    results : pd.DataFrame
        Output from run_walk_forward().

    Returns
    -------
    dict with keys:
        rho, p_value   — Spearman ρ(IS_Rank, OOS_Gross) and p-value
        n_obs          — number of pair-window observations with OOS trades
        quintile_df    — quintile × OOS metric summary
        window_df      — per-window Spearman ρ
        valid          — filtered DataFrame (OOS_Trades > 0)
    """
    empty = {
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

    # Quintile analysis — rank within each window, then group
    valid = valid.copy()

    def _safe_quintile(x):
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
    window_rows = []
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
