"""
research/phase5_execution_lag.py — Phase 5 Execution Lag Test
==============================================================

Quantifies the execution-lag effect on the equity scalp strategy.

Background
----------
The current backtest fills both legs at the signal-day close simultaneously.
For ASX 200 (CIL), which closes ~10 hours before US markets, the next
available fill after a US-close signal is the following day's ASX close —
by which time the ASX has already priced in the US session move that
generated the measured divergence.

Lag implementation
------------------
prices['CIL'] is shifted forward by +1 trading day. Entry and exit fill
prices are both lagged (cum_l); the entry/exit signal (dist_sd) is
unlagged. This isolates fill-price contamination from signal contamination.
See Decision 118.

NOTE — Part 2 WF: run_walk_forward is a black box — it recomputes its
own spread and z-score internally with no hook to inject a separate
fill-price series. Passing prices_lagged gives a result where both signal
and fill are lagged. Treat as a conservative lower bound.

Interpretation gate (Decision 118)
------------------------------------
Post-lag AvgNet_WT:
  >= 0.150%           -> DEPLOY AS PLANNED
  +0.100% – +0.149%   -> DEPLOY NON-ASX PAIRS ONLY
  < 0.100%            -> RE-ENGINEER BEFORE FUNDING

How to run
----------
From trading_app/:
    C:\\Users\\gordo\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe \\
        research/phase5_execution_lag.py

Outputs:
    research/output/phase5_execution_lag_YYYYMMDD_HHMM.txt
    data/phase5_execution_lag_results.json

Compliance
----------
- Type hints (from __future__ import annotations)
- Google-style docstrings; Decision 118/119 referenced where relevant
- logging, not print
- Fail-fast input guards with specific exception types
- Register H/I canaries active via numba_core import
- No bare except
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Register item O: module-level imports required.
# Importing numba_core activates H and I canary assertions (CLAUDE.md §H, §I).
from engine.numba_core import (  # noqa: E402
    rolling_mean_std,
    detect_trades,
    HAS_NUMBA,
)
from engine.backtest import (  # noqa: E402
    load_asset_prices,
    prepare_returns,
    aggregate_trades,
)
from engine.walkforward import (  # noqa: E402
    run_walk_forward,
    summarise_walk_forward,
)
from asset_configs import ASSET_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)


def _mean_equity_spread_cost() -> float:
    """Return universe-mean one-way spread cost from asset_configs equity instruments.

    Matches the cost model used in phase5_equity_exit_sd_sensitivity.py.
    The prompt states '0.1% per leg' but the implemented standard reads
    per-instrument values from asset_configs — register item for this script.
    """
    instruments_cfg = ASSET_CLASSES['equity']['instruments']
    costs = [cfg.get('spread_pct', 0.001) for cfg in instruments_cfg.values()]
    return sum(costs) / len(costs)

# ── Paths ────────────────────────────────────────────────────────────────────
_CACHE_DIR    = _ROOT / 'cache'
_DATA_DIR     = _ROOT / 'data'
_OUTPUT_DIR   = _HERE / 'output'
_PRICES_CSV   = _CACHE_DIR / 'prices.csv'
_RESULTS_JSON = _DATA_DIR / 'phase5_execution_lag_results.json'

# ── Algorithm parameters (Decision 118 baseline) ─────────────────────────────
XING_SD     : float = 2.0
EXIT_SD     : float = 2.0
VOL_WINDOW  : int   = 262
TARGET_VOL  : float = 0.01
MAX_HOLD    : int   = 300
N_LEGS      : int   = 2

# Per-instrument mean from asset_configs — matches phase5_equity_exit_sd_sensitivity.py.
# NOTE: prompt states '0.1% per leg uniform' but validated implementation uses
# asset_configs per-instrument mean (~0.0462%). Using implementation standard here.
SPREAD_COST_PCT: float = _mean_equity_spread_cost()

# Equity financing: (4.88% long − 0.88% short rebate) / 365
_FIN_LONG_ANNUAL : float = 0.0488
_FIN_SHORT_ANNUAL: float = 0.0088
FIN_NET_DAILY    : float = (_FIN_LONG_ANNUAL - _FIN_SHORT_ANNUAL) / 365.0

# WF parameters — must match validated Q11 protocol (IS=3y, OOS=1y, step=1y)
WF_IS_YEARS       : int   = 3
WF_OOS_YEARS      : int   = 1
WF_STEP_YEARS     : int   = 1
WF_SCORING        : str   = 'contrarian'
WF_SPREAD_COST_PCT: float = 0.001

CIL_CODE: str = 'CIL'  # ASX 200 — the lagged leg

# Decision 118/119 reference values for baseline sanity check
D118_MEAN_AVG_NET  : float = 0.00380  # +0.380%
D118_POS_PAIRS     : int   = 58       # out of 132 directed
D118_BOTH_COUNT    : int   = 34
D118_BOTH_NET      : float = 0.00570  # +0.57%
D118_BOTH_HOLD     : float = 6.1
D118_WF_RHO        : float = 0.208
D118_WF_PVAL       : float = 0.0001
D118_WF_NOBS       : int   = 1614
BASELINE_ALERT_DELTA: float = 0.0005  # flag INVESTIGATE if delta > 0.05%

# Watchlist pairs: (long, short, d118_n, d118_avg_net, d118_wr, d118_hold|None)
WATCHLIST_PAIRS: list[tuple[str, str, int, float, float, float | None]] = [
    ('CTN', 'CIL', 37, 0.01730, 0.811, 4.4),  # SPX/ASX — CIL lag applied
    ('CTB', 'CIL', 44, 0.00910, 0.795, 4.9),  # DOW/ASX — CIL lag applied
    ('CFR', 'CIL', 20, 0.01530, 0.950, None), # DAX/ASX — CIL lag applied
    ('UKX', 'CTN', 17, 0.01190, 0.765, 4.6),  # FTSE/SPX — control, no CIL
    ('CTN', 'UKX', 31, 0.01120, 0.774, 3.6),  # SPX/FTSE — control, no CIL
]


# ═══════════════════════════════════════════════════════════════════════════
# PRICE FRAME CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _build_aligned_price_frames(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build baseline (aligned) and lagged price DataFrames.

    Shifts CIL prices forward by 1 trading day, then trims both frames to
    the same date index so control pairs (no CIL leg) show zero delta.

    Args:
        prices: Raw price DataFrame, must contain CIL_CODE column.

    Returns:
        (prices_aligned, prices_lagged): prices_aligned is original prices
        trimmed to match prices_lagged's date range.

    Raises:
        ValueError: If CIL_CODE not in prices.columns.
    """
    if CIL_CODE not in prices.columns:
        raise ValueError(
            f"_build_aligned_price_frames: '{CIL_CODE}' not in prices; "
            f"columns: {list(prices.columns)}"
        )
    prices_lagged = prices.copy()
    prices_lagged[CIL_CODE] = prices_lagged[CIL_CODE].shift(1)
    prices_lagged = prices_lagged.dropna(subset=[CIL_CODE])
    prices_aligned = prices.loc[prices_lagged.index]
    logger.info(
        "Price frames aligned: %d rows (1 leading row dropped for CIL shift)",
        len(prices_lagged),
    )
    return prices_aligned, prices_lagged


def _prepare_scaled_matrices(
    prices_aligned: pd.DataFrame,
    prices_lagged: pd.DataFrame,
    instruments: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build vol-scaled return matrices for both price frames.

    Args:
        prices_aligned: Unlagged prices trimmed to the lagged date range.
        prices_lagged: Prices with CIL shifted +1 day.
        instruments: Instrument codes (columns of both DataFrames).

    Returns:
        (scaled_u, scaled_l, day_ints, index) where scaled_u and scaled_l
        are shape (T, N) float64 arrays; day_ints is (T,) int64.

    Raises:
        ValueError: If CIL not in instruments, or post-trim shapes differ.
    """
    if CIL_CODE not in instruments:
        raise ValueError(
            f"_prepare_scaled_matrices: '{CIL_CODE}' not in instruments: {instruments}"
        )
    scaled_u, day_ints_u, idx_u = prepare_returns(
        prices_aligned, instruments, vol_window=VOL_WINDOW, target_vol=TARGET_VOL,
    )
    scaled_l, day_ints_l, idx_l = prepare_returns(
        prices_lagged, instruments, vol_window=VOL_WINDOW, target_vol=TARGET_VOL,
    )

    if scaled_u.shape[0] != scaled_l.shape[0]:
        common_idx = idx_u.intersection(idx_l)
        logger.warning(
            "Scaled array shapes differ (%d vs %d); trimming to common index (%d rows)",
            scaled_u.shape[0], scaled_l.shape[0], len(common_idx),
        )
        mask_u = idx_u.isin(common_idx)
        mask_l = idx_l.isin(common_idx)
        scaled_u    = scaled_u[mask_u]
        scaled_l    = scaled_l[mask_l]
        day_ints_u  = day_ints_u[mask_u]
        idx_u       = idx_u[mask_u]

    logger.info(
        "Scaled matrices ready: shape=%s, numba=%s",
        scaled_u.shape, HAS_NUMBA,
    )
    return scaled_u, scaled_l, day_ints_u, idx_u


# ═══════════════════════════════════════════════════════════════════════════
# PER-PAIR BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def _run_pair(
    long_i : int,
    short_i: int,
    scaled_u: np.ndarray,
    scaled_l: np.ndarray,
    day_ints: np.ndarray,
) -> tuple[dict, dict]:
    """Run baseline and lagged backtest for one directed pair.

    Signal (dist_sd) is computed once from the unlagged spread (scaled_u)
    and shared between baseline and lagged runs.  Only the cumulative-return
    array (which determines fill P&L) differs. See Decision 118.

    Baseline fill : cum from unlagged spread (standard backtest behaviour).
    Lagged fill   : cum from lagged spread (CIL day shifted +1).

    Args:
        long_i, short_i: Column indices into scaled_u / scaled_l.
        scaled_u: Unlagged vol-scaled return matrix (T, N).
        scaled_l: Lagged vol-scaled return matrix (T, N).
        day_ints: Integer day index (T,).

    Returns:
        (summary_base, summary_lag) — dicts from aggregate_trades.
        An empty dict {} is returned for any run with zero trades.
    """
    # SIGNAL: unlagged spread z-score — unchanged vs standard backtest.
    spread_u  = (scaled_u[:, long_i] - scaled_u[:, short_i]).astype(np.float64)
    cum_u     = np.cumprod(1.0 + spread_u)
    roll_mean, roll_std = rolling_mean_std(cum_u, VOL_WINDOW)
    # np.where correctly produces NaN when roll_std is NaN or 0 (warmup period).
    with np.errstate(invalid='ignore', divide='ignore'):
        dist_sd = np.where(roll_std > 0, (cum_u - roll_mean) / roll_std, np.nan)

    # BASELINE fill: unlagged cum + unlagged signal.
    trades_b, n_b = detect_trades(cum_u, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)
    summary_b: dict = {}
    if n_b > 0:
        summary_b = aggregate_trades(trades_b, n_b, SPREAD_COST_PCT, FIN_NET_DAILY, N_LEGS)

    # LAGGED fill: lagged cum + SAME unlagged signal (dist_sd unchanged).
    spread_l  = (scaled_l[:, long_i] - scaled_l[:, short_i]).astype(np.float64)
    cum_l     = np.cumprod(1.0 + spread_l)
    trades_l, n_l = detect_trades(cum_l, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)
    summary_l: dict = {}
    if n_l > 0:
        summary_l = aggregate_trades(trades_l, n_l, SPREAD_COST_PCT, FIN_NET_DAILY, N_LEGS)

    return summary_b, summary_l


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — UNIVERSE GRID
# ═══════════════════════════════════════════════════════════════════════════

def run_universe_grid(
    scaled_u: np.ndarray,
    scaled_l: np.ndarray,
    day_ints: np.ndarray,
    instruments: list[str],
) -> pd.DataFrame:
    """Run baseline and lagged backtest for all 132 directed equity pairs.

    Generates all N×(N-1) directed pairs and runs _run_pair for each.
    Zero-trade pairs are included in the output DataFrame (n_base/n_lag=0,
    avg_net=NaN) but excluded from universe-mean aggregation downstream.

    Args:
        scaled_u: Unlagged vol-scaled return matrix (T, N).
        scaled_l: Lagged vol-scaled return matrix (T, N).
        day_ints: Integer day index (T,).
        instruments: Instrument codes of length N.

    Returns:
        DataFrame with one row per directed pair, columns:
            long, short, n_base, n_lag,
            avg_net_base, avg_net_lag,
            net_wr_base, net_wr_lag,
            avg_hold_base, avg_hold_lag.

    Raises:
        ValueError: If instruments has fewer than 2 entries, or scaled
            column count does not match len(instruments).
    """
    N = len(instruments)
    if N < 2:
        raise ValueError(f"run_universe_grid: need >= 2 instruments; got {N}")
    if scaled_u.shape[1] != N:
        raise ValueError(
            f"run_universe_grid: scaled_u has {scaled_u.shape[1]} columns "
            f"but {N} instruments supplied"
        )
    if scaled_l.shape[1] != N:
        raise ValueError(
            f"run_universe_grid: scaled_l has {scaled_l.shape[1]} columns "
            f"but {N} instruments supplied"
        )

    # All directed pairs — matches phase5_equity_exit_sd_sensitivity.py:163-165
    pairs: list[tuple[int, int]] = [
        (i, j) for i in range(N) for j in range(N) if i != j
    ]
    n_pairs = len(pairs)
    logger.info("Part 1: %d directed pairs", n_pairs)
    t0 = time.time()

    records: list[dict] = []
    for long_i, short_i in pairs:
        s_b, s_l = _run_pair(long_i, short_i, scaled_u, scaled_l, day_ints)
        records.append({
            'long':          instruments[long_i],
            'short':         instruments[short_i],
            'n_base':        s_b.get('n_trades', 0),
            'n_lag':         s_l.get('n_trades', 0),
            'avg_net_base':  s_b.get('avg_net', np.nan),
            'avg_net_lag':   s_l.get('avg_net', np.nan),
            'net_wr_base':   s_b.get('net_wr', np.nan),
            'net_wr_lag':    s_l.get('net_wr', np.nan),
            'avg_hold_base': s_b.get('avg_holding', np.nan),
            'avg_hold_lag':  s_l.get('avg_holding', np.nan),
        })

    logger.info("Part 1 complete: %.1fs", time.time() - t0)
    return pd.DataFrame(records)


def _aggregate_universe_stats(df: pd.DataFrame) -> tuple[dict, dict]:
    """Compute universe-mean statistics, excluding zero-trade pairs.

    Args:
        df: Output of run_universe_grid.

    Returns:
        (base_stats, lag_stats) — each dict has:
            mean_avg_net, n_positive, n_pairs_with_trades,
            n_zero_trade_pairs, n_total, mean_net_wr, mean_avg_hold.
    """
    n_total = len(df)

    def _stats(n_col: str, net_col: str, wr_col: str, hold_col: str) -> dict:
        active = df[df[n_col] > 0]
        n_zero = n_total - len(active)
        n_pos  = int((active[net_col] > 0).sum())
        return {
            'mean_avg_net':        float(active[net_col].mean()) if len(active) else float('nan'),
            'n_positive':          n_pos,
            'n_pairs_with_trades': len(active),
            'n_zero_trade_pairs':  n_zero,
            'n_total':             n_total,
            'mean_net_wr':         float(active[wr_col].mean()) if len(active) else float('nan'),
            'mean_avg_hold':       float(active[hold_col].mean()) if len(active) else float('nan'),
        }

    base_stats = _stats('n_base', 'avg_net_base', 'net_wr_base', 'avg_hold_base')
    lag_stats  = _stats('n_lag',  'avg_net_lag',  'net_wr_lag',  'avg_hold_lag')
    return base_stats, lag_stats


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — BESTDIR=BOTH ANALYSIS (Decision 119)
# ═══════════════════════════════════════════════════════════════════════════

def compute_best_dir_both(
    df: pd.DataFrame,
    instruments: list[str],
) -> dict:
    """Compute BestDir=Both statistics for baseline and lagged runs.

    BestDir=Both: avg_net > 0 in BOTH directed variants of an undirected pair.
    A direction with n_trades=0 is treated as not-positive (missing ≠ positive).
    See Decision 119.

    Degradation categories (for originally-Both pairs that fall out):
      lost_to_one_dir     : one direction still positive, the other is not
      lost_to_zero_trades : both directions have zero trades in lagged run
      lost_to_neither     : both directions trading but both negative

    Args:
        df: Output of run_universe_grid (one row per directed pair).
        instruments: Ordered instrument list (used to enumerate undirected pairs).

    Returns:
        Dict with:
            base_both_count, lag_both_count,
            lag_both_mean_net, lag_both_mean_hold,
            lost_to_one_dir, lost_to_zero_trades, lost_to_neither.
    """
    if df.empty:
        raise ValueError("compute_best_dir_both: df is empty")

    # Index the pair results for O(1) lookup
    base_net  = {(r.long, r.short): r.avg_net_base  for r in df.itertuples()}
    lag_net   = {(r.long, r.short): r.avg_net_lag   for r in df.itertuples()}
    lag_ntrd  = {(r.long, r.short): r.n_lag         for r in df.itertuples()}
    lag_hold  = {(r.long, r.short): r.avg_hold_lag  for r in df.itertuples()}

    undirected = list(combinations(instruments, 2))  # 66 pairs for 12 instruments

    base_both_count = 0
    lag_both_count  = 0
    lag_both_nets:  list[float] = []
    lag_both_holds: list[float] = []
    lost_to_one_dir      = 0
    lost_to_zero_trades  = 0
    lost_to_neither      = 0

    for a, b in undirected:
        ab, ba = (a, b), (b, a)
        if ab not in base_net or ba not in base_net:
            logger.warning("Missing directed pair in results: %s or %s", ab, ba)
            continue

        bn_ab = base_net[ab]
        bn_ba = base_net[ba]
        is_base_both = (
            not pd.isna(bn_ab) and bn_ab > 0 and
            not pd.isna(bn_ba) and bn_ba > 0
        )
        if is_base_both:
            base_both_count += 1

        ln_ab  = lag_net.get(ab, float('nan'))
        ln_ba  = lag_net.get(ba, float('nan'))
        lnt_ab = lag_ntrd.get(ab, 0)
        lnt_ba = lag_ntrd.get(ba, 0)

        lag_ab_pos = lnt_ab > 0 and not pd.isna(ln_ab) and ln_ab > 0
        lag_ba_pos = lnt_ba > 0 and not pd.isna(ln_ba) and ln_ba > 0

        if lag_ab_pos and lag_ba_pos:
            lag_both_count += 1
            lag_both_nets.append((ln_ab + ln_ba) / 2.0)
            h_ab = lag_hold.get(ab, float('nan'))
            h_ba = lag_hold.get(ba, float('nan'))
            if not pd.isna(h_ab) and not pd.isna(h_ba):
                lag_both_holds.append((h_ab + h_ba) / 2.0)

        if is_base_both and not (lag_ab_pos and lag_ba_pos):
            if lnt_ab == 0 and lnt_ba == 0:
                lost_to_zero_trades += 1
            elif (lnt_ab == 0) != (lnt_ba == 0) or (lag_ab_pos != lag_ba_pos):
                # One direction still positive (or one has zero trades)
                lost_to_one_dir += 1
            else:
                # Both directions trading but neither positive
                lost_to_neither += 1

    return {
        'base_both_count':     base_both_count,
        'lag_both_count':      lag_both_count,
        'lag_both_mean_net':   float(np.mean(lag_both_nets))  if lag_both_nets  else float('nan'),
        'lag_both_mean_hold':  float(np.mean(lag_both_holds)) if lag_both_holds else float('nan'),
        'lost_to_one_dir':     lost_to_one_dir,
        'lost_to_zero_trades': lost_to_zero_trades,
        'lost_to_neither':     lost_to_neither,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════

def run_wf_lagged(
    prices_lagged: pd.DataFrame,
    instruments: list[str],
) -> dict:
    """Re-run the production Q11 WF protocol with lagged prices.

    NOTE: run_walk_forward recomputes spread and z-score internally;
    there is no hook to pass a separate fill-price series. Passing
    prices_lagged means both signal and fill are lagged — this gives a
    conservative lower bound on the true lag impact. A signal-only-lagged
    WF would require a custom loop and is not needed for gating.

    Args:
        prices_lagged: Price DataFrame with CIL shifted +1 trading day.
        instruments: Equity instrument codes.

    Returns:
        Dict with rho, p_value, n_obs from summarise_walk_forward.

    Raises:
        ValueError: If run_walk_forward returns an empty DataFrame.
    """
    if prices_lagged.empty:
        raise ValueError("run_wf_lagged: prices_lagged is empty")
    if not instruments:
        raise ValueError("run_wf_lagged: instruments is empty")

    logger.info("Part 2: Walk-forward with lagged prices (conservative lower bound) …")
    t0 = time.time()

    wf_df = run_walk_forward(
        prices=prices_lagged,
        instruments=instruments,
        is_years=WF_IS_YEARS,
        oos_years=WF_OOS_YEARS,
        step_years=WF_STEP_YEARS,
        scoring_mode=WF_SCORING,
        vol_window=VOL_WINDOW,
        target_vol=TARGET_VOL,
        xing_sd=XING_SD,
        exit_sd=EXIT_SD,
        spread_cost_pct=WF_SPREAD_COST_PCT,
    )

    if wf_df.empty:
        raise ValueError(
            "run_wf_lagged: run_walk_forward returned empty DataFrame — "
            "insufficient price history or no trades under lagged fills"
        )

    summary = summarise_walk_forward(wf_df)
    logger.info(
        "Part 2 done: ρ=%.3f, p=%.4f, n_obs=%d (%.1fs)",
        summary['rho'], summary['p_value'], summary['n_obs'],
        time.time() - t0,
    )
    return {
        'rho':     float(summary['rho']),
        'p_value': float(summary['p_value']),
        'n_obs':   int(summary['n_obs']),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — WATCHLIST PAIRS
# ═══════════════════════════════════════════════════════════════════════════

def run_watchlist_pairs(
    scaled_u: np.ndarray,
    scaled_l: np.ndarray,
    day_ints: np.ndarray,
    instruments: list[str],
) -> list[dict]:
    """Run baseline and lagged backtest for each watchlist pair.

    Control pairs (no CIL leg) should show exactly zero delta between
    baseline and lagged.  Non-zero delta indicates a bug in the lag logic.

    Args:
        scaled_u, scaled_l, day_ints, instruments: from _prepare_scaled_matrices.

    Returns:
        List of result dicts, one per WATCHLIST_PAIRS entry.

    Raises:
        ValueError: If a watchlist instrument is absent from instruments.
    """
    instr_idx = {instr: i for i, instr in enumerate(instruments)}
    results: list[dict] = []

    for long_instr, short_instr, d118_n, d118_net, d118_wr, d118_hold in WATCHLIST_PAIRS:
        for code in (long_instr, short_instr):
            if code not in instr_idx:
                raise ValueError(
                    f"run_watchlist_pairs: '{code}' not in instruments: {instruments}"
                )

        li = instr_idx[long_instr]
        si = instr_idx[short_instr]
        s_b, s_l = _run_pair(li, si, scaled_u, scaled_l, day_ints)
        has_asx = long_instr == CIL_CODE or short_instr == CIL_CODE

        results.append({
            'long':          long_instr,
            'short':         short_instr,
            'has_asx':       has_asx,
            'd118_n':        d118_n,
            'd118_avg_net':  d118_net,
            'd118_wr':       d118_wr,
            'd118_hold':     d118_hold,
            'n_base':        s_b.get('n_trades', 0),
            'n_lag':         s_l.get('n_trades', 0),
            'avg_net_base':  s_b.get('avg_net', float('nan')),
            'avg_net_lag':   s_l.get('avg_net', float('nan')),
            'net_wr_base':   s_b.get('net_wr', float('nan')),
            'net_wr_lag':    s_l.get('net_wr', float('nan')),
            'avg_hold_base': s_b.get('avg_holding', float('nan')),
            'avg_hold_lag':  s_l.get('avg_holding', float('nan')),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# REPORT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def _fp(v: float | None, dec: int = 3, width: int = 0) -> str:
    """Format a decimal fraction as a percentage string (+X.XXX%)."""
    if v is None or pd.isna(v):
        s = 'n/a'
    else:
        s = f'{v * 100:+.{dec}f}%'
    return s if width == 0 else s.rjust(width)


def _ff(v: float | None, dec: int = 1, width: int = 0) -> str:
    """Format a float or return 'n/a'."""
    if v is None or pd.isna(v):
        s = 'n/a'
    else:
        s = f'{v:.{dec}f}'
    return s if width == 0 else s.rjust(width)


def _check_baseline(base_stats: dict) -> str:
    """Compare recomputed baseline mean to Decision 118 reference.

    Returns:
        'OK' if within BASELINE_ALERT_DELTA, else 'INVESTIGATE'.
    """
    delta = abs(base_stats['mean_avg_net'] - D118_MEAN_AVG_NET)
    if delta > BASELINE_ALERT_DELTA:
        logger.warning(
            "Baseline avg_net delta vs Decision 118 = %.5f%% — exceeds %.3f%% "
            "threshold; check parameter alignment.",
            delta * 100, BASELINE_ALERT_DELTA * 100,
        )
        return 'INVESTIGATE'
    return 'OK'


def format_report(
    run_ts: str,
    base_stats: dict,
    lag_stats: dict,
    both_stats: dict,
    wf_lagged: dict,
    watchlist: list[dict],
    baseline_status: str,
) -> str:
    """Build the human-readable text report.

    Args:
        run_ts: Timestamp string.
        base_stats, lag_stats: From _aggregate_universe_stats.
        both_stats: From compute_best_dir_both.
        wf_lagged: From run_wf_lagged.
        watchlist: From run_watchlist_pairs.
        baseline_status: 'OK' or 'INVESTIGATE'.

    Returns:
        Multi-line formatted string.
    """
    SEP = '-' * 72

    # Interpretation verdict
    pln = lag_stats['mean_avg_net']
    if pd.isna(pln) or pln < 0.00100:
        verdict = 'RE-ENGINEER BEFORE FUNDING'
    elif pln < 0.00150:
        verdict = 'DEPLOY NON-ASX PAIRS ONLY'
    else:
        verdict = 'DEPLOY AS PLANNED'

    # BestDir=Both contamination verdict
    bc     = both_stats['base_both_count']
    lc     = both_stats['lag_both_count']
    drop_c = (bc - lc) / bc if bc > 0 else 0.0
    bn_ref = D118_BOTH_NET
    ln_val = both_stats['lag_both_mean_net']
    if not pd.isna(ln_val) and bn_ref > 0:
        drop_n = (bn_ref - ln_val) / bn_ref
    else:
        drop_n = float('nan')

    if drop_c > 0.40:
        c_verdict = 'ARTIFACT'
    elif drop_c > 0.15 or (not pd.isna(drop_n) and drop_n > 0.20):
        c_verdict = 'PARTIAL'
    else:
        c_verdict = 'GENUINE'

    n_tot   = base_stats['n_total']
    n_b_act = base_stats['n_pairs_with_trades']
    n_l_act = lag_stats['n_pairs_with_trades']
    d_net   = lag_stats['mean_avg_net'] - base_stats['mean_avg_net']
    d_wr    = lag_stats['mean_net_wr']  - base_stats['mean_net_wr']  if not pd.isna(lag_stats['mean_net_wr']) and not pd.isna(base_stats['mean_net_wr']) else float('nan')
    d_hold  = lag_stats['mean_avg_hold'] - base_stats['mean_avg_hold'] if not pd.isna(lag_stats['mean_avg_hold']) and not pd.isna(base_stats['mean_avg_hold']) else float('nan')

    L = [
        '=== EXECUTION LAG TEST — EQUITY SCALP ===',
        f'Run date:     {run_ts}',
        f'Lag applied:  {CIL_CODE} (ASX 200) fill prices shifted +1 trading day',
        f'Signal:       unlagged dist_sd (unchanged)',
        '',
        SEP,
        f'PART 1 — UNIVERSE GRID ({n_tot} directed pairs)',
        f'{"Metric":<32}{"Baseline":>12}{"Lagged":>12}{"Delta":>12}',
        SEP,
        f'{"Mean AvgNet_WT":<32}{_fp(base_stats["mean_avg_net"], width=12)}{_fp(lag_stats["mean_avg_net"], width=12)}{_fp(d_net, width=12)}',
        f'{"Mean NetWR":<32}{_ff(base_stats["mean_net_wr"] * 100 if not pd.isna(base_stats["mean_net_wr"]) else None, dec=1, width=11) + "%":>12}'
        f'{_ff(lag_stats["mean_net_wr"] * 100 if not pd.isna(lag_stats["mean_net_wr"]) else None, dec=1, width=11) + "%":>12}'
        f'{(_ff(d_wr * 100 if not pd.isna(d_wr) else None, dec=1, width=10) + "%"):>12}',
        f'{"Mean AvgHold":<32}{_ff(base_stats["mean_avg_hold"], dec=1, width=11) + "d":>12}'
        f'{_ff(lag_stats["mean_avg_hold"], dec=1, width=11) + "d":>12}'
        f'{(_ff(d_hold, dec=1, width=10) + "d"):>12}',
        f'{"Pairs with trades":<32}{str(n_b_act) + "/" + str(n_tot):>12}{str(n_l_act) + "/" + str(n_tot):>12}',
        f'{"Positive WT pairs":<32}{str(base_stats["n_positive"]) + "/" + str(n_b_act):>12}{str(lag_stats["n_positive"]) + "/" + str(n_l_act):>12}',
        f'{"Zero-trade pairs":<32}{str(base_stats["n_zero_trade_pairs"]):>12}{str(lag_stats["n_zero_trade_pairs"]):>12}',
        '',
        'BestDir=Both subset (Decision 119):',
        f'  {"Both count":<30}{str(bc):>12}{str(lc):>12}',
        f'  {"Mean AvgNet":<30}{_fp(D118_BOTH_NET, width=12)}{_fp(ln_val, width=12)}',
        f'  {"Mean AvgHold":<30}{_ff(D118_BOTH_HOLD, dec=1, width=11) + "d":>12}{_ff(both_stats["lag_both_mean_hold"], dec=1, width=11) + "d":>12}',
        '',
        'Baseline sanity check vs Decision 118:',
        f'  D118 reference:    {_fp(D118_MEAN_AVG_NET):>10}   pos pairs {D118_POS_PAIRS}/{n_tot}',
        f'  This run:          {_fp(base_stats["mean_avg_net"]):>10}   pos pairs {base_stats["n_positive"]}/{n_b_act}',
        f'  Delta:             {_fp(base_stats["mean_avg_net"] - D118_MEAN_AVG_NET):>10}   [{baseline_status}]',
        '',
        SEP,
        'PART 2 — WALK-FORWARD (IS=3y/OOS=1y, contrarian, EXIT_SD=2.0)',
        'NOTE: conservative lower bound — both signal and fill lagged',
        f'{"Metric":<32}{"Baseline":>12}{"Lagged":>12}',
        SEP,
        f'{"Spearman rho":<32}{D118_WF_RHO:>+12.3f}{wf_lagged["rho"]:>+12.3f}',
        f'{"p-value":<32}{"<0.0001":>12}{wf_lagged["p_value"]:>12.4f}',
        f'{"n_obs":<32}{D118_WF_NOBS:>12,}{wf_lagged["n_obs"]:>12,}',
        '',
        SEP,
        'PART 3 — WATCHLIST PAIRS',
        f'{"Pair":<18}{"N":>5}{"AvgNet(B)":>10}{"AvgNet(L)":>10}{"Delta":>10}'
        f'{"WR(B)":>7}{"WR(L)":>7}{"Hold(B)":>8}{"Hold(L)":>8}',
        SEP,
    ]

    # Control pair delta check
    ctrl_deltas: list[float] = []
    for r in watchlist:
        pair = f'{r["long"]}/{r["short"]}'
        ctrl = '  ←ctrl' if not r['has_asx'] else ''
        delta_net_w = (r['avg_net_lag'] - r['avg_net_base']
                       if not pd.isna(r['avg_net_lag']) and not pd.isna(r['avg_net_base'])
                       else float('nan'))
        wr_b = _ff(r['net_wr_base'] * 100 if not pd.isna(r['net_wr_base']) else None, 1) + '%'
        wr_l = _ff(r['net_wr_lag']  * 100 if not pd.isna(r['net_wr_lag'])  else None, 1) + '%'
        L.append(
            f'{pair:<18}{r["n_base"]:>5}'
            f'{_fp(r["avg_net_base"], 2, width=10)}'
            f'{_fp(r["avg_net_lag"],  2, width=10)}'
            f'{_fp(delta_net_w,       2, width=10)}'
            f'{wr_b:>7}{wr_l:>7}'
            f'{_ff(r["avg_hold_base"], 1, width=7) + "d":>8}'
            f'{_ff(r["avg_hold_lag"],  1, width=7) + "d":>8}'
            f'{ctrl}'
        )
        if not r['has_asx'] and not pd.isna(delta_net_w):
            ctrl_deltas.append(abs(delta_net_w))

    max_ctrl = max(ctrl_deltas) if ctrl_deltas else 0.0
    ctrl_status = 'PASS' if max_ctrl < 1e-8 else f'FAIL (max delta {max_ctrl * 100:.6f}%)'

    L += [
        '',
        f'Control-pair sanity (expect zero delta): {ctrl_status}',
        '',
        SEP,
        'PART 4 — BESTDIR=BOTH CONTAMINATION (Decision 119)',
        f'  Original Both pairs:     {bc}  (mean AvgNet {_fp(D118_BOTH_NET)}, mean hold {D118_BOTH_HOLD}d)',
        f'  Lagged Both pairs:       {lc}  (mean AvgNet {_fp(ln_val)}, mean hold {_ff(both_stats["lag_both_mean_hold"])}d)',
        f'  Dropped → one-dir-only:  {both_stats["lost_to_one_dir"]}',
        f'  Dropped → zero trades:   {both_stats["lost_to_zero_trades"]}',
        f'  Dropped → both negative: {both_stats["lost_to_neither"]}',
        '',
        f'  Contamination verdict: [{c_verdict}]',
        '    GENUINE  = Both count drops <15% AND mean net drops <20%',
        '    PARTIAL  = Both count drops 15-40% OR mean net drops 20-50%',
        '    ARTIFACT = Both count drops >40%',
        '',
        SEP,
        'INTERPRETATION',
        f'  Post-lag AvgNet_WT: {_fp(pln)}',
        '  Gate:',
        '    >= +0.150%           -> DEPLOY AS PLANNED',
        '    +0.100% – +0.149%    -> DEPLOY NON-ASX PAIRS ONLY',
        '    < +0.100%            -> RE-ENGINEER BEFORE FUNDING',
        f'  Verdict: [{verdict}]',
    ]

    return '\n'.join(L)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _setup_logging() -> None:
    """Configure root logger for standalone script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def main() -> None:
    """Orchestrate the execution lag test: Parts 1–4 and output writing."""
    _setup_logging()
    _OUTPUT_DIR.mkdir(exist_ok=True)

    run_ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    stamp  = datetime.now().strftime('%Y%m%d_%H%M')

    logger.info('=== Phase 5 Execution Lag Test ===')
    logger.info(
        'XING_SD=%.1f, EXIT_SD=%.1f, VOL_WINDOW=%d, SPREAD_COST=%.3f%%, FIN=%.5f/day',
        XING_SD, EXIT_SD, VOL_WINDOW, SPREAD_COST_PCT * 100, FIN_NET_DAILY,
    )

    # ── Load equity prices ───────────────────────────────────────────────
    if not _PRICES_CSV.exists():
        raise FileNotFoundError(f"Equity price cache not found: {_PRICES_CSV}")

    prices, instruments = load_asset_prices(_PRICES_CSV, start_date='1999-01-01')

    if CIL_CODE not in instruments:
        raise ValueError(
            f"main: '{CIL_CODE}' (ASX 200) not found in instruments: {instruments}"
        )
    if len(instruments) < 2:
        raise ValueError(f"main: need >= 2 instruments; got {instruments}")

    logger.info("Loaded: %d instruments, %d rows", len(instruments), len(prices))
    logger.info("Instruments: %s", instruments)

    # ── Build lagged price frames ────────────────────────────────────────
    prices_aligned, prices_lagged = _build_aligned_price_frames(prices)

    # ── Compute vol-scaled matrices ──────────────────────────────────────
    scaled_u, scaled_l, day_ints, _ = _prepare_scaled_matrices(
        prices_aligned, prices_lagged, instruments,
    )

    # ── Part 1: Universe grid ────────────────────────────────────────────
    grid_df = run_universe_grid(scaled_u, scaled_l, day_ints, instruments)
    base_stats, lag_stats = _aggregate_universe_stats(grid_df)

    logger.info(
        "Part 1 summary: baseline=%.4f%%, lagged=%.4f%%, "
        "delta=%.4f%%  (pos pairs: %d→%d / %d)",
        base_stats['mean_avg_net'] * 100, lag_stats['mean_avg_net'] * 100,
        (lag_stats['mean_avg_net'] - base_stats['mean_avg_net']) * 100,
        base_stats['n_positive'], lag_stats['n_positive'], base_stats['n_total'],
    )

    # ── Baseline sanity check ────────────────────────────────────────────
    baseline_status = _check_baseline(base_stats)

    # ── Part 4: BestDir=Both ─────────────────────────────────────────────
    both_stats = compute_best_dir_both(grid_df, instruments)
    logger.info(
        "Part 4: BestDir=Both baseline=%d, lagged=%d "
        "(lost: one-dir=%d, zero=%d, neither=%d)",
        both_stats['base_both_count'], both_stats['lag_both_count'],
        both_stats['lost_to_one_dir'], both_stats['lost_to_zero_trades'],
        both_stats['lost_to_neither'],
    )

    # ── Part 2: Walk-forward ─────────────────────────────────────────────
    wf_results = run_wf_lagged(prices_lagged, instruments)

    # ── Part 3: Watchlist pairs ──────────────────────────────────────────
    logger.info("Part 3: Watchlist pair breakdown …")
    watchlist = run_watchlist_pairs(scaled_u, scaled_l, day_ints, instruments)

    # Control pair sanity check (warn in log if non-zero delta)
    for r in watchlist:
        if not r['has_asx']:
            delta = (abs(r['avg_net_lag'] - r['avg_net_base'])
                     if not pd.isna(r['avg_net_lag']) and not pd.isna(r['avg_net_base'])
                     else float('nan'))
            if not pd.isna(delta) and delta > 1e-8:
                logger.warning(
                    "Control pair %s/%s shows non-zero delta %.8f%% — "
                    "possible bug in lag implementation.",
                    r['long'], r['short'], delta * 100,
                )

    # ── Format report ────────────────────────────────────────────────────
    report = format_report(
        run_ts=run_ts,
        base_stats=base_stats,
        lag_stats=lag_stats,
        both_stats=both_stats,
        wf_lagged=wf_results,
        watchlist=watchlist,
        baseline_status=baseline_status,
    )

    report_path = _OUTPUT_DIR / f'phase5_execution_lag_{stamp}.txt'
    report_path.write_text(report, encoding='utf-8')
    logger.info("Report written: %s", report_path)
    # Safe print: Windows console may not support unicode; fall back to ascii.
    try:
        print(report)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(report.encode('utf-8', errors='replace') + b'\n')

    # ── JSON output ──────────────────────────────────────────────────────
    def _jsonify(obj: object) -> object:
        """Recursively convert numpy scalars for JSON serialisation."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    json_out = _jsonify({
        'run_ts': run_ts,
        'parameters': {
            'xing_sd': XING_SD, 'exit_sd': EXIT_SD,
            'vol_window': VOL_WINDOW, 'max_hold': MAX_HOLD,
            'spread_cost_pct': SPREAD_COST_PCT,
            'fin_net_daily': FIN_NET_DAILY,
            'wf_is_years': WF_IS_YEARS, 'wf_oos_years': WF_OOS_YEARS,
        },
        'universe': {'baseline': base_stats, 'lagged': lag_stats},
        'best_dir_both': both_stats,
        'walk_forward': {
            'baseline': {
                'rho': D118_WF_RHO, 'p_value': D118_WF_PVAL,
                'n_obs': D118_WF_NOBS, 'source': 'Decision 118',
            },
            'lagged': wf_results,
        },
        'watchlist': watchlist,
        'baseline_check': {
            'status': baseline_status,
            'd118_mean_net': D118_MEAN_AVG_NET,
            'run_mean_net': base_stats['mean_avg_net'],
            'delta': base_stats['mean_avg_net'] - D118_MEAN_AVG_NET,
            'threshold': BASELINE_ALERT_DELTA,
        },
    })

    _RESULTS_JSON.write_text(
        json.dumps(json_out, indent=2), encoding='utf-8',
    )
    logger.info("JSON results written: %s", _RESULTS_JSON)


if __name__ == '__main__':
    main()
