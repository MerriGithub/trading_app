"""
research/non_asx_expansion.py — Non-ASX Universe Expansion
===========================================================

Identifies deployable non-ASX equity pairs following the Decision 122
ruling that CIL (ASX 200) pairs are eliminated by the execution-lag test.

Background
----------
Decision 122 (2026-06-11): ASX pairs show material fill-price degradation
under a +1-day lag.  Non-ASX control pairs (e.g. UKX/CTN) show zero delta
and are confirmed deployable without a WF gate.

This script:
  1. Replicates the Phase 5 lag-grid parameters to produce a consistent
     per-pair universe (unlagged signal, lagged fill).
  2. Filters to non-CIL pairs with positive post-lag AvgNet.
  3. Derives the both_dir_caution flag (both directed variants positive).
  4. Runs a single WF call on the top-10 candidate instruments.
  5. Produces a verdict-annotated shortlist.

How to run
----------
From trading_app/:
    C:\\Users\\gordo\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe \\
        research/non_asx_expansion.py

Outputs:
    data/non_asx_candidates.json   — full filtered candidate list
    data/non_asx_wf_results.json   — raw WF pair-window records
    data/non_asx_wf_summary.json   — scalar WF summary metrics
    data/non_asx_shortlist.json    — final verdict-annotated shortlist

Compliance
----------
- from __future__ import annotations
- Type hints on all functions
- Google-style docstrings; Decision 122 and register items H, I, K, O, P referenced
- logging.getLogger(__name__) — no print() in functions
- Fail-fast input guards with specific exception types
- Register H/I canaries active via numba_core import (register item O)
- No bare except
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Register item O: all imports must be module-level.
# Importing numba_core activates H and I canary assertions.
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

# ── Paths ───────────────────────────────────────────────────────────────────
_CACHE_DIR = _ROOT / 'cache'
_DATA_DIR  = _ROOT / 'data'
_PRICES_CSV = _CACHE_DIR / 'prices.csv'

# ── Algorithm parameters — verbatim match to phase5_execution_lag.py ────────
XING_SD         : float = 2.0
EXIT_SD         : float = 2.0
VOL_WINDOW      : int   = 262
TARGET_VOL      : float = 0.01
MAX_HOLD        : int   = 300
N_LEGS          : int   = 2

# Equity financing: (4.88% long − 0.88% short rebate) / 365
_FIN_LONG_ANNUAL : float = 0.0488
_FIN_SHORT_ANNUAL: float = 0.0088
FIN_NET_DAILY    : float = (_FIN_LONG_ANNUAL - _FIN_SHORT_ANNUAL) / 365.0

# WF parameters — must match Q11 equity protocol (IS=3y, OOS=1y, step=1y)
WF_IS_YEARS       : int   = 3   # default in run_walk_forward is 5 — must pass explicitly
WF_OOS_YEARS      : int   = 1   # default in run_walk_forward is 2 — must pass explicitly
WF_STEP_YEARS     : int   = 1
WF_SCORING        : str   = 'contrarian'
WF_SPREAD_COST_PCT: float = 0.001  # 0.1% per leg — matches lag script WF call

CIL_CODE: str = 'CIL'  # ASX 200 — excluded from non-ASX candidates

# Confirmed-deployable pairs from Decision 122 (zero lag delta)
_CONFIRMED_DEPLOYABLE: frozenset[tuple[str, str]] = frozenset({
    ('UKX', 'CTN'),
    ('CTN', 'UKX'),
})

# Scalar keys from summarise_walk_forward — verified against walkforward.py:653-664
# Non-scalar keys (quintile_df, window_df, valid) are DataFrames and cannot be JSON-serialised.
_SCALAR_KEYS: frozenset[str] = frozenset({
    'rho', 'p_value', 'n_obs', 'q1_mean', 'q5_mean', 't_stat', 't_p',
})

# Verdict thresholds
_DEPLOY_OOS_TRADES   : int   = 3
_MONITOR_OOS_TRADES  : int   = 1   # >= 1 but < DEPLOY threshold


def _mean_equity_spread_cost() -> float:
    """Return universe-mean one-way spread cost from asset_configs equity instruments.

    Verbatim copy of the helper in research/phase5_execution_lag.py:94-103.
    Uses .get('spread_pct', 0.001) fallback and sum()/len() (not np.mean)
    to match the lag script exactly.

    Returns:
        Mean one-way spread cost fraction (~0.000462 for the current equity universe).
    """
    instruments_cfg = ASSET_CLASSES['equity']['instruments']
    costs = [cfg.get('spread_pct', 0.001) for cfg in instruments_cfg.values()]
    return sum(costs) / len(costs)


SPREAD_COST_PCT: float = _mean_equity_spread_cost()
# Per-instrument mean from asset_configs (~0.0462% ≈ 0.000462).
# Dynamic — avoids silent drift if instrument configs change.


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — PRICE FRAME CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _build_aligned_price_frames(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned (unlagged) and lagged price DataFrames for CIL.

    Shifts CIL +1 trading day to simulate ASX execution lag, then trims
    both frames to the same date range.  Non-CIL instruments are identical
    in both frames — control pairs will show exactly zero delta.

    See Decision 118 for the lag methodology; Decision 122 for the ruling
    that non-ASX pairs are deployable.

    Args:
        prices: Raw equity price DataFrame; must contain CIL_CODE column.

    Returns:
        (prices_aligned, prices_lagged): prices_aligned is original prices
        trimmed to match prices_lagged's date range.

    Raises:
        ValueError: If CIL_CODE is not in prices.columns.
    """
    if CIL_CODE not in prices.columns:
        raise ValueError(
            f"_build_aligned_price_frames: '{CIL_CODE}' not found in prices; "
            f"columns: {list(prices.columns)}"
        )
    prices_lagged = prices.copy()
    prices_lagged[CIL_CODE] = prices_lagged[CIL_CODE].shift(1)
    prices_lagged = prices_lagged.dropna(subset=[CIL_CODE])
    prices_aligned = prices.loc[prices_lagged.index]
    logger.info(
        "Price frames built: %d rows (1 leading row dropped for CIL shift)",
        len(prices_lagged),
    )
    return prices_aligned, prices_lagged


def _prepare_scaled_matrices(
    prices_aligned: pd.DataFrame,
    prices_lagged: pd.DataFrame,
    instruments: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build vol-scaled return matrices for both price frames.

    Register item O: prepare_returns must be imported at module level.
    prepare_returns returns a 3-tuple (scaled, day_ints, index) — unpack all three.

    Args:
        prices_aligned: Unlagged prices trimmed to the lagged date range.
        prices_lagged: Prices with CIL shifted +1 day.
        instruments: Instrument codes (columns present in both DataFrames).

    Returns:
        (scaled_u, scaled_l, day_ints, index) where scaled_u and scaled_l
        are shape (T, N) float64 arrays; day_ints is (T,) int64.

    Raises:
        ValueError: If CIL not in instruments, or shapes differ after trim.
    """
    if CIL_CODE not in instruments:
        raise ValueError(
            f"_prepare_scaled_matrices: '{CIL_CODE}' not in instruments: {instruments}"
        )

    scaled_u, day_ints, idx_u = prepare_returns(
        prices_aligned, instruments, vol_window=VOL_WINDOW, target_vol=TARGET_VOL,
    )
    scaled_l, _, idx_l = prepare_returns(
        prices_lagged, instruments, vol_window=VOL_WINDOW, target_vol=TARGET_VOL,
    )

    # Shape guard — matches phase5_execution_lag.py:228-239.
    # Fires only in edge cases; idx_u and idx_l share dates after the
    # prices.loc[prices_lagged.index] alignment in _build_aligned_price_frames.
    if scaled_u.shape[0] != scaled_l.shape[0]:
        common_idx = idx_u.intersection(idx_l)
        logger.warning(
            "Scaled array shapes differ (%d vs %d); trimming to common index (%d rows)",
            scaled_u.shape[0], scaled_l.shape[0], len(common_idx),
        )
        mask_u  = idx_u.isin(common_idx)
        mask_l  = idx_l.isin(common_idx)
        scaled_u  = scaled_u[mask_u]
        scaled_l  = scaled_l[mask_l]   # uses idx_l, not idx_u — correct array
        day_ints  = day_ints[mask_u]
        idx_u     = idx_u[mask_u]      # update idx_u to match trimmed shape

    logger.info(
        "Scaled matrices ready: shape=%s, numba=%s",
        scaled_u.shape, HAS_NUMBA,
    )
    return scaled_u, scaled_l, day_ints, idx_u


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — PER-PAIR UNIVERSE GRID
# ═══════════════════════════════════════════════════════════════════════════

def run_universe_grid(
    scaled_u: np.ndarray,
    scaled_l: np.ndarray,
    day_ints: np.ndarray,
    instruments: list[str],
) -> pd.DataFrame:
    """Run baseline and lagged backtest for all N×(N-1) directed equity pairs.

    ⚠️ Signal is computed from UNLAGGED prices; P&L fill is computed from
    LAGGED prices.  This isolates fill-price contamination from signal
    contamination.  See Decision 118 / 122.

    Register item K: uses detect_trades + aggregate_trades directly, not
    run_backtest (which does not expose a separate fill-price series).

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

    pairs: list[tuple[int, int]] = [
        (i, j) for i in range(N) for j in range(N) if i != j
    ]
    logger.info("Universe grid: %d directed pairs", len(pairs))
    t0 = time.time()

    records: list[dict] = []
    for long_i, short_i in pairs:
        # SIGNAL: unlagged z-score — same for baseline and lagged runs.
        spread_u = (scaled_u[:, long_i] - scaled_u[:, short_i]).astype(np.float64)
        cum_u    = np.cumprod(1.0 + spread_u)
        roll_mean, roll_std = rolling_mean_std(cum_u, VOL_WINDOW)
        with np.errstate(invalid='ignore', divide='ignore'):
            dist_sd = np.where(roll_std > 0, (cum_u - roll_mean) / roll_std, np.nan)

        # BASELINE fill: unlagged cum.
        trades_b, n_b = detect_trades(cum_u, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)
        stats_b: dict = {}
        if n_b > 0:
            stats_b = aggregate_trades(trades_b, n_b, SPREAD_COST_PCT, FIN_NET_DAILY, N_LEGS)

        # LAGGED fill: lagged cum, same unlagged signal (dist_sd unchanged).
        spread_l = (scaled_l[:, long_i] - scaled_l[:, short_i]).astype(np.float64)
        cum_l    = np.cumprod(1.0 + spread_l)
        trades_l, n_l = detect_trades(cum_l, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)
        stats_l: dict = {}
        if n_l > 0:
            stats_l = aggregate_trades(trades_l, n_l, SPREAD_COST_PCT, FIN_NET_DAILY, N_LEGS)

        # 'avg_holding' key confirmed at engine/backtest.py:322 — not 'avg_hold'.
        records.append({
            'long':          instruments[long_i],
            'short':         instruments[short_i],
            'n_base':        stats_b.get('n_trades', 0),
            'n_lag':         stats_l.get('n_trades', 0),
            'avg_net_base':  stats_b.get('avg_net',     np.nan),
            'avg_net_lag':   stats_l.get('avg_net',     np.nan),
            'net_wr_base':   stats_b.get('net_wr',      np.nan),
            'net_wr_lag':    stats_l.get('net_wr',      np.nan),
            'avg_hold_base': stats_b.get('avg_holding', np.nan),
            'avg_hold_lag':  stats_l.get('avg_holding', np.nan),
        })

    logger.info("Universe grid complete: %.1fs", time.time() - t0)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — NON-CIL CANDIDATE LIST
# ═══════════════════════════════════════════════════════════════════════════

def build_non_cil_candidates(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Filter the universe grid to non-CIL pairs with positive lagged return.

    Derives both_dir_caution flag: True when BOTH directed variants
    (A→B and B→A) have avg_net_lag > 0, indicating the undirected pair
    is generally profitable post-lag regardless of direction.

    Neither the filter nor the flag looks at baseline stats — only the
    lagged column is used, consistent with Decision 122.

    Args:
        grid_df: Output of run_universe_grid.

    Returns:
        DataFrame filtered to non-CIL, positive-lagged pairs, sorted
        descending by avg_net_lag.  Adds columns:
            delta_avg_net      — avg_net_lag − avg_net_base
            both_dir_caution   — bool

    Raises:
        ValueError: If grid_df is empty.
    """
    if grid_df.empty:
        raise ValueError("build_non_cil_candidates: grid_df is empty")

    # Build lookup for both-direction check before filtering
    lag_net: dict[tuple[str, str], float] = {
        (r.long, r.short): r.avg_net_lag
        for r in grid_df.itertuples()
    }

    # Filter: exclude CIL on either leg; require positive lagged return
    mask = (
        (grid_df['long']  != CIL_CODE) &
        (grid_df['short'] != CIL_CODE) &
        (grid_df['avg_net_lag'] > 0)
    )
    df = grid_df[mask].copy()

    if df.empty:
        logger.warning("build_non_cil_candidates: no non-CIL pairs with positive lagged return")
        return df

    # Delta: lag minus baseline
    df['delta_avg_net'] = df.apply(
        lambda r: (
            r['avg_net_lag'] - r['avg_net_base']
            if not (np.isnan(r['avg_net_lag']) or np.isnan(r['avg_net_base']))
            else np.nan
        ),
        axis=1,
    )

    # both_dir_caution: True when reverse direction also has positive lagged return
    def _both_dir(row: pd.Series) -> bool:
        rev_net = lag_net.get((row['short'], row['long']), np.nan)
        return bool(not np.isnan(rev_net) and rev_net > 0)

    df['both_dir_caution'] = df.apply(_both_dir, axis=1)
    df = df.sort_values('avg_net_lag', ascending=False).reset_index(drop=True)

    logger.info(
        "Non-CIL candidates: %d pairs (%d both-dir-caution)",
        len(df), int(df['both_dir_caution'].sum()),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def run_wf_validation(
    prices: pd.DataFrame,
    candidates_df: pd.DataFrame,
    instruments: list[str],
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Run a single WF call on the top candidate instruments.

    ⚠️ CRITICAL: run_walk_forward generates ALL N×(N-1) directed pairs
    internally.  Do NOT call it once per pair — that produces meaningless
    IS_Rank values (rank 1 of 2).  ONE call with all candidate instruments.

    Acceptance threshold: rho > 0 and p_value < 0.05.
    Reference: Decision 122 post-lag WF = rho=+0.263, p<0.0001, n=1,618.

    Register item P: n_obs from summarise_walk_forward is the number of
    pair-window observations with OOS_Trades > 0, NOT the window count.
    Per-pair OOS trade counts are read directly from the raw wf_results DataFrame.

    Uses unlagged prices — run_walk_forward recomputes its own scaling
    internally with no hook to inject a separate fill series.

    Args:
        prices: Unlagged equity price DataFrame.
        candidates_df: Output of build_non_cil_candidates.
        instruments: Full instrument list from load_asset_prices.
        top_n: Number of top candidates (by avg_net_lag) to include in the WF.
            UKX and CTN are always included regardless of ranking.

    Returns:
        (wf_results, wf_summary): wf_results is the raw DataFrame from
        run_walk_forward; wf_summary is the dict from summarise_walk_forward.

    Raises:
        ValueError: If run_walk_forward returns an empty DataFrame, or if
            required WF columns are absent.
    """
    if candidates_df.empty:
        raise ValueError("run_wf_validation: candidates_df is empty")

    # Collect instruments from top-N candidates
    top_pairs = candidates_df.head(top_n)
    wf_instruments: list[str] = []
    for code in pd.concat([top_pairs['long'], top_pairs['short']]).unique():
        if code not in wf_instruments:
            wf_instruments.append(code)

    # Always include confirmed-deployable benchmarks
    for code in ('UKX', 'CTN'):
        if code not in wf_instruments:
            wf_instruments.append(code)
            logger.info("WF: added benchmark instrument %s", code)

    # Confirm all instruments are present in prices
    missing = [c for c in wf_instruments if c not in prices.columns]
    if missing:
        logger.warning("WF: instruments not in prices, skipping: %s", missing)
        wf_instruments = [c for c in wf_instruments if c not in missing]

    if len(wf_instruments) < 2:
        raise ValueError(
            f"run_wf_validation: fewer than 2 valid WF instruments after missing check: "
            f"{wf_instruments}"
        )

    logger.info(
        "WF call: %d instruments (%s), IS=%dy OOS=%dy",
        len(wf_instruments), wf_instruments, WF_IS_YEARS, WF_OOS_YEARS,
    )

    wf_results = run_walk_forward(
        prices=prices,
        instruments=wf_instruments,
        is_years=WF_IS_YEARS,       # default=5, MUST pass explicitly for equity
        oos_years=WF_OOS_YEARS,     # default=2, MUST pass explicitly for equity
        step_years=WF_STEP_YEARS,
        scoring_mode=WF_SCORING,
        vol_window=VOL_WINDOW,
        target_vol=TARGET_VOL,
        xing_sd=XING_SD,
        exit_sd=EXIT_SD,
        spread_cost_pct=WF_SPREAD_COST_PCT,
    )

    if wf_results.empty:
        raise ValueError(
            "run_wf_validation: run_walk_forward returned empty DataFrame — "
            "insufficient price history or no trades"
        )

    # Guard: OOS_Net column must exist (walkforward.py:317)
    if 'OOS_Net' not in wf_results.columns:
        raise ValueError(
            "run_wf_validation: 'OOS_Net' column absent from wf_results — "
            "check engine/walkforward.py version"
        )

    wf_summary = summarise_walk_forward(wf_results)
    logger.info(
        "WF complete: rho=%.3f, p=%.4f, n_obs=%d",
        wf_summary['rho'], wf_summary['p_value'], wf_summary['n_obs'],
    )

    return wf_results, wf_summary


def _per_pair_oos_stats(
    wf_results: pd.DataFrame,
    candidate_pairs: list[tuple[str, str]],
) -> list[dict]:
    """Aggregate per-pair OOS statistics from the raw WF results DataFrame.

    Register item P: n_obs from summarise_walk_forward counts pair-window
    observations, not per-pair trade counts.  Read per-pair trade counts
    directly from wf_results.

    Args:
        wf_results: Raw DataFrame from run_walk_forward.
        candidate_pairs: List of (long_code, short_code) to aggregate.

    Returns:
        List of dicts with oos_trades, oos_winrate, oos_gross, oos_net
        for each candidate pair.
    """
    stats: list[dict] = []
    for long_code, short_code in candidate_pairs:
        pair_df = wf_results[
            (wf_results['long'] == long_code) &
            (wf_results['short'] == short_code)
        ]
        oos_trades = int(pair_df['OOS_Trades'].sum())
        active     = pair_df[pair_df['OOS_Trades'] > 0]
        oos_wr     = float(active['OOS_WinRate'].mean()) if len(active) else 0.0
        oos_gross  = float(active['OOS_Gross'].mean())   if len(active) else 0.0
        oos_net    = float(active['OOS_Net'].mean())     if len(active) else 0.0
        stats.append({
            'long':        long_code,
            'short':       short_code,
            'oos_trades':  oos_trades,
            'oos_winrate': oos_wr,
            'oos_gross':   oos_gross,
            'oos_net':     oos_net,
        })
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — SHORTLIST
# ═══════════════════════════════════════════════════════════════════════════

def build_shortlist(
    candidates_df: pd.DataFrame,
    wf_results: pd.DataFrame,
    wf_summary: dict,
) -> pd.DataFrame:
    """Combine grid and WF results into a verdict-annotated shortlist.

    Verdict logic:
        'DEPLOY'  — avg_net_lag > 0, oos_net > 0, oos_trades >= 3
        'MONITOR' — avg_net_lag > 0, oos_net > 0, oos_trades < 3
        'REVIEW'  — avg_net_lag > 0, oos_net <= 0
        UKX/CTN and CTN/UKX always 'DEPLOY' (zero-delta confirmed, Decision 122)

    Args:
        candidates_df: Output of build_non_cil_candidates.
        wf_results: Raw DataFrame from run_walk_forward.
        wf_summary: Dict from summarise_walk_forward (scalar keys only needed).

    Returns:
        DataFrame with at least 1 OOS trade, ranked by oos_net descending,
        containing verdict, both_dir_caution, and confirmed_deployable fields.

    Raises:
        ValueError: If wf_results is empty or candidates_df is empty.
    """
    if candidates_df.empty:
        raise ValueError("build_shortlist: candidates_df is empty")
    if wf_results.empty:
        raise ValueError("build_shortlist: wf_results is empty")

    candidate_pairs = list(zip(candidates_df['long'], candidates_df['short']))
    oos_stats = _per_pair_oos_stats(wf_results, candidate_pairs)

    # Build lookup for grid stats
    grid_lookup: dict[tuple[str, str], pd.Series] = {
        (r['long'], r['short']): r
        for _, r in candidates_df.iterrows()
    }

    rows: list[dict] = []
    for s in oos_stats:
        if s['oos_trades'] < 1:
            continue
        key = (s['long'], s['short'])
        grid = grid_lookup.get(key, {})
        avg_net_lag    = grid.get('avg_net_lag', np.nan) if isinstance(grid, pd.Series) else grid.get('avg_net_lag', np.nan)
        both_caution   = bool(grid.get('both_dir_caution', False)) if isinstance(grid, pd.Series) else grid.get('both_dir_caution', False)
        confirmed      = key in _CONFIRMED_DEPLOYABLE

        if confirmed:
            verdict = 'DEPLOY'
        elif s['oos_net'] > 0 and s['oos_trades'] >= _DEPLOY_OOS_TRADES:
            verdict = 'DEPLOY'
        elif s['oos_net'] > 0:
            verdict = 'MONITOR'
        else:
            verdict = 'REVIEW'

        rows.append({
            'long':               s['long'],
            'short':              s['short'],
            'avg_net_lag':        avg_net_lag,
            'oos_trades':         s['oos_trades'],
            'oos_winrate':        s['oos_winrate'],
            'oos_net':            s['oos_net'],
            'both_dir_caution':   both_caution,
            'confirmed_deployable': confirmed,
            'verdict':            verdict,
        })

    if not rows:
        logger.warning("build_shortlist: no pairs with OOS trades")
        return pd.DataFrame()

    shortlist = (
        pd.DataFrame(rows)
        .sort_values('oos_net', ascending=False)
        .reset_index(drop=True)
    )
    shortlist.index += 1  # 1-based rank
    logger.info(
        "Shortlist: %d pairs (%d DEPLOY, %d MONITOR, %d REVIEW)",
        len(shortlist),
        (shortlist['verdict'] == 'DEPLOY').sum(),
        (shortlist['verdict'] == 'MONITOR').sum(),
        (shortlist['verdict'] == 'REVIEW').sum(),
    )
    return shortlist


# ═══════════════════════════════════════════════════════════════════════════
# REPORT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def _fp(v: float | None, dec: int = 3, width: int = 0) -> str:
    """Format a decimal fraction as a percentage string (+X.XXX%)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        s = 'n/a'
    else:
        s = f'{v * 100:+.{dec}f}%'
    return s if width == 0 else s.rjust(width)


def _ff(v: float | None, dec: int = 1, width: int = 0) -> str:
    """Format a float or return 'n/a'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        s = 'n/a'
    else:
        s = f'{v:.{dec}f}'
    return s if width == 0 else s.rjust(width)


def _print_top20(candidates_df: pd.DataFrame) -> None:
    """Print the top-20 non-CIL candidates to stdout.

    Args:
        candidates_df: Output of build_non_cil_candidates.
    """
    SEP = '-' * 108
    header = (
        f"{'Rank':>4}  {'Long':<6}{'Short':<6}"
        f"{'AvgNet(L)':>10}{'AvgNet(B)':>10}{'Delta':>10}{'N(L)':>6}"
        f"{'WR(L)':>7}{'Hold(L)':>8}  {'BothDir':>7}"
    )
    print('\n=== TOP-20 NON-ASX CANDIDATES (sorted by avg_net_lag) ===')
    print(header)
    print(SEP)
    for rank, (_, row) in enumerate(candidates_df.head(20).iterrows(), start=1):
        wr_l = _ff(row['net_wr_lag'] * 100 if not np.isnan(row['net_wr_lag']) else None, 1) + '%'
        print(
            f"{rank:>4}  {row['long']:<6}{row['short']:<6}"
            f"{_fp(row['avg_net_lag'], 3, 10)}{_fp(row['avg_net_base'], 3, 10)}"
            f"{_fp(row['delta_avg_net'], 3, 10)}{int(row['n_lag']):>6}"
            f"{wr_l:>7}{_ff(row['avg_hold_lag'], 1, 7) + 'd':>8}"
            f"  {'Y' if row['both_dir_caution'] else 'N':>7}"
        )
    print(SEP)


def _print_wf_table(
    candidates_df: pd.DataFrame,
    wf_results: pd.DataFrame,
    wf_summary: dict,
) -> None:
    """Print per-pair WF stats to stdout.

    Args:
        candidates_df: Output of build_non_cil_candidates.
        wf_results: Raw WF results DataFrame.
        wf_summary: Dict from summarise_walk_forward.
    """
    candidate_pairs = list(zip(candidates_df['long'], candidates_df['short']))
    oos_stats = _per_pair_oos_stats(wf_results, candidate_pairs)
    grid_lookup = {
        (r['long'], r['short']): r for _, r in candidates_df.iterrows()
    }

    SEP = '-' * 90
    print(f'\n=== WALK-FORWARD RESULTS (rho={wf_summary["rho"]:+.3f}, '
          f'p={wf_summary["p_value"]:.4f}, n_obs={wf_summary["n_obs"]}) ===')
    print(
        f"{'Long':<6}{'Short':<6}{'AvgNet(L)':>10}"
        f"{'OosTrades':>10}{'OosWR':>7}{'OosGross':>9}{'OosNet':>9}  BothDir"
    )
    print(SEP)
    for s in oos_stats:
        key  = (s['long'], s['short'])
        grid = grid_lookup.get(key, {})
        net_lag = grid.get('avg_net_lag', np.nan) if isinstance(grid, pd.Series) else np.nan
        both    = bool(grid.get('both_dir_caution', False)) if isinstance(grid, pd.Series) else False
        wr_str  = _ff(s['oos_winrate'] * 100, 1) + '%'
        print(
            f"{s['long']:<6}{s['short']:<6}{_fp(net_lag, 3, 10)}"
            f"{s['oos_trades']:>10}{wr_str:>7}"
            f"{_fp(s['oos_gross'], 3, 9)}{_fp(s['oos_net'], 3, 9)}"
            f"  {'Y' if both else 'N'}"
        )
    print(SEP)


def _print_shortlist(shortlist: pd.DataFrame) -> None:
    """Print the final verdict-annotated shortlist to stdout.

    Args:
        shortlist: Output of build_shortlist.
    """
    if shortlist.empty:
        print('\n=== SHORTLIST: no qualifying pairs ===')
        return

    SEP = '-' * 100
    print('\n=== FINAL SHORTLIST ===')
    print(
        f"{'Rank':>4}  {'Long':<6}{'Short':<6}{'AvgNet(L)':>10}"
        f"{'OosTrades':>10}{'OosWR':>7}{'OosNet':>9}"
        f"  {'BothDir':>7}  {'Confirmed':>9}  Verdict"
    )
    print(SEP)
    for rank, row in shortlist.iterrows():
        wr_str = _ff(row['oos_winrate'] * 100, 1) + '%'
        print(
            f"{rank:>4}  {row['long']:<6}{row['short']:<6}"
            f"{_fp(row['avg_net_lag'], 3, 10)}"
            f"{int(row['oos_trades']):>10}{wr_str:>7}"
            f"{_fp(row['oos_net'], 3, 9)}"
            f"  {'Y' if row['both_dir_caution'] else 'N':>7}"
            f"  {'Y' if row['confirmed_deployable'] else 'N':>9}"
            f"  {row['verdict']}"
        )
    print(SEP)


# ═══════════════════════════════════════════════════════════════════════════
# JSON SERIALISATION
# ═══════════════════════════════════════════════════════════════════════════

def _jsonify(obj: object) -> object:
    """Recursively convert numpy scalars for JSON serialisation."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


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
    """Orchestrate the non-ASX expansion analysis: Steps 1–5 and output writing."""
    _setup_logging()

    logger.info('=== Non-ASX Universe Expansion (Decision 122) ===')
    logger.info(
        'XING_SD=%.1f, EXIT_SD=%.1f, VOL_WINDOW=%d, SPREAD_COST=%.4f%%, FIN=%.5f/day',
        XING_SD, EXIT_SD, VOL_WINDOW, SPREAD_COST_PCT * 100, FIN_NET_DAILY,
    )

    # ── Load equity prices ─────────────────────────────────────────────────
    if not _PRICES_CSV.exists():
        raise FileNotFoundError(f"Equity price cache not found: {_PRICES_CSV}")

    prices, instruments = load_asset_prices(_PRICES_CSV, start_date='1999-01-01')

    if CIL_CODE not in instruments:
        raise ValueError(
            f"main: '{CIL_CODE}' not found in instruments: {instruments}. "
            f"CIL is required to build the lagged arrays."
        )
    if len(instruments) < 2:
        raise ValueError(f"main: need >= 2 instruments; got {instruments}")

    logger.info("Loaded: %d instruments, %d rows", len(instruments), len(prices))

    # ── Step 1: Build lagged price frames ─────────────────────────────────
    prices_aligned, prices_lagged = _build_aligned_price_frames(prices)

    # ── Compute vol-scaled matrices ────────────────────────────────────────
    scaled_u, scaled_l, day_ints, _ = _prepare_scaled_matrices(
        prices_aligned, prices_lagged, instruments,
    )

    # ── Step 2: Universe grid ──────────────────────────────────────────────
    grid_df = run_universe_grid(scaled_u, scaled_l, day_ints, instruments)

    # ── Step 3: Non-CIL candidate list ────────────────────────────────────
    candidates_df = build_non_cil_candidates(grid_df)

    print(f'\nTotal non-CIL positive-lagged pairs: {len(candidates_df)}')
    _print_top20(candidates_df)

    # Save full candidate list
    cand_path = _DATA_DIR / 'non_asx_candidates.json'
    cand_records = _jsonify(candidates_df.reset_index(drop=True).to_dict('records'))
    cand_path.write_text(json.dumps(cand_records, indent=2), encoding='utf-8')
    logger.info("Candidates saved: %s (%d pairs)", cand_path, len(candidates_df))

    if candidates_df.empty:
        logger.warning("No non-CIL candidates found — stopping.")
        return

    # ── Step 4: WF validation ──────────────────────────────────────────────
    wf_results, wf_summary = run_wf_validation(
        prices=prices,            # unlagged prices — WF handles its own scaling
        candidates_df=candidates_df,
        instruments=instruments,
        top_n=10,
    )

    _print_wf_table(candidates_df, wf_results, wf_summary)

    # Accept/reject universe
    rho     = wf_summary['rho']
    p_value = wf_summary['p_value']
    if rho > 0 and p_value < 0.05:
        print(f'\nWF PASS: rho={rho:+.3f}, p={p_value:.4f} — IS rank predicts OOS performance')
    else:
        print(f'\nWF FAIL: rho={rho:+.3f}, p={p_value:.4f} — no significant IS→OOS predictability')

    # Save WF results
    wf_res_path = _DATA_DIR / 'non_asx_wf_results.json'
    wf_results.reset_index(drop=True).to_json(
        wf_res_path, orient='records', indent=2,
    )
    logger.info("WF results saved: %s (%d rows)", wf_res_path, len(wf_results))

    # Save scalar WF summary — DataFrames excluded to allow JSON serialisation.
    # _SCALAR_KEYS verified against engine/walkforward.py:653-664.
    wf_summary_serial = _jsonify(
        {k: v for k, v in wf_summary.items() if k in _SCALAR_KEYS}
    )
    wf_sum_path = _DATA_DIR / 'non_asx_wf_summary.json'
    wf_sum_path.write_text(json.dumps(wf_summary_serial, indent=2), encoding='utf-8')
    logger.info("WF summary saved: %s", wf_sum_path)

    # ── Step 5: Shortlist ──────────────────────────────────────────────────
    shortlist = build_shortlist(candidates_df, wf_results, wf_summary)
    _print_shortlist(shortlist)

    # Save shortlist
    sl_path = _DATA_DIR / 'non_asx_shortlist.json'
    sl_records = _jsonify(shortlist.reset_index().rename(columns={'index': 'rank'}).to_dict('records'))
    sl_path.write_text(json.dumps(sl_records, indent=2), encoding='utf-8')
    logger.info("Shortlist saved: %s (%d pairs)", sl_path, len(shortlist))


if __name__ == '__main__':
    main()
