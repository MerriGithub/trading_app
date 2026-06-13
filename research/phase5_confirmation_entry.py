"""
research/phase5_confirmation_entry.py — Phase 5 Experiment 3B
==============================================================

Tests whether requiring the spread to *reverse direction* by at least
``confirmation_bars`` bars after crossing the entry threshold improves
risk-adjusted returns for the equity scalp regime.

Background
----------
Phase 2 tested four entry signal improvements under EXIT_SD=0.0 (full
hold to mean).  All were rejected.  Phase 4b subsequently confirmed
EXIT_SD=2.0 as the equity scalp optimum.  With a tight exit, entry
timing dominates because the entire captured return occurs in the first
few bars.  The Phase 2 rejections may not hold under the optimised exit
(register item 3B in Fixable Items Plan, 2026-06-07).

Hypothesis
----------
Entering on the first bar where the z-score reverses after crossing
±XING_SD (rather than at the crossing itself) should:
  - Increase win rate (filter out false breaks that do not revert)
  - Thin the left tail (fewer trades that extend further adverse before exit)
  - Reduce average capture slightly (some trades entered a bar or two later)

If: WR rises meaningfully AND worst-trade improves AND avg_net >= 0.30%,
    accept as a new optimum.  (0.30% is the lower bound of the plateau
    confirmed in Decision 116, Task A grid.)

Design
------
Grid: confirmation_bars {0, 1, 2, 3}  ×  exit_sd {1.5, 1.8, 2.0}

confirmation_bars=0 reproduces the standard crossing signal exactly
(baseline row).  For confirmation_bars=N (N>0), the entry fires on the
bar where the z-score has moved back toward zero for N consecutive bars
after the initial crossing.

Asset class: equities first (highest upside from entry improvement due
to thin per-trade margin).  Run the full equity 1v1 pair universe.

How to run
----------
From the trading_app/ directory:

    C:\\Users\\gordo\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe \\
        research/phase5_confirmation_entry.py

Outputs (written to research/output/):
    phase5_3b_grid_summary.csv        — mean stats across all pairs per grid point
    phase5_3b_pair_detail.csv         — per-pair stats at each grid point
    phase5_3b_baseline_vs_best.csv    — side-by-side comparison (CB=0 vs best CB)

Compliance
----------
- Type hints on all functions (from __future__ import annotations)
- Google-style docstrings
- logging, not print
- Fail-fast input guards
- Specific exception types
- Register item H and I canaries imported and active via numba_core import
- No bare except
"""

from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
# Script lives in research/; add the project root (trading_app/) to sys.path
# so engine/, core/, and asset_configs.py are importable.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Importing numba_core activates the register H and I module-level canary
# assertions.  If either invariant has been violated, the import will raise
# AssertionError and the script will not run.
from engine.numba_core import (  # noqa: E402
    rolling_mean_std,
    detect_trades,
    HAS_NUMBA,
    COL_ENTRY_IDX,
    COL_EXIT_IDX,
    COL_SIDE,
    COL_GROSS_RETURN,
    COL_HOLDING_DAYS,
)

if HAS_NUMBA:
    import numba as _numba
from engine.backtest import (  # noqa: E402
    load_asset_prices,
    prepare_returns,
    aggregate_trades,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_VOL_WINDOW: int = 262
DEFAULT_XING_SD: float = 2.0
DEFAULT_MAX_HOLD_DAYS: int = 300

# Equity scalp: EXIT_SD confirmed plateau 1.5–2.0 (Decision 116).
EXIT_SD_GRID: list[float] = [1.5, 1.8, 2.0]

# Confirmation window grid.  0 = standard crossing (baseline).
CONFIRMATION_BARS_GRID: list[int] = [0, 1, 2, 3]

# Equity financing and spread cost (4.88% p.a. long, 0.88% rebate short)
EQUITY_FINANCING_LONG_ANNUAL: float = 0.0488
EQUITY_FINANCING_SHORT_ANNUAL: float = 0.0088
EQUITY_SPREAD_COST_PCT: float = 0.001  # 0.1% per leg, one-way

# Accept threshold from Fixable Items Plan item 3B
ACCEPT_AVG_NET_THRESHOLD: float = 0.0030  # 0.30%


# ═══════════════════════════════════════════════════════════════════════════
# CORE ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════

# Numba JIT confirmation detector — defined when Numba is available.
# The pure-Python fallback inside _detect_trades_with_confirmation is used
# when Numba is absent.  Semantics are identical; verified by running CB=0
# through detect_trades() which has its own parity test in the engine.
_numba_detect_trades_confirmation = None

if HAS_NUMBA:
    @_numba.njit(cache=True)
    def _numba_detect_confirmation_jit(
        cum, dist_sd, xing_sd, exit_sd, day_ints, confirmation_bars, max_hold_days,
    ):
        """JIT-compiled confirmation-entry trade detection (CB > 0 only).

        Exit condition (REGISTER ITEM H — sign intentionally inverted):
            (side == -1 and d <= exit_sd) or (side == 1 and d >= -exit_sd)
        Do NOT rewrite as side * d <= exit_sd.
        """
        T = len(cum)
        max_trades = T // 2 + 1
        trades = np.zeros((max_trades, 5), dtype=np.float64)
        n = 0

        PENDING_CANCEL_SD = xing_sd + 2.5

        in_trade = False
        pending = False
        pending_side = 0
        prev_abs_d = 0.0
        confirm_count = 0
        entry_idx = 0
        entry_cum = 0.0
        side = 0

        for i in range(T):
            d = dist_sd[i]
            if np.isnan(d):
                continue

            if not in_trade and not pending:
                if d > xing_sd:
                    pending = True
                    pending_side = -1
                    prev_abs_d = abs(d)
                    confirm_count = 0
                elif d < -xing_sd:
                    pending = True
                    pending_side = 1
                    prev_abs_d = abs(d)
                    confirm_count = 0

            elif pending:
                abs_d = abs(d)
                if abs_d >= PENDING_CANCEL_SD:
                    pending = False
                    pending_side = 0
                    confirm_count = 0
                    continue
                if pending_side == -1 and d < -xing_sd:
                    pending = False
                    pending_side = 0
                    confirm_count = 0
                    continue
                if pending_side == 1 and d > xing_sd:
                    pending = False
                    pending_side = 0
                    confirm_count = 0
                    continue
                if abs_d < prev_abs_d:
                    confirm_count += 1
                else:
                    confirm_count = 0
                prev_abs_d = abs_d
                if confirm_count >= confirmation_bars:
                    in_trade = True
                    pending = False
                    entry_idx = i
                    entry_cum = cum[i]
                    side = pending_side
                    pending_side = 0
                    confirm_count = 0

            else:
                # REGISTER ITEM H — exit sign intentionally inverted.
                exit_condition = (
                    (side == -1 and d <= exit_sd)
                    or (side == 1 and d >= -exit_sd)
                    or (day_ints[i] - day_ints[entry_idx] >= max_hold_days)
                )
                if exit_condition:
                    exit_cum = cum[i]
                    if side == 1:
                        gross_ret = (exit_cum - entry_cum) / entry_cum
                    else:
                        gross_ret = (entry_cum - exit_cum) / entry_cum
                    holding = day_ints[i] - day_ints[entry_idx]
                    trades[n, 0] = entry_idx   # COL_ENTRY_IDX
                    trades[n, 1] = i           # COL_EXIT_IDX
                    trades[n, 2] = side        # COL_SIDE
                    trades[n, 3] = gross_ret   # COL_GROSS_RETURN
                    trades[n, 4] = holding     # COL_HOLDING_DAYS
                    n += 1
                    in_trade = False

        return trades[:n], n

    _numba_detect_trades_confirmation = _numba_detect_confirmation_jit


def _detect_trades_with_confirmation(
    cum: np.ndarray,
    dist_sd: np.ndarray,
    xing_sd: float,
    exit_sd: float,
    day_ints: np.ndarray,
    confirmation_bars: int,
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
) -> tuple[np.ndarray, int]:
    """Pure-Python crossing signal trade detection with confirmation entry.

    Extends the standard _ref_detect_trades logic to require the spread to
    move back toward zero for ``confirmation_bars`` bars before the entry
    fires.  When confirmation_bars=0 the behaviour is identical to the
    standard crossing signal.

    Confirmation logic:
        After the spread crosses ±xing_sd, instead of entering immediately,
        the algorithm enters on the first bar where the z-score has been
        moving toward zero (i.e. reducing in absolute value) for N consecutive
        bars, where N = confirmation_bars.

        Concretely: the algorithm enters a pending state at the crossing.
        In each pending bar it checks whether |d[i]| < |d[i-1]|.  If this
        condition holds for confirmation_bars consecutive bars, the trade
        fires on bar i (the last confirmation bar).  If the spread extends
        further beyond a 4.5 SD hard boundary while pending, the pending
        state is cancelled (the move has become a trend, not a snap).

        The entry price used is cum[entry_idx] where entry_idx is the bar
        the trade fires (not the crossing bar), so the confirmation delay
        is reflected in both the gross return and the holding period.

    Exit condition (register item H — sign is intentionally inverted):
        Correct form: (side == -1 and d <= exit_sd) or (side == 1 and d >= -exit_sd)
        Do NOT rewrite as ``side * d <= exit_sd``.

    Entry dislocation (register item I — not stored here; this function
        returns the standard 5-column trade array matching COL_* constants.
        If you extend to store entry_d, always use abs(d).)

    Args:
        cum: Cumulative product of (1 + spread_return), shape (T,), float64.
        dist_sd: Z-score distance from rolling mean, shape (T,), float64.
        xing_sd: Entry threshold in SD units (e.g. 2.0).
        exit_sd: Normal exit threshold in SD units (e.g. 2.0 for equity scalp).
        day_ints: Integer day index per row, shape (T,), int64.
        confirmation_bars: Number of consecutive bars moving toward zero required
            before entry fires. 0 = standard immediate entry.
        max_hold_days: Maximum holding period before forced exit.

    Returns:
        Tuple (trades, n_trades) where trades has shape (n_trades, 5) with
        columns matching COL_ENTRY_IDX, COL_EXIT_IDX, COL_SIDE,
        COL_GROSS_RETURN, COL_HOLDING_DAYS.

    Raises:
        ValueError: If confirmation_bars < 0.
        TypeError: If cum, dist_sd, or day_ints are not numpy ndarrays.
    """
    if not isinstance(cum, np.ndarray):
        raise TypeError(
            f"_detect_trades_with_confirmation: cum must be a numpy ndarray; "
            f"got {type(cum).__name__}"
        )
    if not isinstance(dist_sd, np.ndarray):
        raise TypeError(
            f"_detect_trades_with_confirmation: dist_sd must be a numpy ndarray; "
            f"got {type(dist_sd).__name__}"
        )
    if not isinstance(day_ints, np.ndarray):
        raise TypeError(
            f"_detect_trades_with_confirmation: day_ints must be a numpy ndarray; "
            f"got {type(day_ints).__name__}"
        )
    if confirmation_bars < 0:
        raise ValueError(
            f"_detect_trades_with_confirmation: confirmation_bars must be >= 0; "
            f"got {confirmation_bars}"
        )

    # CB=0: delegate to engine's detect_trades(), which dispatches to Numba
    # when available.  Identical to the standard crossing signal (no delay).
    if confirmation_bars == 0:
        return detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints, max_hold_days)

    # CB>0: use Numba-compiled version if available; else pure-Python below.
    if _numba_detect_trades_confirmation is not None:
        return _numba_detect_trades_confirmation(
            cum, dist_sd, xing_sd, exit_sd, day_ints, confirmation_bars, max_hold_days
        )

    T = len(cum)
    max_trades = T // 2 + 1
    trades = np.zeros((max_trades, 5), dtype=np.float64)
    n = 0

    in_trade = False
    pending = False         # Crossed threshold; awaiting confirmation
    pending_side = 0        # +1 or -1 direction of pending entry
    prev_abs_d = 0.0        # |d| on the bar before current confirmation bar
    confirm_count = 0       # How many consecutive confirming bars so far

    # Cancel pending entry if spread extends this far past entry threshold.
    # Prevents chasing runaway trends.
    PENDING_CANCEL_SD: float = xing_sd + 2.5

    for i in range(T):
        d = dist_sd[i]
        if np.isnan(d):
            continue

        if not in_trade and not pending:
            if d > xing_sd:
                pending = True
                pending_side = -1
                prev_abs_d = abs(d)
                confirm_count = 0
            elif d < -xing_sd:
                pending = True
                pending_side = 1
                prev_abs_d = abs(d)
                confirm_count = 0

        elif pending:
            abs_d = abs(d)

            # Cancel if spread has extended far beyond threshold (trending move).
            if abs_d >= PENDING_CANCEL_SD:
                pending = False
                pending_side = 0
                confirm_count = 0
                continue

            # Cancel if spread crosses to the other side (very rare, noise case).
            if pending_side == -1 and d < -xing_sd:
                pending = False
                pending_side = 0
                confirm_count = 0
                continue
            if pending_side == 1 and d > xing_sd:
                pending = False
                pending_side = 0
                confirm_count = 0
                continue

            # Check whether this bar confirms reversion (|d| is shrinking).
            if abs_d < prev_abs_d:
                confirm_count += 1
            else:
                # Reversal interrupted — reset counter but stay pending.
                confirm_count = 0

            prev_abs_d = abs_d

            if confirm_count >= confirmation_bars:
                # Entry fires on this bar.
                in_trade = True
                pending = False
                entry_idx = i
                entry_cum = cum[i]
                side = pending_side
                pending_side = 0
                confirm_count = 0

        else:  # in_trade
            # Exit condition (REGISTER ITEM H — sign intentionally inverted).
            # Correct form: -side * d <= exit_sd
            # Expanded:
            #   side=-1 exits when d <= exit_sd   (spread reverted toward zero)
            #   side=+1 exits when d >= -exit_sd  (spread reverted toward zero)
            # DO NOT rewrite as `side * d <= exit_sd` — that fires at entry.
            exit_condition = (
                (side == -1 and d <= exit_sd) or
                (side == 1 and d >= -exit_sd) or
                (day_ints[i] - day_ints[entry_idx] >= max_hold_days)
            )
            if exit_condition:
                exit_cum = cum[i]
                if side == 1:
                    gross_ret = (exit_cum - entry_cum) / entry_cum
                else:
                    gross_ret = (entry_cum - exit_cum) / entry_cum
                holding = day_ints[i] - day_ints[entry_idx]

                trades[n, COL_ENTRY_IDX] = entry_idx
                trades[n, COL_EXIT_IDX] = i
                trades[n, COL_SIDE] = side
                trades[n, COL_GROSS_RETURN] = gross_ret
                trades[n, COL_HOLDING_DAYS] = holding
                n += 1
                in_trade = False

    return trades[:n], n


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST PIPELINE FOR A SINGLE SPREAD SERIES
# ═══════════════════════════════════════════════════════════════════════════

def _backtest_precomputed(
    cum: np.ndarray,
    dist_sd_arr: np.ndarray,
    day_ints: np.ndarray,
    xing_sd: float,
    exit_sd: float,
    confirmation_bars: int,
    spread_cost_pct: float,
    financing_daily_pct: float,
    n_legs: int = 2,
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
) -> dict:
    """Run detection + aggregation on pre-computed cum and dist_sd arrays.

    Called by run_pair_grid where cum/dist_sd are hoisted outside the
    grid loop (computed once per pair, reused across all exit_sd × CB
    combinations).  Also used as the shared implementation path for
    backtest_spread_with_confirmation.

    Args:
        cum: Cumulative product array, shape (T,), float64.
        dist_sd_arr: Z-score array, shape (T,), float64.
        day_ints: Integer day index, shape (T,), int64.
        xing_sd: Entry threshold in SD units.
        exit_sd: Exit threshold in SD units.
        confirmation_bars: Consecutive confirming bars required before entry.
        spread_cost_pct: One-way spread cost per leg as fraction.
        financing_daily_pct: Daily financing cost per leg as fraction.
        n_legs: Number of legs (2 for 1v1 pair).
        max_hold_days: Maximum holding period before forced exit.

    Returns:
        Dict with keys trades_raw, n_trades, summary.
    """
    trades, n_trades = _detect_trades_with_confirmation(
        cum, dist_sd_arr, xing_sd, exit_sd, day_ints, confirmation_bars, max_hold_days,
    )
    summary = aggregate_trades(trades, n_trades, spread_cost_pct, financing_daily_pct, n_legs)
    return {'trades_raw': trades, 'n_trades': n_trades, 'summary': summary}

def backtest_spread_with_confirmation(
    spread_returns: np.ndarray,
    day_ints: np.ndarray,
    vol_window: int = DEFAULT_VOL_WINDOW,
    xing_sd: float = DEFAULT_XING_SD,
    exit_sd: float = 2.0,
    confirmation_bars: int = 0,
    spread_cost_pct: float = EQUITY_SPREAD_COST_PCT,
    financing_daily_pct: float = 0.0,
    n_legs: int = 2,
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
) -> dict:
    """Run confirmation-entry backtest on a single spread return series.

    Wraps _detect_trades_with_confirmation and aggregate_trades from
    engine/backtest.py into a single dict result, matching the interface
    of engine.backtest.run_backtest.

    Args:
        spread_returns: Vol-scaled daily spread returns, shape (T,), float64.
        day_ints: Integer day index, shape (T,), int64.
        vol_window: Rolling window for z-score (default 262).
        xing_sd: Entry threshold in SD units (default 2.0).
        exit_sd: Exit threshold in SD units.  For equity scalp use 2.0.
        confirmation_bars: Consecutive confirming bars required before entry.
            0 = standard crossing signal (baseline).
        spread_cost_pct: One-way spread cost per leg as fraction.
        financing_daily_pct: Daily financing cost per leg as fraction.
            Compute as (long_rate - short_rebate) / 365 for net drag.
        n_legs: Number of legs (2 for 1v1 pair).
        max_hold_days: Maximum hold before forced exit.

    Returns:
        Dict with keys:
            trades_raw  — np.ndarray (n_trades, 5)
            n_trades    — int
            summary     — dict from aggregate_trades
    """
    if not isinstance(spread_returns, np.ndarray):
        raise TypeError(
            f"backtest_spread_with_confirmation: spread_returns must be a numpy "
            f"ndarray; got {type(spread_returns).__name__}"
        )
    if len(spread_returns) < vol_window:
        raise ValueError(
            f"backtest_spread_with_confirmation: spread_returns length "
            f"({len(spread_returns)}) must be >= vol_window ({vol_window})"
        )

    cum = np.cumprod(1.0 + spread_returns)
    roll_mean, roll_std = rolling_mean_std(cum, vol_window)

    dist_sd_arr = np.full_like(cum, np.nan)
    valid_mask = (~np.isnan(roll_std)) & (roll_std > 0)
    dist_sd_arr[valid_mask] = (cum[valid_mask] - roll_mean[valid_mask]) / roll_std[valid_mask]

    return _backtest_precomputed(
        cum=cum,
        dist_sd_arr=dist_sd_arr,
        day_ints=day_ints,
        xing_sd=xing_sd,
        exit_sd=exit_sd,
        confirmation_bars=confirmation_bars,
        spread_cost_pct=spread_cost_pct,
        financing_daily_pct=financing_daily_pct,
        n_legs=n_legs,
        max_hold_days=max_hold_days,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAIR-LEVEL GRID RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_pair_grid(
    scaled: np.ndarray,
    day_ints: np.ndarray,
    instruments: list[str],
    exit_sd_grid: list[float] = EXIT_SD_GRID,
    confirmation_bars_grid: list[int] = CONFIRMATION_BARS_GRID,
    vol_window: int = DEFAULT_VOL_WINDOW,
    xing_sd: float = DEFAULT_XING_SD,
    spread_cost_pct: float = EQUITY_SPREAD_COST_PCT,
    financing_long_annual: float = EQUITY_FINANCING_LONG_ANNUAL,
    financing_short_annual: float = EQUITY_FINANCING_SHORT_ANNUAL,
) -> pd.DataFrame:
    """Run the 2D confirmation × exit grid across all 1v1 pairs.

    Iterates all non-overlapping (long, short) 1v1 instrument pairs and
    runs backtest_spread_with_confirmation for each (CB, exit_sd) grid point.

    Net daily financing drag per leg: (long_rate - short_rebate) / 365.
    This is the symmetric cost model used throughout the Q1–Q8 research.

    Args:
        scaled: Vol-normalised return matrix, shape (T, N), float64.
        day_ints: Integer day index, shape (T,), int64.
        instruments: Instrument codes corresponding to scaled columns.
        exit_sd_grid: Exit SD values to sweep.
        confirmation_bars_grid: Confirmation bar counts to sweep.
        vol_window: Rolling window in trading days.
        xing_sd: Entry crossing threshold in SD units.
        spread_cost_pct: One-way spread cost per leg.
        financing_long_annual: Annual long-leg financing rate (e.g. 0.0488).
        financing_short_annual: Annual short-leg rebate rate (e.g. 0.0088).

    Returns:
        DataFrame with one row per (long, short, confirmation_bars, exit_sd)
        combination, with all aggregate_trades summary columns included.
    """
    if scaled.ndim != 2:
        raise ValueError(
            f"run_pair_grid: scaled must be 2-D (T, N); got shape {scaled.shape}"
        )
    if scaled.shape[1] != len(instruments):
        raise ValueError(
            f"run_pair_grid: scaled has {scaled.shape[1]} columns but "
            f"{len(instruments)} instruments supplied"
        )
    if len(instruments) < 2:
        raise ValueError(
            f"run_pair_grid: need at least 2 instruments; got {len(instruments)}"
        )

    # Net daily financing drag (both legs: one long, one short)
    financing_daily = (financing_long_annual - financing_short_annual) / 365.0

    N = len(instruments)
    all_pairs = list(combinations(range(N), 2))
    total_pairs = len(all_pairs)
    total_runs = total_pairs * len(exit_sd_grid) * len(confirmation_bars_grid)

    logger.info(
        "Phase 5 3B grid: %d pairs × %d exit_sd × %d CB = %d runs",
        total_pairs, len(exit_sd_grid), len(confirmation_bars_grid), total_runs,
    )

    records: list[dict] = []
    run_count = 0

    for long_i, short_i in all_pairs:
        long_instr = instruments[long_i]
        short_instr = instruments[short_i]
        spread_returns = (scaled[:, long_i] - scaled[:, short_i]).astype(np.float64)

        # Hoist per-pair computation outside the grid loop.
        # cum and dist_sd_arr depend only on spread_returns and vol_window,
        # both of which are fixed for this pair.  Reusing them across the
        # exit_sd × CB inner loop avoids 11 redundant rolling_mean_std calls
        # per pair (12 grid points → computed once).
        pair_cum = np.cumprod(1.0 + spread_returns)
        pair_roll_mean, pair_roll_std = rolling_mean_std(pair_cum, vol_window)
        pair_dist_sd = np.full_like(pair_cum, np.nan)
        pair_valid = (~np.isnan(pair_roll_std)) & (pair_roll_std > 0)
        pair_dist_sd[pair_valid] = (
            (pair_cum[pair_valid] - pair_roll_mean[pair_valid]) / pair_roll_std[pair_valid]
        )

        for exit_sd in exit_sd_grid:
            for cb in confirmation_bars_grid:
                result = _backtest_precomputed(
                    cum=pair_cum,
                    dist_sd_arr=pair_dist_sd,
                    day_ints=day_ints,
                    xing_sd=xing_sd,
                    exit_sd=exit_sd,
                    confirmation_bars=cb,
                    spread_cost_pct=spread_cost_pct,
                    financing_daily_pct=financing_daily,
                    n_legs=2,
                )
                s = result['summary']
                records.append({
                    'Long':              long_instr,
                    'Short':             short_instr,
                    'confirmation_bars': cb,
                    'exit_sd':           exit_sd,
                    **s,
                })

                run_count += 1
                if run_count % 500 == 0:
                    logger.info("  %d / %d runs complete", run_count, total_runs)

    logger.info("Grid complete: %d runs", run_count)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def build_grid_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-pair detail to per-grid-point summary statistics.

    Computes mean and median of key metrics across all pairs at each
    (confirmation_bars, exit_sd) grid point.  Only pairs with >= 5 trades
    are included to avoid sparse-pair distortion.

    Also flags whether each grid point meets the acceptance criterion:
        avg_net >= ACCEPT_AVG_NET_THRESHOLD (0.30%)

    Args:
        detail_df: Output of run_pair_grid (one row per pair × grid point).

    Returns:
        DataFrame with one row per (confirmation_bars, exit_sd) grid point.
    """
    if detail_df.empty:
        raise ValueError("build_grid_summary: detail_df is empty")

    # Filter sparse pairs
    filtered = detail_df[detail_df['n_trades'] >= 5].copy()

    group_cols = ['confirmation_bars', 'exit_sd']
    metric_cols = ['n_trades', 'gross_wr', 'net_wr', 'avg_gross', 'avg_net',
                   'avg_holding', 'median_holding', 'avg_total_cost']

    summary = (
        filtered
        .groupby(group_cols)[metric_cols]
        .agg(['mean', 'median'])
        .round(4)
    )
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()

    # Pair count at each grid point
    pair_counts = (
        filtered.groupby(group_cols)['n_trades']
        .count()
        .reset_index()
        .rename(columns={'n_trades': 'n_pairs'})
    )
    summary = summary.merge(pair_counts, on=group_cols)

    # Fraction of pairs meeting avg_net threshold
    pos_pairs = (
        filtered[filtered['avg_net'] >= ACCEPT_AVG_NET_THRESHOLD]
        .groupby(group_cols)['n_trades']
        .count()
        .reset_index()
        .rename(columns={'n_trades': 'n_pairs_above_threshold'})
    )
    summary = summary.merge(pos_pairs, on=group_cols, how='left')
    summary['n_pairs_above_threshold'] = summary['n_pairs_above_threshold'].fillna(0).astype(int)
    summary['pct_pairs_above_threshold'] = (
        summary['n_pairs_above_threshold'] / summary['n_pairs'] * 100
    ).round(1)

    summary = summary.sort_values(['exit_sd', 'confirmation_bars']).reset_index(drop=True)
    return summary


def build_baseline_vs_best(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side comparison of CB=0 (baseline) vs best CB per exit_sd.

    For each exit_sd, identifies the confirmation_bars value with the
    highest mean avg_net across all pairs and presents it alongside the
    CB=0 baseline row.

    Args:
        detail_df: Output of run_pair_grid.

    Returns:
        DataFrame with columns indicating baseline and best-CB values for
        each exit_sd, for quick inspection.
    """
    if detail_df.empty:
        raise ValueError("build_baseline_vs_best: detail_df is empty")

    filtered = detail_df[detail_df['n_trades'] >= 5].copy()
    pivot_cols = ['n_trades', 'gross_wr', 'net_wr', 'avg_net', 'avg_holding']

    rows: list[dict] = []
    for exit_sd in sorted(filtered['exit_sd'].unique()):
        sub = filtered[filtered['exit_sd'] == exit_sd]
        grid_means = (
            sub.groupby('confirmation_bars')[pivot_cols]
            .mean()
            .reset_index()
        )

        baseline_row = grid_means[grid_means['confirmation_bars'] == 0].iloc[0]
        best_idx = grid_means['avg_net'].idxmax()
        best_row = grid_means.loc[best_idx]

        row: dict = {'exit_sd': exit_sd}
        for col in pivot_cols:
            row[f'baseline_{col}'] = round(float(baseline_row[col]), 4)
        row['best_cb'] = int(best_row['confirmation_bars'])
        for col in pivot_cols:
            row[f'best_{col}'] = round(float(best_row[col]), 4)
        for col in ['avg_net', 'net_wr', 'gross_wr']:
            delta = row[f'best_{col}'] - row[f'baseline_{col}']
            row[f'delta_{col}'] = round(delta, 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the Phase 5 3B confirmation entry experiment end to end.

    Loads equity prices, runs the full 2D grid across all 1v1 pairs,
    writes three output CSV files, and logs a summary of findings.

    Raises:
        FileNotFoundError: If the equity price CSV is not found.
        RuntimeError: If no pairs produce usable results.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s — %(message)s',
        datefmt='%H:%M:%S',
    )

    # ── Output directory ────────────────────────────────────────────────────
    output_dir = _ROOT / 'research' / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load equity prices ──────────────────────────────────────────────────
    equity_csv = _ROOT / 'cache' / 'prices.csv'
    logger.info("Loading equity prices from: %s", equity_csv)

    prices, instruments = load_asset_prices(str(equity_csv))
    logger.info("Loaded %d instruments, %d trading days", len(instruments), len(prices))

    # ── Prepare vol-scaled returns ──────────────────────────────────────────
    scaled, day_ints, index = prepare_returns(
        prices, instruments, vol_window=DEFAULT_VOL_WINDOW
    )
    logger.info(
        "Returns matrix: shape %s, date range %s → %s",
        scaled.shape,
        index[0].date(),
        index[-1].date(),
    )

    # ── Run grid ────────────────────────────────────────────────────────────
    logger.info(
        "Running Phase 5 3B grid: CB=%s × exit_sd=%s",
        CONFIRMATION_BARS_GRID, EXIT_SD_GRID,
    )
    detail_df = run_pair_grid(
        scaled=scaled,
        day_ints=day_ints,
        instruments=instruments,
        exit_sd_grid=EXIT_SD_GRID,
        confirmation_bars_grid=CONFIRMATION_BARS_GRID,
    )

    if detail_df.empty:
        raise RuntimeError(
            "Phase 5 3B: no pairs produced usable results. "
            "Check price data and equity CSV path."
        )

    # ── Build summaries ─────────────────────────────────────────────────────
    grid_summary = build_grid_summary(detail_df)
    baseline_vs_best = build_baseline_vs_best(detail_df)

    # ── Write outputs ────────────────────────────────────────────────────────
    detail_path = output_dir / 'phase5_3b_pair_detail.csv'
    summary_path = output_dir / 'phase5_3b_grid_summary.csv'
    compare_path = output_dir / 'phase5_3b_baseline_vs_best.csv'

    detail_df.to_csv(detail_path, index=False)
    grid_summary.to_csv(summary_path, index=False)
    baseline_vs_best.to_csv(compare_path, index=False)

    logger.info("Outputs written:")
    logger.info("  %s", detail_path)
    logger.info("  %s", summary_path)
    logger.info("  %s", compare_path)

    # ── Console summary ──────────────────────────────────────────────────────
    logger.info("\n=== GRID SUMMARY (mean avg_net across pairs with ≥5 trades) ===")
    pivot = grid_summary.pivot_table(
        values='avg_net_mean',
        index='confirmation_bars',
        columns='exit_sd',
    )
    logger.info("\n%s", pivot.to_string())

    logger.info("\n=== BASELINE (CB=0) vs BEST CB PER EXIT_SD ===")
    for _, row in baseline_vs_best.iterrows():
        logger.info(
            "exit_sd=%.1f | baseline avg_net=%.4f | best CB=%d avg_net=%.4f "
            "| Δavg_net=%+.4f | Δnet_wr=%+.4f",
            row['exit_sd'],
            row['baseline_avg_net'],
            row['best_cb'],
            row['best_avg_net'],
            row['delta_avg_net'],
            row['delta_net_wr'],
        )

    # ── Accept/reject decision ───────────────────────────────────────────────
    logger.info("\n=== ACCEPT/REJECT (threshold: avg_net >= %.2f%%) ===",
                ACCEPT_AVG_NET_THRESHOLD * 100)
    for _, row in baseline_vs_best.iterrows():
        if (row['best_cb'] > 0
                and row['best_avg_net'] >= ACCEPT_AVG_NET_THRESHOLD
                and row['delta_avg_net'] > 0
                and row['delta_net_wr'] > 0):
            logger.info(
                "exit_sd=%.1f → ACCEPT CB=%d: avg_net %.4f >= threshold "
                "and net_wr improves vs baseline",
                row['exit_sd'], row['best_cb'], row['best_avg_net'],
            )
        else:
            logger.info(
                "exit_sd=%.1f → REJECT / NO IMPROVEMENT: "
                "best CB=%d avg_net %.4f (baseline %.4f)",
                row['exit_sd'], row['best_cb'],
                row['best_avg_net'], row['baseline_avg_net'],
            )


if __name__ == '__main__':
    main()
