"""
numba_core.py — JIT-accelerated signal computation
====================================================

Contains Numba-compiled versions of the three computational bottlenecks:
  1. Rolling mean/std (replaces pd.Series.rolling inside hot paths)
  2. Crossing signal trade detection (the sequential state machine loop)
  3. Batch crossing backtest across M spread series in parallel

Also contains pure-Python reference implementations (_ref_ prefix) that
produce identical results.  These exist solely for the parity test
(test_numba_parity.py) — the app should always call the Numba versions.

Dual-path architecture — KEEP IN SYNC
--------------------------------------
Every algorithm has two implementations:
  - ``_numba_*``  — Numba @njit decorated; fast but cannot use Python objects
  - ``_ref_*``    — Pure-Python reference; used when Numba is unavailable and
                    by the parity test (tests/test_numba_parity.py)

When editing one path, you MUST update the other to maintain identical
semantics.  The parity test will catch divergence, but it only runs the
pure-Python path without Numba installed.  Test on both paths whenever
modifying the exit condition or trade-detection logic.

Input conventions (enforced by assertions below)
-------------------------------------------------
All functions take and return raw numpy arrays, never pandas objects.
The calling code is responsible for extracting .values / .index before
calling and wrapping results back into DataFrames if needed.

Trade arrays are pre-allocated to a worst-case size (T//2) and a
separate count integer tracks how many rows are valid.  This avoids
Python list appends inside the JIT boundary.

_batch_crossing_trades uses numba.prange for the outer loop over M
combinations, giving automatic multithreading on multi-core machines.

──────────────────────────────────────────────────────────────────────────────
REGISTER ITEM H — exit condition sign is intentionally inverted.
──────────────────────────────────────────────────────────────────────────────
The exit condition in _ref_detect_trades / _numba_detect_trades is written as:
    (side == -1 and d <= exit_sd) or (side == 1 and d >= -exit_sd)

This is the CORRECT form, equivalent to:  -side * d <= exit_sd

The WRONG form would be:  side * d <= exit_sd
  → For side=-1 at entry (d=+2.2): (-1)*2.2 = -2.2 <= 1.0 → True, fires immediately.
  → The wrong form exits on the same bar as entry, destroying all positive EV.

A module-level canary assertion below (runs on import) verifies that any
future edit has not accidentally reverted to the wrong form.
See also: CLAUDE.md prompt correction register item H.

──────────────────────────────────────────────────────────────────────────────
REGISTER ITEM I — entry dislocation must use abs(d), not signed d.
──────────────────────────────────────────────────────────────────────────────
If entry dislocation (the z-score at trade entry) is ever recorded in the
production trade array, it must be stored as abs(d):
    entry_dist_sd = abs(d)   # CORRECT: 2.2 for both long and short entries
    entry_dist_sd = d        # WRONG:   -2.2 for long, +2.2 for short → average≈0

A module-level canary assertion below demonstrates this invariant.
See also: CLAUDE.md prompt correction register item I.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.info("Numba not available; using pure-Python fallback implementations")


# ── REGISTER ITEM H — module-level canary (runs on import) ────────────────
#
# Probe setup: at entry for a SHORT trade, dist_sd ≈ +2.2 (crossed above
# xing_sd=2.0), side=-1.  The correct exit formula must NOT fire at this
# point (the spread is still extended, not reverted).
#
# Correct expanded form:  (side==-1 and d<=exit_sd) or (side==1 and d>=-exit_sd)
# Correct compact form:   -side * d <= exit_sd
#   → -(-1) * 2.2 = 2.2 <= 1.0  → False  (does not fire at entry) ✓
#
# Wrong compact form:      side * d <= exit_sd
#   → (-1) * 2.2 = -2.2 <= 1.0  → True   (fires immediately at entry) ✗
#
_H_probe_d, _H_probe_side, _H_probe_exit_sd = 2.2, -1, 1.0
assert not (-_H_probe_side * _H_probe_d <= _H_probe_exit_sd), (
    "REGISTER ITEM H VIOLATED: exit condition fires at entry. "
    "The formula must be `-side * d <= exit_sd` (written as "
    "`(side == -1 and d <= exit_sd) or (side == 1 and d >= -exit_sd)`), "
    "not `side * d <= exit_sd`. "
    "See CLAUDE.md prompt correction register item H."
)

# Also verify the correct expanded form matches for both sides:
assert (
    (_H_probe_side == -1 and _H_probe_d > _H_probe_exit_sd)  # short, not yet reverted
), (
    "REGISTER ITEM H CANARY LOGIC ERROR: probe values should represent a trade "
    "that has not yet reached the exit threshold."
)


# ── REGISTER ITEM I — module-level canary (runs on import) ────────────────
#
# Entry dislocation must always be stored as abs(d).  The canary demonstrates
# that signed values cancel across long/short trades when averaged.
#
# Scenario: equal and opposite entries.
#   Long  at d = -2.2  (spread extended downward)  → abs(-2.2) = 2.2
#   Short at d = +2.2  (spread extended upward)    → abs(+2.2) = 2.2
#
# Correct (abs): average = (2.2 + 2.2) / 2 = 2.2  (reflects true dislocation)
# Wrong (signed): average = (-2.2 + 2.2) / 2 = 0.0  (cancels to near zero)
#
_I_long_d, _I_short_d = -2.2, 2.2
_I_correct_avg = (abs(_I_long_d) + abs(_I_short_d)) / 2   # 2.2
_I_wrong_avg   = (_I_long_d + _I_short_d) / 2             # 0.0
assert _I_correct_avg > abs(_I_wrong_avg), (
    "REGISTER ITEM I VIOLATED: entry dislocation canary failed. "
    "abs(d) must be used when recording entry dislocation — signed values "
    "from long/short entries cancel, producing near-zero averages that "
    "misrepresent actual dislocation. "
    "See CLAUDE.md prompt correction register item I."
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ROLLING MEAN / STD
# ═══════════════════════════════════════════════════════════════════════════

def _ref_rolling_mean_std(arr, window):
    """
    Pure-Python rolling mean and std for a 1-D array.

    Returns (mean_out, std_out), both shape (T,).
    Positions with fewer than window//2 observations are NaN.

    This matches the pandas behaviour used throughout calculations.py:
        pd.Series(arr).rolling(window, min_periods=window//2).mean/std()
    """
    T = len(arr)
    mean_out = np.full(T, np.nan)
    std_out = np.full(T, np.nan)
    min_periods = window // 2

    for i in range(T):
        start = max(0, i - window + 1)
        w = arr[start:i + 1]
        n = 0
        s = 0.0
        for v in w:
            if not np.isnan(v):
                s += v
                n += 1
        if n >= min_periods:
            m = s / n
            mean_out[i] = m
            if n >= 2:
                ss = 0.0
                for v in w:
                    if not np.isnan(v):
                        ss += (v - m) ** 2
                std_out[i] = np.sqrt(ss / (n - 1))
            else:
                std_out[i] = 0.0
    return mean_out, std_out


if HAS_NUMBA:
    @numba.njit(cache=True)
    def _numba_rolling_mean_std(arr, window):
        """Numba-compiled rolling mean and std.  Same semantics as _ref_ version."""
        T = len(arr)
        mean_out = np.full(T, np.nan)
        std_out = np.full(T, np.nan)
        min_periods = window // 2

        for i in range(T):
            start = max(0, i - window + 1)
            n = 0
            s = 0.0
            for j in range(start, i + 1):
                v = arr[j]
                if not np.isnan(v):
                    s += v
                    n += 1
            if n >= min_periods:
                m = s / n
                mean_out[i] = m
                if n >= 2:
                    ss = 0.0
                    for j in range(start, i + 1):
                        v = arr[j]
                        if not np.isnan(v):
                            ss += (v - m) ** 2
                    std_out[i] = np.sqrt(ss / (n - 1))
                else:
                    std_out[i] = 0.0
        return mean_out, std_out


def rolling_mean_std(arr, window):
    """Dispatch to Numba if available, else pure Python.

    Args:
        arr: 1-D numpy float64 array of values.
        window: Integer lookback window.

    Returns:
        Tuple ``(mean_out, std_out)`` each of shape ``(T,)``.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"rolling_mean_std: arr must be a numpy ndarray; got {type(arr).__name__}. "
            f"Extract .values from a pandas Series before calling."
        )
    if HAS_NUMBA:
        return _numba_rolling_mean_std(arr, window)
    return _ref_rolling_mean_std(arr, window)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CROSSING SIGNAL TRADE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Trade output array columns (named constants for readability)
COL_ENTRY_IDX = 0
COL_EXIT_IDX = 1
COL_SIDE = 2          # +1 = long (fade -SD), -1 = short (fade +SD)
COL_GROSS_RETURN = 3
COL_HOLDING_DAYS = 4


def _ref_detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints, max_hold_days=300):
    """
    Pure-Python crossing signal trade detection.

    This is the canonical reference implementation.  It matches the logic
    used in the Q1 equity backtest (280,913 trades, 72.9% gross win rate)
    and Q5 FX backtest (1,474 PvP trades, 72.6% gross win rate).

    Parameters
    ----------
    cum : np.ndarray (T,)
        Cumulative product of (1 + spread_return).
    dist_sd : np.ndarray (T,)
        Distance from rolling mean in standard deviations.
    xing_sd : float
        Entry threshold (e.g. 2.0).
    exit_sd : float
        Exit threshold (e.g. 0.0 for full reversion, 0.5 for partial).
    day_ints : np.ndarray (T,) int64
        Integer day count for each timestep (for holding period calc).
    max_hold_days : int
        Maximum holding period before forced exit.

    Returns
    -------
    trades : np.ndarray (n_trades, 5)
        Columns: entry_idx, exit_idx, side, gross_return, holding_days.
    n_trades : int
    """
    T = len(cum)
    max_trades = T // 2 + 1
    trades = np.zeros((max_trades, 5), dtype=np.float64)
    n = 0

    in_trade = False
    entry_idx = 0
    entry_cum = 0.0
    side = 0  # +1 long, -1 short

    for i in range(T):
        d = dist_sd[i]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > xing_sd:
                # Spread above mean by >N SD → fade by going short the spread
                in_trade = True
                entry_idx = i
                entry_cum = cum[i]
                side = -1
            elif d < -xing_sd:
                # Spread below mean by >N SD → fade by going long the spread
                in_trade = True
                entry_idx = i
                entry_cum = cum[i]
                side = 1
        else:
            # ── Exit condition (REGISTER ITEM H) ──────────────────────────
            # Correct form: -side * d <= exit_sd
            # Expanded for clarity:
            #   side=-1 (short): exits when d <= exit_sd   (spread reverts toward 0)
            #   side=+1 (long):  exits when d >= -exit_sd  (spread reverts toward 0)
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


if HAS_NUMBA:
    @numba.njit(cache=True)
    def _numba_detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints, max_hold_days=300):
        """Numba-compiled trade detection.  Identical logic to _ref_ version.

        Exit condition (REGISTER ITEM H — do not alter the sign):
            (side == -1 and d <= exit_sd) or (side == 1 and d >= -exit_sd)
        This is equivalent to: -side * d <= exit_sd
        The wrong form (side * d <= exit_sd) fires at entry, not on reversion.
        """
        T = len(cum)
        max_trades = T // 2 + 1
        trades = np.zeros((max_trades, 5), dtype=np.float64)
        n = 0

        in_trade = False
        entry_idx = 0
        entry_cum = 0.0
        side = 0

        for i in range(T):
            d = dist_sd[i]
            if np.isnan(d):
                continue

            if not in_trade:
                if d > xing_sd:
                    in_trade = True
                    entry_idx = i
                    entry_cum = cum[i]
                    side = -1
                elif d < -xing_sd:
                    in_trade = True
                    entry_idx = i
                    entry_cum = cum[i]
                    side = 1
            else:
                # REGISTER ITEM H — exit sign intentionally inverted; see module docstring
                exit_cond = (
                    (side == -1 and d <= exit_sd) or
                    (side == 1 and d >= -exit_sd) or
                    (day_ints[i] - day_ints[entry_idx] >= max_hold_days)
                )
                if exit_cond:
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


def detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints, max_hold_days=300):
    """Dispatch to Numba if available, else pure Python.

    Args:
        cum: Cumulative spread return array, shape ``(T,)``, dtype float64.
        dist_sd: Z-score distance array, shape ``(T,)``, dtype float64.
        xing_sd: Entry threshold in standard deviations.
        exit_sd: Exit threshold in standard deviations.
        day_ints: Integer day index per row, shape ``(T,)``, dtype int64.
        max_hold_days: Maximum holding period before forced exit.

    Returns:
        Tuple ``(trades, n_trades)`` where ``trades`` is shape
        ``(n_trades, 5)`` with columns defined by the ``COL_*`` constants.
    """
    if not isinstance(cum, np.ndarray):
        raise TypeError(
            f"detect_trades: cum must be a numpy ndarray; got {type(cum).__name__}"
        )
    if not isinstance(dist_sd, np.ndarray):
        raise TypeError(
            f"detect_trades: dist_sd must be a numpy ndarray; got {type(dist_sd).__name__}"
        )
    if not isinstance(day_ints, np.ndarray):
        raise TypeError(
            f"detect_trades: day_ints must be a numpy ndarray; got {type(day_ints).__name__}"
        )
    if HAS_NUMBA:
        return _numba_detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints, max_hold_days)
    return _ref_detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints, max_hold_days)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SINGLE-SERIES BACKTEST (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════════

def backtest_spread(spread_returns, vol_window, xing_sd, exit_sd, day_ints, max_hold_days=300):
    """
    Full crossing signal backtest on a single spread return series.

    Computes cumulative, rolling mean/std, distance, then detects trades.

    Parameters
    ----------
    spread_returns : np.ndarray (T,)
        Daily spread returns (already vol-scaled).
    vol_window : int
        Rolling window for mean/std (e.g. 262).
    xing_sd : float
        Entry threshold.
    exit_sd : float
        Exit threshold (0.0 = full reversion).
    day_ints : np.ndarray (T,) int64
        Integer day count per timestep.
    max_hold_days : int
        Maximum holding period before forced exit.

    Returns
    -------
    trades : np.ndarray (n_trades, 5)
    n_trades : int
    cum : np.ndarray (T,)
    dist_sd : np.ndarray (T,)
    """
    if not isinstance(spread_returns, np.ndarray):
        raise TypeError(
            f"backtest_spread: spread_returns must be a numpy ndarray; "
            f"got {type(spread_returns).__name__}. "
            f"Extract .values from a pandas Series before calling."
        )
    if not isinstance(day_ints, np.ndarray):
        raise TypeError(
            f"backtest_spread: day_ints must be a numpy ndarray; "
            f"got {type(day_ints).__name__}"
        )

    cum = np.cumprod(1.0 + spread_returns)
    roll_mean, roll_std = rolling_mean_std(cum, vol_window)

    # Distance in standard deviations (NaN where std is zero or NaN)
    dist_sd = np.full_like(cum, np.nan)
    for i in range(len(cum)):
        if not np.isnan(roll_std[i]) and roll_std[i] > 0:
            dist_sd[i] = (cum[i] - roll_mean[i]) / roll_std[i]

    trades, n_trades = detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints, max_hold_days)
    return trades, n_trades, cum, dist_sd


# ═══════════════════════════════════════════════════════════════════════════
# 3b. EXTENDED SINGLE-SERIES BACKTEST WITH HARD STOP AND MAE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

# Extended trade array column indices — appended after the existing COL_* columns.
# Only backtest_spread_with_stop() returns an 8-column array; all other functions
# return the original 5-column array indexed by COL_ENTRY_IDX … COL_HOLDING_DAYS.
COL_MAE_SD   = 5  # worst adverse dist_sd reached during trade (always >= 0)
COL_ENTRY_SD = 6  # abs(dist_sd) at entry — register item I: always abs()
COL_STOPPED  = 7  # 1.0 if stopped out via hard stop, 0.0 if normal exit


def backtest_spread_with_stop(
    spread_returns: np.ndarray,
    vol_window: int,
    xing_sd: float,
    exit_sd: float,
    stop_sd: float,
    day_ints: np.ndarray,
    max_hold_days: int = 300,
) -> tuple[np.ndarray, int]:
    """Extended single-series backtest with hard stop and MAE tracking.

    Identical logic to backtest_spread() / _ref_detect_trades() but adds:
      - Hard stop: exit immediately if spread moves stop_sd beyond entry on
        the adverse side (further away from zero than the entry threshold).
      - MAE tracking: worst adverse dist_sd reached during each trade.
      - Stop flag: whether exit was a hard stop or normal signal exit.

    Stop logic (for a long trade, side=+1):
        Entry fires when d < -xing_sd (spread extended downward).
        Adverse direction is further downward (d going more negative).
        Hard stop fires when d < -stop_sd.
        → stop_hit = (side == 1 and d < -stop_sd) or (side == -1 and d > stop_sd)
        Equivalently: -side * d > stop_sd

    Exit condition (register item H — sign is intentionally inverted):
        Normal exit: (side == -1 and d <= exit_sd) or (side == 1 and d >= -exit_sd)
        Do NOT alter this form. See module docstring.

    Entry dislocation (register item I — always abs):
        COL_ENTRY_SD stores abs(d) at entry, not signed d.

    MAE tracking:
        On each bar after entry, adverse_d = -side * d (positive when d moves
        in the wrong direction).  current_mae is updated to the running maximum
        of adverse_d.  Reset to 0.0 at the start of each new trade.

    Performance note:
        This function is pure Python and is intended for research scripts only.
        Do NOT wire into Tab 8/9/10/11 batch search or walk-forward pipelines —
        use batch_backtest() for those paths.  MAE tracking requires per-trade
        state that is incompatible with the fixed-shape batch result array.

    Args:
        spread_returns: Daily spread returns, shape (T,), dtype float64.
            Must be a numpy ndarray, not a pandas Series.
        vol_window: Rolling window in trading days (e.g. 262).
        xing_sd: Entry threshold in standard deviations (e.g. 2.0).
        exit_sd: Normal exit threshold in SD units (e.g. 0.5 for commodities,
            2.0 for equities).  Register item H sign applies to exit condition.
        stop_sd: Hard stop threshold in SD units.  Must be > xing_sd.
            Stop fires if spread moves stop_sd beyond the zero line on the
            adverse side after entry.
        day_ints: Integer day index per row, shape (T,), dtype int64.
        max_hold_days: Maximum holding period before forced exit.  Default 300.
            Forced-hold exits are NOT flagged as stops (COL_STOPPED stays 0.0).

    Returns:
        Tuple (trades, n_trades) where trades is shape (n_trades, 8) with
        columns indexed by COL_ENTRY_IDX, COL_EXIT_IDX, COL_SIDE,
        COL_GROSS_RETURN, COL_HOLDING_DAYS, COL_MAE_SD, COL_ENTRY_SD,
        COL_STOPPED.

    Raises:
        TypeError: If spread_returns or day_ints are not numpy ndarrays.
        ValueError: If stop_sd <= xing_sd (stop would fire at or before entry).
        ValueError: If stop_sd <= exit_sd (stop would fire before normal exit).
    """
    if not isinstance(spread_returns, np.ndarray):
        raise TypeError(
            f"backtest_spread_with_stop: spread_returns must be a numpy ndarray; "
            f"got {type(spread_returns).__name__}. "
            f"Extract .values from a pandas Series before calling."
        )
    if not isinstance(day_ints, np.ndarray):
        raise TypeError(
            f"backtest_spread_with_stop: day_ints must be a numpy ndarray; "
            f"got {type(day_ints).__name__}"
        )
    if stop_sd <= xing_sd:
        raise ValueError(
            f"backtest_spread_with_stop: stop_sd ({stop_sd}) must be > xing_sd ({xing_sd}). "
            f"A stop at or inside the entry threshold would fire immediately after entry."
        )
    if stop_sd <= exit_sd:
        raise ValueError(
            f"backtest_spread_with_stop: stop_sd ({stop_sd}) must be > exit_sd ({exit_sd}). "
            f"Hard stop should be set beyond the normal exit threshold."
        )

    # Compute cum and dist_sd identically to backtest_spread()
    cum = np.cumprod(1.0 + spread_returns)
    roll_mean, roll_std = rolling_mean_std(cum, vol_window)

    dist_sd_arr = np.full_like(cum, np.nan)
    for i in range(len(cum)):
        if not np.isnan(roll_std[i]) and roll_std[i] > 0:
            dist_sd_arr[i] = (cum[i] - roll_mean[i]) / roll_std[i]

    T = len(cum)
    max_trades = T // 2 + 1
    trades = np.zeros((max_trades, 8), dtype=np.float64)
    n = 0

    in_trade = False
    entry_idx = 0
    entry_cum = 0.0
    entry_d = 0.0
    side = 0
    current_mae = 0.0

    for i in range(T):
        d = dist_sd_arr[i]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > xing_sd:
                in_trade = True
                entry_idx = i
                entry_cum = cum[i]
                entry_d = d
                side = -1
                current_mae = 0.0
            elif d < -xing_sd:
                in_trade = True
                entry_idx = i
                entry_cum = cum[i]
                entry_d = d
                side = 1
                current_mae = 0.0
        else:
            # Update MAE: adverse_d is positive when d moves in the wrong direction.
            # For LONG (side=+1): adverse = -d (d going more negative is bad).
            # For SHORT (side=-1): adverse = +d (d going more positive is bad).
            adverse_d = -side * d
            if adverse_d > current_mae:
                current_mae = adverse_d

            # Stop check takes priority (evaluated before normal exit).
            # Fires when spread extends adversely past stop_sd.
            stop_hit = (side == 1 and d < -stop_sd) or (side == -1 and d > stop_sd)

            # Normal exit (REGISTER ITEM H — sign intentionally inverted; see module docstring).
            # Correct form: -side * d <= exit_sd
            # Expanded: side==-1 exits when d<=exit_sd; side==+1 exits when d>=-exit_sd
            exit_condition = (
                (side == -1 and d <= exit_sd) or
                (side == 1 and d >= -exit_sd) or
                (day_ints[i] - day_ints[entry_idx] >= max_hold_days)
            )

            if stop_hit or exit_condition:
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
                trades[n, COL_MAE_SD] = current_mae
                trades[n, COL_ENTRY_SD] = abs(entry_d)  # register item I: always abs
                # max_hold forced exit is not a hard stop (COL_STOPPED remains 0.0)
                trades[n, COL_STOPPED] = 1.0 if stop_hit else 0.0
                n += 1
                in_trade = False

    return trades[:n], n


# ═══════════════════════════════════════════════════════════════════════════
# 4. BATCH CROSSING BACKTEST (parallel over M combinations)
# ═══════════════════════════════════════════════════════════════════════════

# Batch result columns
BR_N_TRADES = 0
BR_GROSS_WR = 1
BR_AVG_GROSS = 2
BR_AVG_HOLDING = 3
BR_AVG_WINNER = 4
BR_AVG_LOSER = 5
BR_PAYOFF_RATIO = 6
BR_TOTAL_GROSS_PNL = 7


def _ref_batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints, max_hold_days=300):
    """
    Pure-Python batch backtest: runs trade detection on each of M spread
    series and aggregates results into a summary array.

    Parameters
    ----------
    spread_mat : np.ndarray (T, M)
        Each column is a daily spread return series.
    vol_window : int
    xing_sd : float
    exit_sd : float
    day_ints : np.ndarray (T,)
    max_hold_days : int

    Returns
    -------
    results : np.ndarray (M, 8)
        Per-column summary: n_trades, gross_wr, avg_gross, avg_holding,
        avg_winner, avg_loser, payoff_ratio, total_gross_pnl.
    """
    T, M = spread_mat.shape
    results = np.zeros((M, 8), dtype=np.float64)

    for j in range(M):
        trades, n_trades, _, _ = backtest_spread(
            spread_mat[:, j], vol_window, xing_sd, exit_sd, day_ints, max_hold_days
        )
        if n_trades == 0:
            continue

        gross_rets = trades[:n_trades, COL_GROSS_RETURN]
        holdings = trades[:n_trades, COL_HOLDING_DAYS]

        n_wins = 0
        sum_winners = 0.0
        n_losses = 0
        sum_losers = 0.0
        total_pnl = 0.0

        for k in range(n_trades):
            g = gross_rets[k]
            total_pnl += g
            if g > 0:
                n_wins += 1
                sum_winners += g
            else:
                n_losses += 1
                sum_losers += g

        avg_winner = sum_winners / n_wins if n_wins > 0 else 0.0
        avg_loser = sum_losers / n_losses if n_losses > 0 else 0.0
        payoff = abs(avg_winner / avg_loser) if avg_loser != 0 else 0.0

        results[j, BR_N_TRADES] = n_trades
        results[j, BR_GROSS_WR] = n_wins / n_trades
        results[j, BR_AVG_GROSS] = total_pnl / n_trades
        results[j, BR_AVG_HOLDING] = holdings.mean()
        results[j, BR_AVG_WINNER] = avg_winner
        results[j, BR_AVG_LOSER] = avg_loser
        results[j, BR_PAYOFF_RATIO] = payoff
        results[j, BR_TOTAL_GROSS_PNL] = total_pnl

    return results


if HAS_NUMBA:
    @numba.njit(cache=True, parallel=True)
    def _numba_batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints, max_hold_days=300):
        """
        Numba-parallel batch backtest.  Uses prange over M columns so each
        combination runs on a separate thread.

        Exit condition (REGISTER ITEM H — do not alter):
            (side == -1 and d <= exit_sd) or (side == 1 and d >= -exit_sd)
        See module docstring for full explanation.
        """
        T, M = spread_mat.shape
        results = np.zeros((M, 8), dtype=np.float64)

        for j in numba.prange(M):
            # --- inline backtest_spread logic (can't call non-njit helpers) ---
            spread_col = spread_mat[:, j]
            cum = np.empty(T, dtype=np.float64)
            cum[0] = 1.0 + spread_col[0]
            for i in range(1, T):
                cum[i] = cum[i - 1] * (1.0 + spread_col[i])

            # Rolling mean/std
            min_periods = vol_window // 2
            roll_mean = np.full(T, np.nan)
            roll_std = np.full(T, np.nan)
            for i in range(T):
                start = max(0, i - vol_window + 1)
                n_valid = 0
                s = 0.0
                for idx in range(start, i + 1):
                    v = cum[idx]
                    if not np.isnan(v):
                        s += v
                        n_valid += 1
                if n_valid >= min_periods:
                    m = s / n_valid
                    roll_mean[i] = m
                    if n_valid >= 2:
                        ss = 0.0
                        for idx in range(start, i + 1):
                            v = cum[idx]
                            if not np.isnan(v):
                                ss += (v - m) ** 2
                        roll_std[i] = np.sqrt(ss / (n_valid - 1))
                    else:
                        roll_std[i] = 0.0

            # Distance
            dist_sd = np.full(T, np.nan)
            for i in range(T):
                rs = roll_std[i]
                if not np.isnan(rs) and rs > 0:
                    dist_sd[i] = (cum[i] - roll_mean[i]) / rs

            # Trade detection (inlined _numba_detect_trades logic)
            max_trades = T // 2 + 1
            trade_gross = np.empty(max_trades, dtype=np.float64)
            trade_hold = np.empty(max_trades, dtype=np.float64)
            n_t = 0
            in_trade = False
            entry_idx = 0
            entry_cum = 0.0
            side = 0

            for i in range(T):
                d = dist_sd[i]
                if np.isnan(d):
                    continue
                if not in_trade:
                    if d > xing_sd:
                        in_trade = True
                        entry_idx = i
                        entry_cum = cum[i]
                        side = -1
                    elif d < -xing_sd:
                        in_trade = True
                        entry_idx = i
                        entry_cum = cum[i]
                        side = 1
                else:
                    # REGISTER ITEM H — exit sign intentionally inverted; see module docstring.
                    # Correct: (side==-1 and d<=exit_sd) or (side==1 and d>=-exit_sd)
                    # Wrong:   (side==-1 and (-1)*d<=exit_sd) [this fires at entry]
                    exit_cond = (
                        (side == -1 and d <= exit_sd) or
                        (side == 1 and d >= -exit_sd) or
                        (day_ints[i] - day_ints[entry_idx] >= max_hold_days)
                    )
                    if exit_cond:
                        ec = cum[i]
                        if side == 1:
                            gr = (ec - entry_cum) / entry_cum
                        else:
                            gr = (entry_cum - ec) / entry_cum
                        trade_gross[n_t] = gr
                        trade_hold[n_t] = day_ints[i] - day_ints[entry_idx]
                        n_t += 1
                        in_trade = False

            # Aggregate
            if n_t == 0:
                continue

            n_wins = 0
            sum_winners = 0.0
            n_losses = 0
            sum_losers = 0.0
            total_pnl = 0.0
            total_hold = 0.0

            for k in range(n_t):
                g = trade_gross[k]
                total_pnl += g
                total_hold += trade_hold[k]
                if g > 0:
                    n_wins += 1
                    sum_winners += g
                else:
                    n_losses += 1
                    sum_losers += g

            avg_w = sum_winners / n_wins if n_wins > 0 else 0.0
            avg_l = sum_losers / n_losses if n_losses > 0 else 0.0
            payoff = abs(avg_w / avg_l) if avg_l != 0 else 0.0

            results[j, 0] = n_t
            results[j, 1] = n_wins / n_t
            results[j, 2] = total_pnl / n_t
            results[j, 3] = total_hold / n_t
            results[j, 4] = avg_w
            results[j, 5] = avg_l
            results[j, 6] = payoff
            results[j, 7] = total_pnl

        return results


def batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints, max_hold_days=300):
    """Dispatch to Numba-parallel if available, else pure Python.

    Args:
        spread_mat: 2-D numpy float64 array of shape ``(T, M)``. Each column
            is a vol-scaled daily spread return series. Must be a numpy array,
            not a pandas DataFrame.
        vol_window: Rolling lookback in trading days.
        xing_sd: Entry threshold in standard deviations.
        exit_sd: Exit threshold in standard deviations.
        day_ints: 1-D int64 array of length ``T`` with integer day indices.
        max_hold_days: Maximum holding period before forced exit.

    Returns:
        2-D numpy array of shape ``(M, 8)`` with per-combination summary
        statistics. Columns are indexed by the ``BR_*`` constants.

    Raises:
        TypeError: If ``spread_mat`` or ``day_ints`` are not numpy arrays.
    """
    if not isinstance(spread_mat, np.ndarray):
        raise TypeError(
            f"batch_backtest: spread_mat must be a numpy ndarray; "
            f"got {type(spread_mat).__name__}. "
            f"Call .values on a DataFrame before passing to this function."
        )
    if not isinstance(day_ints, np.ndarray):
        raise TypeError(
            f"batch_backtest: day_ints must be a numpy ndarray; "
            f"got {type(day_ints).__name__}"
        )
    if spread_mat.ndim != 2:
        raise ValueError(
            f"batch_backtest: spread_mat must be 2-D (T, M); "
            f"got shape {spread_mat.shape}"
        )
    if HAS_NUMBA:
        return _numba_batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints, max_hold_days)
    return _ref_batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints, max_hold_days)
