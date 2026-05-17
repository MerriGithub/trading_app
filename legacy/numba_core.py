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

Design notes
------------
- All functions take and return raw numpy arrays, never pandas objects.
  The calling code is responsible for extracting .values / .index before
  calling and wrapping results back into DataFrames if needed.
- Trade arrays are pre-allocated to a worst-case size (T//2) and a
  separate count integer tracks how many rows are valid.  This avoids
  Python list appends inside the JIT boundary.
- _batch_crossing_trades uses numba.prange for the outer loop over M
  combinations, giving automatic multithreading on multi-core machines.
"""

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


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
    """Dispatch to Numba if available, else pure Python."""
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


def _ref_detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints):
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
            # Exit when spread reverts past the exit threshold
            # side=+1 (long): entered below -xing_sd, exit when dist >= -exit_sd
            # side=-1 (short): entered above +xing_sd, exit when dist <= exit_sd
            exit_condition = (
                (side == -1 and d <= exit_sd) or
                (side == 1 and d >= -exit_sd)
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
    def _numba_detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints):
        """Numba-compiled trade detection.  Identical logic to _ref_ version."""
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
                exit_cond = (
                    (side == -1 and d <= exit_sd) or
                    (side == 1 and d >= -exit_sd)
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


def detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints):
    """Dispatch to Numba if available, else pure Python."""
    if HAS_NUMBA:
        return _numba_detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints)
    return _ref_detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SINGLE-SERIES BACKTEST (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════════

def backtest_spread(spread_returns, vol_window, xing_sd, exit_sd, day_ints):
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

    Returns
    -------
    trades : np.ndarray (n_trades, 5)
    n_trades : int
    cum : np.ndarray (T,)
    dist_sd : np.ndarray (T,)
    """
    cum = np.cumprod(1.0 + spread_returns)
    roll_mean, roll_std = rolling_mean_std(cum, vol_window)

    # Distance in standard deviations (NaN where std is zero or NaN)
    dist_sd = np.full_like(cum, np.nan)
    for i in range(len(cum)):
        if not np.isnan(roll_std[i]) and roll_std[i] > 0:
            dist_sd[i] = (cum[i] - roll_mean[i]) / roll_std[i]

    trades, n_trades = detect_trades(cum, dist_sd, xing_sd, exit_sd, day_ints)
    return trades, n_trades, cum, dist_sd


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


def _ref_batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints):
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
            spread_mat[:, j], vol_window, xing_sd, exit_sd, day_ints
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
    def _numba_batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints):
        """
        Numba-parallel batch backtest.  Uses prange over M columns so each
        combination runs on a separate thread.
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
                    exit_cond = (
                        (side == -1 and d <= exit_sd) or
                        (side == 1 and d >= -exit_sd)
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


def batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints):
    """Dispatch to Numba-parallel if available, else pure Python."""
    if HAS_NUMBA:
        return _numba_batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints)
    return _ref_batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints)
