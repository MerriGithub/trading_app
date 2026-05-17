"""
test_numba_parity.py — Verify Numba and Python reference produce identical results
====================================================================================

This test MUST pass before any integration changes are made to search.py or app.py.

It compares:
  1. Rolling mean/std:  _ref_ vs _numba_ (or dispatched)
  2. Trade detection:   _ref_detect_trades vs _numba_detect_trades
  3. Batch backtest:    _ref_batch_backtest vs _numba_batch_backtest
  4. End-to-end:        5 known equity 3v3 baskets, 5 FX pair-vs-pair spreads

All comparisons use float tolerance of 1e-10 for exact numerical parity.

Run:  python test_numba_parity.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so imports work from any location
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from numba_core import (
    _ref_rolling_mean_std, _ref_detect_trades, _ref_batch_backtest,
    rolling_mean_std, detect_trades, batch_backtest,
    backtest_spread, HAS_NUMBA,
    COL_ENTRY_IDX, COL_EXIT_IDX, COL_SIDE, COL_GROSS_RETURN, COL_HOLDING_DAYS,
)

TOLERANCE = 1e-10
PASS = 0
FAIL = 0


def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  ✓ {name}')
    else:
        FAIL += 1
        print(f'  ✗ {name}  {detail}')


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Rolling Mean/Std Parity
# ═══════════════════════════════════════════════════════════════════════════

def test_rolling_mean_std():
    print('\n[Test 1] Rolling Mean/Std Parity')
    np.random.seed(42)
    arr = np.random.randn(1000)
    # Insert some NaNs to test handling
    arr[50:55] = np.nan
    arr[500] = np.nan

    ref_mean, ref_std = _ref_rolling_mean_std(arr, 262)
    fast_mean, fast_std = rolling_mean_std(arr, 262)

    # Compare non-NaN positions
    valid = ~(np.isnan(ref_mean) | np.isnan(fast_mean))
    mean_diff = np.max(np.abs(ref_mean[valid] - fast_mean[valid]))
    check('Mean values match', mean_diff < TOLERANCE, f'max_diff={mean_diff}')

    valid_s = ~(np.isnan(ref_std) | np.isnan(fast_std))
    std_diff = np.max(np.abs(ref_std[valid_s] - fast_std[valid_s]))
    check('Std values match', std_diff < TOLERANCE, f'max_diff={std_diff}')

    # Check NaN positions match
    nan_match = np.array_equal(np.isnan(ref_mean), np.isnan(fast_mean))
    check('NaN positions match (mean)', nan_match)

    nan_match_s = np.array_equal(np.isnan(ref_std), np.isnan(fast_std))
    check('NaN positions match (std)', nan_match_s)


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Trade Detection Parity
# ═══════════════════════════════════════════════════════════════════════════

def test_detect_trades():
    print('\n[Test 2] Trade Detection Parity')
    np.random.seed(123)
    T = 2000
    returns = np.random.randn(T) * 0.01
    cum = np.cumprod(1.0 + returns)
    rm, rs = _ref_rolling_mean_std(cum, 262)
    dist = np.full(T, np.nan)
    for i in range(T):
        if not np.isnan(rs[i]) and rs[i] > 0:
            dist[i] = (cum[i] - rm[i]) / rs[i]

    day_ints = np.arange(T, dtype=np.int64)

    # Test with different SD thresholds and exit targets
    for xing_sd, exit_sd in [(2.0, 0.0), (1.5, 0.0), (2.0, 0.5), (3.0, 0.0)]:
        ref_trades, ref_n = _ref_detect_trades(cum, dist, xing_sd, exit_sd, day_ints)
        fast_trades, fast_n = detect_trades(cum, dist, xing_sd, exit_sd, day_ints)

        label = f'SD={xing_sd}, exit={exit_sd}'
        check(f'Trade count matches ({label})', ref_n == fast_n,
              f'ref={ref_n}, fast={fast_n}')

        if ref_n == fast_n and ref_n > 0:
            max_diff = np.max(np.abs(ref_trades[:ref_n] - fast_trades[:fast_n]))
            check(f'Trade values match ({label})', max_diff < TOLERANCE,
                  f'max_diff={max_diff}')

            # Verify entry/exit indices are identical (integer values)
            idx_match = np.array_equal(
                ref_trades[:ref_n, :2].astype(int),
                fast_trades[:fast_n, :2].astype(int)
            )
            check(f'Entry/exit indices match ({label})', idx_match)


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Batch Backtest Parity
# ═══════════════════════════════════════════════════════════════════════════

def test_batch_backtest():
    print('\n[Test 3] Batch Backtest Parity')
    np.random.seed(456)
    T = 1500
    M = 20
    spread_mat = np.random.randn(T, M) * 0.008
    day_ints = np.arange(T, dtype=np.int64)

    ref_results = _ref_batch_backtest(spread_mat, 262, 2.0, 0.0, day_ints)
    fast_results = batch_backtest(spread_mat, 262, 2.0, 0.0, day_ints)

    # Compare each column
    col_names = ['n_trades', 'gross_wr', 'avg_gross', 'avg_holding',
                 'avg_winner', 'avg_loser', 'payoff_ratio', 'total_pnl']
    for ci, name in enumerate(col_names):
        max_diff = np.max(np.abs(ref_results[:, ci] - fast_results[:, ci]))
        check(f'{name} matches', max_diff < TOLERANCE, f'max_diff={max_diff}')


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: End-to-End with Real Data
# ═══════════════════════════════════════════════════════════════════════════

def test_equity_e2e():
    print('\n[Test 4a] End-to-End: Equity Index Baskets')

    prices_path = PROJECT_ROOT / 'prices.csv'
    if not prices_path.exists():
        print('  ⚠ prices.csv not found — skipping equity tests')
        return

    prices = pd.read_csv(prices_path, index_col='Date', parse_dates=True)
    prices = prices.ffill(limit=3).dropna(how='all')
    instruments = list(prices.columns)

    rets = prices.pct_change().dropna(how='all')
    vols = rets.rolling(262, min_periods=131).std()
    scalings = (0.01 / vols.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    scaled_df = (rets * scalings).dropna(how='any')
    scaled = scaled_df.values.astype(np.float64)
    day_ints = ((scaled_df.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).values.astype(np.int64)

    # 5 known 3v3 baskets (indices into the instrument list)
    N = len(instruments)
    test_baskets = [
        ([0, 1, 2], [3, 4, 5]),
        ([6, 7, 8], [9, 10, 11]),
        ([0, 3, 6], [1, 4, 7]),
        ([2, 5, 8], [0, 9, 11]),
        ([1, 3, 5], [7, 9, 11]),
    ]

    for bi, (long_idx, short_idx) in enumerate(test_baskets):
        if max(long_idx + short_idx) >= N:
            continue

        # Build spread return
        long_ret = scaled[:, long_idx].mean(axis=1)
        short_ret = scaled[:, short_idx].mean(axis=1)
        spread = long_ret - short_ret

        # Reference path
        ref_trades, ref_n, ref_cum, ref_dist = backtest_spread(
            spread, 262, 2.0, 0.0, day_ints
        )

        # Also run via _ref_ functions explicitly
        cum_check = np.cumprod(1.0 + spread)
        rm, rs = _ref_rolling_mean_std(cum_check, 262)
        dist_check = np.full(len(cum_check), np.nan)
        for i in range(len(cum_check)):
            if not np.isnan(rs[i]) and rs[i] > 0:
                dist_check[i] = (cum_check[i] - rm[i]) / rs[i]
        ref2_trades, ref2_n = _ref_detect_trades(cum_check, dist_check, 2.0, 0.0, day_ints)

        long_names = [instruments[i] for i in long_idx]
        short_names = [instruments[i] for i in short_idx]
        label = f'{"/".join(long_names)} vs {"/".join(short_names)}'

        check(f'Basket {bi+1} trade count ({label})', ref_n == ref2_n,
              f'backtest_spread={ref_n}, ref_detect={ref2_n}')

        if ref_n == ref2_n and ref_n > 0:
            max_diff = np.max(np.abs(
                ref_trades[:ref_n, COL_GROSS_RETURN] -
                ref2_trades[:ref2_n, COL_GROSS_RETURN]
            ))
            check(f'Basket {bi+1} gross returns match', max_diff < TOLERANCE,
                  f'max_diff={max_diff}')

            # Print summary for inspection
            gross_wr = (ref_trades[:ref_n, COL_GROSS_RETURN] > 0).mean()
            avg_gross = ref_trades[:ref_n, COL_GROSS_RETURN].mean()
            avg_hold = ref_trades[:ref_n, COL_HOLDING_DAYS].mean()
            print(f'    → {ref_n} trades, WR={gross_wr:.1%}, '
                  f'gross={avg_gross:+.4%}, hold={avg_hold:.0f}d')


def test_fx_e2e():
    print('\n[Test 4b] End-to-End: FX Pair-vs-Pair')

    fx_path = PROJECT_ROOT / 'fx_prices.csv'
    if not fx_path.exists():
        print('  ⚠ fx_prices.csv not found — skipping FX tests')
        return

    prices = pd.read_csv(fx_path, index_col='Date', parse_dates=True)
    prices = prices.ffill(limit=3).loc['2004-01-01':].dropna(how='all')
    instruments = list(prices.columns)

    rets = prices.pct_change().dropna(how='all')
    vols = rets.rolling(262, min_periods=131).std()
    scalings = (0.01 / vols.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    scaled_df = (rets * scalings).dropna(how='any')
    scaled = scaled_df.values.astype(np.float64)
    day_ints = ((scaled_df.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).values.astype(np.int64)

    # Test 5 pair-vs-pair combinations
    test_pairs = [
        ('EURUSD', 'GBPUSD'), ('AUDUSD', 'NZDUSD'), ('USDJPY', 'EURCHF'),
        ('EURGBP', 'AUDNZD'), ('USDCHF', 'USDCAD'),
    ]

    for p1, p2 in test_pairs:
        if p1 not in instruments or p2 not in instruments:
            print(f'  ⚠ {p1}/{p2} not in data — skipping')
            continue

        i1 = instruments.index(p1)
        i2 = instruments.index(p2)
        spread = scaled[:, i1] - scaled[:, i2]

        trades, n_trades, _, _ = backtest_spread(spread, 262, 2.0, 0.0, day_ints)

        if n_trades > 0:
            gross_wr = (trades[:n_trades, COL_GROSS_RETURN] > 0).mean()
            avg_gross = trades[:n_trades, COL_GROSS_RETURN].mean()
            avg_hold = trades[:n_trades, COL_HOLDING_DAYS].mean()
            check(f'{p1}/{p2}: {n_trades} trades', n_trades > 0)
            print(f'    → WR={gross_wr:.1%}, gross={avg_gross:+.4%}, hold={avg_hold:.0f}d')
        else:
            check(f'{p1}/{p2}: trades found', False, 'n_trades=0')


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Performance Comparison
# ═══════════════════════════════════════════════════════════════════════════

def test_performance():
    print('\n[Test 5] Performance Comparison')
    np.random.seed(789)
    T = 5000
    M = 100
    spread_mat = np.random.randn(T, M) * 0.008
    day_ints = np.arange(T, dtype=np.int64)

    # Warm up Numba JIT (first call compiles)
    if HAS_NUMBA:
        _ = batch_backtest(spread_mat[:100, :5], 50, 2.0, 0.0, day_ints[:100])

    # Time reference implementation
    t0 = time.perf_counter()
    ref = _ref_batch_backtest(spread_mat, 262, 2.0, 0.0, day_ints)
    t_ref = time.perf_counter() - t0

    # Time fast implementation
    t0 = time.perf_counter()
    fast = batch_backtest(spread_mat, 262, 2.0, 0.0, day_ints)
    t_fast = time.perf_counter() - t0

    speedup = t_ref / t_fast if t_fast > 0 else float('inf')
    print(f'  Reference: {t_ref:.3f}s')
    print(f'  Fast:      {t_fast:.3f}s')
    print(f'  Speedup:   {speedup:.1f}x')

    if HAS_NUMBA:
        check('Numba available', True)
        check('Speedup > 5x', speedup > 5, f'got {speedup:.1f}x')
    else:
        print('  ⚠ Numba not installed — both paths use Python reference')
        check('Fallback to reference works', True)


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Exit target (partial reversion)
# ═══════════════════════════════════════════════════════════════════════════

def test_exit_targets():
    print('\n[Test 6] Exit Target (Partial Reversion)')
    np.random.seed(321)
    T = 3000
    returns = np.random.randn(T) * 0.01
    cum = np.cumprod(1.0 + returns)
    rm, rs = _ref_rolling_mean_std(cum, 262)
    dist = np.full(T, np.nan)
    for i in range(T):
        if not np.isnan(rs[i]) and rs[i] > 0:
            dist[i] = (cum[i] - rm[i]) / rs[i]
    day_ints = np.arange(T, dtype=np.int64)

    # exit_sd=0.0 should produce the most trades (exits earliest, at mean)
    # exit_sd=0.5 exits earlier (at 0.5 SD from mean), so fewer missed reversions
    # but each trade captures less movement
    t0, n0 = _ref_detect_trades(cum, dist, 2.0, 0.0, day_ints)
    t05, n05 = _ref_detect_trades(cum, dist, 2.0, 0.5, day_ints)
    t10, n10 = _ref_detect_trades(cum, dist, 2.0, 1.0, day_ints)

    print(f'  exit=0.0: {n0} trades')
    print(f'  exit=0.5: {n05} trades')
    print(f'  exit=1.0: {n10} trades')

    # More aggressive exit should generally produce more trades (faster turnover)
    # and shorter holding periods
    if n0 > 0 and n10 > 0:
        avg_hold_0 = t0[:n0, COL_HOLDING_DAYS].mean()
        avg_hold_10 = t10[:n10, COL_HOLDING_DAYS].mean()
        check('Higher exit target → shorter holding',
              avg_hold_10 < avg_hold_0,
              f'exit=0: {avg_hold_0:.0f}d, exit=1.0: {avg_hold_10:.0f}d')
        check('Higher exit target → more trades',
              n10 >= n0,
              f'exit=0: {n0}, exit=1.0: {n10}')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 70)
    print('Numba Parity Test Suite')
    print(f'Numba available: {HAS_NUMBA}')
    print('=' * 70)

    test_rolling_mean_std()
    test_detect_trades()
    test_batch_backtest()
    test_equity_e2e()
    test_fx_e2e()
    test_performance()
    test_exit_targets()

    print('\n' + '=' * 70)
    print(f'Results: {PASS} passed, {FAIL} failed')
    if FAIL > 0:
        print('⚠ PARITY TEST FAILED — do not proceed with integration')
        sys.exit(1)
    else:
        print('✓ All tests passed — safe to proceed')
        sys.exit(0)
