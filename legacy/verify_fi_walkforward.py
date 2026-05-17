"""
Standalone verification that FI walk-forward produces correct results
after the FI_EXCLUDE fix. Expected gross win rate: ~65-75% (not ~45%).
"""
from pathlib import Path
import pandas as pd

from backtest import load_asset_prices
from walkforward import run_walk_forward
from asset_configs import FI_EXCLUDE

# --- 1. Load prices ---
fi_csv = Path('cache/fi_prices.csv')
assert fi_csv.exists(), f"fi_prices.csv not found at {fi_csv}"

prices, instruments = load_asset_prices(fi_csv)

# --- 2. Apply the FI_EXCLUDE filter ---
instruments_before = list(instruments)
instruments = [i for i in instruments if i not in FI_EXCLUDE]
prices      = prices[[c for c in prices.columns if c not in FI_EXCLUDE]]

print(f"Instruments before filter : {len(instruments_before)} — {instruments_before}")
print(f"Instruments after filter  : {len(instruments)} — {instruments}")
print(f"Excluded                  : {sorted(set(instruments_before) - set(instruments))}")
print()

assert len(instruments) == 11, (
    f"Expected 11 tradeable FI instruments after filter, got {len(instruments)}: {instruments}"
)
assert len(prices.columns) == 11, (
    f"Expected 11 price columns after filter, got {len(prices.columns)}: {list(prices.columns)}"
)
print("[PASS] Instrument count correct: 11 tradeable instruments, 4 excluded")
print()

# --- 3. Run walk-forward ---
print("Running walk-forward (IS=3y, OOS=1y, step=1y, SD=2.0, exit=0.0)...")
print("This may take 1–3 minutes...")
print()

results = run_walk_forward(
    prices,
    instruments,
    is_years=3,
    oos_years=1,
    step_years=1,
    scoring_mode='composite',
    vol_window=262,
    xing_sd=2.0,
    exit_sd=0.0,
)

# --- 4. Check results ---
if results is None or len(results) == 0:
    print("✗ FAIL: run_walk_forward() returned empty results.")
    raise SystemExit(1)

print(f"Walk-forward complete: {len(results)} window-pair observations")
print()
print(f"Columns: {list(results.columns)}")
print()

# OOS win rate column is 'OOS_WinRate' (per walkforward.py line 181)
avg_gwr = results['OOS_WinRate'].mean()
print(f"Average OOS gross win rate (OOS_WinRate): {avg_gwr:.1%}")

if avg_gwr >= 0.60:
    print(f"[PASS] GWR {avg_gwr:.1%} is in the expected range (>=60%)")
elif avg_gwr >= 0.45:
    print(f"[MARGINAL] GWR {avg_gwr:.1%} -- above the broken ~45% but below expected ~65-75%")
else:
    print(f"[FAIL] GWR {avg_gwr:.1%} -- too low, filter may not be working")

print()
print("Summary statistics (OOS columns, ALL pairs including zero-trade):")
print(results[['OOS_Trades', 'OOS_WinRate', 'OOS_Gross', 'OOS_Net']].describe().round(4))

# Filter to active pairs only (OOS_Trades > 0) — zero-trade pairs are trivially WR=0
active = results[results['OOS_Trades'] > 0]
print(f"\nActive pairs (OOS_Trades > 0): {len(active)} of {len(results)} ({len(active)/len(results):.1%})")
if len(active) > 0:
    avg_gwr_active = active['OOS_WinRate'].mean()
    print(f"Average OOS win rate (active pairs only): {avg_gwr_active:.1%}")
    if avg_gwr_active >= 0.60:
        print(f"[PASS] Active pair GWR {avg_gwr_active:.1%} is in expected range (>=60%)")
    elif avg_gwr_active >= 0.45:
        print(f"[MARGINAL] Active pair GWR {avg_gwr_active:.1%}")
    else:
        print(f"[FAIL] Active pair GWR {avg_gwr_active:.1%} -- too low")
    print("\nActive pair summary:")
    print(active[['OOS_Trades', 'OOS_WinRate', 'OOS_Gross', 'OOS_Net']].describe().round(4))

# Also run the proper summariser
from walkforward import summarise_walk_forward
summary = summarise_walk_forward(results)
print(f"\nSpearman rho (IS_Rank vs OOS_Gross): {summary['rho']:.3f} (p={summary['p_value']:.3f})")
print(f"Valid observations (OOS_Trades>0): {summary['n_obs']}")
if not summary['quintile_df'].empty:
    print("\nQuintile analysis (Q1=best IS rank, Q5=worst):")
    print(summary['quintile_df'])
