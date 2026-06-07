"""
One-shot script: Q11 walk-forward for equity scalp regime (EXIT_SD=2.0).
Runs composite then contrarian mode, prints results matching the recording template.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from engine.walkforward import run_walk_forward, summarise_walk_forward
from engine.backtest import load_asset_prices

CACHE_DIR = Path(__file__).parent / 'cache'
CSV_PATH  = CACHE_DIR / 'prices.csv'

IS_YEARS  = 3
OOS_YEARS = 1
STEP      = 1
XING_SD   = 2.0
EXIT_SD   = 2.0
VOL_WIN   = 262


def run_mode(scoring_mode: str):
    print(f"\n{'='*60}")
    print(f"Run: {scoring_mode.upper()}, IS={IS_YEARS}y, OOS={OOS_YEARS}y, EXIT_SD={EXIT_SD}")
    print('='*60)

    prices, instruments = load_asset_prices(CSV_PATH)
    print(f"Instruments ({len(instruments)}): {instruments}")
    print(f"Price history: {prices.index[0].date()} to {prices.index[-1].date()}  ({len(prices)} days)")

    n_windows_est = max(0, (len(prices) - IS_YEARS * 262 - OOS_YEARS * 262) // (STEP * 262))
    n_pairs_est   = len(instruments) * (len(instruments) - 1)
    print(f"Estimated: ~{n_windows_est} windows × {n_pairs_est} pairs = ~{n_windows_est * n_pairs_est:,} observations")
    print("Running walk-forward (this may take several minutes)...")

    last_pct = [-1]
    def progress_cb(pct):
        p = int(pct * 10) * 10
        if p != last_pct[0]:
            print(f"  {p}%", flush=True)
            last_pct[0] = p

    results = run_walk_forward(
        prices, instruments,
        is_years=IS_YEARS,
        oos_years=OOS_YEARS,
        step_years=STEP,
        scoring_mode=scoring_mode,
        vol_window=VOL_WIN,
        xing_sd=XING_SD,
        exit_sd=EXIT_SD,
        progress_cb=progress_cb,
    )

    if results.empty:
        print("ERROR: No results returned — insufficient data for window lengths.")
        return

    s = summarise_walk_forward(results)
    rho, pval, n_obs = s['rho'], s['p_value'], s['n_obs']

    # Interpretation matching app logic
    significant = pval < 0.05
    mag = abs(rho)
    if not significant:
        interp = "No predictive power"
    elif rho > 0:
        interp = "Positive predictive power"
    else:
        interp = "Negative predictor"

    print(f"\nSpearman rho:    {rho:+.6f}")
    print(f"p-value:         {pval:.6f}")
    print(f"N observations:  {n_obs:,}")
    print(f"Interpretation:  {interp}")

    q = s['quintile_df']
    if not q.empty:
        # Sanity check: OOS_AvgHold
        q1_hold = q.loc['Q1', 'OOS_AvgHold'] if 'Q1' in q.index else float('nan')
        q3_hold = q.loc['Q3', 'OOS_AvgHold'] if 'Q3' in q.index else float('nan')
        print(f"OOS_AvgHold Q1:  {q1_hold:.1f}d  (sanity: expect 5–15d)")
        print(f"OOS_AvgHold Q3:  {q3_hold:.1f}d")

        print("\nQuintile table:")
        print(f"{'Q':<4} {'OOS_Gross':>10} {'OOS_Net':>10} {'OOS_GrossWR':>12} {'OOS_AvgHold':>12} {'N':>6}")
        print('-' * 58)
        for qi in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            if qi not in q.index:
                continue
            row = q.loc[qi]
            print(f"{qi:<4} {row['OOS_Gross']:>+10.4%} {row['OOS_Net']:>+10.4%} "
                  f"{row['OOS_GrossWR']:>12.1%} {row['OOS_AvgHold']:>11.1f}d "
                  f"{int(row['N']):>6,}")
    else:
        print("No quintile data.")


if __name__ == '__main__':
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run the Backtest tab first to cache equity prices.")
        sys.exit(1)

    run_mode('composite')
    run_mode('contrarian')
    print("\nDone.")
