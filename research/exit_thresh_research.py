"""
exit_thresh_research.py
=======================
Phase 3, Step 3.2 — Entry Threshold Grid on Commodity Spreads.

Tests whether a higher entry threshold (XING_SD=2.5 or 3.0) improves on the
Step 3.1 winner (EXIT_SD=0.5, XING_SD=2.0, total_EV=7.37). EXIT_SD is fixed
at 0.5 throughout.

Usage:
    python research/exit_thresh_research.py

Outputs (all in results/):
    exit_thresh_commodities.csv
    exit_thresh_per_pair_commodities.csv
    exit_thresh_holdperiod_commodities.csv
    exit_thresh_32_inputs_commodities.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

RESEARCH_DIR = Path(__file__).parent
ROOT         = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

from trailing_stop_research import DAILY_FIN, MAX_HOLD, RESULTS_DIR
from peak_analysis_research import (
    build_spread,
    load_prices,
    COMMODITY_PAIRS,
    DATA_CONFIGS,
    VOL_WINDOW,
)

# ── Constants ─────────────────────────────────────────────────────────────────

BID_ASK        = DATA_CONFIGS['commodities']['bid_ask']   # = 0.002
EXIT_SD        = 0.5                                      # fixed — Step 3.1 winner
XING_SD_VALUES = [2.0, 2.5, 3.0]

STEP31_EV    = 7.37
BASELINE_EV  = 5.60
REPL_TOL     = 0.05

W    = 78
SEP  = '═' * W
SEP2 = '─' * W


# ── State machine ─────────────────────────────────────────────────────────────

def simulate_xing_sd(cum, dist_sd, day_ints, long_label, short_label, xing_sd):
    """
    IDLE → ENTERED → IDLE state machine.

    Entry:  |dist_sd[t]| > xing_sd (level-check, no crossing guard)
    Exit:   (side == -1 and dist_sd[t] <= 0.5) or (side == +1 and dist_sd[t] >= -0.5)
            OR hold_cal >= MAX_HOLD (calendar days).
    Tie-break: if target and max_hold both fire on the same bar → 'target'.
    """
    trades   = []
    in_trade = False

    for t in range(VOL_WINDOW, len(dist_sd)):
        d = dist_sd[t]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > xing_sd:
                in_trade       = True
                entry_idx      = t
                entry_cum      = cum[t]
                side           = -1
                entry_dist_sd  = d
            elif d < -xing_sd:
                in_trade       = True
                entry_idx      = t
                entry_cum      = cum[t]
                side           = +1
                entry_dist_sd  = d
        else:
            hold_cal  = int(day_ints[t] - day_ints[entry_idx])
            exit_cond = (side == -1 and d <= 0.5) or (side == +1 and d >= -0.5)
            max_hit   = hold_cal >= MAX_HOLD

            if exit_cond or max_hit:
                exit_cum       = cum[t]
                gross          = ((exit_cum - entry_cum) / entry_cum if side == +1
                                  else (entry_cum - exit_cum) / entry_cum)
                financing_cost = DAILY_FIN * hold_cal
                net            = gross - BID_ASK - financing_cost
                hold_td        = t - entry_idx

                entry_date = (pd.Timestamp('1970-01-01')
                              + pd.Timedelta(days=int(day_ints[entry_idx])))
                exit_date  = (pd.Timestamp('1970-01-01')
                              + pd.Timedelta(days=int(day_ints[t])))

                trades.append({
                    'entry_date':       entry_date,
                    'exit_date':        exit_date,
                    'side':             side,
                    'long_label':       long_label,
                    'short_label':      short_label,
                    'xing_sd':          xing_sd,
                    'gross_return':     gross,
                    'net_return':       net,
                    'hold_cal':         hold_cal,
                    'hold_td':          hold_td,
                    'financing_cost':   financing_cost,
                    'exit_type':        ('max_hold' if (max_hit and not exit_cond)
                                         else 'target'),
                    'dist_sd_at_entry': entry_dist_sd,
                    'dist_sd_at_exit':  d,
                })
                in_trade = False

    return trades


# ── Data collection ───────────────────────────────────────────────────────────

def collect_all():
    prices     = load_prices('commodities')
    all_trades = []

    n_pairs = len(COMMODITY_PAIRS)
    n_xings = len(XING_SD_VALUES)
    print(f"\nRunning {n_pairs} pairs × {n_xings} XING_SD values...")

    for idx, (long_inst, short_inst) in enumerate(COMMODITY_PAIRS):
        if long_inst not in prices.columns or short_inst not in prices.columns:
            print(f"  [{idx+1:02d}] {long_inst}/{short_inst}  SKIPPED (not in data)")
            continue

        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)

        for xing_sd in XING_SD_VALUES:
            trades = simulate_xing_sd(
                cum, dist_sd, day_ints, long_inst, short_inst, xing_sd
            )
            all_trades.extend(trades)

    return pd.DataFrame(all_trades)


# ── Summary tables ────────────────────────────────────────────────────────────

def build_main_grid(df):
    rows = []
    for xing_sd in XING_SD_VALUES:
        sub = df[df['xing_sd'] == xing_sd]
        if len(sub) == 0:
            rows.append({'xing_sd': xing_sd, 'n_trades': 0})
            continue

        n            = len(sub)
        pct_target   = (sub['exit_type'] == 'target').mean()
        avg_gross    = sub['gross_return'].mean()
        avg_net      = sub['net_return'].mean()
        avg_hold_cal = sub['hold_cal'].mean()
        avg_fin      = sub['financing_cost'].mean()
        avg_spread   = BID_ASK
        avg_total    = avg_fin + avg_spread
        win_rate     = (sub['net_return'] > 0).mean()
        avg_entry_sd = sub['dist_sd_at_entry'].mean()
        total_ev     = avg_net * n

        rows.append({
            'xing_sd':              xing_sd,
            'n_trades':             n,
            'pct_target_exit':      pct_target,
            'avg_gross':            avg_gross,
            'avg_net':              avg_net,
            'avg_hold_cal':         avg_hold_cal,
            'avg_financing_cost':   avg_fin,
            'avg_spread_cost':      avg_spread,
            'avg_total_cost':       avg_total,
            'win_rate':             win_rate,
            'avg_dist_sd_at_entry': avg_entry_sd,
            'total_EV':             total_ev,
            'vs_31_EV':             total_ev - STEP31_EV,
            'vs_baseline_EV':       total_ev - BASELINE_EV,
            'beats_31':             total_ev > STEP31_EV,
            'beats_baseline':       total_ev > BASELINE_EV,
        })

    return pd.DataFrame(rows)


def build_per_pair(df):
    rows = []
    for (xing_sd, long_label, short_label), sub in df.groupby(
        ['xing_sd', 'long_label', 'short_label'], dropna=False
    ):
        n        = len(sub)
        avg_net  = sub['net_return'].mean()
        avg_hold = sub['hold_cal'].mean()
        win_rate = (sub['net_return'] > 0).mean()
        total_ev = avg_net * n
        rows.append({
            'xing_sd':     xing_sd,
            'long_label':  long_label,
            'short_label': short_label,
            'n_trades':    n,
            'avg_gross':   sub['gross_return'].mean(),
            'avg_net':     avg_net,
            'avg_hold_cal': avg_hold,
            'win_rate':    win_rate,
            'total_EV':    total_ev,
        })
    return pd.DataFrame(rows)


def build_hold_distribution(df):
    rows = []
    for xing_sd in XING_SD_VALUES:
        sub = df[df['xing_sd'] == xing_sd]
        if len(sub) == 0:
            rows.append({'xing_sd': xing_sd, 'n_trades': 0})
            continue
        hold   = sub['hold_cal']
        pct_mh = (sub['exit_type'] == 'max_hold').mean()
        rows.append({
            'xing_sd':            xing_sd,
            'n_trades':           len(sub),
            'hold_p10':           hold.quantile(0.10),
            'hold_p25':           hold.quantile(0.25),
            'hold_p50':           hold.quantile(0.50),
            'hold_p75':           hold.quantile(0.75),
            'hold_p90':           hold.quantile(0.90),
            'hold_mean':          hold.mean(),
            'pct_max_hold_exits': pct_mh,
        })
    return pd.DataFrame(rows)


def build_32_inputs(main_grid):
    df       = main_grid.copy()
    best_idx = df['total_EV'].idxmax()
    df['recommended'] = False
    df.loc[best_idx, 'recommended'] = True
    cols = [
        'xing_sd', 'n_trades', 'avg_net', 'avg_hold_cal',
        'avg_dist_sd_at_entry', 'win_rate', 'total_EV',
        'vs_31_EV', 'vs_baseline_EV', 'beats_31', 'recommended',
    ]
    return df[cols]


# ── Printed output ────────────────────────────────────────────────────────────

def print_results(main_grid, hold_dist, df):
    ref_row  = main_grid[main_grid['xing_sd'] == 2.0].iloc[0]
    best_row = main_grid.loc[main_grid['total_EV'].idxmax()]
    hd_ref   = hold_dist[hold_dist['xing_sd'] == 2.0].iloc[0]
    hd_best  = hold_dist[hold_dist['xing_sd'] == best_row['xing_sd']].iloc[0]

    repl_ev   = ref_row['total_EV']
    repl_diff = abs(repl_ev - STEP31_EV)
    repl_pass = repl_diff <= REPL_TOL

    print()
    print(SEP)
    print('STEP 3.2 — EXIT + THRESHOLD GRID — COMMODITY SPREADS')
    print(SEP)
    print()
    print('Benchmarks:')
    print(f'  Primary   (Step 3.1 winner): total_EV = {STEP31_EV:.2f}  [EXIT_SD=0.5, XING_SD=2.0]')
    print(f'  Secondary (Phase 2.4 base):  total_EV = {BASELINE_EV:.2f}  [EXIT_SD=0.0, XING_SD=2.0]')
    print()
    print('Replication check (XING_SD=2.0 must match 3.1 within ±0.05):')
    status = 'PASS' if repl_pass else 'FAIL — investigate before reading 2.5/3.0'
    print(f'  Replicated total_EV: {repl_ev:.2f}  [{status}]')
    print()
    print('Main results:')
    print()
    print(f"  {'XING_SD':>7}  {'n_trades':>8}  {'%target':>7}  "
          f"{'avg_gross':>9}  {'avg_net':>7}  {'avg_hold':>8}  "
          f"{'avg_cost':>8}  {'win_rate':>8}  {'avg_entry_SD':>12}  "
          f"{'total_EV':>8}  {'Δ_vs_3.1':>9}")
    print('  ' + '─' * 106)

    for _, row in main_grid.iterrows():
        delta = ('baseline' if row['xing_sd'] == 2.0
                 else f"{row['vs_31_EV']:+.2f}")
        print(
            f"  {row['xing_sd']:>7.1f}  {int(row['n_trades']):>8d}  "
            f"{row['pct_target_exit']*100:>6.1f}%  "
            f"{row['avg_gross']:>+9.4f}  {row['avg_net']:>+7.4f}  "
            f"{row['avg_hold_cal']:>7.0f} d  "
            f"{row['avg_total_cost']*100:>7.4f}%  "
            f"{row['win_rate']*100:>7.1f}%  "
            f"{row['avg_dist_sd_at_entry']:>9.2f} SD  "
            f"{row['total_EV']:>8.2f}  {delta:>9}"
        )

    print()
    print('Hold period distribution (calendar days):')
    print()
    print(f"  {'XING_SD':>7}  {'p10':>5}  {'p25':>5}  {'p50':>5}  "
          f"{'p75':>5}  {'p90':>5}  {'mean':>5}  {'%max_hold':>9}")
    print('  ' + SEP2)
    for _, row in hold_dist.iterrows():
        print(
            f"  {row['xing_sd']:>7.1f}  "
            f"{row['hold_p10']:>5.0f}  {row['hold_p25']:>5.0f}  "
            f"{row['hold_p50']:>5.0f}  {row['hold_p75']:>5.0f}  "
            f"{row['hold_p90']:>5.0f}  {row['hold_mean']:>5.0f}  "
            f"{row['pct_max_hold_exits']*100:>8.1f}%"
        )

    print()
    print('Gross/net decomposition:')
    print()
    print(f"  {'XING_SD':>7}  {'avg_gross':>9}  {'avg_spread':>10}  "
          f"{'avg_financing':>13}  {'avg_net':>8}  {'gross_retention%':>16}")
    print('  ' + SEP2)
    for _, row in main_grid.iterrows():
        gross_ret_pct = (row['avg_net'] / row['avg_gross'] * 100
                         if row['avg_gross'] != 0 else float('nan'))
        print(
            f"  {row['xing_sd']:>7.1f}  "
            f"{row['avg_gross']:>+9.4f}  "
            f"{row['avg_spread_cost']*100:>9.4f}%  "
            f"{row['avg_financing_cost']*100:>12.4f}%  "
            f"{row['avg_net']:>+8.4f}  "
            f"{gross_ret_pct:>15.1f}%"
        )

    print()
    print('Entry distance analysis (does higher XING_SD select materially larger dislocations?):')
    print()
    print(f"  {'XING_SD':>7}  {'avg_dist_sd_at_entry':>20}  {'avg_hold_cal':>12}  implication")
    print('  ' + SEP2)
    ref_entry_sd  = ref_row['avg_dist_sd_at_entry']
    ref_hold      = ref_row['avg_hold_cal']
    for _, row in main_grid.iterrows():
        if row['xing_sd'] == 2.0:
            impl = 'reference'
        else:
            d_entry = row['avg_dist_sd_at_entry'] - ref_entry_sd
            d_hold  = row['avg_hold_cal'] - ref_hold
            impl    = f'Δ entry: {d_entry:+.2f} SD, Δ hold: {d_hold:+.0f} d'
        print(
            f"  {row['xing_sd']:>7.1f}  "
            f"{row['avg_dist_sd_at_entry']:>17.2f} SD  "
            f"{row['avg_hold_cal']:>10.0f} d  "
            f"{impl}"
        )

    print()
    print(SEP)
    print()
    print('DECISION SUMMARY')
    print(SEP)
    print()

    any_beats = bool((main_grid['total_EV'] > STEP31_EV).any())
    print(f'Does any XING_SD beat the primary benchmark (total_EV = {STEP31_EV:.2f})?  '
          f'[{"YES" if any_beats else "NO"}]')
    print()
    print('Best XING_SD by total_EV:')

    b  = best_row
    r  = ref_row
    n_lost     = int(r['n_trades']) - int(b['n_trades'])
    n_lost_pct = (n_lost / r['n_trades'] * 100) if r['n_trades'] > 0 else float('nan')
    net_gained     = b['avg_net'] - r['avg_net']
    net_gained_pct = (net_gained / abs(r['avg_net']) * 100
                      if r['avg_net'] != 0 else float('nan'))
    hold_delta     = b['avg_hold_cal'] - r['avg_hold_cal']
    ev_delta       = b['total_EV'] - STEP31_EV

    print(f"  XING_SD = {b['xing_sd']:.1f}")
    print(f"  n_trades:              {int(b['n_trades'])} (vs {int(r['n_trades'])} at XING_SD=2.0)")
    print(f"  avg_dist_sd_at_entry:  {b['avg_dist_sd_at_entry']:.2f} SD "
          f"(vs {r['avg_dist_sd_at_entry']:.2f} at XING_SD=2.0)")
    print(f"  avg_gross:             {b['avg_gross']:+.4f} (vs {r['avg_gross']:+.4f})")
    print(f"  avg_net:               {b['avg_net']:+.4f} (vs {r['avg_net']:+.4f})")
    print(f"  avg_hold_cal:          {b['avg_hold_cal']:.0f} d   "
          f"(vs {r['avg_hold_cal']:.0f} d, Δ = {hold_delta:+.0f} d)")
    print(f"  avg_financing:         {b['avg_financing_cost']*100:.4f}% "
          f"(vs {r['avg_financing_cost']*100:.4f}%)")
    print(f"  win_rate:              {b['win_rate']*100:.1f}%   "
          f"(vs {r['win_rate']*100:.1f}%)")
    print(f"  total_EV:              {b['total_EV']:.2f}    "
          f"(vs {STEP31_EV:.2f} primary benchmark, Δ = {ev_delta:+.2f})")
    print()
    print('Trade-off analysis:')
    compensates = (b['total_EV'] >= STEP31_EV)
    direction   = 'improved' if ev_delta >= 0 else 'declined'
    print(f"  n_trades lost vs XING_SD=2.0:  {n_lost} (−{n_lost_pct:.1f}%)")
    print(f"  avg_net gained per trade:       {net_gained:+.4f} ({net_gained_pct:+.1f}%)")
    print(f"  Does per-trade improvement compensate for trade count loss?  "
          f"[{'YES' if compensates else 'NO'}]")
    print(f"  Net direction: total_EV {direction} by {abs(ev_delta):.2f}")
    print()
    print('Holding period effect at best XING_SD:')
    hold_pct = (hold_delta / r['avg_hold_cal'] * 100
                if r['avg_hold_cal'] != 0 else float('nan'))
    if abs(hold_pct) < 5:
        interp = 'similar — entry threshold change does not materially shift hold duration'
    elif hold_delta < 0:
        interp = 'faster — larger entry dislocations reverted to EXIT_SD=0.5 more quickly'
    else:
        interp = 'slower — larger entry dislocations took longer to revert to EXIT_SD=0.5'
    print(f"  Does larger entry dislocation lead to faster reversion to EXIT_SD=0.5?")
    print(f"  avg_hold at XING_SD=2.0: {r['avg_hold_cal']:.0f} d → "
          f"XING_SD={b['xing_sd']:.1f}: {b['avg_hold_cal']:.0f} d "
          f"(Δ = {hold_delta:+.0f} d, {hold_pct:+.1f}%)")
    print(f"  Interpretation: [{interp}]")
    print()
    print('Phase 3 combined optimum:')
    best_overall_ev   = b['total_EV']
    best_overall_xing = b['xing_sd']
    vs_baseline_pct   = ((best_overall_ev - BASELINE_EV) / abs(BASELINE_EV) * 100
                          if BASELINE_EV != 0 else float('nan'))
    print(f"  Best configuration found across Steps 3.1 + 3.2:")
    print(f"    EXIT_SD = 0.5, XING_SD = {best_overall_xing:.1f}")
    print(f"    total_EV = {best_overall_ev:.2f} "
          f"(vs {BASELINE_EV:.2f} original baseline, {vs_baseline_pct:+.1f}% improvement)")
    print(f"    avg_net   = {b['avg_net']:+.4f} per trade")
    print(f"    avg_hold  = {b['avg_hold_cal']:.0f} d")
    print(f"    n_trades  = {int(b['n_trades'])}")
    print()
    print('Phase 3 conclusion:')
    if best_row['xing_sd'] == 2.0:
        print(f"  EXIT_SD=0.5 is the only lever. Higher thresholds sacrifice too many trades.")
        print(f"  Combined optimum is EXIT_SD=0.5 + XING_SD=2.0, total_EV={STEP31_EV:.2f}.")
    else:
        xbest = best_row['xing_sd']
        print(f"  Higher entry threshold improves total_EV.")
        print(f"  Combined optimum is EXIT_SD=0.5 + XING_SD={xbest:.1f}, "
              f"total_EV={best_overall_ev:.2f}. "
              f"Additional gain vs Step 3.1: {ev_delta:+.2f}.")
    print()
    print('Next step:')
    print(f"  Phase 3 is complete. Proceed to Phase 4 — cross-asset extension.")
    print(f"  Apply best Phase 3 configuration (EXIT_SD=0.5, "
          f"XING_SD={best_overall_xing:.1f}) to FX, equity")
    print(f"  indices, and fixed income as a parameter swap (same state machine, asset-")
    print(f"  specific data and cost constants).")
    print()
    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = collect_all()

    if df.empty:
        print("No trades generated. Check data file and pair list.")
        return

    main_grid = build_main_grid(df)
    per_pair  = build_per_pair(df)
    hold_dist = build_hold_distribution(df)
    inputs_32 = build_32_inputs(main_grid)

    main_grid.to_csv(RESULTS_DIR / 'exit_thresh_commodities.csv',           index=False)
    per_pair.to_csv( RESULTS_DIR / 'exit_thresh_per_pair_commodities.csv',  index=False)
    hold_dist.to_csv(RESULTS_DIR / 'exit_thresh_holdperiod_commodities.csv',index=False)
    inputs_32.to_csv(RESULTS_DIR / 'exit_thresh_32_inputs_commodities.csv', index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")

    print_results(main_grid, hold_dist, df)


if __name__ == '__main__':
    main()
