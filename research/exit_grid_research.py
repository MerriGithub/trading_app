"""
exit_grid_research.py
=====================
Phase 3, Step 3.1 — Exit SD Grid on Commodity Spreads.

Tests whether exiting at a higher SD threshold (before 0.0 SD mean-reversion)
improves total expected value by cutting financing drag and locking in gross
return from the fast initial reversion phase.

Phase 3 benchmark: total_EV = 5.60 (Phase 2.4 baseline, EXIT_SD=0.0, n=583)

Usage:
    python research/exit_grid_research.py

Outputs (all in results/):
    exit_grid_commodities.csv
    exit_grid_per_pair_commodities.csv
    exit_grid_holdperiod_commodities.csv
    exit_grid_31_inputs_commodities.csv
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

from trailing_stop_research import (
    DAILY_FIN, MAX_HOLD, RESULTS_DIR,
)
from peak_analysis_research import (
    build_spread,
    load_prices,
    COMMODITY_PAIRS,
    DATA_CONFIGS,
    VOL_WINDOW,
)

# ── Constants ─────────────────────────────────────────────────────────────────

EXIT_SD_VALUES      = [0.0, 0.5, 1.0, 1.5]
XING_SD             = 2.0
BID_ASK             = DATA_CONFIGS['commodities']['bid_ask']   # = 0.002

PHASE3_BENCHMARK_EV = 5.60

W    = 78
SEP  = '═' * W
SEP2 = '─' * W


# ── State machine ─────────────────────────────────────────────────────────────

def simulate_exit_sd(cum, dist_sd, day_ints, long_label, short_label, exit_sd):
    """
    IDLE → ENTERED → IDLE state machine.

    Entry:  |dist_sd| first crosses above XING_SD = 2.0 (from below).
    Exit:   -side * dist_sd[t] <= exit_sd  (the asymmetric general form)
            OR hold_cal >= MAX_HOLD.

    Correct exit formula: (-side * d <= exit_sd)
      side=-1, exit_sd=1.0 → d <= +1.0  (spread fading back from +2.x)
      side=+1, exit_sd=1.0 → d >= -1.0  (spread fading back from -2.x)
      exit_sd=0.0           → fires at mean-crossing for both sides (baseline)
    """
    trades   = []
    in_trade = False

    for t in range(VOL_WINDOW, len(dist_sd)):
        d = dist_sd[t]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > XING_SD:
                in_trade  = True
                entry_idx = t
                entry_cum = cum[t]
                side      = -1
            elif d < -XING_SD:
                in_trade  = True
                entry_idx = t
                entry_cum = cum[t]
                side      = +1
        else:
            hold_cal  = int(day_ints[t] - day_ints[entry_idx])
            exit_cond = (-side * d <= exit_sd)
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
                    'entry_date':      entry_date,
                    'exit_date':       exit_date,
                    'side':            side,
                    'long_label':      long_label,
                    'short_label':     short_label,
                    'exit_sd':         exit_sd,
                    'gross_return':    gross,
                    'net_return':      net,
                    'hold_cal':        hold_cal,
                    'hold_td':         hold_td,
                    'financing_cost':  financing_cost,
                    'exit_type':       ('max_hold' if (max_hit and not exit_cond)
                                        else 'target'),
                    'dist_sd_at_exit': d,
                })
                in_trade = False

    return trades


# ── Data collection ───────────────────────────────────────────────────────────

def collect_all():
    prices     = load_prices('commodities')
    all_trades = []

    n_pairs = len(COMMODITY_PAIRS)
    n_exits = len(EXIT_SD_VALUES)
    print(f"\nRunning {n_pairs} pairs × {n_exits} EXIT_SD values...")

    for idx, (long_inst, short_inst) in enumerate(COMMODITY_PAIRS):
        if long_inst not in prices.columns or short_inst not in prices.columns:
            print(f"  [{idx+1:02d}] {long_inst}/{short_inst}  SKIPPED (not in data)")
            continue

        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)

        for exit_sd in EXIT_SD_VALUES:
            trades = simulate_exit_sd(
                cum, dist_sd, day_ints, long_inst, short_inst, exit_sd
            )
            all_trades.extend(trades)

    return pd.DataFrame(all_trades)


# ── Summary tables ────────────────────────────────────────────────────────────

def build_main_grid(df):
    rows        = []
    baseline_ev = None

    for exit_sd in EXIT_SD_VALUES:
        sub = df[df['exit_sd'] == exit_sd]
        if len(sub) == 0:
            rows.append({'exit_sd': exit_sd, 'n_trades': 0})
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
        total_ev     = avg_net * n

        if exit_sd == 0.0:
            baseline_ev = total_ev

        rows.append({
            'exit_sd':            exit_sd,
            'n_trades':           n,
            'pct_target_exit':    pct_target,
            'avg_gross':          avg_gross,
            'avg_net':            avg_net,
            'avg_hold_cal':       avg_hold_cal,
            'avg_financing_cost': avg_fin,
            'avg_spread_cost':    avg_spread,
            'avg_total_cost':     avg_total,
            'win_rate':           win_rate,
            'total_EV':           total_ev,
        })

    result = pd.DataFrame(rows)
    if baseline_ev is not None:
        result['vs_baseline_EV'] = result['total_EV'] - baseline_ev
        result['beats_baseline'] = result['total_EV'] > baseline_ev
    return result


def build_per_pair(df):
    rows = []
    for (exit_sd, long_label, short_label), sub in df.groupby(
        ['exit_sd', 'long_label', 'short_label'], dropna=False
    ):
        n        = len(sub)
        avg_net  = sub['net_return'].mean()
        avg_hold = sub['hold_cal'].mean()
        win_rate = (sub['net_return'] > 0).mean()
        total_ev = avg_net * n
        rows.append({
            'exit_sd':     exit_sd,
            'long_label':  long_label,
            'short_label': short_label,
            'n_trades':    n,
            'avg_net':     avg_net,
            'avg_hold_cal': avg_hold,
            'win_rate':    win_rate,
            'total_EV':    total_ev,
        })
    return pd.DataFrame(rows)


def build_hold_distribution(df):
    rows = []
    for exit_sd in EXIT_SD_VALUES:
        sub = df[df['exit_sd'] == exit_sd]
        if len(sub) == 0:
            rows.append({'exit_sd': exit_sd, 'n_trades': 0})
            continue
        hold   = sub['hold_cal']
        pct_mh = (sub['exit_type'] == 'max_hold').mean()
        rows.append({
            'exit_sd':            exit_sd,
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


def build_31_inputs(main_grid):
    df       = main_grid.copy()
    best_idx = df['total_EV'].idxmax()
    df['recommended'] = False
    df.loc[best_idx, 'recommended'] = True
    cols = [
        'exit_sd', 'n_trades', 'avg_net', 'avg_hold_cal', 'win_rate',
        'total_EV', 'vs_baseline_EV', 'beats_baseline', 'recommended',
    ]
    return df[cols]


# ── Printed output ────────────────────────────────────────────────────────────

def print_results(main_grid, hold_dist, df):
    baseline = main_grid[main_grid['exit_sd'] == 0.0].iloc[0]
    best_row = main_grid.loc[main_grid['total_EV'].idxmax()]
    hd_base  = hold_dist[hold_dist['exit_sd'] == 0.0].iloc[0]
    hd_best  = hold_dist[hold_dist['exit_sd'] == best_row['exit_sd']].iloc[0]

    print()
    print(SEP)
    print('STEP 3.1 — EXIT SD GRID — COMMODITY SPREADS')
    print(SEP)
    print()
    print(f'Benchmark: total_EV = {PHASE3_BENCHMARK_EV:.2f} '
          f'(Phase 2.4 baseline, EXIT_SD=0.0, n=583 trades)')
    print()
    print('Main results:')
    print()
    print(f"  {'EXIT_SD':>7}  {'n_trades':>8}  {'%target':>7}  "
          f"{'avg_gross':>9}  {'avg_net':>7}  {'avg_hold':>8}  "
          f"{'avg_cost':>8}  {'win_rate':>8}  {'total_EV':>8}  {'Δ_EV':>8}")
    print('  ' + SEP2)

    for _, row in main_grid.iterrows():
        delta = ('baseline' if row['exit_sd'] == 0.0
                 else f"{row['vs_baseline_EV']:+.2f}")
        print(
            f"  {row['exit_sd']:>7.1f}  {int(row['n_trades']):>8d}  "
            f"{row['pct_target_exit']*100:>6.1f}%  "
            f"{row['avg_gross']:>+9.4f}  {row['avg_net']:>+7.4f}  "
            f"{row['avg_hold_cal']:>7.0f} d  "
            f"{row['avg_total_cost']*100:>7.4f}%  "
            f"{row['win_rate']*100:>7.1f}%  "
            f"{row['total_EV']:>8.2f}  {delta:>8}"
        )

    print()
    print('Hold period distribution (calendar days):')
    print()
    print(f"  {'EXIT_SD':>7}  {'p10':>5}  {'p25':>5}  {'p50':>5}  "
          f"{'p75':>5}  {'p90':>5}  {'mean':>5}  {'%max_hold':>9}")
    print('  ' + SEP2)
    for _, row in hold_dist.iterrows():
        print(
            f"  {row['exit_sd']:>7.1f}  "
            f"{row['hold_p10']:>5.0f}  {row['hold_p25']:>5.0f}  "
            f"{row['hold_p50']:>5.0f}  {row['hold_p75']:>5.0f}  "
            f"{row['hold_p90']:>5.0f}  {row['hold_mean']:>5.0f}  "
            f"{row['pct_max_hold_exits']*100:>8.1f}%"
        )

    print()
    print('Gross/net decomposition:')
    print()
    print(f"  {'EXIT_SD':>7}  {'avg_gross':>9}  {'avg_spread_cost':>15}  "
          f"{'avg_financing':>13}  {'avg_net':>8}  {'gross_retention%':>16}")
    print('  ' + SEP2)
    for _, row in main_grid.iterrows():
        gross_ret_pct = (row['avg_net'] / row['avg_gross'] * 100
                         if row['avg_gross'] != 0 else float('nan'))
        print(
            f"  {row['exit_sd']:>7.1f}  "
            f"{row['avg_gross']:>+9.4f}  "
            f"{row['avg_spread_cost']*100:>14.4f}%  "
            f"{row['avg_financing_cost']*100:>12.4f}%  "
            f"{row['avg_net']:>+8.4f}  "
            f"{gross_ret_pct:>15.1f}%"
        )

    print()
    print(f"  gross_retention% = avg_net / avg_gross * 100")
    print()
    print(SEP)
    print()
    print('DECISION SUMMARY')
    print(SEP)
    print()

    any_beats = bool((main_grid['total_EV'] > PHASE3_BENCHMARK_EV).any())
    print(f'Does any EXIT_SD beat the total_EV benchmark of '
          f'{PHASE3_BENCHMARK_EV:.2f}? [{"YES" if any_beats else "NO"}]')
    print()
    print('Best EXIT_SD by total_EV:')

    b          = best_row
    bl         = baseline
    gross_delta = b['avg_gross'] - bl['avg_gross']
    gross_pct   = (gross_delta / bl['avg_gross'] * 100
                   if bl['avg_gross'] != 0 else float('nan'))
    net_delta   = b['avg_net'] - bl['avg_net']
    hold_delta  = b['avg_hold_cal'] - bl['avg_hold_cal']
    ev_delta    = b['total_EV'] - PHASE3_BENCHMARK_EV

    print(f"  EXIT_SD = {b['exit_sd']:.1f}")
    print(f"  n_trades:      {int(b['n_trades'])} (vs {int(bl['n_trades'])} baseline)")
    print(f"  avg_gross:     {b['avg_gross']:+.4f} (vs {bl['avg_gross']:+.4f} baseline)")
    print(f"  avg_net:       {b['avg_net']:+.4f} (vs {bl['avg_net']:+.4f} baseline)")
    print(f"  avg_hold_cal:  {b['avg_hold_cal']:.0f} d  "
          f"(vs {bl['avg_hold_cal']:.0f} d baseline, Δ = {hold_delta:+.0f} d)")
    print(f"  avg_financing: {b['avg_financing_cost']*100:.4f}% "
          f"(vs {bl['avg_financing_cost']*100:.4f}% baseline)")
    print(f"  win_rate:      {b['win_rate']*100:.1f}%  "
          f"(vs {bl['win_rate']*100:.1f}% baseline)")
    print(f"  total_EV:      {b['total_EV']:.2f}  "
          f"(vs {PHASE3_BENCHMARK_EV:.2f} benchmark, Δ = {ev_delta:+.2f})")
    print()
    print('Hold reduction at best EXIT_SD:')
    med_delta  = hd_best['hold_p50'] - hd_base['hold_p50']
    med_pct    = (med_delta / hd_base['hold_p50'] * 100
                  if hd_base['hold_p50'] != 0 else float('nan'))
    mean_delta = hd_best['hold_mean'] - hd_base['hold_mean']
    mean_pct   = (mean_delta / hd_base['hold_mean'] * 100
                  if hd_base['hold_mean'] != 0 else float('nan'))
    print(f"  Median hold: {hd_base['hold_p50']:.0f} d → {hd_best['hold_p50']:.0f} d "
          f"(Δ = {med_delta:+.0f} d, {med_pct:+.1f}%)")
    print(f"  Mean hold:   {hd_base['hold_mean']:.0f} d → {hd_best['hold_mean']:.0f} d "
          f"(Δ = {mean_delta:+.0f} d, {mean_pct:+.1f}%)")
    print(f"  %max_hold exits: {hd_base['pct_max_hold_exits']*100:.1f}% → "
          f"{hd_best['pct_max_hold_exits']*100:.1f}%")
    print()
    print('Gross return retained at best EXIT_SD:')
    print(f"  avg_gross:  {bl['avg_gross']:+.4f} → {b['avg_gross']:+.4f} "
          f"(Δ = {gross_delta:+.4f}, {gross_pct:+.1f}% of gross)")
    net_dir = 'IMPROVED' if net_delta > 0 else 'DECLINED'
    print(f"  avg_net:    {bl['avg_net']:+.4f} → {b['avg_net']:+.4f} "
          f"(Δ = {net_delta:+.4f} — net {net_dir} despite lower gross)")
    print()

    net_improved = net_delta > 0
    exceeds      = 'exceeds' if net_improved else 'does not exceed'
    print(f'Is the hold reduction more valuable than the gross return given up? '
          f'[{"YES" if net_improved else "NO"}]')
    print(f'  Net change per trade: Δ_net = {net_delta:+.4f}')
    print(f'  This {exceeds} the cost saving from hold reduction.')
    print()

    # Monotone trend check across EXIT_SD_VALUES
    ev_by_sd = main_grid.set_index('exit_sd')['total_EV']
    diffs    = [ev_by_sd[EXIT_SD_VALUES[i+1]] - ev_by_sd[EXIT_SD_VALUES[i]]
                for i in range(len(EXIT_SD_VALUES) - 1)]
    best_sd  = b['exit_sd']

    if all(d > 0 for d in diffs):
        trend = 'MONOTONE improvement'
        impl  = ('total_EV increases with EXIT_SD across the full grid — '
                 'consider testing EXIT_SD > 1.5.')
    elif all(d < 0 for d in diffs):
        trend = 'MONOTONE decline'
        impl  = ('Exiting earlier consistently hurts — '
                 'baseline exit at 0.0 SD is optimal.')
    else:
        peak_sd = EXIT_SD_VALUES[list(ev_by_sd[EXIT_SD_VALUES]).index(
            max(ev_by_sd[EXIT_SD_VALUES])
        )]
        trend = f'PEAKS at EXIT_SD={peak_sd:.1f} then falls'
        impl  = (f'Optimal exit threshold is EXIT_SD={peak_sd:.1f}; '
                 f'earlier exits give diminishing returns past that point.')

    print('Monotone trend check (does total_EV peak and then fall, or monotone?):')
    print(f'  [{trend}]')
    print(f'  Implication: [{impl}]')
    print()
    print('Recommendation for Step 3.2:')
    if any_beats:
        print(f'  Proceed to Step 3.2 — combine best EXIT_SD={best_sd:.1f} '
              f'with threshold entry T=2.5/3.0')
    else:
        fin_saving    = bl['avg_financing_cost'] - b['avg_financing_cost']
        gross_giveup  = abs(gross_delta)
        factor        = ('gross give-up dominates the financing saving'
                         if gross_giveup > fin_saving
                         else 'n_trades loss at higher EXIT_SD reduces total_EV')
        print(f'  Diagnose: {factor}.')
        print('  No EXIT_SD improves on the 5.60 benchmark. Phase 3 closes.')
        print('  Proceed to Phase 4 cross-asset extension.')
    print()
    print('Next step:')
    if any_beats:
        print(f'  Step 3.2 — best exit (EXIT_SD={best_sd:.1f}) + threshold entry grid')
    else:
        print('  Phase 3 closes. Proceed to Phase 4 cross-asset extension.')
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
    inputs_31 = build_31_inputs(main_grid)

    main_grid.to_csv(RESULTS_DIR / 'exit_grid_commodities.csv', index=False)
    per_pair.to_csv( RESULTS_DIR / 'exit_grid_per_pair_commodities.csv', index=False)
    hold_dist.to_csv(RESULTS_DIR / 'exit_grid_holdperiod_commodities.csv', index=False)
    inputs_31.to_csv(RESULTS_DIR / 'exit_grid_31_inputs_commodities.csv', index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")

    print_results(main_grid, hold_dist, df)


if __name__ == '__main__':
    main()
