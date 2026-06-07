"""
phase4b_grid_research.py
========================
Phase 4b — Equity EXIT_SD extension to [2.0, 2.5].

Phase 4 showed monotone improvement through EXIT_SD=1.5 with no peak in range.
Phase 4b extends to EXIT_SD=[2.0, 2.5] to confirm EXIT_SD=1.5 as the equity
scalp regime optimum and characterise the degenerate regimes at 2.0 and 2.5.

Usage (from project root):
    python research/phase4b_grid_research.py

Outputs (all in results/):
    phase4b_grid_equities.csv
    phase4b_holdperiod_equities.csv
    phase4b_inputs_equities.csv
"""

import sys
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

RESEARCH_DIR = Path(__file__).parent
ROOT         = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

from peak_analysis_research import build_spread, load_prices, DATA_CONFIGS, VOL_WINDOW
from trailing_stop_research  import DAILY_FIN, MAX_HOLD, RESULTS_DIR

# ── Constants ─────────────────────────────────────────────────────────────────

XING_SD        = 2.0
EXIT_SD_VALUES = [1.5, 2.0, 2.5]   # 1.5 = replication check; 2.0 and 2.5 are new
BID_ASK        = DATA_CONFIGS['equities']['bid_ask']  # 0.0014

# Phase 4 reference values (hardcoded for printed table and delta columns)
P4_REF_15_TOTAL_EV   =  48.07
P4_REF_00_TOTAL_EV   = -55.69
P4_REF_15_N          =  9829
P4_REF_15_AVG_NET    =  0.00489
P4_REF_15_AVG_HOLD   =  28.0

W    = 78
SEP  = '═' * W
SEP2 = '─' * W


# ── State machine ─────────────────────────────────────────────────────────────

def simulate_pair(cum, dist_sd, day_ints, exit_sd):
    trades   = []
    in_trade = False

    for t in range(VOL_WINDOW, len(dist_sd)):
        d = dist_sd[t]
        if np.isnan(d):
            continue

        if not in_trade:
            if abs(d) > XING_SD:
                in_trade      = True
                entry_idx     = t
                entry_cum     = cum[t]
                side          = +1 if d < -XING_SD else -1
                entry_dist_sd = abs(d)
        else:
            hold_cal    = int(day_ints[t] - day_ints[entry_idx])
            exit_target = ((side == -1 and d <= exit_sd)
                           or (side == +1 and d >= -exit_sd))
            exit_max    = hold_cal >= MAX_HOLD

            if exit_target or exit_max:
                gross          = ((cum[t] - entry_cum) / entry_cum if side == +1
                                  else (entry_cum - cum[t]) / entry_cum)
                financing_cost = DAILY_FIN * hold_cal
                net            = gross - BID_ASK - financing_cost

                entry_date = (pd.Timestamp('1970-01-01')
                              + pd.Timedelta(days=int(day_ints[entry_idx])))
                exit_date  = (pd.Timestamp('1970-01-01')
                              + pd.Timedelta(days=int(day_ints[t])))

                trades.append({
                    'exit_sd':        exit_sd,
                    'entry_date':     entry_date,
                    'exit_date':      exit_date,
                    'side':           side,
                    'gross_return':   gross,
                    'financing_cost': financing_cost,
                    'spread_cost':    BID_ASK,
                    'net_return':     net,
                    'hold_cal':       hold_cal,
                    'exit_type':      ('max_hold' if (exit_max and not exit_target)
                                       else 'target'),
                    'entry_dist_sd':  entry_dist_sd,
                })
                in_trade = False

    return trades


# ── Data collection ────────────────────────────────────────────────────────────

def collect_all(prices):
    tickers    = sorted(prices.columns)
    pairs      = list(itertools.combinations(tickers, 2))
    n_unique   = len(pairs)
    n_dir      = n_unique * 2
    print(f"  {n_unique} unique pairs × 2 directions = {n_dir} directional pairs "
          f"× {len(EXIT_SD_VALUES)} EXIT_SD values...")

    all_trades = []
    for a, b in pairs:
        for long_inst, short_inst in [(a, b), (b, a)]:
            if long_inst not in prices.columns or short_inst not in prices.columns:
                continue
            try:
                _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
            except Exception as e:
                print(f"    SKIP {long_inst}/{short_inst}: {e}")
                continue

            for exit_sd in EXIT_SD_VALUES:
                trades = simulate_pair(cum, dist_sd, day_ints, exit_sd)
                all_trades.extend(trades)

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


# ── Summary builders ──────────────────────────────────────────────────────────

def build_main_grid(df):
    rows = []
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
        avg_spread   = sub['spread_cost'].mean()
        avg_total    = avg_fin + avg_spread
        win_rate     = (sub['net_return'] > 0).mean()
        avg_entry_sd = sub['entry_dist_sd'].mean()
        total_ev     = avg_net * n

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
            'avg_entry_dist_sd':  avg_entry_sd,
            'total_EV':           total_ev,
            'vs_15_EV':           total_ev - P4_REF_15_TOTAL_EV,
            'vs_baseline_EV':     total_ev - P4_REF_00_TOTAL_EV,
        })

    return pd.DataFrame(rows)


def build_hold_distribution(df):
    rows = []
    for exit_sd in EXIT_SD_VALUES:
        sub = df[df['exit_sd'] == exit_sd]
        if len(sub) == 0:
            rows.append({'exit_sd': exit_sd, 'n_trades': 0})
            continue
        hold          = sub['hold_cal']
        pct_max_hold  = (sub['exit_type'] == 'max_hold').mean()
        pct_target    = (sub['exit_type'] == 'target').mean()
        rows.append({
            'exit_sd':            exit_sd,
            'n_trades':           len(sub),
            'hold_p10':           hold.quantile(0.10),
            'hold_p25':           hold.quantile(0.25),
            'hold_p50':           hold.quantile(0.50),
            'hold_p75':           hold.quantile(0.75),
            'hold_p90':           hold.quantile(0.90),
            'hold_mean':          hold.mean(),
            'pct_max_hold_exits': pct_max_hold,
            'pct_target_exits':   pct_target,
        })
    return pd.DataFrame(rows)


def build_inputs(main_grid):
    rows = []
    for _, row in main_grid.iterrows():
        exit_sd = row['exit_sd']
        rows.append({
            'exit_sd':       exit_sd,
            'n_trades':      row['n_trades'],
            'avg_net':       row['avg_net'],
            'avg_hold_cal':  row['avg_hold_cal'],
            'win_rate':      row['win_rate'],
            'total_EV':      row['total_EV'],
            'vs_15_EV':      row['vs_15_EV'],
            'is_degenerate': exit_sd in (2.0, 2.5),
            'zero_trades':   False,
        })
    return pd.DataFrame(rows)


# ── Printed output ─────────────────────────────────────────────────────────────

# Phase 4 reference rows for the full printed table (0.0, 0.5, 1.0)
P4_ROWS = [
    (0.0,  4375,  76.2, +0.5520, -1.2730, 154, 60.5, -55.69, -103.76, '—'),
    (0.5,  5031,  89.8, +0.7910, -0.5500, 110, 65.4, -27.65,  -75.72, '—'),
    (1.0,  6430,  98.3, +0.9230, +0.0840,  64, 70.1,  +5.42,  -42.65, '—'),
    (1.5,  9829, 100.0, +0.9350, +0.4890,  28, 75.4, +48.07,    0.00, '—'),
]

P4_HOLD_ROWS = [
    (0.0,  22,  50, 130, 288, 300, 154, 23.8, 76.2),
    (0.5,  11,  29,  76, 175, 300, 110, 10.2, 89.8),
    (1.0,   5,  14,  37,  91, 165,  64,  1.7, 98.3),
    (1.5,   2,   5,  14,  37,  76,  28,  0.0, 100.0),
]


def flag_for(exit_sd):
    if exit_sd == 2.0:
        return '⚠️ DEGENERATE (exit on bar t+1)'
    if exit_sd == 2.5:
        return '⚠️ DOUBLY DEGENERATE (exit before reversion begins)'
    return '—'


def print_results(main_grid, hold_dist, repl_ev):
    repl_pass = abs(repl_ev - P4_REF_15_TOTAL_EV) / abs(P4_REF_15_TOTAL_EV) <= 0.02
    repl_label = 'PASS' if repl_pass else 'FAIL'

    print()
    print(SEP)
    print('PHASE 4b — EQUITY EXIT_SD EXTENSION — EXIT_SD=[2.0, 2.5]')
    print(SEP)
    print()
    print('Phase 4 reference (EXIT_SD=1.5 optimum-in-range):')
    print(f'  n_trades={P4_REF_15_N}  avg_net=+{P4_REF_15_AVG_NET*100:.3f}%  '
          f'avg_hold={P4_REF_15_AVG_HOLD:.0f}d  total_EV=+{P4_REF_15_TOTAL_EV:.2f}')
    print()
    print('Replication check (EXIT_SD=1.5 — must match within ±2% total_EV):')
    print(f'  Replicated total_EV: {repl_ev:.2f}  [{repl_label}]')
    if not repl_pass:
        print(f'  ⚠️ FAIL — diverges by '
              f'{abs(repl_ev - P4_REF_15_TOTAL_EV)/abs(P4_REF_15_TOTAL_EV)*100:.1f}% '
              f'from Phase 4 reference. Investigate before interpreting 2.0 and 2.5.')
    print()
    print('Full equity EXIT_SD curve (Phase 4 + Phase 4b):')
    print()
    hdr = (f"  {'EXIT_SD':>7}  {'n_trades':>8}  {'%target':>7}  "
           f"{'avg_gross':>9}  {'avg_net':>8}  {'avg_hold':>8}  "
           f"{'win_rate':>8}  {'total_EV':>8}  {'Δ_vs_1.5':>9}  Flag")
    print(hdr)
    print('  ' + SEP2)

    # Phase 4 reference rows
    for (sd, n, pct, gross, net, hold, wr, ev, delta, flg) in P4_ROWS:
        delta_str = 'reference' if sd == 1.5 else f'{delta:+.2f}'
        print(f"  {sd:>7.1f}  {n:>8d}  {pct:>6.1f}%  "
              f"{gross/100:>+9.4f}  {net/100:>+8.4f}  {hold:>7.0f} d  "
              f"{wr:>7.1f}%  {ev:>8.2f}  {delta_str:>9}  {flg}")

    # Phase 4b computed rows
    for _, row in main_grid[main_grid['exit_sd'] > 1.5].iterrows():
        flg = flag_for(row['exit_sd'])
        print(f"  {row['exit_sd']:>7.1f}  {int(row['n_trades']):>8d}  "
              f"{row['pct_target_exit']*100:>6.1f}%  "
              f"{row['avg_gross']:>+9.4f}  {row['avg_net']:>+8.4f}  "
              f"{row['avg_hold_cal']:>7.0f} d  "
              f"{row['win_rate']*100:>7.1f}%  "
              f"{row['total_EV']:>8.2f}  {row['vs_15_EV']:>+9.2f}  {flg}")

    print()
    print('Hold period distribution:')
    print()
    print(f"  {'EXIT_SD':>7}  {'p10':>4}  {'p25':>4}  {'p50':>4}  "
          f"{'p75':>4}  {'p90':>4}  {'mean':>4}  {'%max_hold':>9}  {'%target':>7}")
    print('  ' + SEP2)

    for (sd, p10, p25, p50, p75, p90, mn, pct_mh, pct_tg) in P4_HOLD_ROWS:
        print(f"  {sd:>7.1f}  {p10:>4.0f}d  {p25:>4.0f}d  {p50:>4.0f}d  "
              f"{p75:>4.0f}d  {p90:>4.0f}d  {mn:>4.0f}d  "
              f"{pct_mh:>8.1f}%  {pct_tg:>6.1f}%")

    for _, row in hold_dist[hold_dist['exit_sd'] > 1.5].iterrows():
        note = ('← ~1–2d, immediate exit' if row['exit_sd'] == 2.0
                else '← ~1d, exit before reversion')
        print(f"  {row['exit_sd']:>7.1f}  "
              f"{row['hold_p10']:>4.0f}d  {row['hold_p25']:>4.0f}d  "
              f"{row['hold_p50']:>4.0f}d  {row['hold_p75']:>4.0f}d  "
              f"{row['hold_p90']:>4.0f}d  {row['hold_mean']:>4.0f}d  "
              f"{row['pct_max_hold_exits']*100:>8.1f}%  "
              f"{row['pct_target_exits']*100:>6.1f}%  {note}")

    print()
    print(SEP)
    print('PHASE 4b DECISION')
    print(SEP)
    print()

    for _, row in main_grid[main_grid['exit_sd'] > 1.5].iterrows():
        exit_sd  = row['exit_sd']
        label    = 'DEGENERATE' if exit_sd == 2.0 else 'DOUBLY DEGENERATE'
        expected = '~1–2d — degenerate immediate exit' if exit_sd == 2.0 else '~1d — exit before reversion'
        degen_confirmed = 'YES' if row['avg_hold_cal'] <= 5 else 'NO — investigate'

        print(f"EXIT_SD={exit_sd:.1f} result ({label}):")
        print(f"  n_trades:   {int(row['n_trades'])}")
        print(f"  avg_hold:   {row['avg_hold_cal']:.1f} d  (expected: {expected})")
        print(f"  avg_gross:  {row['avg_gross']:+.4f}  ({row['avg_gross']*100:+.3f}%)")
        print(f"  avg_net:    {row['avg_net']:+.4f}  ({row['avg_net']*100:+.3f}%)  "
              f"(expected: ~−BID_ASK ≈ −0.14%)")
        print(f"  total_EV:   {row['total_EV']:.2f}")
        print(f"  Δ vs EXIT_SD=1.5:  {row['vs_15_EV']:+.2f}")
        print(f"  Degenerate confirmed?  [{degen_confirmed}]")
        if exit_sd == 2.0:
            direction = 'rises vs 1.5' if row['total_EV'] > P4_REF_15_TOTAL_EV else 'falls vs 1.5'
            print(f"  Interpretation: total_EV {direction}. "
                  f"At ~1-bar hold, only cost is BID_ASK=0.14%; gross ≈ 0. "
                  f"{'Unexpected — check logic.' if row['total_EV'] > P4_REF_15_TOTAL_EV else 'Expected negative net.'}")
        else:
            print(f"  Interpretation: EXIT_SD=2.5 is more degenerate than 2.0 — exit condition "
                  f"already satisfied at d_entry≈2.22 ≤ 2.5 from bar t+1. "
                  f"{'Higher n_trades amplifies negative total_EV.' if int(row['n_trades']) > P4_REF_15_N else 'n_trades similar to 1.5.'}")
        print()

    # Identify peak
    computed_evs = {row['exit_sd']: row['total_EV'] for _, row in main_grid.iterrows()}
    all_evs = {**{sd: ev for (sd, _, _, _, _, _, _, ev, _, _) in P4_ROWS}, **computed_evs}
    peak_sd = max(all_evs, key=all_evs.get)
    peak_ev = all_evs[peak_sd]

    row_20 = main_grid[main_grid['exit_sd'] == 2.0].iloc[0]
    row_25 = main_grid[main_grid['exit_sd'] == 2.5].iloc[0]
    both_degenerate = row_20['total_EV'] < P4_REF_15_TOTAL_EV and row_25['total_EV'] < P4_REF_15_TOTAL_EV
    optimum_confirmed = 'YES' if (peak_sd == 1.5 and both_degenerate) else 'NO — investigate'

    print('Complete equity EXIT_SD curve — peak identification:')
    print(f'  Peak EXIT_SD by total_EV: {peak_sd:.1f} (total_EV = {peak_ev:.2f})')
    evs_sorted = [all_evs[sd] for sd in sorted(all_evs)]
    mono_to_15 = all(evs_sorted[i] < evs_sorted[i+1] for i in range(3))
    collapses   = row_20['total_EV'] < P4_REF_15_TOTAL_EV
    more_neg    = row_25['total_EV'] < row_20['total_EV']
    shape_parts = []
    if mono_to_15:
        shape_parts.append('monotone improvement through EXIT_SD=1.5')
    if collapses:
        shape_parts.append('collapses at EXIT_SD=2.0')
    if more_neg:
        shape_parts.append('more negative at EXIT_SD=2.5')
    print(f"  Shape: {', '.join(shape_parts) if shape_parts else 'see data'}")
    print(f'  Is EXIT_SD=1.5 confirmed as the practical optimum?  [{optimum_confirmed}]')
    if both_degenerate:
        print('  Both EXIT_SD=2.0 and 2.5 confirm EXIT_SD=1.5 as peak — both are degenerate '
              'immediate-exit regimes with large negative total_EV.')
    print()

    print('Scalp regime characterisation (EXIT_SD=1.5 optimum):')
    print('  avg_hold: 28d (median 14d) — exits within ~2 weeks of a 2 SD dislocation')
    print('  win_rate: 75.4% — most dislocations partially revert quickly')
    print('  avg_net: +0.489% per trade — positive because financing cost ~0.31% over 28d')
    print('  is below avg_gross ~0.94%')
    print('  Strategy identity: This is NOT the WF-validated medium-term pairs trade. This is a')
    print('  short-term mean-reversion scalp from a 2 SD dislocation. The WF pipeline (Tab 9/10/11)')
    print('  was not designed for this regime and has not validated it. Walk-forward validation')
    print('  would require a separate IS/OOS framework appropriate to ~2-week hold periods.')
    print()

    print('Recommendation:')
    if peak_sd == 1.5:
        print('  EXIT_SD=1.5 is the equity scalp regime optimum. Walk-forward validation required')
        print('  before live consideration — this is a different strategy from the commodity/')
        print('  cross-asset regime validated by the existing WF pipeline.')
    else:
        print(f'  ⚠️ EXIT_SD={peak_sd:.1f} beats 1.5 — contradicts degenerate-exit expectation.')
        print('  Warrant a logic check before accepting the result.')
    print()

    print('Phase 4 / 4b complete. Equity EXIT_SD curve fully characterised.')
    print()
    print('Next step: Phase 5 — strict crossing vs level-check (commodities only,')
    print('EXIT_SD=0.5, XING_SD=2.0; max-hold re-entry rate 6.8% — effect expected small).')
    print()
    print(SEP)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('\nLoading equity prices...')
    prices = load_prices('equities')
    print(f"  Instruments ({len(prices.columns)}): {sorted(prices.columns)}")
    print(f"  BID_ASK: {BID_ASK:.4f}  XING_SD: {XING_SD}  EXIT_SD_VALUES: {EXIT_SD_VALUES}")
    print()

    print('Running state machine...')
    df = collect_all(prices)

    if df.empty:
        print('No trades generated. Check data and config.')
        return

    print(f'  Total trade records: {len(df)}')
    print()

    main_grid = build_main_grid(df)
    hold_dist = build_hold_distribution(df)
    inputs    = build_inputs(main_grid)

    # Replication EV from EXIT_SD=1.5 row
    repl_row = main_grid[main_grid['exit_sd'] == 1.5]
    repl_ev  = repl_row.iloc[0]['total_EV'] if len(repl_row) > 0 else float('nan')

    main_grid.to_csv(RESULTS_DIR / 'phase4b_grid_equities.csv',       index=False)
    hold_dist.to_csv(RESULTS_DIR / 'phase4b_holdperiod_equities.csv', index=False)
    inputs.to_csv(   RESULTS_DIR / 'phase4b_inputs_equities.csv',     index=False)

    print_results(main_grid, hold_dist, repl_ev)

    print(f'\nResults saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
