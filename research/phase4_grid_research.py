"""
phase4_grid_research.py
=======================
Phase 4 — EXIT_SD grid on FX, equities, and fixed income.

Applies the Phase 3 exit threshold grid [0.0, 0.5, 1.0, 1.5] to three non-commodity
asset classes to test whether EXIT_SD=0.5 (commodity optimum, total_EV=7.37) transfers.

Usage (from project root):
    python research/phase4_grid_research.py

Outputs (all in results/):
    phase4_grid_{fx,equities,fi}.csv
    phase4_per_pair_{fx,equities,fi}.csv
    phase4_holdperiod_{fx,equities,fi}.csv
    phase4_inputs_{fx,equities,fi}.csv
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

from peak_analysis_research import build_spread, DATA_CONFIGS, VOL_WINDOW
from trailing_stop_research  import DAILY_FIN, MAX_HOLD, RESULTS_DIR

# ── Constants ─────────────────────────────────────────────────────────────────

XING_SD        = 2.0
EXIT_SD_VALUES = [0.0, 0.5, 1.0, 1.5]

COMM_REF_EXIT_SD  = 0.5
COMM_REF_TOTAL_EV = 7.37
COMM_REF_AVG_NET  = 0.01137
COMM_REF_N        = 648
COMM_REF_HOLD     = 104

W    = 78
SEP  = '═' * W
SEP2 = '─' * W


# ── State machine ─────────────────────────────────────────────────────────────

def simulate_pair(cum, dist_sd, day_ints, long_label, short_label,
                  exit_sd, bid_ask, asset_class_name):
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
                net            = gross - bid_ask - financing_cost
                hold_td        = t - entry_idx

                entry_date = (pd.Timestamp('1970-01-01')
                              + pd.Timedelta(days=int(day_ints[entry_idx])))
                exit_date  = (pd.Timestamp('1970-01-01')
                              + pd.Timedelta(days=int(day_ints[t])))

                trades.append({
                    'asset_class':    asset_class_name,
                    'long_label':     long_label,
                    'short_label':    short_label,
                    'exit_sd':        exit_sd,
                    'entry_date':     entry_date,
                    'exit_date':      exit_date,
                    'side':           side,
                    'gross_return':   gross,
                    'financing_cost': financing_cost,
                    'spread_cost':    bid_ask,
                    'net_return':     net,
                    'hold_cal':       hold_cal,
                    'hold_td':        hold_td,
                    'exit_type':      ('max_hold' if (exit_max and not exit_target)
                                       else 'target'),
                    'entry_dist_sd':  entry_dist_sd,
                })
                in_trade = False

    return trades


# ── Data collection ────────────────────────────────────────────────────────────

def collect_all(prices, pairs, bid_ask, asset_class_name, label):
    all_trades = []
    n_unique   = len(pairs)
    n_dir      = n_unique * 2
    n_exits    = len(EXIT_SD_VALUES)
    print(f"  {n_unique} unique pairs × 2 directions = {n_dir} directional pairs "
          f"× {n_exits} EXIT_SD values...")

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
                trades = simulate_pair(
                    cum, dist_sd, day_ints,
                    long_inst, short_inst, exit_sd, bid_ask, asset_class_name
                )
                all_trades.extend(trades)

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


# ── Summary builders ──────────────────────────────────────────────────────────

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
        avg_spread   = sub['spread_cost'].mean()
        avg_total    = avg_fin + avg_spread
        win_rate     = (sub['net_return'] > 0).mean()
        avg_entry_sd = sub['entry_dist_sd'].mean()
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
            'avg_entry_dist_sd':  avg_entry_sd,
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
        n         = len(sub)
        avg_gross = sub['gross_return'].mean()
        avg_net   = sub['net_return'].mean()
        avg_hold  = sub['hold_cal'].mean()
        win_rate  = (sub['net_return'] > 0).mean()
        total_ev  = avg_net * n
        rows.append({
            'exit_sd':     exit_sd,
            'long_label':  long_label,
            'short_label': short_label,
            'n_trades':    n,
            'avg_gross':   avg_gross,
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


def build_inputs(main_grid):
    df       = main_grid.copy()
    best_idx = df['total_EV'].idxmax()
    df['recommended'] = False
    df.loc[best_idx, 'recommended'] = True
    cols = [
        'exit_sd', 'n_trades', 'avg_net', 'avg_hold_cal', 'avg_entry_dist_sd',
        'win_rate', 'total_EV', 'vs_baseline_EV', 'beats_baseline', 'recommended',
    ]
    return df[[c for c in cols if c in df.columns]]


# ── Printed output ─────────────────────────────────────────────────────────────

def print_asset_class(label, ref_gross, ref_hold, main_grid, hold_dist):
    baseline = main_grid[main_grid['exit_sd'] == 0.0].iloc[0]
    best_row = main_grid.loc[main_grid['total_EV'].idxmax()]
    bl       = baseline
    b        = best_row

    print()
    print(SEP)
    print(f'PHASE 4 — EXIT_SD GRID — {label.upper()}')
    print(SEP)
    print()
    print(f'Reference: Commodity Phase 3 optimum — EXIT_SD={COMM_REF_EXIT_SD}, '
          f'total_EV={COMM_REF_TOTAL_EV:.2f}, avg_net=+{COMM_REF_AVG_NET*100:.3f}%')
    print()

    # Baseline check
    dev_pct = abs(bl['avg_gross'] - ref_gross) / ref_gross * 100 if ref_gross != 0 else 0
    consistency = ('CONSISTENT' if dev_pct <= 20
                   else f'CHECK — {dev_pct:.0f}% deviation from reference gross')
    print('Baseline check (EXIT_SD=0.0 — should approximate prior research):')
    print(f"  n_trades: {int(bl['n_trades'])}  "
          f"avg_gross: {bl['avg_gross']:+.4f}  "
          f"avg_net: {bl['avg_net']:+.4f}  "
          f"avg_hold: {bl['avg_hold_cal']:.0f} d  "
          f"total_EV: {bl['total_EV']:.2f}")
    print(f"  Research reference: avg_gross ~+{ref_gross*100:.2f}%, avg_hold ~{ref_hold} d")
    print(f"  [{consistency}]")
    print()

    # Main results table
    print('Main results:')
    print()
    print(f"  {'EXIT_SD':>7}  {'n_trades':>8}  {'%target':>7}  "
          f"{'avg_gross':>9}  {'avg_net':>8}  {'avg_hold':>8}  "
          f"{'avg_cost':>9}  {'win_rate':>8}  {'avg_entry_SD':>12}  "
          f"{'total_EV':>8}  {'Δ_vs_baseline':>13}")
    print('  ' + SEP2)
    for _, row in main_grid.iterrows():
        delta = ('  baseline' if row['exit_sd'] == 0.0
                 else f"{row['vs_baseline_EV']:>+13.2f}")
        print(
            f"  {row['exit_sd']:>7.1f}  {int(row['n_trades']):>8d}  "
            f"{row['pct_target_exit']*100:>6.1f}%  "
            f"{row['avg_gross']:>+9.4f}  {row['avg_net']:>+8.4f}  "
            f"{row['avg_hold_cal']:>7.0f} d  "
            f"{row['avg_total_cost']*100:>8.4f}%  "
            f"{row['win_rate']*100:>7.1f}%  "
            f"{row['avg_entry_dist_sd']:>10.2f} SD  "
            f"{row['total_EV']:>8.2f}  {delta}"
        )
    print()

    # Hold period distribution
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

    # Gross/net decomposition
    print('Gross/net decomposition:')
    print()
    print(f"  {'EXIT_SD':>7}  {'avg_gross':>9}  {'avg_spread':>10}  "
          f"{'avg_financing':>13}  {'avg_net':>8}  {'gross_retention%':>16}")
    print('  ' + SEP2)
    for _, row in main_grid.iterrows():
        gross_ret = (row['avg_net'] / row['avg_gross'] * 100
                     if row['avg_gross'] != 0 else float('nan'))
        ret_label = '100% (reference)' if row['exit_sd'] == 0.0 else f'{gross_ret:>+.1f}%'
        print(
            f"  {row['exit_sd']:>7.1f}  "
            f"{row['avg_gross']:>+9.4f}  "
            f"{row['avg_spread_cost']*100:>9.4f}%  "
            f"{row['avg_financing_cost']*100:>12.4f}%  "
            f"{row['avg_net']:>+8.4f}  "
            f"{ret_label:>16}"
        )

    # Decision block
    print()
    print(SEP)
    print(f'DECISION — {label.upper()}')
    print(SEP)
    print()

    ev_delta   = b['total_EV']     - bl['total_EV']
    net_delta  = b['avg_net']      - bl['avg_net']
    hold_delta = b['avg_hold_cal'] - bl['avg_hold_cal']
    hold_pct   = hold_delta / bl['avg_hold_cal'] * 100 if bl['avg_hold_cal'] != 0 else 0.0

    print(f"Optimal EXIT_SD for this asset class: {b['exit_sd']:.1f}")
    print(f"  total_EV:   {b['total_EV']:.2f}  "
          f"(vs {bl['total_EV']:.2f} at EXIT_SD=0.0, Δ = {ev_delta:+.2f})")
    print(f"  avg_net:    {b['avg_net']*100:+.4f}%  "
          f"(vs {bl['avg_net']*100:+.4f}% at EXIT_SD=0.0)")
    print(f"  avg_hold:   {b['avg_hold_cal']:.0f} d  "
          f"(vs {bl['avg_hold_cal']:.0f} d at EXIT_SD=0.0, "
          f"Δ = {hold_delta:+.0f} d, {hold_pct:+.1f}%)")
    print(f"  n_trades:   {int(b['n_trades'])}  "
          f"(vs {int(bl['n_trades'])} at EXIT_SD=0.0)")
    print(f"  win_rate:   {b['win_rate']*100:.1f}%")
    print()

    # Does EXIT_SD=0.5 transfer?
    row_05 = main_grid[main_grid['exit_sd'] == 0.5]
    if len(row_05) > 0:
        ev_05      = row_05.iloc[0]['total_EV']
        optimal_sd = b['exit_sd']
        if optimal_sd == 0.5 and ev_05 > bl['total_EV']:
            transfer = 'YES — EXIT_SD=0.5 is also optimal here, total_EV improved'
        elif ev_05 > bl['total_EV']:
            transfer = 'PARTIAL — EXIT_SD=0.5 improves EV but a different EXIT_SD is better'
        else:
            transfer = 'NO — EXIT_SD=0.5 does not improve EV vs baseline for this asset class'
    else:
        transfer = 'UNKNOWN'

    print(f'Does EXIT_SD=0.5 (commodity optimum) transfer to this asset class?')
    print(f'  [{transfer}]')
    print()

    # Shape vs commodity
    ev_by_sd = main_grid.set_index('exit_sd')['total_EV']
    diffs    = [ev_by_sd[EXIT_SD_VALUES[i+1]] - ev_by_sd[EXIT_SD_VALUES[i]]
                for i in range(len(EXIT_SD_VALUES) - 1)]
    if all(d > 0 for d in diffs):
        shape = 'monotone improvement across full grid — optimum may be beyond 1.5'
    elif all(d < 0 for d in diffs):
        shape = 'monotone decline — EXIT_SD=0.0 is optimal'
    else:
        peak_sd = EXIT_SD_VALUES[list(ev_by_sd[EXIT_SD_VALUES]).index(
            max(ev_by_sd[EXIT_SD_VALUES])
        )]
        shape = f'peaks at EXIT_SD={peak_sd:.1f} then declines'

    thin = ref_gross < 0.02
    print('Shape vs commodity:')
    print(f'  Commodity: peaks at EXIT_SD=0.5, falls monotonically at 1.0 and 1.5.')
    print(f'  {label}: {shape}.')
    print(f'  Interpretation: gross per trade ~+{ref_gross*100:.2f}% vs ~+4.83% for commodities — '
          f'{"thin gross margin means financing saving is unlikely to dominate gross give-up" if thin else "comparable gross margin — similar dynamic to commodities expected"}.')
    print()

    # Financing saving vs gross give-up
    fin_saved    = bl['avg_financing_cost'] - b['avg_financing_cost']
    gross_giveup = bl['avg_gross'] - b['avg_gross']
    net_dir      = ('financing saving exceeds gross give-up'
                    if net_delta > 0 else 'gross give-up exceeds financing saving')

    print('Financing saving vs gross give-up:')
    print(f"  At optimal EXIT_SD: financing saved = {fin_saved*100:+.4f}% per trade "
          f"vs EXIT_SD=0.0 baseline.")
    print(f"  Gross given up:     {gross_giveup*100:+.4f}% per trade.")
    print(f"  Net direction:      [{net_dir}]")
    print(f"  Note: with gross expectancy ~+{ref_gross*100:.2f}% "
          f"(vs ~+4.83% for commodities), EXIT_SD=0.5 "
          f"{'cannot' if thin else 'can'} generate enough gross to make the saving meaningful.")
    if asset_label_is_fx(label):
        print(f"  (FX note: actual FX cost advantage is slightly larger than modelled — "
              f"DAILY_FIN overstates FX financing by ~0.21% per trade at a {ref_hold}-day hold.)")
    print()
    print(SEP)


def asset_label_is_fx(label):
    return label.upper() == 'FX'


# ── Cross-asset summary ────────────────────────────────────────────────────────

def print_cross_asset(results):
    print()
    print(SEP)
    print('PHASE 4 — CROSS-ASSET SUMMARY')
    print(SEP)
    print()
    print('Phase 3 + 4 combined optimum by asset class:')
    print()
    print(f"  {'Asset class':<14}  {'Optimal EXIT_SD':>15}  {'n_trades':>8}  "
          f"{'avg_net':>9}  {'avg_hold':>9}  {'total_EV':>8}  {'Δ_vs_baseline':>15}")
    print('  ' + SEP2)

    # Commodity anchor row (Phase 3 result)
    print(f"  {'Commodities':<14}  {COMM_REF_EXIT_SD:>15.1f}  {COMM_REF_N:>8d}  "
          f"{COMM_REF_AVG_NET*100:>+8.3f}%  {COMM_REF_HOLD:>7d} d  "
          f"{COMM_REF_TOTAL_EV:>8.2f}  +1.77 (+31.5%)")

    for r in results:
        ev_delta    = r['vs_baseline_EV']
        baseline_ev = r['total_EV'] - ev_delta
        ev_pct      = (ev_delta / abs(baseline_ev) * 100) if baseline_ev != 0 else 0.0
        print(
            f"  {r['label']:<14}  {r['best_exit_sd']:>15.1f}  {r['n_trades']:>8d}  "
            f"{r['avg_net']*100:>+8.3f}%  {r['avg_hold']:>7.0f} d  "
            f"{r['total_EV']:>8.2f}  {ev_delta:+.2f} ({ev_pct:+.1f}%)"
        )

    print()

    n_optimal_05 = sum(1 for r in results if r['best_exit_sd'] == 0.5) + 1  # +1 commodity
    n_total      = len(results) + 1
    if n_optimal_05 == n_total:
        transfer_verdict = 'Yes — optimal for all asset classes'
    elif n_optimal_05 > 1:
        transfer_verdict = (f'Partially — optimal for {n_optimal_05}/{n_total} classes, '
                            f'see per-class decisions above')
    else:
        transfer_verdict = 'No — commodity-specific result'

    print('Does EXIT_SD=0.5 transfer universally?')
    print(f'  [{transfer_verdict}]')
    print()

    any_positive = any(r['avg_net'] > 0 for r in results)
    viable       = [r['label'] for r in results if r['avg_net'] > 0]
    not_viable   = [r['label'] for r in results if r['avg_net'] <= 0]

    print('Key structural finding:')
    print('  The EXIT_SD=0.5 improvement for commodities is driven by high gross per trade')
    print('  (+4.83%). For FX (+0.49%) and equities (+0.36%), gross is 10–13x smaller —')
    print('  the financing saving from early exit may be insufficient to compensate.')
    if any_positive:
        print(f'  [Partially confirmed: EXIT_SD optimisation helps {", ".join(viable)} '
              f'achieve positive avg_net; {", ".join(not_viable) if not_viable else "none"} '
              f'remain negative.]')
    else:
        print('  [Confirmed: no non-commodity asset class achieves positive avg_net at any EXIT_SD.]')
    print()

    print('Phase 4 conclusion:')
    if viable:
        print(f'  EXIT_SD optimisation lifts total_EV for all classes. '
              f'{", ".join(viable)} achieve positive avg_net at optimal EXIT_SD.')
        print(f'  {", ".join(not_viable) if not_viable else "No classes"} remain structurally '
              f'unviable at the 4.88% CFD financing rate regardless of exit threshold.')
    else:
        print('  Commodities remain the uniquely viable asset class at retail CFD rates.')
        print('  EXIT_SD optimisation improves total_EV for all classes but does not overcome')
        print('  the structural financing barrier for low-gross asset classes (FX, equities, FI).')
    print()
    print('Next step: Phase 5 — strict crossing vs level-check (deferred; max-hold re-entry')
    print('rate at Phase 3 optimum is 6.8% — effect is small; test only after Phase 4 confirmed).')
    print()
    print(SEP)


# ── Run one asset class ────────────────────────────────────────────────────────

def run_asset_class(name, label, prices, bid_ask, ref_gross, ref_hold):
    print(f"\nRunning {label}...")

    tickers = sorted(prices.columns)
    pairs   = [(a, b) for a, b in itertools.combinations(tickers, 2)]

    df = collect_all(prices, pairs, bid_ask, name, label)

    if df.empty:
        print(f"  No trades generated for {label}. Check data.")
        return None

    main_grid = build_main_grid(df)
    per_pair  = build_per_pair(df)
    hold_dist = build_hold_distribution(df)
    inputs    = build_inputs(main_grid)

    main_grid.to_csv(RESULTS_DIR / f'phase4_grid_{name}.csv',       index=False)
    per_pair.to_csv( RESULTS_DIR / f'phase4_per_pair_{name}.csv',   index=False)
    hold_dist.to_csv(RESULTS_DIR / f'phase4_holdperiod_{name}.csv', index=False)
    inputs.to_csv(   RESULTS_DIR / f'phase4_inputs_{name}.csv',     index=False)

    print_asset_class(label, ref_gross, ref_hold, main_grid, hold_dist)

    best_row = main_grid.loc[main_grid['total_EV'].idxmax()]
    bl       = main_grid[main_grid['exit_sd'] == 0.0].iloc[0]

    return {
        'label':          label,
        'best_exit_sd':   best_row['exit_sd'],
        'n_trades':       int(best_row['n_trades']),
        'avg_net':        best_row['avg_net'],
        'avg_hold':       best_row['avg_hold_cal'],
        'total_EV':       best_row['total_EV'],
        'vs_baseline_EV': best_row['vs_baseline_EV'],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('\nLoading data...')

    fx_prices = pd.read_csv(ROOT / 'cache' / 'fx_prices.csv',
                            parse_dates=['Date'], index_col='Date')
    eq_prices = pd.read_csv(ROOT / 'cache' / 'prices.csv',
                            parse_dates=['Date'], index_col='Date')
    fi_prices = pd.read_csv(ROOT / 'cache' / 'fi_prices.csv',
                            parse_dates=['Date'], index_col='Date')

    fi_prices = fi_prices.drop(
        columns=['IBTM', 'UST10Y', 'UST30Y', 'UST5Y'], errors='ignore'
    )

    print(f"  FX instruments ({len(fx_prices.columns)}):           {sorted(fx_prices.columns)}")
    print(f"  Equity instruments ({len(eq_prices.columns)}):       {sorted(eq_prices.columns)}")
    print(f"  Fixed income instruments ({len(fi_prices.columns)}): {sorted(fi_prices.columns)}")

    cross_asset = []

    r = run_asset_class(
        name='fx', label='FX', prices=fx_prices,
        bid_ask=DATA_CONFIGS['fx']['bid_ask'],
        ref_gross=0.0049, ref_hold=189,
    )
    if r:
        cross_asset.append(r)

    r = run_asset_class(
        name='equities', label='Equities', prices=eq_prices,
        bid_ask=DATA_CONFIGS['equities']['bid_ask'],
        ref_gross=0.0036, ref_hold=135,
    )
    if r:
        cross_asset.append(r)

    r = run_asset_class(
        name='fi', label='Fixed Income', prices=fi_prices,
        bid_ask=0.0004,
        ref_gross=0.0039, ref_hold=129,
    )
    if r:
        cross_asset.append(r)

    if cross_asset:
        print_cross_asset(cross_asset)

    print(f'\nResults saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
