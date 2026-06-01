"""
velocity_retest_research.py
===========================
Phase 2, Step 2.1 — Short-lookback velocity reversal retest.

Phase 1 confirmed vel_5 >= 0 at peak in 100% of trades (the spread turns too
fast for a 5-day window to detect deceleration).  This script retests with
1-day and 2-day lookbacks only to determine whether the velocity reversal
signal is viable at shorter resolution before abandoning it entirely.

Part A: Diagnostic — % of trades where vel_1 / vel_2 < 0 at and before peak.
Part B: Simulation — delayed entry until vel_N first turns negative; compare
        net return and hold vs. baseline.
Part C: Decision gate — verdict on signal viability.

Inputs:
  results/peak_trades_full_commodities.csv   (Phase 1 output — provides peak_step)

Outputs:
  results/velocity_retest_diagnostic.csv     — pct_neg / mean_vel by lookback & offset
  results/velocity_retest_sim_lb1.csv        — per-trade sim results, lookback = 1
  results/velocity_retest_sim_lb2.csv        — per-trade sim results, lookback = 2
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).parent
ROOT         = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

# trailing_stop_research: only what is directly used in this script
# DAILY_FIN: used in Part B net formula (net = gross - BID_ASK - DAILY_FIN * hold_cal)
# EXIT_SD:   used in Part B exit scan (dist_path[k] <= EXIT_SD / >= EXIT_SD)
# MAX_HOLD:  used in Part B exit scan (hold_cal >= MAX_HOLD)
# RESULTS_DIR: used to locate peak_trades_full_commodities.csv
from trailing_stop_research import (
    DAILY_FIN, EXIT_SD, MAX_HOLD, RESULTS_DIR,
)

# peak_analysis_research: all shared utilities — import directly, do NOT redefine.
# These functions use DATA_CONFIGS internally; redefining them here without DATA_CONFIGS
# will cause NameError at runtime.
# find_peak is NOT imported — peak_step comes from the Phase 1 CSV join, not find_peak.
# VOL_WINDOW is imported (not redefined) — run_baseline_with_paths uses its own module's value.
from peak_analysis_research import (
    run_baseline_with_paths,   # uses DATA_CONFIGS[asset_class]['bid_ask'] internally
    compute_velocity_at_step,  # used for vel_1/vel_2 computation
    load_prices,               # load commodity price data via DATA_CONFIGS
    COMMODITY_PAIRS,           # canonical pair list — do NOT redefine
    DATA_CONFIGS,              # needed for bid_ask in gross/net formula
    VOL_WINDOW,                # 262 — defined in peak_analysis_research; don't redefine
)

W    = 78
SEP  = '=' * W
SEP2 = '-' * W

PHASE1_CSV = RESULTS_DIR / 'peak_trades_full_commodities.csv'


def _pct(arr, condition_fn):
    """Fraction of non-NaN elements satisfying condition_fn."""
    valid = [x for x in arr if not np.isnan(x)]
    if not valid:
        return np.nan
    return sum(1 for x in valid if condition_fn(x)) / len(valid)


def _fmt_pct(v):
    return f"{v:.1%}" if not np.isnan(v) else "  N/A"


# ==============================================================================
# Step 0 — Collect baseline trades with paths; join Phase 1 peak_step
# ==============================================================================

def collect_trades_with_peak(asset_class='commodities'):
    """
    Re-run run_baseline_with_paths to obtain dist_path / cum_path / dayint_path,
    then join with Phase 1 CSV on (pair, entry_idx, exit_idx) to attach peak_step.
    """
    if not PHASE1_CSV.exists():
        raise FileNotFoundError(
            f"Phase 1 CSV not found: {PHASE1_CSV}\n"
            "Run peak_analysis_research.py first."
        )

    phase1      = pd.read_csv(PHASE1_CSV)
    peak_lookup = {}
    for _, row in phase1.iterrows():
        key = (row['pair'], int(row['entry_idx']), int(row['exit_idx']))
        peak_lookup[key] = int(row['peak_step'])

    prices = load_prices(asset_class)
    cfg    = DATA_CONFIGS[asset_class]

    all_trades = []
    for (long_inst, short_inst) in cfg['pairs']:
        if long_inst not in prices.columns or short_inst not in prices.columns:
            print(f"  Skipping {long_inst}/{short_inst} — not in data")
            continue
        trades = run_baseline_with_paths(long_inst, short_inst, prices, asset_class)
        for t in trades:
            key           = (t['pair'], int(t['entry_idx']), int(t['exit_idx']))
            t['peak_step'] = peak_lookup.get(key)
        all_trades.extend(trades)

    n_missing = sum(1 for t in all_trades if t['peak_step'] is None)
    print(f"\n  {asset_class}: {len(all_trades)} trades collected, "
          f"{n_missing} unmatched in Phase 1 CSV")
    return all_trades


# ==============================================================================
# Part A — Short-lookback velocity diagnostic at and before peak
# ==============================================================================

def part_a_diagnostic(trades, asset_class='commodities'):
    """
    Compute vel_1 and vel_2 at peak (offset=0) and up to 3 steps before peak
    (offset=1,2,3).  Report % negative and mean per cell.

    Modifies trades in-place — adds vel_{lb}_at_peak_minus_{offset} keys.
    Returns (trades, diag_rows) where diag_rows is a list of dicts for CSV output.
    """
    print(f"\n{SEP}")
    print(f"PART A -- SHORT-LOOKBACK VELOCITY AT PEAK  ({asset_class})")
    print(f"Phase 1 reference: vel_5 < 0 at peak = 0.0% (all positive at 5-day)")
    print(f"{SEP}")

    for t in trades:
        ps   = t.get('peak_step')
        dp   = t['dist_path']
        side = t['side']
        for lb in [1, 2]:
            for offset in [0, 1, 2, 3]:
                key  = f'vel_{lb}_at_peak_minus_{offset}'
                step = (ps - offset) if ps is not None else -1
                if step < 0:
                    t[key] = np.nan
                else:
                    t[key] = compute_velocity_at_step(dp, step, side, lb)

    print(f"\n  {'lookback':<10}  {'offset':<14}  {'n_valid':>7}  "
          f"{'pct_neg':>8}  {'mean_vel':>9}  {'median_vel':>10}")
    print(f"  {'-'*65}")

    diag_rows = []
    for lb in [1, 2]:
        for offset in [0, 1, 2, 3]:
            key   = f'vel_{lb}_at_peak_minus_{offset}'
            vals  = [t[key] for t in trades if not np.isnan(t[key])]
            nv    = len(vals)
            pneg  = _pct(vals, lambda x: x < 0)
            mv    = np.mean(vals)   if vals else np.nan
            medv  = np.median(vals) if vals else np.nan
            label = 'at peak' if offset == 0 else f'peak-{offset}'
            print(f"  {'vel_'+str(lb)+'d':<10}  {label:<14}  {nv:>7}  "
                  f"{_fmt_pct(pneg):>8}  {mv:>+9.3f}  {medv:>+10.3f}")
            diag_rows.append({
                'lookback':   lb,
                'offset':     offset,
                'n_valid':    nv,
                'pct_neg':    pneg,
                'mean_vel':   mv,
                'median_vel': medv,
            })

    out  = pd.DataFrame(diag_rows)
    path = RESULTS_DIR / 'velocity_retest_diagnostic.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}")

    return trades, diag_rows


# ==============================================================================
# Part B — Simulated velocity-entry
# ==============================================================================

def simulate_velocity_entry(trade, lookback, bid_ask):
    """
    Within the baseline dist_path, scan forward from step `lookback` and enter
    when vel_N first turns negative (velocity reversal signal fires).

    Exit when:
      - dist_path reaches EXIT_SD (normal exit), or
      - hold days from new entry >= MAX_HOLD, or
      - path ends (original trade window closes before either condition met).

    Returns a result dict, or None if vel_N never turns negative.
    """
    dp   = trade['dist_path']
    cp   = trade['cum_path']
    dip  = trade['dayint_path']
    side = trade['side']

    # Find first step where vel_N < 0
    new_entry_step = None
    for k in range(lookback, len(dp)):
        v = compute_velocity_at_step(dp, k, side, lookback)
        if not np.isnan(v) and v < 0:
            new_entry_step = k
            break

    if new_entry_step is None:
        return None

    # Scan for exit from new_entry_step + 1 onward
    result_k     = None
    max_hit_flag = False
    for k in range(new_entry_step + 1, len(dp)):
        hold_cal    = dip[k] - dip[new_entry_step]
        normal_exit = (side == -1 and dp[k] <= EXIT_SD) or \
                      (side == +1 and dp[k] >= EXIT_SD)
        max_hit     = hold_cal >= MAX_HOLD
        if normal_exit or max_hit:
            result_k     = k
            max_hit_flag = max_hit
            break

    # If path ends before exit triggers (original window closed),
    # record result at last step — most common for max-hold baseline trades
    # where delayed entry reduces apparent hold below MAX_HOLD.
    if result_k is None:
        result_k = len(dp) - 1

    hold_cal     = dip[result_k] - dip[new_entry_step]
    max_hit_flag = max_hit_flag or (hold_cal >= MAX_HOLD)

    gross = ((cp[result_k] - cp[new_entry_step]) / cp[new_entry_step]) if side == +1 \
            else ((cp[new_entry_step] - cp[result_k]) / cp[new_entry_step])
    net   = gross - bid_ask - DAILY_FIN * hold_cal

    return {
        'pair':           trade['pair'],
        'side':           side,
        'bl_net':         trade['net_final'],
        'bl_hold':        trade['total_hold_cal'],
        'entry_step_vel': new_entry_step,
        'vel_hold':       hold_cal,
        'vel_gross':      gross,
        'vel_net':        net,
        'is_winner_vel':  net > 0,
        'max_hold_exit':  max_hit_flag,
    }


def part_b_simulation(trades, asset_class='commodities'):
    print(f"\n{SEP}")
    print(f"PART B -- SIMULATED VELOCITY-ENTRY  ({asset_class})")
    print(f"Enter when vel_N first turns negative within dist_path.")
    print(f"Exit at EXIT_SD={EXIT_SD} or MAX_HOLD={MAX_HOLD} d (same rules as baseline).")
    print(f"{SEP}")

    bid_ask = DATA_CONFIGS[asset_class]['bid_ask']
    n_bl    = len(trades)

    sim_results = {}
    for lb in [1, 2]:
        print(f"\nLookback = {lb} day{'s' if lb > 1 else ''}")
        print(SEP2)

        results  = []
        skipped  = 0
        for t in trades:
            r = simulate_velocity_entry(t, lb, bid_ask)
            if r is None:
                skipped += 1
            else:
                results.append(r)

        n_sig = len(results)
        print(f"\n  Baseline trades:              {n_bl}")
        print(f"  Signal fired (vel_N < 0):     {n_sig:>5}  ({n_sig / n_bl:.1%})")
        print(f"  No signal (vel never neg):    {skipped:>5}  ({skipped / n_bl:.1%})")

        if not results:
            print(f"\n  No results — signal never fires at {lb}-day resolution.")
            sim_results[lb] = []
            continue

        bl_nets    = [r['bl_net']        for r in results]
        vel_nets   = [r['vel_net']        for r in results]
        bl_holds   = [r['bl_hold']        for r in results]
        vel_holds  = [r['vel_hold']       for r in results]
        bl_wins    = [r['bl_net'] > 0     for r in results]
        vel_wins   = [r['is_winner_vel']  for r in results]
        delays     = [r['entry_step_vel'] for r in results]

        d_net  = np.mean(vel_nets)  - np.mean(bl_nets)
        d_win  = np.mean(vel_wins)  - np.mean(bl_wins)
        d_hold = np.mean(vel_holds) - np.mean(bl_holds)

        print(f"\n  {'Metric':<32}  {'Baseline':>10}  {'Vel-entry':>10}  {'Delta':>9}")
        print(f"  {'-'*65}")
        print(f"  {'mean_net':<32}  {np.mean(bl_nets):>+10.4f}  "
              f"{np.mean(vel_nets):>+10.4f}  {d_net:>+9.4f}")
        print(f"  {'win_rate':<32}  {np.mean(bl_wins):>10.1%}  "
              f"{np.mean(vel_wins):>10.1%}  {d_win:>+9.1%}")
        print(f"  {'mean_hold_cal (d)':<32}  {np.mean(bl_holds):>10.0f}  "
              f"{np.mean(vel_holds):>10.0f}  {d_hold:>+9.0f}")
        print(f"  {'mean_entry_delay (steps)':<32}  {'--':>10}  {np.mean(delays):>10.1f}  {'--':>9}")
        print(f"  {'median_entry_delay (steps)':<32}  {'--':>10}  {np.median(delays):>10.1f}  {'--':>9}")

        out  = pd.DataFrame(results)
        path = RESULTS_DIR / f'velocity_retest_sim_lb{lb}.csv'
        out.to_csv(path, index=False)
        print(f"\n  -> {path.name}")

        sim_results[lb] = results

    return sim_results


# ==============================================================================
# Part C — Decision gate
# ==============================================================================

def part_c_decision_gate(diag_rows, sim_results):
    print(f"\n{SEP}")
    print(f"PART C -- DECISION GATE")
    print(f"{SEP}")

    by_key = {(r['lookback'], r['offset']): r for r in diag_rows}

    pneg = {
        (lb, off): by_key.get((lb, off), {}).get('pct_neg', np.nan)
        for lb in [1, 2] for off in [0, 1, 2, 3]
    }

    print(f"\n  % velocity < 0 at and before peak:")
    print(f"  {'':16}  {'vel_1d':>8}  {'vel_2d':>8}")
    for off in [0, 1, 2, 3]:
        label = 'at peak' if off == 0 else f'peak-{off}'
        print(f"  {label:<16}  {_fmt_pct(pneg[(1, off)]):>8}  {_fmt_pct(pneg[(2, off)]):>8}")

    # Net improvement from Part B (using the matched subset that fired the signal)
    net_delta = {}
    for lb in [1, 2]:
        rs = sim_results.get(lb, [])
        if rs:
            d = np.mean([r['vel_net'] for r in rs]) - np.mean([r['bl_net'] for r in rs])
            net_delta[lb] = d
        else:
            net_delta[lb] = np.nan

    candidates        = [v for v in [pneg[(1, 0)], pneg[(2, 0)]] if not np.isnan(v)]
    best_pneg_at_peak = max(candidates) if candidates else np.nan

    print(f"\n  Net delta vs baseline (signal-matched subset):")
    for lb in [1, 2]:
        d = net_delta.get(lb, np.nan)
        print(f"    vel_{lb}d: {d:>+.4f}" if not np.isnan(d) else f"    vel_{lb}d:    N/A")

    # Verdict
    VIABLE_THRESHOLD   = 0.50
    MARGINAL_THRESHOLD = 0.20

    if best_pneg_at_peak >= VIABLE_THRESHOLD:
        verdict = 'VIABLE'
        impl    = (f'vel_N < 0 at peak in {_fmt_pct(best_pneg_at_peak).strip()} of trades — '
                   f'velocity reversal fires reliably at peak at short resolution. '
                   f'Proceed to Step 2.2: backtest delayed entry triggered by vel_N sign change.')
    elif best_pneg_at_peak >= MARGINAL_THRESHOLD:
        verdict = 'MARGINAL'
        impl    = (f'Signal fires at peak in {_fmt_pct(best_pneg_at_peak).strip()} of trades — '
                   f'better than 5-day but not reliable on its own. '
                   f'Combine with a peak_sd filter (e.g. peak_sd > 3.0) or time gate '
                   f'before committing; otherwise move to Step 2.2 (time-based delay).')
    else:
        verdict = 'NOT VIABLE'
        impl    = (f'vel_N < 0 at peak in only {_fmt_pct(best_pneg_at_peak).strip()} of trades '
                   f'even at 1-day resolution. Velocity reversal does not distinguish the peak. '
                   f'Abandon velocity-based entry timing. '
                   f'Proceed to Step 2.2: time-based delay calibrated from peak_cal distribution.')

    print(f"\n  VERDICT: [{verdict}]")
    print(f"  {impl}")
    print(f"\n{SEP}")


# ==============================================================================
# Main
# ==============================================================================

def run_all(asset_class='commodities'):
    print(f"\n{SEP}")
    print(f"VELOCITY RETEST -- Phase 2, Step 2.1  ({asset_class.upper()})")
    print(f"Lookbacks: 1-day, 2-day only")
    print(f"{SEP}")

    trades      = collect_trades_with_peak(asset_class)
    trades, rows = part_a_diagnostic(trades, asset_class)
    sim_results = part_b_simulation(trades, asset_class)
    part_c_decision_gate(rows, sim_results)


if __name__ == '__main__':
    run_all(asset_class='commodities')
