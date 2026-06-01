"""
stall_gate_research.py
======================
Phase 2, Step 2.2 — Stall-gate entry with peak-SD quality filter.

After the crossing fires at 2.0 SD, the spread typically extends to a peak
(~3.1 SD mean) before reverting.  This script delays entry until the peak has
clearly formed: N consecutive steps with no new extreme (the "stall gate").
The observed peak SD at stall time is then used as a quality filter — Phase 1
showed loser peak = 3.41 SD vs winner peak = 2.77 SD.

Part A: Stall diagnostic — % of trades that stall per stall_days, distribution
        of observed_peak_sd by winner/loser.
Part B: Grid simulation — stall_days × peak_sd_gate, compare mean_net,
        win_rate, and hold vs baseline on the matched entered subset.
Part C: Decision gate — best combo, coverage/filtering tradeoff summary.

Inputs:
  results/peak_trades_full_commodities.csv   (Phase 1 output)

Outputs:
  results/stall_gate_diagnostic.csv
  results/stall_gate_grid.csv
  results/stall_gate_best.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product

RESEARCH_DIR = Path(__file__).parent
ROOT         = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

# trailing_stop_research: only constants used directly in this script's logic
# DAILY_FIN: used in net formula
# EXIT_SD:   used in exit scan condition
# MAX_HOLD:  used in exit scan condition
# RESULTS_DIR: used to locate peak_trades_full_commodities.csv
# build_spread, SPREAD_COST, DATA_PATH, XING_SD are NOT imported — they are either
# used internally by run_baseline_with_paths or superseded by DATA_CONFIGS.
from trailing_stop_research import (
    DAILY_FIN, EXIT_SD, MAX_HOLD, RESULTS_DIR,
)

# peak_analysis_research: all shared utilities — import directly, do NOT redefine.
# DATA_CONFIGS provides bid_ask cost; load_prices handles the data path.
# VOL_WINDOW and COMMODITY_PAIRS must be imported, not redefined.
from peak_analysis_research import (
    run_baseline_with_paths,   # uses DATA_CONFIGS[asset_class]['bid_ask'] internally
    load_prices,               # loads commodity prices via DATA_CONFIGS
    COMMODITY_PAIRS,           # canonical 20-pair list — do NOT redefine
    DATA_CONFIGS,              # provides DATA_CONFIGS['commodities']['bid_ask']
    VOL_WINDOW,                # 262 — do NOT redefine
)

W    = 78
SEP  = '=' * W
SEP2 = '-' * W

PHASE1_CSV = RESULTS_DIR / 'peak_trades_full_commodities.csv'

STALL_DAYS_VALUES   = [3, 5, 7, 10]
PEAK_SD_GATE_VALUES = [None, 2.5, 3.0, 3.5, 4.0]


def _pct(arr, condition_fn):
    """Fraction of non-NaN elements satisfying condition_fn."""
    valid = [x for x in arr if not np.isnan(x)]
    if not valid:
        return np.nan
    return sum(1 for x in valid if condition_fn(x)) / len(valid)


def _fmt_pct(v):
    return f"{v:.1%}" if not np.isnan(v) else "  N/A"


def _gate_label(g):
    return f"<={g:.1f}" if g is not None else "  none"


# ==============================================================================
# Step 0 — Collect baseline trades; join Phase 1 peak data
# ==============================================================================

def collect_trades_with_peak(asset_class='commodities'):
    """
    Re-run run_baseline_with_paths to get dist_path / cum_path / dayint_path,
    then join Phase 1 CSV on (pair, entry_idx, exit_idx) to attach peak_step
    and peak_sd.
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
        peak_lookup[key] = (int(row['peak_step']), float(row['peak_sd']))

    prices = load_prices(asset_class)
    cfg    = DATA_CONFIGS[asset_class]

    all_trades = []
    for (long_inst, short_inst) in cfg['pairs']:
        if long_inst not in prices.columns or short_inst not in prices.columns:
            print(f"  Skipping {long_inst}/{short_inst} — not in data")
            continue
        trades = run_baseline_with_paths(long_inst, short_inst, prices, asset_class)
        for t in trades:
            key   = (t['pair'], int(t['entry_idx']), int(t['exit_idx']))
            match = peak_lookup.get(key)
            t['peak_step'] = match[0] if match else None
            t['peak_sd']   = match[1] if match else np.nan
        all_trades.extend(trades)

    n_missing = sum(1 for t in all_trades if t['peak_step'] is None)
    print(f"\n  {asset_class}: {len(all_trades)} trades collected, "
          f"{n_missing} unmatched in Phase 1 CSV")
    return all_trades


# ==============================================================================
# Core: stall detection
# ==============================================================================

def detect_stall(dist_path, side, stall_days):
    """
    Scan dist_path from step 0 and detect when the running extreme (max for
    side==-1, min for side==+1) has not updated for stall_days consecutive steps.

    Returns (stall_step, observed_peak_sd) or None if no stall within path.
      stall_step       = step at which the stall is first confirmed
      observed_peak_sd = abs(extreme) at the prior peak step
    """
    if len(dist_path) <= stall_days:
        return None

    extreme           = dist_path[0]
    last_extreme_step = 0

    for k in range(1, len(dist_path)):
        if side == -1:
            if dist_path[k] > extreme:
                extreme           = dist_path[k]
                last_extreme_step = k
        else:
            if dist_path[k] < extreme:
                extreme           = dist_path[k]
                last_extreme_step = k

        if k - last_extreme_step >= stall_days:
            return k, abs(extreme)

    return None


# ==============================================================================
# Part A — Stall diagnostic
# ==============================================================================

def part_a_diagnostic(trades, asset_class='commodities'):
    print(f"\n{SEP}")
    print(f"PART A -- STALL DIAGNOSTIC  ({asset_class})")
    print(f"Phase 1 reference: loser peak_sd = 3.41, winner peak_sd = 2.77, "
          f"mean peak_cal ~23 d")
    print(f"{SEP}")

    diag_rows = []

    for sd in STALL_DAYS_VALUES:
        stall_results = []
        for t in trades:
            r = detect_stall(t['dist_path'], t['side'], sd)
            if r is not None:
                stall_step, obs_peak = r
                stall_results.append({
                    'stall_step':       stall_step,
                    'observed_peak_sd': obs_peak,
                    'is_winner':        t['is_winner'],
                })

        n_total  = len(trades)
        n_stall  = len(stall_results)
        pct_fire = n_stall / n_total if n_total else np.nan

        obs_all   = [r['observed_peak_sd'] for r in stall_results]
        obs_win   = [r['observed_peak_sd'] for r in stall_results if     r['is_winner']]
        obs_lose  = [r['observed_peak_sd'] for r in stall_results if not r['is_winner']]
        step_vals = [r['stall_step']        for r in stall_results]

        mean_obs  = np.mean(obs_all)   if obs_all   else np.nan
        mean_win  = np.mean(obs_win)   if obs_win   else np.nan
        mean_lose = np.mean(obs_lose)  if obs_lose  else np.nan
        mean_step = np.mean(step_vals) if step_vals else np.nan
        pct_lt30  = _pct(obs_all, lambda x: x < 3.0)
        pct_lt35  = _pct(obs_all, lambda x: x < 3.5)

        print(f"\n  stall_days = {sd}")
        print(f"  {'stalls fired':>24}  {n_stall:>5} / {n_total}  ({pct_fire:.1%})")
        print(f"  {'mean stall_step':>24}  {mean_step:>5.1f} steps after entry")
        print(f"  {'mean observed_peak_sd':>24}  {mean_obs:>5.2f} SD")
        print(f"  {'  winners':>24}  {mean_win:>5.2f} SD")
        print(f"  {'  losers':>24}  {mean_lose:>5.2f} SD")
        print(f"  {'  gap (loser - winner)':>24}  {mean_lose - mean_win:>+5.2f} SD")
        print(f"  {'pct obs_peak < 3.0 SD':>24}  {_fmt_pct(pct_lt30):>8}")
        print(f"  {'pct obs_peak < 3.5 SD':>24}  {_fmt_pct(pct_lt35):>8}")

        diag_rows.append({
            'stall_days':            sd,
            'n_baseline':            n_total,
            'n_stall_fires':         n_stall,
            'pct_fires':             round(pct_fire, 4),
            'mean_stall_step':       round(mean_step, 1) if not np.isnan(mean_step) else np.nan,
            'mean_observed_peak_sd': round(mean_obs,  2) if not np.isnan(mean_obs)  else np.nan,
            'mean_obs_peak_winners': round(mean_win,  2) if not np.isnan(mean_win)  else np.nan,
            'mean_obs_peak_losers':  round(mean_lose, 2) if not np.isnan(mean_lose) else np.nan,
            'pct_obs_peak_lt_3.0':   round(pct_lt30,  4) if not np.isnan(pct_lt30)  else np.nan,
            'pct_obs_peak_lt_3.5':   round(pct_lt35,  4) if not np.isnan(pct_lt35)  else np.nan,
        })

    out  = pd.DataFrame(diag_rows)
    path = RESULTS_DIR / 'stall_gate_diagnostic.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}")

    return diag_rows


# ==============================================================================
# Part B — Grid simulation
# ==============================================================================

def simulate_stall_gate(trade, stall_days, peak_sd_gate, bid_ask):
    """
    Detect stall in dist_path.  Apply peak_sd_gate filter.  Enter at stall_step.
    Exit at EXIT_SD or MAX_HOLD (same rules as baseline).

    Returns (result_dict, status) where status is 'entered', 'filtered', or 'no_stall'.
    """
    r = detect_stall(trade['dist_path'], trade['side'], stall_days)
    if r is None:
        return None, 'no_stall'

    stall_step, obs_peak = r

    if peak_sd_gate is not None and obs_peak > peak_sd_gate:
        return None, 'filtered'

    dp   = trade['dist_path']
    cp   = trade['cum_path']
    dip  = trade['dayint_path']
    side = trade['side']

    result_k     = None
    max_hit_flag = False
    for k in range(stall_step + 1, len(dp)):
        hold_cal    = dip[k] - dip[stall_step]
        normal_exit = (side == -1 and dp[k] <= EXIT_SD) or \
                      (side == +1 and dp[k] >= EXIT_SD)
        max_hit     = hold_cal >= MAX_HOLD
        if normal_exit or max_hit:
            result_k     = k
            max_hit_flag = max_hit
            break

    if result_k is None:
        result_k = len(dp) - 1

    hold_cal     = dip[result_k] - dip[stall_step]
    max_hit_flag = max_hit_flag or (hold_cal >= MAX_HOLD)

    gross = ((cp[result_k] - cp[stall_step]) / cp[stall_step]) if side == +1 \
            else ((cp[stall_step] - cp[result_k]) / cp[stall_step])
    net   = gross - bid_ask - DAILY_FIN * hold_cal

    return {
        'pair':             trade['pair'],
        'side':             side,
        'bl_net':           trade['net_final'],
        'bl_hold':          trade['total_hold_cal'],
        'is_winner_bl':     trade['is_winner'],
        'observed_peak_sd': obs_peak,
        'stall_step':       stall_step,
        'stall_hold':       hold_cal,
        'stall_gross':      gross,
        'stall_net':        net,
        'is_winner_stall':  net > 0,
        'max_hold_exit':    max_hit_flag,
    }, 'entered'


def part_b_grid(trades, asset_class='commodities'):
    print(f"\n{SEP}")
    print(f"PART B -- GRID SIMULATION  ({asset_class})")
    print(f"stall_days × peak_sd_gate  |  mean_net, win_rate, hold delta vs baseline")
    print(f"{SEP}")

    bid_ask  = DATA_CONFIGS[asset_class]['bid_ask']
    n_bl_all = len(trades)

    hdr = (f"  {'stall_d':>7}  {'gate':>7}  {'n_enter':>7}  {'cover':>6}  "
           f"{'n_filt':>6}  {'n_nostl':>7}  {'bl_net':>8}  {'sg_net':>8}  "
           f"{'delta':>8}  {'win_bl':>7}  {'win_sg':>7}  {'hold_d':>7}")
    print(f"\n{hdr}")
    print(f"  {'-'*96}")

    grid_rows = []

    for sd, gate in product(STALL_DAYS_VALUES, PEAK_SD_GATE_VALUES):
        n_entered  = 0
        n_filtered = 0
        n_no_stall = 0
        entered    = []

        for t in trades:
            res, status = simulate_stall_gate(t, sd, gate, bid_ask)
            if status == 'entered':
                n_entered += 1
                entered.append(res)
            elif status == 'filtered':
                n_filtered += 1
            else:
                n_no_stall += 1

        coverage = n_entered / n_bl_all if n_bl_all else np.nan

        if not entered:
            grid_rows.append({
                'stall_days': sd, 'peak_sd_gate': str(gate),
                'n_baseline': n_bl_all, 'n_entered': 0,
                'n_filtered': n_filtered, 'n_no_stall': n_no_stall,
                'coverage': 0.0, 'bl_mean_net': np.nan, 'sg_mean_net': np.nan,
                'net_delta': np.nan, 'bl_win_rate': np.nan,
                'sg_win_rate': np.nan, 'mean_hold_delta': np.nan,
            })
            print(f"  {sd:>7}  {_gate_label(gate):>7}  {'0':>7}  {'0.0%':>6}  "
                  f"{n_filtered:>6}  {n_no_stall:>7}  "
                  f"{'—':>8}  {'—':>8}  {'—':>8}  {'—':>7}  {'—':>7}  {'—':>7}")
            continue

        bl_nets  = [r['bl_net']          for r in entered]
        sg_nets  = [r['stall_net']       for r in entered]
        bl_holds = [r['bl_hold']         for r in entered]
        sg_holds = [r['stall_hold']      for r in entered]
        bl_wins  = [r['is_winner_bl']    for r in entered]
        sg_wins  = [r['is_winner_stall'] for r in entered]

        bl_mn = np.mean(bl_nets)
        sg_mn = np.mean(sg_nets)
        delta = sg_mn - bl_mn
        bl_wr = np.mean(bl_wins)
        sg_wr = np.mean(sg_wins)
        dhold = np.mean(sg_holds) - np.mean(bl_holds)

        print(f"  {sd:>7}  {_gate_label(gate):>7}  {n_entered:>7}  "
              f"{coverage:>5.1%}  {n_filtered:>6}  {n_no_stall:>7}  "
              f"{bl_mn:>+8.4f}  {sg_mn:>+8.4f}  {delta:>+8.4f}  "
              f"{bl_wr:>7.1%}  {sg_wr:>7.1%}  {dhold:>+7.0f}")

        grid_rows.append({
            'stall_days':      sd,
            'peak_sd_gate':    str(gate),
            'n_baseline':      n_bl_all,
            'n_entered':       n_entered,
            'n_filtered':      n_filtered,
            'n_no_stall':      n_no_stall,
            'coverage':        round(coverage, 4),
            'bl_mean_net':     round(bl_mn,   4),
            'sg_mean_net':     round(sg_mn,   4),
            'net_delta':       round(delta,   4),
            'bl_win_rate':     round(bl_wr,   4),
            'sg_win_rate':     round(sg_wr,   4),
            'mean_hold_delta': round(dhold,   1),
        })

    out  = pd.DataFrame(grid_rows)
    path = RESULTS_DIR / 'stall_gate_grid.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}")

    return grid_rows


# ==============================================================================
# Part C — Decision gate
# ==============================================================================

def part_c_decision(diag_rows, grid_rows):
    print(f"\n{SEP}")
    print(f"PART C -- DECISION GATE")
    print(f"{SEP}")

    df = pd.DataFrame(grid_rows).dropna(subset=['net_delta'])
    if df.empty:
        print("\n  No valid results — cannot determine verdict.")
        print(f"{SEP}")
        return

    best_idx   = df['net_delta'].idxmax()
    best       = df.loc[best_idx]
    best_sd    = int(best['stall_days'])
    best_gate  = best['peak_sd_gate']
    best_delta = float(best['net_delta'])
    best_cov   = float(best['coverage'])

    timing_df = df[df['peak_sd_gate'] == 'None'].copy()
    gate_df   = df[df['peak_sd_gate'] != 'None'].copy()

    print(f"\n  Best combo:  stall_days={best_sd}, peak_sd_gate={best_gate}")
    print(f"  Net delta vs baseline (matched entered subset):  {best_delta:>+.4f}")
    print(f"  Coverage (% of baseline trades entered):         {best_cov:.1%}")
    print(f"  n_entered: {int(best['n_entered'])}")

    print(f"\n  Timing effect only (gate=None):")
    if not timing_df.empty:
        bt = timing_df.loc[timing_df['net_delta'].idxmax()]
        print(f"    best stall_days={int(bt['stall_days'])}, "
              f"delta={float(bt['net_delta']):>+.4f}, coverage={float(bt['coverage']):.1%}")

    print(f"\n  Decomposition: timing vs filter (per stall_days, best gate):")
    if not gate_df.empty and not timing_df.empty:
        print(f"  {'stall_d':>7}  {'timing_d':>10}  {'best_gate':>9}  "
              f"{'filter_d':>10}  {'total_d':>10}  {'cover':>7}")
        for sd in STALL_DAYS_VALUES:
            t_rows = timing_df[timing_df['stall_days'] == sd]
            g_rows = gate_df[gate_df['stall_days']   == sd]
            if t_rows.empty or g_rows.empty:
                continue
            t_delta  = float(t_rows['net_delta'].iloc[0])
            best_g   = g_rows.loc[g_rows['net_delta'].idxmax()]
            total_d  = float(best_g['net_delta'])
            filter_d = total_d - t_delta
            print(f"  {sd:>7}  {t_delta:>+10.4f}  {best_g['peak_sd_gate']:>9}  "
                  f"{filter_d:>+10.4f}  {total_d:>+10.4f}  "
                  f"{float(best_g['coverage']):>6.1%}")

    print(f"\n  Summary (net_delta × coverage, all combos):")
    print(f"  {'stall_d':>7}  {'gate':>7}  {'delta':>8}  {'cover':>7}")
    for _, row in df.iterrows():
        flag = '  <-- best' if (int(row['stall_days']) == best_sd and
                                row['peak_sd_gate'] == best_gate) else ''
        print(f"  {int(row['stall_days']):>7}  {row['peak_sd_gate']:>7}  "
              f"{float(row['net_delta']):>+8.4f}  {float(row['coverage']):>6.1%}{flag}")

    MATERIAL_THRESHOLD = 0.001
    COVERAGE_FLOOR     = 0.50

    print(f"\n  VERDICT:")
    if best_delta > MATERIAL_THRESHOLD and best_cov >= COVERAGE_FLOOR:
        verdict = 'VIABLE'
        impl    = (f"stall_days={best_sd}, gate={best_gate} improves net by "
                   f"{best_delta:+.4f} with {best_cov:.1%} coverage. "
                   f"Both timing delay and peak_sd filter contribute. "
                   f"Recommend forward-testing this parameter set.")
    elif best_delta > MATERIAL_THRESHOLD and best_cov < COVERAGE_FLOOR:
        verdict = 'FILTER TOO AGGRESSIVE'
        impl    = (f"Best improvement is {best_delta:+.4f} but only {best_cov:.1%} coverage. "
                   f"The gate is discarding too many trades. "
                   f"Relax peak_sd_gate or reduce stall_days to recover coverage.")
    elif 0 < best_delta <= MATERIAL_THRESHOLD:
        verdict = 'MARGINAL'
        impl    = (f"Best delta is only {best_delta:+.4f}. Timing delay and gate "
                   f"provide minimal benefit. Consider combining with velocity filter "
                   f"or testing on longer holding periods.")
    else:
        verdict = 'NOT VIABLE'
        impl    = (f"No combo improves net expectancy (best delta = {best_delta:+.4f}). "
                   f"Stall gate does not add value over immediate entry. "
                   f"Check Part A winner/loser obs_peak separation for diagnosis.")

    print(f"  [{verdict}]")
    print(f"  {impl}")
    print(f"\n{SEP}")

    best_rows = []
    for sd in STALL_DAYS_VALUES:
        sub = df[df['stall_days'] == sd]
        if sub.empty:
            continue
        best_rows.append(sub.loc[sub['net_delta'].idxmax()].to_dict())
    best_df  = pd.DataFrame(best_rows)
    best_path = RESULTS_DIR / 'stall_gate_best.csv'
    best_df.to_csv(best_path, index=False)
    print(f"  -> {best_path.name}")


# ==============================================================================
# Main
# ==============================================================================

def run_all(asset_class='commodities'):
    print(f"\n{SEP}")
    print(f"STALL GATE RESEARCH -- Phase 2, Step 2.2  ({asset_class.upper()})")
    print(f"stall_days: {STALL_DAYS_VALUES}  |  peak_sd_gates: {PEAK_SD_GATE_VALUES}")
    print(f"{SEP}")

    trades    = collect_trades_with_peak(asset_class)
    diag_rows = part_a_diagnostic(trades, asset_class)
    grid_rows = part_b_grid(trades, asset_class)
    part_c_decision(diag_rows, grid_rows)


if __name__ == '__main__':
    run_all(asset_class='commodities')
