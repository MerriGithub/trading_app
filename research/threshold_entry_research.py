"""
threshold_entry_research.py
===========================
Phase 2, Step 2.3 — Higher-threshold entry rules.

Tests entering only when |dist_SD| first crosses a threshold above the
baseline 2.0 SD (thresholds: 2.0, 2.5, 3.0, 3.5).  EXIT_SD = 0.0 is
kept constant across all thresholds so entry level is the single variable.

Quantifies per-trade effect on the 20-pair research subset and produces
key inputs for the Step 2.4 reversion-trigger design.

Usage:
    python research/threshold_entry_research.py

Outputs (all in results/):
    threshold_entry_aggregate.csv
    threshold_entry_per_pair.csv
    threshold_entry_decision.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).parent
ROOT         = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

# trailing_stop_research: only constants used directly in this script's logic
# build_spread, SPREAD_COST, DATA_PATH, XING_SD are NOT imported:
#   build_spread  — import from peak_analysis_research instead (it re-exports from trailing_stop_research)
#   SPREAD_COST   — superseded by DATA_CONFIGS['commodities']['bid_ask']
#   DATA_PATH     — superseded by load_prices
#   XING_SD       — used only inside run_baseline_with_paths
# EXIT_SD = 0.0 is used for ALL thresholds including baseline (T=2.0).
# This keeps entry threshold as the single variable — a clean apples-to-apples comparison.
# A higher entry threshold does NOT imply a different exit; that interaction is the
# subject of Step 2.4 (reversion-trigger), not this step.
from trailing_stop_research import (
    DAILY_FIN, EXIT_SD, MAX_HOLD, RESULTS_DIR,
)

# peak_analysis_research: shared utilities — import directly, do NOT redefine.
# VOL_WINDOW, COMMODITY_PAIRS must be imported, not redefined locally.
from peak_analysis_research import (
    build_spread,              # used directly in simulate_threshold_entry
    load_prices,               # loads commodity prices via DATA_CONFIGS
    COMMODITY_PAIRS,           # canonical 20-pair list — do NOT redefine
    DATA_CONFIGS,              # provides DATA_CONFIGS['commodities']['bid_ask']
    VOL_WINDOW,                # 262 — do NOT redefine
)

THRESHOLDS = [2.0, 2.5, 3.0, 3.5]   # 2.0 = baseline; others = higher-threshold entry
BID_ASK    = DATA_CONFIGS['commodities']['bid_ask']

W   = 78
SEP = '=' * W


# ==============================================================================
# Core simulation
# ==============================================================================

def simulate_threshold_entry(cum, dist_sd, day_ints, threshold):
    """
    Enter when |dist_sd| first crosses ±threshold (outward).
    Exit at EXIT_SD (0.0) or MAX_HOLD.  Same exit rule as baseline (T=2.0).

    Returns list of trade dicts.
    """
    trades   = []
    in_trade = False

    for i in range(len(cum)):
        d = dist_sd[i]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > threshold:
                in_trade  = True
                entry_idx = i
                entry_cum = cum[i]
                side      = -1
            elif d < -threshold:
                in_trade  = True
                entry_idx = i
                entry_cum = cum[i]
                side      = +1
        else:
            hold_cal    = int(day_ints[i] - day_ints[entry_idx])
            normal_exit = (side == -1 and d <= EXIT_SD) or (side == +1 and d >= EXIT_SD)
            max_hit     = hold_cal >= MAX_HOLD

            if normal_exit or max_hit:
                exit_cum = cum[i]
                gross = ((exit_cum - entry_cum) / entry_cum if side == +1
                         else (entry_cum - exit_cum) / entry_cum)
                net = gross - BID_ASK - DAILY_FIN * hold_cal
                trades.append({
                    'side':          side,
                    'entry_idx':     entry_idx,
                    'entry_dist_sd': abs(dist_sd[entry_idx]),
                    'gross':         gross,
                    'net':           net,
                    'hold_cal':      hold_cal,
                    'is_winner':     net > 0,
                    'max_hold_exit': max_hit and not normal_exit,
                })
                in_trade = False

    return trades


def _stats(trades):
    if not trades:
        return dict(n=0, avg_gross=np.nan, avg_net=np.nan, win_rate=np.nan,
                    avg_hold=np.nan, avg_entry_dist_sd=np.nan, pct_max_hold=np.nan)
    gross_arr = np.array([t['gross']          for t in trades])
    net_arr   = np.array([t['net']            for t in trades])
    hold_arr  = np.array([t['hold_cal']       for t in trades])
    esd_arr   = np.array([t['entry_dist_sd']  for t in trades])
    win_arr   = np.array([t['is_winner']      for t in trades], dtype=float)
    mh_arr    = np.array([t['max_hold_exit']  for t in trades], dtype=float)
    return dict(
        n                 = len(trades),
        avg_gross         = float(gross_arr.mean()),
        avg_net           = float(net_arr.mean()),
        win_rate          = float(win_arr.mean()),
        avg_hold          = float(hold_arr.mean()),
        avg_entry_dist_sd = float(esd_arr.mean()),
        pct_max_hold      = float(mh_arr.mean()),
    )


# ==============================================================================
# Data collection
# ==============================================================================

def collect_all(asset_class='commodities'):
    """
    Run simulate_threshold_entry for every (pair, threshold) combo.
    Returns pair_results: pair_label -> {threshold -> [trade dicts]}
    """
    prices = load_prices(asset_class)
    cfg    = DATA_CONFIGS[asset_class]

    pair_results = {}
    print(f"\nRunning {len(cfg['pairs'])} pairs × {len(THRESHOLDS)} thresholds...")

    for idx, (long_inst, short_inst) in enumerate(cfg['pairs']):
        lbl = f"{long_inst}/{short_inst}"
        if long_inst not in prices.columns or short_inst not in prices.columns:
            print(f"  [{idx+1:02d}] {lbl}  SKIPPED (not in data)")
            continue

        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)

        threshold_trades = {T: simulate_threshold_entry(cum, dist_sd, day_ints, T)
                            for T in THRESHOLDS}

        counts = '  '.join(f"T={T:.1f}:{len(threshold_trades[T])}" for T in THRESHOLDS)
        print(f"  [{idx+1:02d}] {lbl}  {counts}")
        pair_results[lbl] = threshold_trades

    return pair_results


# ==============================================================================
# Part A — Aggregate across all pairs
# ==============================================================================

def part_a_aggregate(pair_results):
    print(f"\n{SEP}")
    print("PART A -- AGGREGATE ACROSS ALL PAIRS")
    print(f"{SEP}")
    print(f"\n  {'T':>5}  {'n_trades':>8}  {'coverage':>9}  {'avg_net':>9}  "
          f"{'net_delta':>10}  {'win%':>7}  {'avg_hold':>9}  "
          f"{'avg_entry_sd':>13}  {'max_hold%':>10}")

    agg_rows = []
    bl_n     = None
    bl_net   = None

    for T in THRESHOLDS:
        all_trades = []
        for threshold_trades in pair_results.values():
            all_trades.extend(threshold_trades.get(T, []))

        st = _stats(all_trades)

        if T == 2.0:
            bl_n     = st['n']
            bl_net   = st['avg_net']
            coverage = 1.0
            delta    = 0.0
        else:
            coverage = st['n'] / bl_n if bl_n else np.nan
            delta    = (st['avg_net'] - bl_net
                        if not np.isnan(st['avg_net']) and bl_net is not None
                        else np.nan)

        marker = '  <-- baseline' if T == 2.0 else ''
        print(f"  {T:>5.1f}  {st['n']:>8}  {coverage:>8.1%}  "
              f"{st['avg_net']:>+9.4f}  {delta:>+10.4f}  "
              f"{st['win_rate']:>6.1%}  {st['avg_hold']:>8.1f}d  "
              f"{st['avg_entry_dist_sd']:>12.3f}  {st['pct_max_hold']:>9.1%}{marker}")

        agg_rows.append(dict(
            threshold         = T,
            n_trades          = st['n'],
            coverage          = round(coverage, 4),
            avg_gross         = round(st['avg_gross'], 4),
            avg_net           = round(st['avg_net'],   4),
            net_delta         = round(delta,           4),
            win_rate          = round(st['win_rate'],  4),
            avg_hold          = round(st['avg_hold'],  1),
            avg_entry_dist_sd = round(st['avg_entry_dist_sd'], 3),
            pct_max_hold      = round(st['pct_max_hold'], 4),
        ))

    return agg_rows


# ==============================================================================
# Part B — Per-pair breakdown
# ==============================================================================

def part_b_per_pair(pair_results):
    print(f"\n{SEP}")
    print("PART B -- PER-PAIR BREAKDOWN  (avg_net at each threshold)")
    print(f"{SEP}")

    th_labels = ''.join(f"  {'T='+str(T):>9}" for T in THRESHOLDS)
    print(f"\n  {'pair':>22}{th_labels}  {'best_T':>7}  {'best_delta':>11}")

    pair_rows = []

    for lbl, threshold_trades in pair_results.items():
        bl_st  = _stats(threshold_trades.get(2.0, []))
        bl_net = bl_st['avg_net']

        per_row    = {'pair': lbl}
        best_T     = 2.0
        best_delta = 0.0
        net_strs   = []

        for T in THRESHOLDS:
            st    = _stats(threshold_trades.get(T, []))
            delta = (st['avg_net'] - bl_net
                     if not np.isnan(st['avg_net']) and not np.isnan(bl_net)
                     else np.nan)

            per_row[f'n_{T}']            = st['n']
            per_row[f'avg_net_{T}']      = round(st['avg_net'],           4) if not np.isnan(st['avg_net'])           else np.nan
            per_row[f'net_delta_{T}']    = round(delta,                   4) if not np.isnan(delta)                   else np.nan
            per_row[f'win_rate_{T}']     = round(st['win_rate'],          4) if not np.isnan(st['win_rate'])          else np.nan
            per_row[f'avg_hold_{T}']     = round(st['avg_hold'],          1) if not np.isnan(st['avg_hold'])          else np.nan
            per_row[f'avg_entry_sd_{T}'] = round(st['avg_entry_dist_sd'], 3) if not np.isnan(st['avg_entry_dist_sd']) else np.nan

            if not np.isnan(delta) and delta > best_delta:
                best_T     = T
                best_delta = delta

            net_strs.append(f"  {st['avg_net']:>+9.4f}" if not np.isnan(st['avg_net'])
                            else f"  {'  N/A':>9}")

        per_row['best_T']     = best_T
        per_row['best_delta'] = round(best_delta, 4)
        pair_rows.append(per_row)
        print(f"  {lbl:>22}{''.join(net_strs)}  {best_T:>7.1f}  {best_delta:>+11.4f}")

    print(f"\n  Pairs where T>2.0 beats baseline (avg_net):")
    for T in THRESHOLDS[1:]:
        key        = f'net_delta_{T}'
        n_improved = sum(1 for r in pair_rows
                         if not np.isnan(r.get(key, np.nan)) and r[key] > 0)
        print(f"    T={T:.1f}: {n_improved}/{len(pair_rows)}")

    return pair_rows


# ==============================================================================
# Part C — Decision gate + Step 2.4 inputs
# ==============================================================================

def part_c_decision(agg_rows, pair_rows):
    print(f"\n{SEP}")
    print("PART C -- DECISION GATE  +  STEP 2.4 INPUTS")
    print(f"{SEP}")

    df     = pd.DataFrame(agg_rows)
    non_bl = df[df['threshold'] > 2.0].copy()

    if non_bl.empty:
        print("\n  No non-baseline thresholds to evaluate.")
        print(f"{SEP}")
        return []

    best_idx   = non_bl['avg_net'].idxmax()
    best_T     = float(non_bl.loc[best_idx, 'threshold'])
    best_delta = float(non_bl.loc[best_idx, 'net_delta'])
    best_cov   = float(non_bl.loc[best_idx, 'coverage'])

    bl_row = df[df['threshold'] == 2.0].iloc[0]
    bl_net = float(bl_row['avg_net'])

    # Monotonicity across the full threshold sequence
    nets = df.sort_values('threshold')['avg_net'].tolist()
    monotone_up = all(nets[i] <= nets[i + 1] for i in range(len(nets) - 1))
    monotone_dn = all(nets[i] >= nets[i + 1] for i in range(len(nets) - 1))
    if monotone_up:
        mono_label = 'MONOTONE INCREASING  (higher T -> better avg_net)'
    elif monotone_dn:
        mono_label = 'MONOTONE DECREASING  (higher T -> worse avg_net)'
    else:
        mono_label = 'NON-MONOTONE  (peak at intermediate threshold)'

    print(f"\n  Monotonicity:  {mono_label}")
    print(f"  avg_net by threshold: "
          + '  '.join(f"T={r['threshold']:.1f}:{r['avg_net']:+.4f}"
                      for _, r in df.iterrows()))

    print(f"\n  Coverage vs net improvement (vs baseline T=2.0):")
    print(f"  {'T':>5}  {'n_trades':>8}  {'coverage':>9}  {'net_delta':>10}  "
          f"{'avg_entry_sd':>13}")
    for _, r in df.iterrows():
        marker = '  <-- best' if r['threshold'] == best_T else ''
        print(f"  {r['threshold']:>5.1f}  {r['n_trades']:>8}  {r['coverage']:>8.1%}  "
              f"{r['net_delta']:>+10.4f}  {r['avg_entry_dist_sd']:>13.3f}{marker}")

    print(f"\n  STEP 2.4 INPUTS  (reversion-trigger design):")
    print(f"  avg_entry_dist_sd = spread location at entry - the 'starting point'")
    print(f"  for the reversion trigger.  Higher T begins closer to the peak.")
    print(f"  {'T':>5}  {'avg_entry_sd':>13}  {'avg_hold':>9}  {'pct_max_hold':>13}")
    for _, r in df.iterrows():
        print(f"  {r['threshold']:>5.1f}  {r['avg_entry_dist_sd']:>13.3f}  "
              f"{r['avg_hold']:>8.1f}d  {r['pct_max_hold']:>12.1%}")

    MATERIAL_THRESHOLD = 0.001
    COVERAGE_FLOOR     = 0.50

    print(f"\n  VERDICT:")
    if best_delta > MATERIAL_THRESHOLD and best_cov >= COVERAGE_FLOOR:
        verdict = 'VIABLE'
        impl    = (f"T={best_T:.1f} improves avg_net by {best_delta:+.4f} "
                   f"with {best_cov:.1%} coverage vs baseline.  "
                   f"Recommend T={best_T:.1f} as the Step 2.4 entry threshold.")
    elif best_delta > MATERIAL_THRESHOLD and best_cov < COVERAGE_FLOOR:
        verdict = 'FILTER TOO AGGRESSIVE'
        impl    = (f"T={best_T:.1f} improves avg_net by {best_delta:+.4f} "
                   f"but only {best_cov:.1%} coverage.  "
                   f"Consider T=2.5 for better coverage with partial benefit.")
    elif 0 < best_delta <= MATERIAL_THRESHOLD:
        verdict = 'MARGINAL'
        impl    = (f"Best delta {best_delta:+.4f} below material threshold.  "
                   f"Higher-threshold entry provides limited benefit in isolation; "
                   f"combine with reversion trigger in Step 2.4.")
    else:
        verdict = 'NOT VIABLE'
        impl    = (f"No higher threshold improves avg_net (best delta={best_delta:+.4f}).  "
                   f"Use T=2.0 baseline for Step 2.4.")

    print(f"  [{verdict}]")
    print(f"  {impl}")
    print(f"\n{SEP}")

    decision_rows = []
    for _, r in df.iterrows():
        key            = f'net_delta_{r["threshold"]}'
        n_pairs_impr   = sum(1 for pr in pair_rows
                             if not np.isnan(pr.get(key, np.nan))
                             and pr[key] > 0)
        decision_rows.append(dict(
            threshold         = r['threshold'],
            n_trades          = r['n_trades'],
            coverage          = r['coverage'],
            avg_net           = r['avg_net'],
            net_delta         = r['net_delta'],
            win_rate          = r['win_rate'],
            avg_hold          = r['avg_hold'],
            avg_entry_dist_sd = r['avg_entry_dist_sd'],
            pct_max_hold      = r['pct_max_hold'],
            n_pairs_improved  = n_pairs_impr,
            is_best           = int(r['threshold'] == best_T),
            verdict           = verdict if r['threshold'] == best_T else '',
        ))

    return decision_rows


# ==============================================================================
# Main
# ==============================================================================

def run_all(asset_class='commodities'):
    print(f"\n{SEP}")
    print(f"THRESHOLD ENTRY RESEARCH -- Phase 2, Step 2.3  "
          f"({asset_class.upper()})")
    print(f"Thresholds: {THRESHOLDS}  |  EXIT_SD={EXIT_SD}  |  MAX_HOLD={MAX_HOLD}")
    print(f"{SEP}")

    pair_results = collect_all(asset_class)
    if not pair_results:
        print("No pairs collected — check data and pair list.")
        return

    agg_rows      = part_a_aggregate(pair_results)
    pair_rows     = part_b_per_pair(pair_results)
    decision_rows = part_c_decision(agg_rows, pair_rows)

    pd.DataFrame(agg_rows).to_csv(
        RESULTS_DIR / 'threshold_entry_aggregate.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(
        RESULTS_DIR / 'threshold_entry_per_pair.csv', index=False)
    if decision_rows:
        pd.DataFrame(decision_rows).to_csv(
            RESULTS_DIR / 'threshold_entry_decision.csv', index=False)

    print(f"\nCSVs saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    run_all(asset_class='commodities')
