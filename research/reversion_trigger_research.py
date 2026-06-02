"""
reversion_trigger_research.py
==============================
Phase 2, Step 2.4 -- Reversion-trigger entry with peak quality gate.

Combines lessons from Steps 2.1 (velocity), 2.2 (stall gate), and 2.3
(threshold selection) into a single coherent entry rule:
  - Wait for spread to peak at >= PEAK_MIN SD after the 2.0 SD crossing
  - Enter only after spread pulls back to REVERT_THRESH SD (on the way back)

Tests 8 valid (PEAK_MIN, REVERT_THRESH) combinations against the baseline
and Step 2.3 threshold results.

Usage:
    python research/reversion_trigger_research.py

Outputs (all in results/):
    reversion_trigger_grid_commodities.csv
    reversion_trigger_coverage_commodities.csv
    reversion_trigger_wl_commodities.csv
    reversion_trigger_24_inputs_commodities.csv
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

# trailing_stop_research: only constants used directly in this script.
# build_spread, SPREAD_COST, DATA_PATH, XING_SD are NOT imported.
from trailing_stop_research import (
    DAILY_FIN, EXIT_SD, MAX_HOLD, RESULTS_DIR,
)

# peak_analysis_research: shared utilities -- import directly, do NOT redefine.
from peak_analysis_research import (
    build_spread,    # called directly; re-exported from trailing_stop_research
    load_prices,     # loads commodity prices via DATA_CONFIGS
    COMMODITY_PAIRS, # canonical 20-pair list -- do NOT redefine
    DATA_CONFIGS,    # provides DATA_CONFIGS['commodities']['bid_ask']
    VOL_WINDOW,      # 262 -- do NOT redefine
)

# CROSSING_SD: baseline 2.0 SD entry threshold, defined locally.
# Do NOT import XING_SD -- it is only used inside run_baseline_with_paths.
CROSSING_SD = 2.0
BID_ASK     = DATA_CONFIGS['commodities']['bid_ask']

W    = 106
SEP  = '=' * W
SEP2 = '-' * W

# ── Parameter Grid ─────────────────────────────────────────────────────────────

PEAK_MIN_VALUES      = [2.5, 3.0, None]
REVERT_THRESH_VALUES = [1.5, 2.0, 2.5]

# Valid combinations: 8 total
#   PEAK_MIN=2.5:  REVERT in [1.5, 2.0]       -- 2.5 excluded (must be strictly <)
#   PEAK_MIN=3.0:  REVERT in [1.5, 2.0, 2.5]  -- all three valid
#   PEAK_MIN=None: REVERT in [1.5, 2.0, 2.5]  -- no peak constraint; all three valid
# Excluded: PEAK_MIN=2.5 + REVERT=2.5 (equal -- trigger fires at same level as gate)
VALID_COMBOS = [
    (peak_min, revert)
    for peak_min, revert in product(PEAK_MIN_VALUES, REVERT_THRESH_VALUES)
    if peak_min is None or revert < peak_min
]
# 8 combinations total


# ── Core Simulation ────────────────────────────────────────────────────────────

def simulate_reversion_trigger(dist_sd, day_ints, cum,
                                long_inst, short_inst,
                                peak_min, revert_thresh):
    """Caller passes pre-built arrays (built once per pair, outside combo loop)."""
    results        = []
    in_signal      = False
    entered        = False
    crossing_idx   = None
    side           = None
    running_max_sd = None
    peak_qualified = None
    entry_idx      = None

    for i in range(VOL_WINDOW, len(dist_sd)):
        d = dist_sd[i]
        if np.isnan(d):
            continue

        # ---------------------------------------------------------------- IDLE
        # Note: after no_signal or maxhold_skip, in_signal resets to False.
        # If abs(d) > CROSSING_SD, a new crossing fires on the NEXT iteration (i+1),
        # not the same step -- elif branches are mutually exclusive within one i.
        # EXCEPTION: PEAK_MIN=3.0 + REVERT=2.5 (the only combo where REVERT_THRESH >
        # CROSSING_SD). Spreads stalling in 2.0–2.9 SD can cascade across iterations.
        # Documented; not patched. Grid reveals materiality.
        if not in_signal:
            if d > CROSSING_SD:
                in_signal, entered = True, False
                crossing_idx, side = i, -1
                running_max_sd     = abs(d)
                peak_qualified     = (peak_min is None) or (running_max_sd >= peak_min)
            elif d < -CROSSING_SD:
                in_signal, entered = True, False
                crossing_idx, side = i, +1
                running_max_sd     = abs(d)
                peak_qualified     = (peak_min is None) or (running_max_sd >= peak_min)

        # -------------------------------------------------------------- SEEKING
        elif not entered:
            current_abs = abs(d)
            if current_abs > running_max_sd:
                running_max_sd = current_abs
                if peak_min is not None and running_max_sd >= peak_min:
                    peak_qualified = True

            cal_from_crossing = int(day_ints[i] - day_ints[crossing_idx])
            reversion_hit = (side == -1 and d <= revert_thresh) or \
                            (side == +1 and d >= -revert_thresh)

            if reversion_hit:
                if not peak_qualified:
                    # Reverted without reaching PEAK_MIN -- no_signal; terminate.
                    results.append({
                        'pair': f"{long_inst}/{short_inst}", 'side': side,
                        'crossing_idx': crossing_idx,
                        'peak_min': peak_min, 'revert_thresh': revert_thresh,
                        'entered': False, 'no_signal': True, 'maxhold_skip': False,
                        'peak_sd_at_entry': np.nan, 'actual_entry_step': np.nan,
                        'total_hold_cal': np.nan, 'gross_final': np.nan,
                        'net_final': np.nan, 'is_winner': np.nan,
                    })
                    in_signal = False
                else:
                    # peak_qualified=True -- flip to ENTERED state
                    entered   = True
                    entry_idx = i

            elif cal_from_crossing >= MAX_HOLD:
                results.append({
                    'pair': f"{long_inst}/{short_inst}", 'side': side,
                    'crossing_idx': crossing_idx,
                    'peak_min': peak_min, 'revert_thresh': revert_thresh,
                    'entered': False,
                    'no_signal':    not peak_qualified,
                    'maxhold_skip': peak_qualified,
                    'peak_sd_at_entry': np.nan, 'actual_entry_step': np.nan,
                    'total_hold_cal': np.nan, 'gross_final': np.nan,
                    'net_final': np.nan, 'is_winner': np.nan,
                })
                in_signal = False

        # -------------------------------------------------------------- ENTERED
        else:
            cal_from_crossing = int(day_ints[i] - day_ints[crossing_idx])
            # EXIT_SD ASYMMETRY -- Phase 3 TODO:
            # EXIT_SD=0.0 is symmetric: side=-1 exits at d<=0.0, side=+1 at d>=0.0 ✓
            # For Phase 3 partial exit at EXIT_SD=1.0, side=+1 must use d >= -EXIT_SD
            # (reversion from negative side through -1.0), NOT d >= +1.0 (wrong direction).
            # Fix before copying this block to Phase 3.
            normal_exit = (side == -1 and d <= EXIT_SD) or \
                          (side == +1 and d >= EXIT_SD)   # correct only for EXIT_SD=0.0
            max_hit     = cal_from_crossing >= MAX_HOLD

            if normal_exit or max_hit:
                hold_cal = int(day_ints[i] - day_ints[entry_idx])
                gross = (cum[i] - cum[entry_idx]) / cum[entry_idx] if side == +1 \
                        else (cum[entry_idx] - cum[i]) / cum[entry_idx]
                net   = gross - BID_ASK - DAILY_FIN * hold_cal
                results.append({
                    'pair': f"{long_inst}/{short_inst}", 'side': side,
                    'crossing_idx': crossing_idx,
                    'peak_min': peak_min, 'revert_thresh': revert_thresh,
                    'entered': True, 'no_signal': False, 'maxhold_skip': False,
                    'peak_sd_at_entry':  running_max_sd,
                    'actual_entry_step': entry_idx - crossing_idx,
                    'total_hold_cal':    hold_cal,
                    'gross_final': gross, 'net_final': net,
                    'is_winner':   net > 0,
                })
                in_signal = False

    return results


# ── Grid Runner ────────────────────────────────────────────────────────────────

def run_grid(prices):
    """Run all 8 valid combos across all commodity pairs. Returns flat list of result dicts."""
    all_results = []
    for (long_inst, short_inst) in COMMODITY_PAIRS:
        if long_inst not in prices.columns or short_inst not in prices.columns:
            continue
        # Build once per pair -- same arrays for all 8 combos
        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)

        for (peak_min, revert_thresh) in VALID_COMBOS:
            results = simulate_reversion_trigger(
                dist_sd, day_ints, cum,
                long_inst, short_inst,
                peak_min, revert_thresh,
            )
            all_results.extend(results)

    return all_results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_peak(v):
    return '-   ' if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f} SD"


def _fmt_pct_or_dash(v, is_none_row):
    if is_none_row:
        return '    -'
    if pd.isna(v):
        return '    -'
    return f"{v:.1%}"


def _sort_key(col):
    return col.map(lambda x: float('inf') if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Load baseline data ────────────────────────────────────────────────────
    csv_path      = RESULTS_DIR / 'peak_trades_full_commodities.csv'
    phase1_df     = pd.read_csv(csv_path)
    baseline_ev   = phase1_df['net_final'].mean() * len(phase1_df)
    n_baseline    = len(phase1_df)
    w_peak_phase1 = phase1_df[phase1_df['is_winner']]['peak_sd'].mean()
    l_peak_phase1 = phase1_df[~phase1_df['is_winner']]['peak_sd'].mean()
    prices        = load_prices('commodities')

    # ── Baseline summary ──────────────────────────────────────────────────────
    print()
    print(SEP)
    print("REVERSION-TRIGGER ENTRY RESEARCH -- PHASE 2, STEP 2.4")
    print(SEP)
    print()
    print("Baseline summary (commodities, 2.0 SD entry, 0.0 SD exit):")
    print(f"  n_signals: {n_baseline}")
    print(f"  avg_net:   {phase1_df['net_final'].mean():.4f}")
    print(f"  avg_hold:  {phase1_df['total_hold_cal'].mean():.0f} d")
    print(f"  win_rate:  {(phase1_df['net_final'] > 0).mean()*100:.1f}%")
    print(f"  total_EV:  {baseline_ev:.2f}")
    print(f"  avg_gross: {phase1_df['gross_final'].mean():.4f}")
    print()
    print("Step 2.3 reference (threshold selection, same subset):")
    print("  T=2.5:  465 trades (68.7%), avg_net +0.0103, total_EV 4.79")
    print("  T=3.0:  307 trades (45.3%), avg_net +0.0112, total_EV 3.44")
    print("  T=3.5:  183 trades (27.0%), avg_net +0.0254, total_EV 4.65")
    print("  baseline_ev was not beaten by any threshold. This is the hurdle.")
    print()

    # ── Single case first: PEAK_MIN=2.5, REVERT=2.0 ──────────────────────────
    print(SEP2)
    print("Single case: PEAK_MIN=2.5 SD, REVERT_THRESH=2.0 SD")
    print(SEP2)

    single_results = []
    for (long_inst, short_inst) in COMMODITY_PAIRS:
        if long_inst not in prices.columns or short_inst not in prices.columns:
            continue
        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
        single_results.extend(
            simulate_reversion_trigger(dist_sd, day_ints, cum,
                                       long_inst, short_inst,
                                       peak_min=2.5, revert_thresh=2.0)
        )

    sc_df      = pd.DataFrame(single_results)
    sc_entered = sc_df[sc_df['entered']]
    sc_n       = len(sc_entered)
    sc_avg_net = sc_entered['net_final'].mean()
    sc_avg_hld = sc_entered['total_hold_cal'].mean()
    sc_total   = sc_avg_net * sc_n
    sc_wr      = (sc_entered['net_final'] > 0).mean() * 100
    sc_peak    = sc_entered['peak_sd_at_entry'].mean()
    sc_no_sig  = int(sc_df['no_signal'].sum())
    sc_mh_skip = int(sc_df['maxhold_skip'].sum())
    sc_n_total = sc_n + sc_no_sig + sc_mh_skip

    bl_avg_net = phase1_df['net_final'].mean()
    bl_avg_hld = phase1_df['total_hold_cal'].mean()

    print(f"  n_entered: {sc_n} ({sc_n/n_baseline:.1%} of baseline)")
    print(f"  no_signal: {sc_no_sig} ({sc_no_sig/sc_n_total:.1%} of combo signals) -- peak not reached before reversion")
    print(f"  mh_skip:   {sc_mh_skip} ({sc_mh_skip/sc_n_total:.1%} of combo signals) -- peak reached, reversion never triggered")
    print(f"  avg_net:   {sc_avg_net:+.4f}  (vs baseline {bl_avg_net:+.4f},  delta = {sc_avg_net - bl_avg_net:+.4f})")
    print(f"  avg_hold:  {sc_avg_hld:.0f} d (vs baseline {bl_avg_hld:.0f} d,    delta = {sc_avg_hld - bl_avg_hld:+.0f} d)")
    print(f"  total_EV:  {sc_total:.2f} (vs baseline_ev {baseline_ev:.2f},  delta = {sc_total - baseline_ev:+.2f})")
    print(f"  win_rate:  {sc_wr:.1f}%")
    print(f"  avg_peak_sd_entry: {sc_peak:.2f} SD")
    print(f"  Phase 1 hindsight gap: {l_peak_phase1 - w_peak_phase1:.2f} SD")
    if sc_total > baseline_ev:
        print("  *** BEATS BASELINE TOTAL EV ***")
    else:
        print(f"  Does not beat baseline total EV ({baseline_ev:.2f})")
    print()

    # ── Full grid ─────────────────────────────────────────────────────────────
    print("Running full 8-combination grid...")
    all_results = run_grid(prices)
    df          = pd.DataFrame(all_results)
    entered_df  = df[df['entered']].copy()
    # Cast is_winner to float so .mean() works correctly in pandas 2.x groupby.
    # Object-dtype bool columns return wrong results from groupby mean().
    entered_df['is_winner'] = pd.to_numeric(entered_df['is_winner'], errors='coerce')

    # Aggregate
    agg = entered_df.groupby(['peak_min', 'revert_thresh'], dropna=False).agg(
        n_entered      = ('net_final', 'count'),
        avg_net        = ('net_final', 'mean'),
        avg_hold       = ('total_hold_cal', 'mean'),
        win_rate       = ('is_winner', 'mean'),
        avg_peak_entry = ('peak_sd_at_entry', 'mean'),
        avg_step       = ('actual_entry_step', 'mean'),
    ).reset_index()

    # Count no_signal and maxhold_skip per combo
    ns_counts = df.groupby(['peak_min', 'revert_thresh'], dropna=False)['no_signal'].sum().reset_index()
    ns_counts.columns = ['peak_min', 'revert_thresh', 'n_no_sig']
    mh_counts = df.groupby(['peak_min', 'revert_thresh'], dropna=False)['maxhold_skip'].sum().reset_index()
    mh_counts.columns = ['peak_min', 'revert_thresh', 'n_mh_skip']

    grid = agg.merge(ns_counts, on=['peak_min', 'revert_thresh']).merge(mh_counts, on=['peak_min', 'revert_thresh'])
    grid['total_EV']    = grid['avg_net'] * grid['n_entered']
    grid['vs_base_EV']  = grid['total_EV'] - baseline_ev
    grid['pct_base']    = grid['n_entered'] / n_baseline
    # pct_no_sig and pct_mh_skip use total combo signals as denominator so values
    # stay in [0,1]: the machine can fire more crossings than n_baseline when no_signal
    # resets in_signal=False mid-baseline-trade (multiple 2.0 SD crosses per trade).
    grid['n_total_combo'] = grid['n_entered'] + grid['n_no_sig'] + grid['n_mh_skip']
    grid['pct_no_sig']  = grid['n_no_sig']  / grid['n_total_combo']
    grid['pct_mh_skip'] = grid['n_mh_skip'] / grid['n_total_combo']
    grid['beats']       = grid['total_EV'] > baseline_ev

    # Sort: floats first, None last
    grid_sorted = grid.sort_values(
        by=['peak_min', 'revert_thresh'],
        key=_sort_key
    ).reset_index(drop=True)

    # ── Main output table ──────────────────────────────────────────────────────
    print()
    print("Reversion-Trigger Entry -- Full Grid Results (commodities)")
    print(SEP)
    hdr = (f"{'peak_min':>8}  {'revert':>6}  {'n_entered':>9}  {'pct_base':>8}  "
           f"{'pct_no_sig':>10}  {'pct_mh_skip':>11}  {'avg_net':>8}  {'avg_hold':>9}  "
           f"{'win_rate':>9}  {'peak_sd_entry':>13}  {'total_EV':>9}  {'vs_base_EV':>11}")
    print(hdr)
    print(SEP2)

    for _, row in grid_sorted.iterrows():
        pm   = row['peak_min']
        rt   = row['revert_thresh']
        is_none = pd.isna(pm)

        pm_s  = ' None  ' if is_none else f"{pm:.1f} SD"
        rt_s  = f"{rt:.1f} SD"
        ns_s  = _fmt_pct_or_dash(row['pct_no_sig'], is_none)
        beat_s = '  <- BEATS BASELINE?' if row['beats'] else ''

        # Notes
        note = ''
        if not is_none and pm == 3.0 and rt == 2.5:
            note = '  [cascade risk - see v5 note]'
        elif is_none and rt == 2.5:
            note = '  [near-baseline timing]'

        line = (f"{pm_s:>8}  {rt_s:>6}  {int(row['n_entered']):>9}  "
                f"{row['pct_base']:>8.1%}  {ns_s:>10}  {row['pct_mh_skip']:>11.1%}  "
                f"{row['avg_net']:>+8.4f}  {row['avg_hold']:>7.0f} d  "
                f"{row['win_rate']:>9.1%}  {row['avg_peak_entry']:>11.2f} SD  "
                f"{row['total_EV']:>9.2f}  {row['vs_base_EV']:>+11.2f}"
                f"{beat_s}{note}")
        print(line)

    print(SEP2)
    bl_line = (f"{'Baseline':>8}  {'-':>6}  {n_baseline:>9}  {'100%':>8}  "
               f"{'-':>10}  {'-':>11}  {bl_avg_net:>+8.4f}  {bl_avg_hld:>7.0f} d  "
               f"{(phase1_df['net_final']>0).mean():>9.1%}  {'2.00 SD':>13}  "
               f"{baseline_ev:>9.2f}  {'-':>11}")
    print(bl_line)
    print(SEP)
    print()
    print("Note: baseline peak_sd_entry = 2.00 SD: entry IS the crossing.")
    print("      pct_no_sig = '--' for PEAK_MIN=None rows (peak_qualified always True).")
    print()

    # ── Secondary analysis: coverage delta vs Step 2.3 ────────────────────────
    print(SEP2)
    print("Coverage delta -- PEAK_MIN=2.5 vs Step 2.3 T=2.5:")
    print(SEP2)
    step23_n   = 465
    step23_pct = 0.687
    step23_net = 0.0103
    step23_ev  = 4.79

    pm25_rows = grid_sorted[grid_sorted['peak_min'] == 2.5]

    print(f"  {'Rule':<40}  {'n_entered':>9}  {'pct_baseline':>12}  {'avg_net':>8}  {'total_EV':>9}")
    print(f"  {'2.3 threshold T=2.5':<40}  {step23_n:>9}  {step23_pct:>12.1%}  {step23_net:>+8.4f}  {step23_ev:>9.2f}")

    for _, row in pm25_rows.sort_values('revert_thresh').iterrows():
        label = f"2.4 PEAK_MIN=2.5, R={row['revert_thresh']:.1f}"
        print(f"  {label:<40}  {int(row['n_entered']):>9}  {row['pct_base']:>12.1%}  "
              f"{row['avg_net']:>+8.4f}  {row['total_EV']:>9.2f}")

    # Best R for PEAK_MIN=2.5 is R=2.0 (the focal case)
    r20_row = pm25_rows[pm25_rows['revert_thresh'] == 2.0]
    if len(r20_row):
        delta_n = int(r20_row['n_entered'].values[0]) - step23_n
        print()
        print(f"  Coverage delta: n_entered(2.4 R=2.0) - {step23_n} = {delta_n:+d} trades (signed; may be negative)")
    print()
    print("  Note: per-trade avg_net of the delta subset is NOT computed.")
    print("  Reason: threshold_entry_research.py outputs only aggregate CSVs -- no per-trade")
    print("  records with crossing indices exist for 2.3 T=2.5. Even with per-trade data,")
    print("  2.3's entry_idx (outbound 2.5 SD crossing) and 2.4's crossing_idx (outbound 2.0 SD")
    print("  crossing) are different indices for the same signal -- a direct set-difference yields")
    print("  an empty intersection. The four-column comparison above is the correct diagnostic.")
    print()

    # ── Secondary analysis: effect of REVERT_THRESH ───────────────────────────
    print(SEP2)
    print("Effect of reversion threshold on hold period and entry timing:")
    print(SEP2)

    revert_effect = entered_df.groupby(['peak_min', 'revert_thresh'], dropna=False).agg(
        avg_entry_step    = ('actual_entry_step', 'mean'),
        avg_hold_cal      = ('total_hold_cal', 'mean'),
        avg_peak_sd_entry = ('peak_sd_at_entry', 'mean'),
        avg_net           = ('net_final', 'mean'),
    ).reset_index()

    re_sorted = revert_effect.sort_values(
        by=['peak_min', 'revert_thresh'], key=_sort_key
    )

    print(f"  {'PEAK_MIN':>8}  {'REVERT':>6}  {'avg_entry_step':>14}  {'avg_hold_cal':>12}  "
          f"{'avg_peak_sd_entry':>17}  {'avg_net':>8}")
    for _, row in re_sorted.iterrows():
        pm = row['peak_min']
        if pd.isna(pm):
            continue  # None rows have no meaningful entry_step pattern; skip in this section
        print(f"  {pm:>6.1f} SD  {row['revert_thresh']:>4.1f} SD  "
              f"{row['avg_entry_step']:>12.1f} steps  {row['avg_hold_cal']:>10.0f} d  "
              f"{row['avg_peak_sd_entry']:>15.2f} SD  {row['avg_net']:>+8.4f}")
    print()
    print("  Looser REVERT_THRESH (larger, closer to peak) = earlier entry,")
    print("  more SD remaining to travel to 0.0.")
    print("  Tighter REVERT_THRESH (smaller, closer to zero) = later entry, less SD remaining.")
    print()

    # ── Secondary analysis: winner/loser split for best combo ─────────────────
    best_row   = grid_sorted.loc[grid_sorted['total_EV'].idxmax()]
    best_pm    = best_row['peak_min']
    best_rt    = best_row['revert_thresh']

    if pd.isna(best_pm):
        best_combo_df = entered_df[
            entered_df['peak_min'].isna() &
            (entered_df['revert_thresh'] == best_rt)
        ]
    else:
        best_combo_df = entered_df[
            (entered_df['peak_min'] == best_pm) &
            (entered_df['revert_thresh'] == best_rt)
        ]

    wl = best_combo_df.groupby('is_winner', dropna=False).agg(
        n          = ('net_final', 'count'),
        avg_net    = ('net_final', 'mean'),
        avg_hold   = ('total_hold_cal', 'mean'),
        avg_peak   = ('peak_sd_at_entry', 'mean'),
    ).reset_index()

    # DataFrame.agg() doesn't support named-tuple syntax; compute scalars directly.
    all_n        = len(best_combo_df)
    all_avg_net  = best_combo_df['net_final'].mean()
    all_avg_hold = best_combo_df['total_hold_cal'].mean()
    all_avg_peak = best_combo_df['peak_sd_at_entry'].mean()
    all_win_rate = best_combo_df['is_winner'].mean()

    pm_label = 'None' if pd.isna(best_pm) else f"{best_pm:.1f}"
    print(SEP2)
    print(f"Winner/loser split for best combination: PEAK_MIN={pm_label}, REVERT_THRESH={best_rt:.1f}")
    print(SEP2)
    print(f"  {'outcome':<10}  {'n':>5}  {'avg_net':>8}  {'avg_hold_cal':>12}  {'peak_sd_entry':>13}  {'win_rate':>9}")

    winners_row = wl[wl['is_winner'] == True]
    losers_row  = wl[wl['is_winner'] == False]

    if len(winners_row):
        wr = winners_row.iloc[0]
        print(f"  {'Winners':<10}  {int(wr['n']):>5}  {wr['avg_net']:>+8.4f}  "
              f"{wr['avg_hold']:>10.0f} d  {wr['avg_peak']:>11.2f} SD  {'100%':>9}")
        w_peak_obs = wr['avg_peak']
    else:
        w_peak_obs = np.nan

    if len(losers_row):
        lr = losers_row.iloc[0]
        print(f"  {'Losers':<10}  {int(lr['n']):>5}  {lr['avg_net']:>+8.4f}  "
              f"{lr['avg_hold']:>10.0f} d  {lr['avg_peak']:>11.2f} SD  {'  0%':>9}")
        l_peak_obs = lr['avg_peak']
    else:
        l_peak_obs = np.nan

    print(f"  {'All':<10}  {all_n:>5}  {all_avg_net:>+8.4f}  "
          f"{all_avg_hold:>10.0f} d  {all_avg_peak:>11.2f} SD  "
          f"{all_win_rate:>9.1%}")
    print()

    obs_gap      = l_peak_obs - w_peak_obs if not (np.isnan(l_peak_obs) or np.isnan(w_peak_obs)) else np.nan
    hinsight_gap = l_peak_phase1 - w_peak_phase1

    print(f"  Compare to Phase 1 hindsight (computed from phase1_df, not hardcoded):")
    print(f"  Winner peak (Phase 1): {w_peak_phase1:.2f} SD  |  peak_sd_entry winners: "
          f"{w_peak_obs:.2f} SD" if not np.isnan(w_peak_obs) else
          f"  Winner peak (Phase 1): {w_peak_phase1:.2f} SD  |  peak_sd_entry winners: N/A")
    print(f"  Loser  peak (Phase 1): {l_peak_phase1:.2f} SD  |  peak_sd_entry losers:  "
          f"{l_peak_obs:.2f} SD" if not np.isnan(l_peak_obs) else
          f"  Loser  peak (Phase 1): {l_peak_phase1:.2f} SD  |  peak_sd_entry losers:  N/A")
    if not np.isnan(obs_gap):
        print(f"  Observable gap: {obs_gap:.2f} SD  vs  hindsight gap: {hinsight_gap:.2f} SD")
    print()

    # ── Decision summary block ────────────────────────────────────────────────
    best_total_ev  = best_row['total_EV']
    best_avg_net   = best_row['avg_net']
    best_n         = int(best_row['n_entered'])
    best_avg_hold  = best_row['avg_hold']
    best_pct_base  = best_row['pct_base']
    best_peak_entry= best_row['avg_peak_entry']

    # None combo for quality gate comparison
    none_r20 = grid_sorted[
        grid_sorted['peak_min'].isna() &
        (grid_sorted['revert_thresh'] == 2.0)
    ]
    if len(none_r20):
        none_r20_ev  = none_r20['total_EV'].values[0]
        none_r20_net = none_r20['avg_net'].values[0]
        none_r20_n   = int(none_r20['n_entered'].values[0])
    else:
        none_r20_ev = none_r20_net = none_r20_n = np.nan

    pm25_r20 = grid_sorted[
        (grid_sorted['peak_min'] == 2.5) &
        (grid_sorted['revert_thresh'] == 2.0)
    ]
    if len(pm25_r20):
        pm25_r20_ev  = pm25_r20['total_EV'].values[0]
        pm25_r20_net = pm25_r20['avg_net'].values[0]
        pm25_r20_n   = int(pm25_r20['n_entered'].values[0])
    else:
        pm25_r20_ev = pm25_r20_net = pm25_r20_n = np.nan

    gate_contribution = pm25_r20_ev - none_r20_ev if not (np.isnan(pm25_r20_ev) or np.isnan(none_r20_ev)) else np.nan

    print()
    print('=' * 78)
    print("STEP 2.4 -- REVERSION TRIGGER + QUALITY GATE -- DECISION SUMMARY")
    print('=' * 78)
    print()
    print("Single case result (PEAK_MIN=2.5, REVERT=2.0):")
    print(f"  n_entered: {sc_n} ({sc_n/n_baseline:.1%} of baseline)")
    print(f"  avg_net:   {sc_avg_net:+.4f} (vs baseline {bl_avg_net:+.4f}, delta = {sc_avg_net - bl_avg_net:+.4f})")
    print(f"  avg_hold:  {sc_avg_hld:.0f} d   (vs baseline {bl_avg_hld:.0f} d,   delta = {sc_avg_hld - bl_avg_hld:+.0f} d)")
    print(f"  total_EV:  {sc_total:.2f}    (vs baseline_ev {baseline_ev:.2f}, delta = {sc_total - baseline_ev:+.2f})")
    print(f"  win_rate:  {sc_wr:.1f}%")
    print(f"  avg_peak_sd_entry: {sc_peak:.2f} SD")
    print(f"  Phase 1 hindsight gap: {hinsight_gap:.2f} SD -- is observable gap close?")
    print()

    beats_str = "YES" if sc_total > baseline_ev else "NO"
    print(f"Does the reversion trigger beat baseline total EV? [{beats_str}]")
    print(f"  If YES: mechanism works; grid refines parameters. Proceed to Phase 3.")
    print(f"  If NO:  diagnose -- is the shortfall from coverage loss or per-trade quality?")
    print()

    print(f"Best grid combination (by total_EV):")
    print(f"  PEAK_MIN={pm_label}, REVERT={best_rt:.1f}")
    print(f"  total_EV: {best_total_ev:.2f} (delta = {best_total_ev - baseline_ev:+.2f} vs baseline_ev)")
    print(f"  avg_net:  {best_avg_net:+.4f}, n_entered: {best_n} ({best_pct_base:.1%}), avg_hold: {best_avg_hold:.0f} d")
    print()

    print("Quality gate value (PEAK_MIN vs None):")
    if not np.isnan(none_r20_ev):
        print(f"  PEAK_MIN=None, REVERT=2.0: total_EV = {none_r20_ev:.2f}, avg_net = {none_r20_net:+.4f}, n_entered = {none_r20_n}")
    if not np.isnan(pm25_r20_ev):
        print(f"  PEAK_MIN=2.5,  REVERT=2.0: total_EV = {pm25_r20_ev:.2f}, avg_net = {pm25_r20_net:+.4f}, n_entered = {pm25_r20_n}")
    if not np.isnan(gate_contribution):
        print(f"  Gate contribution to total_EV: delta = {gate_contribution:+.2f}")
    print()

    print("Observable winner/loser gap at entry:")
    if not np.isnan(obs_gap):
        print(f"  Winners: {w_peak_obs:.2f} SD  |  Losers: {l_peak_obs:.2f} SD  |  Gap: {obs_gap:.2f} SD")
        print(f"  vs hindsight: {hinsight_gap:.2f} SD")
    print()

    print("Next step: Phase 3.1 (best 2.4 entry + 1.0 SD exit vs 0.0 SD exit)")
    print('=' * 78)
    print()

    # ── Save CSVs ─────────────────────────────────────────────────────────────

    # 1. Full grid
    grid_out = grid_sorted[[
        'peak_min', 'revert_thresh', 'n_entered', 'pct_base', 'pct_no_sig',
        'pct_mh_skip', 'avg_net', 'avg_hold', 'win_rate', 'avg_peak_entry',
        'total_EV', 'vs_base_EV', 'beats',
    ]].copy()
    grid_out.to_csv(RESULTS_DIR / 'reversion_trigger_grid_commodities.csv', index=False)

    # 2. Coverage delta vs 2.3
    cov_rows = []
    cov_rows.append({'rule': '2.3 T=2.5', 'peak_min': 2.5, 'revert_thresh': np.nan,
                     'n_entered': step23_n, 'pct_baseline': step23_pct,
                     'avg_net': step23_net, 'total_EV': step23_ev})
    for _, row in pm25_rows.sort_values('revert_thresh').iterrows():
        cov_rows.append({
            'rule': f"2.4 PEAK_MIN=2.5 R={row['revert_thresh']:.1f}",
            'peak_min': 2.5, 'revert_thresh': row['revert_thresh'],
            'n_entered': int(row['n_entered']), 'pct_baseline': row['pct_base'],
            'avg_net': row['avg_net'], 'total_EV': row['total_EV'],
        })
    pd.DataFrame(cov_rows).to_csv(
        RESULTS_DIR / 'reversion_trigger_coverage_commodities.csv', index=False)

    # 3. Winner/loser split for best combo
    wl_out = wl.copy()
    wl_out['peak_min']      = best_pm
    wl_out['revert_thresh'] = best_rt
    wl_out.to_csv(RESULTS_DIR / 'reversion_trigger_wl_commodities.csv', index=False)

    # 4. Machine-readable inputs for Phase 3
    phase3_rows = []
    for _, row in grid_sorted.iterrows():
        phase3_rows.append({
            'is_baseline':       False,
            'peak_min':          row['peak_min'],
            'revert_thresh':     row['revert_thresh'],
            'n_entered':         int(row['n_entered']),
            'pct_baseline':      row['pct_base'],
            'pct_no_sig':        row['pct_no_sig'],
            'pct_mh_skip':       row['pct_mh_skip'],
            'avg_net':           row['avg_net'],
            'avg_hold_cal':      row['avg_hold'],
            'win_rate':          row['win_rate'],
            'avg_peak_sd_entry': row['avg_peak_entry'],
            'total_EV':          row['total_EV'],
            'vs_base_EV':        row['vs_base_EV'],
            'beats_baseline_ev': row['beats'],
            'recommended':       (row['total_EV'] == grid_sorted['total_EV'].max()),
        })

    baseline_row_df = pd.DataFrame([{
        'is_baseline': True, 'peak_min': np.nan, 'revert_thresh': np.nan,
        'n_entered': n_baseline, 'pct_baseline': 1.0,
        'pct_no_sig': np.nan, 'pct_mh_skip': np.nan,
        'avg_net': phase1_df['net_final'].mean(),
        'avg_hold_cal': phase1_df['total_hold_cal'].mean(),
        'win_rate': (phase1_df['net_final'] > 0).mean(),
        'avg_peak_sd_entry': 2.00,
        'total_EV': baseline_ev, 'vs_base_EV': 0.0,
        'beats_baseline_ev': False, 'recommended': False,
    }])

    phase3_df = pd.concat([pd.DataFrame(phase3_rows), baseline_row_df], ignore_index=True)
    phase3_df.to_csv(
        RESULTS_DIR / 'reversion_trigger_24_inputs_commodities.csv', index=False)

    print(f"Saved: reversion_trigger_grid_commodities.csv        ({len(grid_sorted)} rows)")
    print(f"Saved: reversion_trigger_coverage_commodities.csv    ({len(cov_rows)} rows)")
    print(f"Saved: reversion_trigger_wl_commodities.csv          ({len(wl_out)} rows)")
    print(f"Saved: reversion_trigger_24_inputs_commodities.csv   ({len(phase3_df)} rows, incl. baseline)")


if __name__ == '__main__':
    main()
