"""
reversion_path_research.py
==========================
Measures gross return accrual shape over commodity spread trade holding periods.

Follows partial exit backtest (May 2026, best +0.0003). Sweeps all timing fractions
to find the theoretical sweet spot for a 50% partial exit.

Key question: front-loaded (frac_cap_50 > 0.60), linear (~0.50), or back-loaded (<0.40)?

Usage (from project root):
  C:\\Users\\gordo\\AppData\\Local\\Python\\bin\\python.exe research/reversion_path_research.py

Outputs:
  results/reversion_path_aggregate.csv   -- Section 1 full profile
  results/reversion_path_per_pair.csv    -- Section 2 per-pair at t=0.25/0.50/0.75
  results/sweet_spot_per_pair.csv        -- Section 3 per-pair improvement sweep
  results/sweet_spot_aggregate.csv       -- Section 3 aggregate
  results/proximity_interaction.csv      -- Section 4 gate at peak timing
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).parent
ROOT         = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

from engine.backtest  import load_asset_prices
from engine.numba_core import detect_trades, COL_ENTRY_IDX, COL_EXIT_IDX, COL_SIDE
from trailing_stop_research import (
    build_spread, DAILY_FIN, SPREAD_COST,
    DATA_PATH, RESULTS_DIR, XING_SD, EXIT_SD, MAX_HOLD,
)

# -- Config --------------------------------------------------------------------

ALL_PAIRS = [
    ("NATGAS",   "COPPER"),   ("NATGAS",   "BRENT"),
    ("NATGAS",   "COFFEE"),   ("NATGAS",   "SUGAR"),
    ("NATGAS",   "SOYBEANS"), ("SILVER",   "COFFEE"),
    ("WHEAT",    "BRENT"),    ("SOYBEANS", "WHEAT"),
    ("SOYBEANS", "PLATINUM"), ("GOLD",     "NATGAS"),
    ("COPPER",   "NATGAS"),   ("BRENT",    "NATGAS"),
    ("COFFEE",   "NATGAS"),   ("SUGAR",    "NATGAS"),
    ("SOYBEANS", "NATGAS"),   ("COFFEE",   "SILVER"),
    ("BRENT",    "WHEAT"),    ("WHEAT",    "SOYBEANS"),
    ("PLATINUM", "SOYBEANS"), ("NATGAS",   "GOLD"),
]

TIME_FRACS_PROFILE = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
TIME_FRACS_SWEEP   = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
SD_GATES_S4        = [0.25, 0.50, 0.75, 1.00]
MIN_HOLD_TD        = 10     # exclude trades shorter than this many trading days
PARTIAL_F          = 0.50
PRIOR_BEST_PE      = 0.0003   # partial exit best: tgp=0.50, sd=0.75, F=0.75
PRIOR_BEST_TS      = -0.0011  # trailing stop best


# -- Data collection -----------------------------------------------------------

def collect_pair_data(prices):
    """
    Run baseline trades for all 20 pairs. Record full daily path per trade.

    Returns dict: pair_label -> (trade_list, dist_sd, day_ints)

    path[j] = gross return after (j+1) trading days from entry, j in 0..total_hold-1
    path[0]   = return after 1 trading day
    path[-1]  = gross_final (return at exit)
    """
    pair_data = {}
    for i, (long_inst, short_inst) in enumerate(ALL_PAIRS):
        lbl = f"{long_inst}/{short_inst}"
        print(f"  [{i+1:02d}/{len(ALL_PAIRS)}] {lbl}", end="", flush=True)

        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
        trades_raw, n = detect_trades(cum, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)

        if n == 0:
            print("  -> 0 trades")
            pair_data[lbl] = ([], dist_sd, day_ints)
            continue

        trade_list = []
        for k in range(n):
            entry_idx  = int(trades_raw[k, COL_ENTRY_IDX])
            exit_idx   = int(trades_raw[k, COL_EXIT_IDX])
            side       = int(trades_raw[k, COL_SIDE])
            total_hold = exit_idx - entry_idx            # trading days
            cal_hold   = int(day_ints[exit_idx] - day_ints[entry_idx])

            if total_hold < MIN_HOLD_TD:
                continue

            path = np.empty(total_hold)
            for j in range(total_hold):
                idx = entry_idx + j + 1   # +1: path[0] = day 1, path[-1] = exit day
                if side == +1:
                    path[j] = (cum[idx] - cum[entry_idx]) / cum[entry_idx]
                else:
                    path[j] = (cum[entry_idx] - cum[idx]) / cum[entry_idx]

            trade_list.append({
                'entry_idx':   entry_idx,
                'exit_idx':    exit_idx,
                'side':        side,
                'total_hold':  total_hold,
                'cal_hold':    cal_hold,
                'gross_final': float(path[-1]),
                'path':        path,
            })

        pair_data[lbl] = (trade_list, dist_sd, day_ints)
        print(f"  -> {len(trade_list)} valid trades")

    return pair_data


# -- Section 1 & 2 helpers ----------------------------------------------------

def _frac_cap(trade, t_frac):
    """Fraction of gross_final captured at t_frac ? total_hold trading days."""
    H  = trade['total_hold']
    gf = trade['gross_final']
    if abs(gf) < 1e-12:
        return None
    N     = max(1, min(int(round(t_frac * H)), H))   # days elapsed, clamped [1, H]
    r_at_t = trade['path'][N - 1]                     # path index = days - 1
    return r_at_t / gf


def _profile_rows(trades, time_fracs, group_label):
    rows = []
    for tf in time_fracs:
        fcs = [_frac_cap(tr, tf) for tr in trades]
        fcs = [f for f in fcs if f is not None]
        if not fcs:
            continue
        arr = np.array(fcs)
        rows.append({
            'group':                group_label,
            'time_frac':            tf,
            'mean_frac_captured':   round(float(arr.mean()), 4),
            'median_frac_captured': round(float(np.median(arr)), 4),
            'n_trades':             len(arr),
            'pct_above_linear':     round(float((arr > tf).mean() * 100), 1),
        })
    return rows


def section1(pair_data):
    winners, losers = [], []
    for lbl, (trades, _, _) in pair_data.items():
        for tr in trades:
            (winners if tr['gross_final'] > 0 else losers).append(tr)
    win_rows  = _profile_rows(winners, TIME_FRACS_PROFILE, "winners")
    loss_rows = _profile_rows(losers,  TIME_FRACS_PROFILE, "losers")
    return win_rows, loss_rows


def section2(pair_data):
    fracs = [0.25, 0.50, 0.75]
    rows = []
    for lbl, (trades, _, _) in pair_data.items():
        winners = [tr for tr in trades if tr['gross_final'] > 0]
        if not winners:
            rows.append(dict(pair=lbl, n_win=0,
                             frac_cap_25=None, frac_cap_50=None, frac_cap_75=None,
                             profile_type='n/a'))
            continue
        vals = {}
        for tf in fracs:
            fcs = [_frac_cap(tr, tf) for tr in winners]
            fcs = [f for f in fcs if f is not None]
            vals[tf] = float(np.median(fcs)) if fcs else 0.0

        fc50 = vals[0.50]
        ptype = 'front' if fc50 > 0.60 else ('back' if fc50 < 0.40 else 'linear')
        rows.append(dict(
            pair         = lbl,
            n_win        = len(winners),
            frac_cap_25  = round(vals[0.25], 4),
            frac_cap_50  = round(vals[0.50], 4),
            frac_cap_75  = round(vals[0.75], 4),
            profile_type = ptype,
        ))
    return pd.DataFrame(rows)


# -- Section 3: sweet spot sweep -----------------------------------------------

def _improvement_at_N(trade, N_td):
    """
    Theoretical net improvement (vs hold-to-zero) for F=0.50 partial exit at
    trading day N_td. Calendar days for financing computed proportionally.
    """
    H     = trade['total_hold']
    cal_H = trade['cal_hold']
    gf    = trade['gross_final']

    N_actual_td  = min(N_td, H)
    N_actual_cal = round(N_actual_td * cal_H / H) if H > 0 else 0

    r_partial = trade['path'][N_actual_td - 1]
    gross_pe  = PARTIAL_F * r_partial + (1 - PARTIAL_F) * gf
    fin_pe    = DAILY_FIN * (PARTIAL_F * N_actual_cal + (1 - PARTIAL_F) * cal_H)
    net_pe    = gross_pe - SPREAD_COST - fin_pe
    net_bl    = gf       - SPREAD_COST - DAILY_FIN * cal_H
    return net_pe - net_bl


def section3(pair_data):
    per_pair_rows = []
    agg_by_tf     = {tf: [] for tf in TIME_FRACS_SWEEP}

    for lbl, (trades, _, _) in pair_data.items():
        if not trades:
            continue
        td_holds  = [tr['total_hold'] for tr in trades]
        median_td = float(np.median(td_holds))

        for tf in TIME_FRACS_SWEEP:
            N_td    = max(1, int(round(tf * median_td)))
            imps    = [_improvement_at_N(tr, N_td) for tr in trades]
            avg_imp = float(np.mean(imps))
            agg_by_tf[tf].append(avg_imp)
            per_pair_rows.append(dict(
                pair            = lbl,
                median_hold     = round(median_td, 1),
                time_frac       = tf,
                N_td            = N_td,
                avg_improvement = round(avg_imp, 5),
            ))

    agg_rows = [
        dict(time_frac=tf,
             avg_improvement=round(float(np.mean(agg_by_tf[tf])), 5)
             if agg_by_tf[tf] else 0.0)
        for tf in TIME_FRACS_SWEEP
    ]
    return pd.DataFrame(per_pair_rows), pd.DataFrame(agg_rows)


# -- Section 4: proximity gate -------------------------------------------------

def section4(pair_data, peak_t_frac):
    """
    At the peak timing fraction, check what fraction of trades have
    |dist_sd[entry_idx + N_td]| <= sd_gate, and whether the gate selects
    better-than-average trades.
    """
    unfilt_imps = []
    gate_hits   = {g: [] for g in SD_GATES_S4}

    for lbl, (trades, dist_sd, day_ints) in pair_data.items():
        if not trades:
            continue
        median_td = float(np.median([tr['total_hold'] for tr in trades]))
        N_td      = max(1, int(round(peak_t_frac * median_td)))

        for tr in trades:
            H         = tr['total_hold']
            entry_idx = tr['entry_idx']

            imp = _improvement_at_N(tr, N_td)
            unfilt_imps.append(imp)

            if N_td < H:
                d = dist_sd[entry_idx + N_td]
                for g in SD_GATES_S4:
                    gate_hits[g].append(not np.isnan(d) and abs(d) <= g)
            else:
                for g in SD_GATES_S4:
                    gate_hits[g].append(False)

    avg_unfilt = float(np.mean(unfilt_imps)) if unfilt_imps else 0.0
    unfilt_arr = np.array(unfilt_imps)

    rows = []
    for g in SD_GATES_S4:
        hits      = np.array(gate_hits[g], dtype=bool)
        fire_rate = float(hits.mean()) if len(hits) else 0.0
        gated_avg = float(unfilt_arr[hits].mean()) if hits.sum() > 0 else 0.0
        net_filt  = gated_avg * fire_rate   # portfolio-level view

        if fire_rate < 0.20:
            verdict = 'too selective'
        elif gated_avg > avg_unfilt * 1.10:
            verdict = 'right (selects better trades)'
        elif gated_avg < avg_unfilt * 0.90:
            verdict = 'adverse (selects worse trades)'
        else:
            verdict = 'neutral'

        rows.append(dict(
            sd_gate            = g,
            fire_rate_pct      = round(fire_rate * 100, 1),
            net_imp_unfiltered = round(avg_unfilt, 5),
            gated_avg_imp      = round(gated_avg, 5),
            net_imp_filtered   = round(net_filt, 5),
            verdict            = verdict,
        ))
    return pd.DataFrame(rows)


# -- Terminal reporting --------------------------------------------------------

def _print_profile_table(rows, label):
    print(f"\n  {label}:")
    print(f"  {'time_frac':>10}  {'mean_frac_cap':>14}  "
          f"{'median_frac_cap':>16}  {'n_trades':>9}  {'pct_above_linear':>17}")
    for r in rows:
        print(f"  {r['time_frac']:>10.2f}  {r['mean_frac_captured']:>14.4f}  "
              f"{r['median_frac_captured']:>16.4f}  {r['n_trades']:>9}  "
              f"{r['pct_above_linear']:>16.1f}%")


def print_section1(win_rows, loss_rows):
    W = 80
    print("\n" + "=" * W)
    print("SECTION 1 -- AGGREGATE REVERSION PROFILE (all pairs combined)")
    print("=" * W)
    _print_profile_table(win_rows,  "WINNING trades (gross_final > 0)")
    _print_profile_table(loss_rows, "LOSING trades  (gross_final <= 0)")


def print_section2(s2_df):
    W = 80
    print("\n" + "=" * W)
    print("SECTION 2 -- PER-PAIR PROFILES  (winning trades, condensed at t=0.25/0.50/0.75)")
    print("=" * W)
    print(f"\n  {'pair':>20}  {'n_win':>6}  {'fc_25':>7}  {'fc_50':>7}  "
          f"{'fc_75':>7}  {'type':>8}")
    for _, r in s2_df.iterrows():
        if r['n_win'] == 0 or r['frac_cap_50'] is None:
            print(f"  {r['pair']:>20}  {int(r['n_win'] or 0):>6}  "
                  f"{'n/a':>7}  {'n/a':>7}  {'n/a':>7}  {'n/a':>8}")
        else:
            print(f"  {r['pair']:>20}  {int(r['n_win']):>6}  "
                  f"{r['frac_cap_25']:>7.4f}  {r['frac_cap_50']:>7.4f}  "
                  f"{r['frac_cap_75']:>7.4f}  {r['profile_type']:>8}")
    valid = s2_df[s2_df['n_win'] > 0]
    if len(valid):
        vc = valid['profile_type'].value_counts().to_dict()
        print(f"\n  Profile types (of {len(valid)} pairs with winners): {vc}")


def print_section3(s3_pair_df, s3_agg_df):
    W = 80
    print("\n" + "=" * W)
    print("SECTION 3 -- SWEET SPOT ANALYSIS  (F=0.50, no proximity filter)")
    print("=" * W)

    tf_cols = sorted(s3_pair_df['time_frac'].unique())
    header_tfs = "  ".join(f"t={tf:.2f}" for tf in tf_cols)
    print(f"\n  {'pair':>20}  {'med_hold':>9}  {header_tfs}")

    for lbl in s3_pair_df['pair'].unique():
        sub = s3_pair_df[s3_pair_df['pair'] == lbl].sort_values('time_frac')
        med = sub['median_hold'].iloc[0]
        vals = "  ".join(f"{v:+.4f}" for v in sub['avg_improvement'])
        print(f"  {lbl:>20}  {med:>9.1f}  {vals}")

    print(f"\n  Aggregate (mean across all pairs):")
    agg_line = "  ".join(
        f"t={r['time_frac']:.2f}: {r['avg_improvement']:+.5f}"
        for _, r in s3_agg_df.iterrows()
    )
    print(f"  {agg_line}")

    peak_idx = s3_agg_df['avg_improvement'].idxmax()
    peak_row = s3_agg_df.loc[peak_idx]
    print(f"\n  Peak timing: t={peak_row['time_frac']:.2f}  "
          f"->  avg net improvement = {peak_row['avg_improvement']:+.5f}")

    return float(peak_row['time_frac'])


def print_section4(s4_df, peak_t):
    W = 80
    print("\n" + "=" * W)
    print(f"SECTION 4 -- PROXIMITY GATE INTERACTION  "
          f"(peak timing t={peak_t:.2f}, F={PARTIAL_F})")
    print("=" * W)
    print(f"\n  {'sd_gate':>8}  {'fire_rate':>10}  {'unfilt_imp':>11}  "
          f"{'gated_avg':>10}  {'filtered':>9}  verdict")
    for _, r in s4_df.iterrows():
        print(f"  {r['sd_gate']:>8.2f}  {r['fire_rate_pct']:>9.1f}%  "
              f"{r['net_imp_unfiltered']:>11.5f}  {r['gated_avg_imp']:>10.5f}  "
              f"{r['net_imp_filtered']:>9.5f}  {r['verdict']}")


def print_section5(win_rows, s2_df, s3_agg_df, s4_df, peak_t):
    W = 80
    print("\n" + "=" * W)
    print("SECTION 5 -- RECOMMENDATION")
    print("=" * W)

    # Path shape from aggregate t=0.50
    fc50_row = next((r for r in win_rows if abs(r['time_frac'] - 0.50) < 0.01), None)
    fc50_mean = fc50_row['mean_frac_captured'] if fc50_row else 0.5
    fc50_med  = fc50_row['median_frac_captured'] if fc50_row else 0.5
    shape = ('front-loaded' if fc50_mean > 0.60
             else 'back-loaded' if fc50_mean < 0.40
             else 'approximately linear')

    valid = s2_df[s2_df['n_win'] > 0]
    vc    = valid['profile_type'].value_counts().to_dict() if len(valid) else {}

    peak_row    = s3_agg_df.loc[s3_agg_df['avg_improvement'].idxmax()]
    peak_imp    = float(peak_row['avg_improvement'])

    row_75 = s4_df[s4_df['sd_gate'] == 0.75]
    if len(row_75):
        r75 = row_75.iloc[0]
        gate_line = (f"sd_gate=0.75: fire_rate={r75['fire_rate_pct']:.1f}%  "
                     f"gated_avg={r75['gated_avg_imp']:+.5f}  "
                     f"filtered={r75['net_imp_filtered']:+.5f}  ({r75['verdict']})")
    else:
        gate_line = "n/a"

    if peak_imp > PRIOR_BEST_PE + 0.0005:
        verdict = (
            f"Sweet spot ({peak_imp:+.5f}) materially exceeds prior partial exit best "
            f"({PRIOR_BEST_PE:+.4f}). Implement partial exit at t={peak_t:.2f} ? median_hold."
        )
    elif abs(peak_imp) < 0.0005:
        verdict = (
            f"Even at the theoretical sweet spot, improvement ({peak_imp:+.5f}) is "
            "negligible. The near-linear reversion path means financing saving and "
            "gross return sacrifice roughly cancel at any timing. "
            "CONCLUSION: hold-to-zero is the correct exit rule for this strategy."
        )
    else:
        verdict = (
            f"Sweet spot ({peak_imp:+.5f}) is marginal and near the prior best "
            f"({PRIOR_BEST_PE:+.4f}). The {shape} path leaves limited room for "
            "timing improvement. Not worth implementing."
        )

    print(f"""
  1. PATH SHAPE
     {shape} (winning trades: mean frac_cap at t=0.50 = {fc50_mean:.4f}, median = {fc50_med:.4f})
     Per-pair: front={vc.get('front', 0)}, linear={vc.get('linear', 0)}, back={vc.get('back', 0)}

  2. THEORETICAL SWEET SPOT
     F=0.50, no proximity filter
     Peak timing: t={peak_t:.2f} of median hold
     Avg net improvement: {peak_imp:+.5f}

  3. PROXIMITY GATE AT PEAK TIMING
     {gate_line}

  4. COMPARISON
     Hold-to-zero baseline:                0.00000
     Prior trailing stop best:             {PRIOR_BEST_TS:+.5f}
     Prior partial exit best:              {PRIOR_BEST_PE:+.5f}
     This analysis sweet spot (unfiltered):{peak_imp:+.5f}

  5. VERDICT
     {verdict}
""")


# -- Main ----------------------------------------------------------------------

def main():
    print("Reversion Path Analysis")
    print("=" * 60)
    print("Loading commodity prices...")
    prices, _ = load_asset_prices(DATA_PATH)
    print(f"  {len(prices)} trading days")

    print(f"\nCollecting trajectories (min_hold={MIN_HOLD_TD} trading days)...")
    pair_data = collect_pair_data(prices)

    n_total = sum(len(t) for t, _, _ in pair_data.values())
    print(f"  Total valid trades: {n_total}")

    print("\nSection 1 -- aggregate profile...")
    win_rows, loss_rows = section1(pair_data)
    print_section1(win_rows, loss_rows)

    print("\nSection 2 -- per-pair profiles...")
    s2_df = section2(pair_data)
    print_section2(s2_df)

    print("\nSection 3 -- sweet spot sweep...")
    s3_pair_df, s3_agg_df = section3(pair_data)
    peak_t = print_section3(s3_pair_df, s3_agg_df)

    print("\nSection 4 -- proximity gate interaction...")
    s4_df = section4(pair_data, peak_t)
    print_section4(s4_df, peak_t)

    print_section5(win_rows, s2_df, s3_agg_df, s4_df, peak_t)

    # Save CSVs
    agg_csv = pd.DataFrame(win_rows + loss_rows)
    agg_csv.to_csv(RESULTS_DIR / "reversion_path_aggregate.csv",  index=False)
    s2_df.to_csv(  RESULTS_DIR / "reversion_path_per_pair.csv",   index=False)
    s3_pair_df.to_csv(RESULTS_DIR / "sweet_spot_per_pair.csv",    index=False)
    s3_agg_df.to_csv( RESULTS_DIR / "sweet_spot_aggregate.csv",   index=False)
    s4_df.to_csv(     RESULTS_DIR / "proximity_interaction.csv",  index=False)

    print(f"CSVs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
