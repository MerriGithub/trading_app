"""
entry_velocity_research.py
==========================
Entry velocity analysis for commodity spread strategy.

Tests whether the rate of change of dist_SD at the +/-2.0 SD crossing predicts
trade outcome. A spread crossing while still accelerating outward (high velocity)
may be a worse entry than one where the spread is already slowing or turning.

Usage (from project root):
  C:\\Users\\gordo\\AppData\\Local\\Python\\bin\\python.exe research/entry_velocity_research.py

Outputs (all in results/):
  entry_velocity_distribution.csv
  entry_velocity_quintiles.csv
  entry_velocity_filter_sim.csv
  entry_extension_distribution.csv
  entry_extension_by_quintile.csv
  entry_extension_vs_outcome.csv
  entry_exit_combined_sim.csv
  entry_velocity_per_pair.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).parent
ROOT         = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

from engine.backtest   import load_asset_prices
from engine.numba_core import (
    detect_trades,
    COL_ENTRY_IDX, COL_EXIT_IDX, COL_SIDE, COL_GROSS_RETURN,
)
from trailing_stop_research import (
    build_spread, DAILY_FIN, SPREAD_COST,
    DATA_PATH, RESULTS_DIR, XING_SD, EXIT_SD, MAX_HOLD,
)

# ── Config ────────────────────────────────────────────────────────────────────

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

LOOKBACKS    = [3, 5, 10]
V_THRESHOLDS = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

# Best partial exit combo from prior backtest
PE_TGP    = 0.50
PE_SDGATE = 0.75
PE_F      = 0.75

PRIOR_BEST_PE = 0.0003
PRIOR_BEST_TS = -0.0011


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_velocity(dist_sd, entry_idx, lookback, side):
    """
    Outward velocity of dist_SD over [entry_idx-lookback, entry_idx].
    Positive = spread was moving AWAY from mean in the window.
    Negative = spread was already turning BACK before crossing.
    Returns NaN if lookback data unavailable.
    """
    if entry_idx < lookback:
        return np.nan
    d_prev = dist_sd[entry_idx - lookback]
    if np.isnan(d_prev):
        return np.nan
    d_entry = dist_sd[entry_idx]
    if side == -1:   # above +2 SD: outward = dist_sd increasing
        return float(d_entry - d_prev)
    else:            # below -2 SD: outward = dist_sd decreasing (more negative)
        return float(d_prev - d_entry)


def _pe_outcome(tr, dist_sd, cum, day_ints, N_cal, F, sd_gate):
    """
    Simulate partial exit outcome for a single trade.
    Scans from entry to exit looking for the first day where
    calendar_days_elapsed >= N_cal AND |dist_sd| <= sd_gate.
    """
    ei = tr['entry_idx']
    xi = tr['exit_idx']
    s  = tr['side']
    ec = cum[ei]
    fc = cum[xi]

    r_final  = (fc - ec) / ec if s == +1 else (ec - fc) / ec
    cal_full = int(day_ints[xi] - day_ints[ei])

    partial_fired = False
    for i in range(ei, xi + 1):
        cd = int(day_ints[i] - day_ints[ei])
        if cd >= N_cal and abs(dist_sd[i]) <= sd_gate:
            partial_fired    = True
            partial_hold_cal = cd
            r_partial        = ((cum[i] - ec) / ec if s == +1 else (ec - cum[i]) / ec)
            break

    if partial_fired:
        gross = F * r_partial + (1 - F) * r_final
        fin   = DAILY_FIN * (F * partial_hold_cal + (1 - F) * cal_full)
    else:
        gross = r_final
        fin   = DAILY_FIN * cal_full

    return float(gross - SPREAD_COST - fin)


# ── Data collection ────────────────────────────────────────────────────────────

def collect_pair_data(prices):
    """
    Returns (pair_data, all_trades).
    pair_data: pair_label -> (trade_list, dist_sd, day_ints, cum)
    all_trades: flat list of all trade dicts across all pairs
    """
    pair_data  = {}
    all_trades = []

    for i, (long_inst, short_inst) in enumerate(ALL_PAIRS):
        lbl = f"{long_inst}/{short_inst}"
        print(f"  [{i+1:02d}/{len(ALL_PAIRS)}] {lbl}", end="", flush=True)

        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
        trades_raw, n = detect_trades(cum, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)

        if n == 0:
            print("  -> 0 trades")
            pair_data[lbl] = ([], dist_sd, day_ints, cum)
            continue

        trade_list = []
        for k in range(n):
            entry_idx      = int(trades_raw[k, COL_ENTRY_IDX])
            exit_idx       = int(trades_raw[k, COL_EXIT_IDX])
            side           = int(trades_raw[k, COL_SIDE])
            gross          = float(trades_raw[k, COL_GROSS_RETURN])
            total_hold_cal = int(day_ints[exit_idx] - day_ints[entry_idx])
            total_hold_td  = exit_idx - entry_idx

            net = gross - SPREAD_COST - DAILY_FIN * total_hold_cal

            # Velocity for each lookback
            vels = {lb: compute_velocity(dist_sd, entry_idx, lb, side)
                    for lb in LOOKBACKS}

            # max_ext and days_to_peak: scan from entry (inclusive) to exit (inclusive)
            seg     = np.abs(dist_sd[entry_idx : exit_idx + 1])
            valid   = ~np.isnan(seg)
            if valid.any():
                peak_rel    = int(np.nanargmax(seg))
                max_ext     = float(np.nanmax(seg))
                days_to_peak = peak_rel
            else:
                max_ext = float(abs(dist_sd[entry_idx]))
                days_to_peak = 0

            tr = dict(
                pair           = lbl,
                side           = side,
                entry_idx      = entry_idx,
                exit_idx       = exit_idx,
                total_hold_td  = total_hold_td,
                total_hold_cal = total_hold_cal,
                gross_return   = gross,
                net_return     = net,
                is_winner      = bool(net > 0),
                velocity_3     = vels[3],
                velocity_5     = vels[5],
                velocity_10    = vels[10],
                entry_dist_sd  = float(abs(dist_sd[entry_idx])),
                max_ext        = max_ext,
                days_to_peak   = days_to_peak,
            )
            trade_list.append(tr)
            all_trades.append(tr)

        pair_data[lbl] = (trade_list, dist_sd, day_ints, cum)
        print(f"  -> {len(trade_list)} trades")

    return pair_data, all_trades


# ── Section 1: Velocity distribution ─────────────────────────────────────────

def section1(all_trades):
    valid = [t for t in all_trades if not np.isnan(t['velocity_5'])]
    vels  = np.array([t['velocity_5'] for t in valid])

    agg = dict(
        pair='all_pairs', n_trades=len(valid),
        mean_velocity=round(float(vels.mean()), 4),
        median_velocity=round(float(np.median(vels)), 4),
        pct_neg=round(float((vels < 0).mean() * 100), 1),
        pct_0_to_05=round(float(((vels >= 0) & (vels < 0.5)).mean() * 100), 1),
        pct_05_to_10=round(float(((vels >= 0.5) & (vels < 1.0)).mean() * 100), 1),
        pct_gt10=round(float((vels >= 1.0).mean() * 100), 1),
        pct_gt15=round(float((vels >= 1.5).mean() * 100), 1),
    )

    pair_rows = []
    for lbl in [p for p in dict.fromkeys(t['pair'] for t in all_trades)]:
        pv = [t['velocity_5'] for t in valid if t['pair'] == lbl]
        if not pv:
            continue
        a = np.array(pv)
        pair_rows.append(dict(
            pair='all_pairs',
            **{k: None for k in agg if k not in ('pair',)},  # placeholder
        ))
        pair_rows[-1] = dict(
            pair=lbl, n_trades=len(a),
            mean_velocity=round(float(a.mean()), 4),
            median_velocity=round(float(np.median(a)), 4),
            pct_neg=round(float((a < 0).mean() * 100), 1),
            pct_0_to_05=round(float(((a >= 0) & (a < 0.5)).mean() * 100), 1),
            pct_05_to_10=round(float(((a >= 0.5) & (a < 1.0)).mean() * 100), 1),
            pct_gt10=round(float((a >= 1.0).mean() * 100), 1),
            pct_gt15=round(float((a >= 1.5).mean() * 100), 1),
        )

    return [agg] + pair_rows, valid, vels


# ── Section 2: Quintile analysis ──────────────────────────────────────────────

def _quintile_df(all_trades, vel_key, lookback):
    valid = [t for t in all_trades if not np.isnan(t[vel_key])]
    if len(valid) < 5:
        return pd.DataFrame()
    vels    = np.array([t[vel_key] for t in valid])
    bounds  = np.percentile(vels, [20, 40, 60, 80])
    qlabels = np.digitize(vels, bounds)   # 0=Q1 .. 4=Q5

    rows = []
    for q in range(5):
        mask = qlabels == q
        qt   = [valid[i] for i in range(len(valid)) if mask[i]]
        if not qt:
            continue
        qv   = vels[mask]
        nets  = np.array([t['net_return']   for t in qt])
        gross = np.array([t['gross_return'] for t in qt])
        holds = np.array([t['total_hold_cal'] for t in qt])
        mexts = np.array([t['max_ext']      for t in qt])
        dtps  = np.array([t['days_to_peak'] for t in qt])
        wins  = np.array([t['is_winner']    for t in qt])
        rows.append(dict(
            lookback        = lookback,
            quintile        = f"Q{q+1}",
            vel_range       = f"[{qv.min():.2f},{qv.max():.2f}]",
            n_trades        = len(qt),
            win_rate        = round(float(wins.mean() * 100), 1),
            avg_gross       = round(float(gross.mean()), 4),
            avg_net         = round(float(nets.mean()), 4),
            avg_hold        = round(float(holds.mean()), 1),
            avg_max_ext     = round(float(mexts.mean()), 3),
            avg_days_to_peak= round(float(dtps.mean()), 1),
        ))

    # Baseline row (all trades, not just valid-velocity ones)
    all_nets  = np.array([t['net_return']     for t in all_trades])
    all_gross = np.array([t['gross_return']   for t in all_trades])
    all_holds = np.array([t['total_hold_cal'] for t in all_trades])
    all_mexts = np.array([t['max_ext']        for t in all_trades])
    all_dtps  = np.array([t['days_to_peak']   for t in all_trades])
    all_wins  = np.array([t['is_winner']      for t in all_trades])
    rows.append(dict(
        lookback=lookback, quintile='All',
        vel_range='[all]',
        n_trades=len(all_trades),
        win_rate=round(float(all_wins.mean() * 100), 1),
        avg_gross=round(float(all_gross.mean()), 4),
        avg_net=round(float(all_nets.mean()), 4),
        avg_hold=round(float(all_holds.mean()), 1),
        avg_max_ext=round(float(all_mexts.mean()), 3),
        avg_days_to_peak=round(float(all_dtps.mean()), 1),
    ))
    return pd.DataFrame(rows)


def section2(all_trades):
    frames = [_quintile_df(all_trades, f'velocity_{lb}', lb) for lb in LOOKBACKS]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


# ── Section 3: Filter simulation ─────────────────────────────────────────────

def section3(all_trades):
    valid   = [t for t in all_trades if not np.isnan(t['velocity_5'])]
    n_valid = len(valid)
    if n_valid == 0:
        return pd.DataFrame()

    bl_nets  = np.array([t['net_return']     for t in valid])
    bl_wins  = np.array([t['is_winner']      for t in valid])
    bl_holds = np.array([t['total_hold_cal'] for t in valid])
    bl_mexts = np.array([t['max_ext']        for t in valid])
    bl_avg   = float(bl_nets.mean())

    rows = []
    for V in V_THRESHOLDS:
        filt = [t for t in valid if t['velocity_5'] < V]
        if not filt:
            rows.append(dict(
                V_threshold=V, pct_kept=0.0, win_rate=0.0, avg_net=None,
                avg_hold=None, avg_max_ext=None, net_vs_baseline=None))
            continue
        nets  = np.array([t['net_return']     for t in filt])
        wins  = np.array([t['is_winner']      for t in filt])
        holds = np.array([t['total_hold_cal'] for t in filt])
        mexts = np.array([t['max_ext']        for t in filt])
        rows.append(dict(
            V_threshold    = V,
            pct_kept       = round(len(filt) / n_valid * 100, 1),
            win_rate       = round(float(wins.mean() * 100), 1),
            avg_net        = round(float(nets.mean()), 4),
            avg_hold       = round(float(holds.mean()), 1),
            avg_max_ext    = round(float(mexts.mean()), 3),
            net_vs_baseline= round(float(nets.mean()) - bl_avg, 4),
        ))

    # Baseline row
    rows.append(dict(
        V_threshold='baseline',
        pct_kept=100.0,
        win_rate=round(float(bl_wins.mean() * 100), 1),
        avg_net=round(bl_avg, 4),
        avg_hold=round(float(bl_holds.mean()), 1),
        avg_max_ext=round(float(bl_mexts.mean()), 3),
        net_vs_baseline=0.0,
    ))
    return pd.DataFrame(rows)


# ── Section 4: Extension analysis ────────────────────────────────────────────

def section4(all_trades):
    mexts = np.array([t['max_ext']        for t in all_trades])
    dtps  = np.array([t['days_to_peak']   for t in all_trades])
    nets  = np.array([t['net_return']     for t in all_trades])
    wins  = np.array([t['is_winner']      for t in all_trades])
    holds = np.array([t['total_hold_cal'] for t in all_trades])

    buckets = [
        ('<2.0',   mexts < 2.0),
        ('2.0-2.5',(mexts >= 2.0) & (mexts < 2.5)),
        ('2.5-3.0',(mexts >= 2.5) & (mexts < 3.0)),
        ('3.0-3.5',(mexts >= 3.0) & (mexts < 3.5)),
        ('>3.5',   mexts >= 3.5),
    ]

    # 4a - distribution
    dist_rows = []
    for lbl, mask in buckets:
        dist_rows.append(dict(
            range=lbl, n_trades=int(mask.sum()),
            pct_trades=round(float(mask.mean() * 100), 1),
            avg_max_ext=round(float(mexts[mask].mean()), 3) if mask.sum() else None,
            avg_days_to_peak=round(float(dtps[mask].mean()), 1) if mask.sum() else None,
        ))

    # 4b - extension by velocity quintile
    valid = [t for t in all_trades if not np.isnan(t['velocity_5'])]
    vels  = np.array([t['velocity_5'] for t in valid])
    qbounds = np.percentile(vels, [20, 40, 60, 80])
    qlabels = np.digitize(vels, qbounds)

    ext_q_rows = []
    for q in range(5):
        mask = qlabels == q
        qt   = [valid[i] for i in range(len(valid)) if mask[i]]
        if not qt:
            continue
        qv   = vels[mask]
        qm   = np.array([t['max_ext']      for t in qt])
        qd   = np.array([t['days_to_peak'] for t in qt])
        ext_q_rows.append(dict(
            quintile        = f"Q{q+1}",
            vel_range       = f"[{qv.min():.2f},{qv.max():.2f}]",
            n_trades        = len(qt),
            avg_max_ext     = round(float(qm.mean()), 3),
            avg_days_to_peak= round(float(qd.mean()), 1),
        ))

    # 4c - extension vs outcome
    out_rows = []
    for lbl, mask in buckets:
        if not mask.any():
            continue
        out_rows.append(dict(
            range   = lbl,
            n_trades= int(mask.sum()),
            win_rate= round(float(wins[mask].mean() * 100), 1),
            avg_net = round(float(nets[mask].mean()), 4),
            avg_hold= round(float(holds[mask].mean()), 1),
        ))

    return pd.DataFrame(dist_rows), pd.DataFrame(ext_q_rows), pd.DataFrame(out_rows)


# ── Section 5: Combined entry + exit simulation ───────────────────────────────

def section5(all_trades, pair_data, best_V):
    # Per-pair median calendar hold (for PE N_cal computation)
    pair_med_cal = {
        lbl: float(np.median([t['total_hold_cal'] for t in trades]))
        for lbl, (trades, *_) in pair_data.items() if trades
    }

    # Baseline net (all trades, including NaN velocity)
    bl_all = np.array([t['net_return'] for t in all_trades])

    # Velocity-filter-only: filter valid trades by best_V, take baseline net
    vf_trades = [t for t in all_trades
                 if not np.isnan(t['velocity_5']) and t['velocity_5'] < best_V]
    vf_nets = np.array([t['net_return'] for t in vf_trades]) if vf_trades else np.array([])

    # Partial-exit-only: all trades, apply PE
    pe_nets = []
    for lbl, (trades, dist_sd, day_ints, cum) in pair_data.items():
        if not trades or lbl not in pair_med_cal:
            continue
        N_cal = max(1, round(PE_TGP * pair_med_cal[lbl]))
        for tr in trades:
            pe_nets.append(_pe_outcome(tr, dist_sd, cum, day_ints, N_cal, PE_F, PE_SDGATE))
    pe_nets = np.array(pe_nets) if pe_nets else np.array([])

    # Combined: velocity filter + PE
    comb_nets = []
    for lbl, (trades, dist_sd, day_ints, cum) in pair_data.items():
        if not trades or lbl not in pair_med_cal:
            continue
        N_cal = max(1, round(PE_TGP * pair_med_cal[lbl]))
        for tr in trades:
            if np.isnan(tr['velocity_5']) or tr['velocity_5'] >= best_V:
                continue
            comb_nets.append(_pe_outcome(tr, dist_sd, cum, day_ints, N_cal, PE_F, PE_SDGATE))
    comb_nets = np.array(comb_nets) if comb_nets else np.array([])

    def row(label, arr):
        if len(arr) == 0:
            return dict(component=label, avg_net=None, n_trades=0, win_rate=None)
        return dict(
            component = label,
            avg_net   = round(float(arr.mean()), 4),
            n_trades  = len(arr),
            win_rate  = round(float((arr > 0).mean() * 100), 1),
        )

    return pd.DataFrame([
        row('Baseline (no filter, no PE)',            bl_all),
        row(f'Velocity filter only (V<{best_V:.2f})', vf_nets),
        row('Partial exit only (all trades)',          pe_nets),
        row('Velocity filter + partial exit',          comb_nets),
    ])


# ── Section 6: Per-pair best threshold ────────────────────────────────────────

def section6(all_trades):
    pairs = list(dict.fromkeys(t['pair'] for t in all_trades))
    rows  = []
    for lbl in pairs:
        pair_trades = [t for t in all_trades if t['pair'] == lbl]
        valid       = [t for t in pair_trades if not np.isnan(t['velocity_5'])]
        if not valid:
            continue
        bl_avg = float(np.array([t['net_return'] for t in valid]).mean())

        best_V_val = None
        best_avg   = bl_avg
        best_n     = len(valid)

        for V in V_THRESHOLDS:
            filt = [t for t in valid if t['velocity_5'] < V]
            if len(filt) < max(3, len(valid) * 0.1):
                continue
            avg = float(np.array([t['net_return'] for t in filt]).mean())
            if avg > best_avg:
                best_avg   = avg
                best_V_val = V
                best_n     = len(filt)

        pct_kept = round(best_n / len(valid) * 100, 1)
        net_imp  = round(best_avg - bl_avg, 4)
        flag = ('robust'        if pct_kept > 50 and net_imp > 0
                else 'over-aggressive' if pct_kept < 30
                else 'marginal')

        rows.append(dict(
            pair             = lbl,
            bl_avg_net       = round(bl_avg, 4),
            best_V           = best_V_val if best_V_val is not None else 'none',
            filtered_avg_net = round(best_avg, 4),
            net_improvement  = net_imp,
            pct_kept         = pct_kept,
            flag             = flag,
        ))
    return pd.DataFrame(rows)


# ── Terminal reporting ────────────────────────────────────────────────────────

W = 100

def print_section1(s1_rows, valid, vels):
    print("\n" + "=" * W)
    print("SECTION 1 -- VELOCITY DISTRIBUTION  (lookback=5, valid-velocity trades)")
    print("=" * W)
    agg = s1_rows[0]
    print(f"\n  n_trades         : {agg['n_trades']}  "
          f"(of {agg['n_trades']} total with valid velocity_5)")
    print(f"  mean velocity    : {agg['mean_velocity']:+.3f}")
    print(f"  median velocity  : {agg['median_velocity']:+.3f}")
    print(f"  pct < 0          : {agg['pct_neg']:5.1f}%  (already turning at entry)")
    print(f"  pct [0, 0.5)     : {agg['pct_0_to_05']:5.1f}%  (slow / stalling)")
    print(f"  pct [0.5, 1.0)   : {agg['pct_05_to_10']:5.1f}%  (moderate)")
    print(f"  pct >= 1.0       : {agg['pct_gt10']:5.1f}%  (fast / still extending hard)")
    print(f"  pct >= 1.5       : {agg['pct_gt15']:5.1f}%  (very fast)")
    print(f"\n  Per-pair (mean velocity, pct<0):")
    print(f"  {'pair':>20}  {'n_valid':>8}  {'mean_vel':>9}  {'pct_neg':>8}")
    for r in s1_rows[1:]:
        print(f"  {r['pair']:>20}  {r['n_trades']:>8}  "
              f"{r['mean_velocity']:>9.3f}  {r['pct_neg']:>7.1f}%")


def print_section2(s2_df):
    print("\n" + "=" * W)
    print("SECTION 2 -- VELOCITY QUINTILE ANALYSIS")
    print("=" * W)
    for lb in LOOKBACKS:
        sub = s2_df[s2_df['lookback'] == lb]
        if not len(sub):
            continue
        print(f"\n  Lookback = {lb} days:")
        print(f"  {'Q':>4}  {'vel_range':>16}  {'n':>5}  {'win%':>5}  "
              f"{'avg_gross':>10}  {'avg_net':>9}  {'hold':>6}  "
              f"{'max_ext':>8}  {'dtp':>5}")
        for _, r in sub.iterrows():
            print(f"  {r['quintile']:>4}  {r['vel_range']:>16}  {r['n_trades']:>5}  "
                  f"{r['win_rate']:>4.1f}%  {r['avg_gross']:>10.4f}  "
                  f"{r['avg_net']:>9.4f}  {r['avg_hold']:>5.0f}d  "
                  f"{r['avg_max_ext']:>7.3f}  {r['avg_days_to_peak']:>4.1f}d")


def print_section3(s3_df):
    print("\n" + "=" * W)
    print("SECTION 3 -- VELOCITY THRESHOLD FILTER SIMULATION  (lookback=5)")
    print("=" * W)
    print(f"\n  {'V_thresh':>9}  {'pct_kept':>9}  {'win%':>5}  "
          f"{'avg_net':>8}  {'hold':>6}  {'max_ext':>8}  {'vs_baseline':>12}")
    for _, r in s3_df.iterrows():
        v = r['V_threshold']
        if v == 'baseline' or v is None:
            vstr = 'baseline'
            net  = f"{r['avg_net']:>8.4f}" if r['avg_net'] is not None else '       -'
            hold = f"{r['avg_hold']:>5.0f}d" if r['avg_hold'] is not None else '      -'
            mext = f"{r['avg_max_ext']:>8.3f}" if r['avg_max_ext'] is not None else '       -'
            imp  = '            -'
        else:
            vstr = f"{float(v):.2f}"
            net  = f"{r['avg_net']:>8.4f}" if r['avg_net'] is not None else '       -'
            hold = f"{r['avg_hold']:>5.0f}d" if r['avg_hold'] is not None else '      -'
            mext = f"{r['avg_max_ext']:>8.3f}" if r['avg_max_ext'] is not None else '       -'
            imp  = (f"{r['net_vs_baseline']:>+12.4f}"
                    if r['net_vs_baseline'] is not None else '            -')
        print(f"  {vstr:>9}  {r['pct_kept']:>8.1f}%  {r['win_rate']:>4.1f}%  "
              f"{net}  {hold}  {mext}  {imp}")


def print_section4(dist_df, ext_q_df, out_df):
    print("\n" + "=" * W)
    print("SECTION 4 -- POST-ENTRY EXTENSION ANALYSIS")
    print("=" * W)

    all_mexts = np.array([r['avg_max_ext'] for _, r in dist_df.iterrows()
                          if r['avg_max_ext'] is not None])
    print(f"\n  4a -- Extension distribution (all {dist_df['n_trades'].sum()} trades):")
    print(f"  {'range':>10}  {'n':>6}  {'pct':>6}  {'avg_max_ext':>12}  {'avg_dtp':>8}")
    for _, r in dist_df.iterrows():
        me = f"{r['avg_max_ext']:.3f}" if r['avg_max_ext'] is not None else '-'
        dp = f"{r['avg_days_to_peak']:.1f}d" if r['avg_days_to_peak'] is not None else '-'
        print(f"  {r['range']:>10}  {r['n_trades']:>6}  {r['pct_trades']:>5.1f}%  "
              f"{me:>12}  {dp:>8}")

    print(f"\n  4b -- Extension by velocity quintile (lookback=5):")
    print(f"  {'Q':>4}  {'vel_range':>16}  {'n':>5}  {'avg_max_ext':>12}  {'avg_dtp':>8}")
    for _, r in ext_q_df.iterrows():
        print(f"  {r['quintile']:>4}  {r['vel_range']:>16}  {r['n_trades']:>5}  "
              f"{r['avg_max_ext']:>12.3f}  {r['avg_days_to_peak']:>7.1f}d")

    print(f"\n  4c -- Extension vs trade outcome:")
    print(f"  {'range':>10}  {'n':>6}  {'win%':>5}  {'avg_net':>8}  {'avg_hold':>9}")
    for _, r in out_df.iterrows():
        print(f"  {r['range']:>10}  {r['n_trades']:>6}  {r['win_rate']:>4.1f}%  "
              f"{r['avg_net']:>8.4f}  {r['avg_hold']:>8.0f}d")


def print_section5(s5_df, best_V):
    print("\n" + "=" * W)
    print(f"SECTION 5 -- COMBINED ENTRY + EXIT  "
          f"(best V={best_V:.2f}, PE: tgp={PE_TGP}, sd={PE_SDGATE}, F={PE_F})")
    print("=" * W)
    print(f"\n  {'component':45}  {'avg_net':>8}  {'n_trades':>9}  {'win%':>5}")
    for _, r in s5_df.iterrows():
        net = f"{r['avg_net']:>8.4f}" if r['avg_net'] is not None else '       -'
        wr  = f"{r['win_rate']:>4.1f}%" if r['win_rate'] is not None else '     -'
        print(f"  {r['component']:45}  {net}  {r['n_trades']:>9}  {wr}")


def print_section6(s6_df):
    print("\n" + "=" * W)
    print("SECTION 6 -- PER-PAIR BEST VELOCITY THRESHOLD  (lookback=5)")
    print("=" * W)
    print(f"\n  {'pair':>20}  {'bl_net':>7}  {'best_V':>7}  "
          f"{'filt_net':>8}  {'imp':>7}  {'kept':>6}  {'flag':>15}")
    for _, r in s6_df.iterrows():
        print(f"  {r['pair']:>20}  {r['bl_avg_net']:>7.4f}  {str(r['best_V']):>7}  "
              f"{r['filtered_avg_net']:>8.4f}  {r['net_improvement']:>+7.4f}  "
              f"{r['pct_kept']:>5.1f}%  {r['flag']:>15}")
    n_robust = (s6_df['flag'] == 'robust').sum()
    n_over   = (s6_df['flag'] == 'over-aggressive').sum()
    print(f"\n  Robust: {n_robust}/{len(s6_df)}    Over-aggressive: {n_over}/{len(s6_df)}")


def print_section7(s2_df, s3_df, s5_df, s6_df, best_V):
    print("\n" + "=" * W)
    print("SECTION 7 -- RECOMMENDATION")
    print("=" * W)

    q5_sub = s2_df[(s2_df['lookback'] == 5) & (s2_df['quintile'] == 'Q1')]
    q1_net = float(q5_sub['avg_net'].iloc[0]) if len(q5_sub) else 0.0
    q5_sub = s2_df[(s2_df['lookback'] == 5) & (s2_df['quintile'] == 'Q5')]
    q5_net = float(q5_sub['avg_net'].iloc[0]) if len(q5_sub) else 0.0

    bl_row   = s3_df[s3_df['V_threshold'] == 'baseline']
    bl_avg   = float(bl_row['avg_net'].iloc[0]) if len(bl_row) else 0.0

    filt_rows = s3_df[s3_df['V_threshold'] != 'baseline'].dropna(subset=['net_vs_baseline'])
    if len(filt_rows):
        best_row = filt_rows.loc[filt_rows['net_vs_baseline'].idxmax()]
        best_imp = float(best_row['net_vs_baseline'])
        best_pct = float(best_row['pct_kept'])
    else:
        best_imp = 0.0; best_pct = 100.0

    n_robust  = int((s6_df['flag'] == 'robust').sum())
    n_total   = len(s6_df)

    bl_comb = s5_df[s5_df['component'].str.contains('Baseline')]
    vf_row  = s5_df[s5_df['component'].str.contains('Velocity filter only')]
    pe_row  = s5_df[s5_df['component'].str.contains('Partial exit only')]
    cb_row  = s5_df[s5_df['component'].str.contains('filter \+ partial')]

    bl_net = float(bl_comb['avg_net'].iloc[0]) if len(bl_comb) else 0.0
    vf_net = float(vf_row['avg_net'].iloc[0])  if len(vf_row) else 0.0
    pe_net = float(pe_row['avg_net'].iloc[0])  if len(pe_row) else 0.0
    cb_net = float(cb_row['avg_net'].iloc[0])  if len(cb_row) else 0.0

    q_gap  = q1_net - q5_net
    predictive = abs(q_gap) > 0.002

    print(f"""
  1. VELOCITY PREDICTS OUTCOME?  (lookback=5 quintile analysis)
     Q1 (slowest) avg_net: {q1_net:+.4f}   Q5 (fastest) avg_net: {q5_net:+.4f}
     Q1 - Q5 gap: {q_gap:+.4f}
     {'YES -- slow entries outperform fast.' if predictive and q1_net > q5_net
      else 'INVERTED -- fast entries outperform slow (momentum signal).' if predictive and q5_net > q1_net
      else 'WEAK -- gap below actionable threshold.'}

  2. OPTIMAL VELOCITY THRESHOLD  (lookback=5)
     Best V: {best_V:.2f} SD/window  ({best_pct:.0f}% of trades kept)
     Per-trade improvement: {best_imp:+.4f}

  3. ROBUSTNESS
     {n_robust}/{n_total} pairs show robust improvement (pct_kept>50% AND improvement>0)

  4. EXTENSION PROFILE  (see Section 4 tables)

  5. COMBINED IMPROVEMENT  (velocity filter V<{best_V:.2f} + partial exit)
     Baseline:                {bl_net:+.4f}
     Velocity filter only:    {vf_net:+.4f}   (delta: {vf_net - bl_net:+.4f})
     Partial exit only:       {pe_net:+.4f}   (delta: {pe_net - bl_net:+.4f})
     Combined:                {cb_net:+.4f}   (delta: {cb_net - bl_net:+.4f})
     Sum of individual deltas:{(vf_net - bl_net) + (pe_net - bl_net):+.4f}
     {'ADDITIVE' if cb_net - bl_net >= 0.9 * ((vf_net - bl_net) + (pe_net - bl_net)) else 'SUBSTITUTIVE'}

  6. COMPARISON VS PRIOR RESEARCH
     Trailing stop best:     {PRIOR_BEST_TS:+.4f}
     Partial exit best:      {PRIOR_BEST_PE:+.4f}
     Entry velocity best:    {best_imp:+.4f}  (per-trade; applies to {best_pct:.0f}% of trades)

  7. VERDICT""")

    if predictive and q1_net > q5_net and best_imp > 0.001:
        v = (f"Entry velocity IS predictive. Filter at V<{best_V:.2f} improves per-trade "
             f"net by {best_imp:+.4f} while keeping {best_pct:.0f}% of opportunities. "
             f"Robust across {n_robust}/{n_total} pairs.")
    elif best_imp > 0 and not predictive:
        v = (f"Velocity filter shows marginal positive improvement ({best_imp:+.4f}) "
             "but Q1/Q5 quintile analysis does not confirm a strong predictive relationship. "
             "The filter may be removing thin-edge trades by chance.")
    elif q5_net > q1_net and predictive:
        v = ("INVERTED: faster entries outperform slower ones. Velocity is a momentum "
             "confirmation signal, not a fade filter. Do not apply a velocity ceiling; "
             "consider a velocity floor instead.")
    else:
        v = (f"Entry velocity does not materially predict outcome ({q_gap:+.4f} Q1-Q5 gap). "
             "The ±2.0 SD crossing is the signal; velocity at the crossing is noise. "
             "No velocity filter recommended.")

    print(f"     {v}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Entry Velocity Analysis")
    print("=" * 60)
    print("Loading commodity prices...")
    prices, _ = load_asset_prices(DATA_PATH)
    print(f"  {len(prices)} trading days")

    print(f"\nCollecting trades with velocity data ({len(ALL_PAIRS)} pairs)...")
    pair_data, all_trades = collect_pair_data(prices)
    n_valid_v5 = sum(1 for t in all_trades if not np.isnan(t['velocity_5']))
    print(f"  Total trades: {len(all_trades)}  |  valid velocity_5: {n_valid_v5}")

    # Section 1
    print("\nSection 1 -- velocity distribution...")
    s1_rows, s1_valid, s1_vels = section1(all_trades)
    print_section1(s1_rows, s1_valid, s1_vels)

    # Section 2
    print("\nSection 2 -- quintile analysis...")
    s2_df = section2(all_trades)
    print_section2(s2_df)

    # Section 3
    print("\nSection 3 -- filter simulation...")
    s3_df = section3(all_trades)
    print_section3(s3_df)

    # Determine best_V for Section 5/7
    filt_rows = s3_df[s3_df['V_threshold'] != 'baseline'].dropna(subset=['net_vs_baseline'])
    if len(filt_rows):
        best_V = float(filt_rows.loc[filt_rows['net_vs_baseline'].idxmax(), 'V_threshold'])
    else:
        best_V = 0.50  # fallback

    # Section 4
    print("\nSection 4 -- extension analysis...")
    s4_dist_df, s4_extq_df, s4_out_df = section4(all_trades)
    print_section4(s4_dist_df, s4_extq_df, s4_out_df)

    # Section 5
    print(f"\nSection 5 -- combined simulation (best_V={best_V:.2f})...")
    s5_df = section5(all_trades, pair_data, best_V)
    print_section5(s5_df, best_V)

    # Section 6
    print("\nSection 6 -- per-pair best threshold...")
    s6_df = section6(all_trades)
    print_section6(s6_df)

    # Section 7
    print_section7(s2_df, s3_df, s5_df, s6_df, best_V)

    # Save CSVs
    pd.DataFrame(s1_rows).to_csv(
        RESULTS_DIR / "entry_velocity_distribution.csv", index=False)
    s2_df.to_csv(
        RESULTS_DIR / "entry_velocity_quintiles.csv", index=False)
    s3_df.to_csv(
        RESULTS_DIR / "entry_velocity_filter_sim.csv", index=False)
    s4_dist_df.to_csv(
        RESULTS_DIR / "entry_extension_distribution.csv", index=False)
    s4_extq_df.to_csv(
        RESULTS_DIR / "entry_extension_by_quintile.csv", index=False)
    s4_out_df.to_csv(
        RESULTS_DIR / "entry_extension_vs_outcome.csv", index=False)
    s5_df.to_csv(
        RESULTS_DIR / "entry_exit_combined_sim.csv", index=False)
    s6_df.to_csv(
        RESULTS_DIR / "entry_velocity_per_pair.csv", index=False)

    print(f"CSVs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
