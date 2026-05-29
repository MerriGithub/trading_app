"""
extension_stop_research.py
==========================
Extension stop analysis: does a hard stop on continued divergence improve
net expectancy vs hold-to-zero?

Signal: ±2.0 SD crossing, exit at 0 SD. MAX_HOLD = 300.
Stop condition: exit if |dist_SD| exceeds EXT_STOP after entry.
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
    build_spread, DAILY_FIN, SPREAD_COST, DATA_PATH,
    RESULTS_DIR, XING_SD, EXIT_SD, MAX_HOLD,
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

EXT_STOPS = [2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
LOOKBACK  = 5   # velocity lookback (matches prior research)

W = 110   # print width


# ── Velocity helper ───────────────────────────────────────────────────────────

def compute_velocity(dist_sd, entry_idx, lookback, side):
    if entry_idx < lookback:
        return np.nan
    d_prev = dist_sd[entry_idx - lookback]
    if np.isnan(d_prev):
        return np.nan
    d_entry = dist_sd[entry_idx]
    if side == -1:
        return float(d_entry - d_prev)
    else:
        return float(d_prev - d_entry)


# ── Step 0 -- Collect trade trajectories ──────────────────────────────────────

def collect_trades(prices):
    all_trades = []

    for i, (long_inst, short_inst) in enumerate(ALL_PAIRS):
        lbl = f"{long_inst}/{short_inst}"
        print(f"  [{i+1:02d}/{len(ALL_PAIRS)}] {lbl}", end="", flush=True)

        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
        trades_raw, n = detect_trades(cum, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)

        if n == 0:
            print("  -> 0 trades")
            continue

        for k in range(n):
            entry_idx      = int(trades_raw[k, COL_ENTRY_IDX])
            exit_idx       = int(trades_raw[k, COL_EXIT_IDX])
            side           = int(trades_raw[k, COL_SIDE])
            gross          = float(trades_raw[k, COL_GROSS_RETURN])
            total_hold     = exit_idx - entry_idx          # time-step count
            total_hold_cal = int(day_ints[exit_idx] - day_ints[entry_idx])

            net = gross - SPREAD_COST - DAILY_FIN * total_hold_cal

            dist_path  = dist_sd[entry_idx : exit_idx + 1].copy()
            cum_path   = cum[entry_idx     : exit_idx + 1].copy()
            dayint_path = (day_ints[entry_idx : exit_idx + 1]
                           - day_ints[entry_idx]).astype(np.float64)

            velocity_5 = compute_velocity(dist_sd, entry_idx, LOOKBACK, side)

            # max_ext: max absolute dist_sd during the trade
            seg   = np.abs(dist_path)
            valid = ~np.isnan(seg)
            max_ext = float(np.nanmax(seg)) if valid.any() else float(abs(dist_sd[entry_idx]))

            all_trades.append({
                'pair':           lbl,
                'side':           side,
                'entry_idx':      entry_idx,
                'total_hold':     total_hold,
                'total_hold_cal': total_hold_cal,
                'entry_cum':      float(cum[entry_idx]),
                'gross_final':    gross,
                'net_final':      net,
                'velocity_5':     velocity_5,
                'max_ext':        max_ext,
                'dist_path':      dist_path,
                'cum_path':       cum_path,
                'dayint_path':    dayint_path,
            })

        print(f"  -> {n} trades")

    return all_trades


# ── Step 1 -- Extension stop simulation ───────────────────────────────────────

def simulate_extension_stop(trade, EXT_STOP):
    side = trade['side']
    dp   = trade['dist_path']
    cp   = trade['cum_path']
    dip  = trade['dayint_path']

    if side == -1:
        stop_days = np.where(dp > EXT_STOP)[0]
    else:
        stop_days = np.where(dp < -EXT_STOP)[0]

    if len(stop_days) == 0:
        return {
            'stopped':       False,
            'net_with_stop': trade['net_final'],
            'net_if_held':   trade['net_final'],
            'hold_cal_stop': trade['total_hold_cal'],
        }

    stop_day    = stop_days[0]
    stop_cum    = cp[stop_day]
    cal_at_stop = dip[stop_day]

    if side == -1:
        gross_stop = (trade['entry_cum'] - stop_cum) / trade['entry_cum']
    else:
        gross_stop = (stop_cum - trade['entry_cum']) / trade['entry_cum']

    net_stop = gross_stop - SPREAD_COST - DAILY_FIN * cal_at_stop

    return {
        'stopped':          True,
        'net_with_stop':    net_stop,
        'net_if_held':      trade['net_final'],
        'gross_stop':       gross_stop,
        'hold_cal_stop':    cal_at_stop,
        'stop_improvement': net_stop - trade['net_final'],
    }


# ── Section 1 -- Aggregate threshold sweep ────────────────────────────────────

def section1(all_trades):
    n_total  = len(all_trades)
    bl_avg   = float(np.mean([t['net_final'] for t in all_trades]))
    bl_hold  = float(np.mean([t['total_hold_cal'] for t in all_trades]))

    rows = []
    for ext in EXT_STOPS:
        results = [simulate_extension_stop(t, ext) for t in all_trades]

        stopped     = [r for r in results if r['stopped']]
        n_stopped   = len(stopped)
        pct_stopped = n_stopped / n_total * 100

        net_all_with = float(np.mean([r['net_with_stop'] for r in results]))
        net_all_held = float(np.mean([r['net_if_held']   for r in results]))

        if n_stopped > 0:
            avg_net_sw   = float(np.mean([r['net_with_stop'] for r in stopped]))
            avg_net_sh   = float(np.mean([r['net_if_held']   for r in stopped]))
            avg_imp      = float(np.mean([r['stop_improvement'] for r in stopped]))
            avg_hold_stp = float(np.mean([r['hold_cal_stop']  for r in stopped]))
        else:
            avg_net_sw = avg_net_sh = avg_imp = avg_hold_stp = float('nan')

        rows.append({
            'EXT_STOP':             ext,
            'n_stopped':            n_stopped,
            'pct_stopped':          round(pct_stopped, 1),
            'avg_net_stopped_with': round(avg_net_sw, 4) if n_stopped else None,
            'avg_net_stopped_held': round(avg_net_sh, 4) if n_stopped else None,
            'stop_improvement':     round(avg_imp, 4)    if n_stopped else None,
            'avg_net_all_with':     round(net_all_with, 4),
            'avg_net_all_held':     round(net_all_held, 4),
            'net_vs_baseline':      round(net_all_with - net_all_held, 4),
            'avg_hold_cal_stopped': round(avg_hold_stp, 1) if n_stopped else None,
        })

    return pd.DataFrame(rows), bl_avg, bl_hold, n_total


def print_section1(df, bl_avg, bl_hold):
    print("\n" + "=" * W)
    print("SECTION 1 -- AGGREGATE EXTENSION STOP THRESHOLD SWEEP")
    print("=" * W)
    hdr = (f"  {'EXT_STOP':>8}  {'n_stp':>6}  {'pct_stp':>8}  "
           f"{'net_stp_with':>13}  {'net_stp_held':>13}  {'stp_impr':>9}  "
           f"{'net_all_with':>13}  {'net_all_held':>13}  {'vs_baseline':>12}  "
           f"{'hold_cal_stp':>13}")
    print(hdr)
    print("  " + "-" * (W - 2))
    for _, r in df.iterrows():
        sw   = f"{r['avg_net_stopped_with']:>+13.4f}" if r['avg_net_stopped_with'] is not None else f"{'--':>13}"
        sh   = f"{r['avg_net_stopped_held']:>+13.4f}" if r['avg_net_stopped_held'] is not None else f"{'--':>13}"
        imp  = f"{r['stop_improvement']:>+9.4f}"      if r['stop_improvement']     is not None else f"{'--':>9}"
        hld  = f"{r['avg_hold_cal_stopped']:>12.1f}d" if r['avg_hold_cal_stopped'] is not None else f"{'--':>13}"
        print(f"  {r['EXT_STOP']:>8.2f}  {r['n_stopped']:>6}  {r['pct_stopped']:>7.1f}%  "
              f"{sw}  {sh}  {imp}  "
              f"{r['avg_net_all_with']:>+13.4f}  {r['avg_net_all_held']:>+13.4f}  "
              f"{r['net_vs_baseline']:>+12.4f}  {hld}")
    print("  " + "-" * (W - 2))
    print(f"  {'Baseline':>8}  {'':>6}  {'':>8}  {'':>13}  {'':>13}  {'':>9}  "
          f"{'':>13}  {bl_avg:>+13.4f}  {'':>12}  {bl_hold:>12.1f}d")

    opt = df.loc[df['net_vs_baseline'].idxmax()]
    print(f"\n  Optimal EXT_STOP: {opt['EXT_STOP']:.2f} SD  "
          f"(net_vs_baseline = {opt['net_vs_baseline']:+.4f}, "
          f"fires on {opt['pct_stopped']:.1f}% of trades)")


# ── Section 2 -- Hindsight W/L split ──────────────────────────────────────────

def section2(all_trades):
    n_total = len(all_trades)
    rows = []
    for ext in EXT_STOPS:
        results  = [simulate_extension_stop(t, ext) for t in all_trades]
        stopped  = [r for r in results if r['stopped']]
        if not stopped:
            continue
        n_stp = len(stopped)

        W_bucket = [r for r in stopped if r['net_if_held'] > 0]
        L_bucket = [r for r in stopped if r['net_if_held'] <= 0]
        pct_W = len(W_bucket) / n_stp * 100
        pct_L = len(L_bucket) / n_stp * 100

        def safe_mean(lst, key):
            return float(np.mean([r[key] for r in lst])) if lst else float('nan')

        avg_W_with = safe_mean(W_bucket, 'net_with_stop')
        avg_W_held = safe_mean(W_bucket, 'net_if_held')
        avg_L_with = safe_mean(L_bucket, 'net_with_stop')
        avg_L_held = safe_mean(L_bucket, 'net_if_held')

        W_damage = avg_W_held - avg_W_with if W_bucket else float('nan')
        L_saving = avg_L_with - avg_L_held if L_bucket else float('nan')

        net_benefit = (
            (pct_L / 100) * L_saving - (pct_W / 100) * W_damage
            if W_bucket and L_bucket else float('nan')
        )

        rows.append({
            'EXT_STOP':         ext,
            'n_stopped':        n_stp,
            'pct_W':            round(pct_W, 1),
            'pct_L':            round(pct_L, 1),
            'avg_net_W_stopped': round(avg_W_with, 4) if W_bucket else None,
            'avg_net_W_held':    round(avg_W_held, 4) if W_bucket else None,
            'W_damage':          round(W_damage, 4)   if W_bucket else None,
            'avg_net_L_stopped': round(avg_L_with, 4) if L_bucket else None,
            'avg_net_L_held':    round(avg_L_held, 4) if L_bucket else None,
            'L_saving':          round(L_saving, 4)   if L_bucket else None,
            'net_benefit':       round(net_benefit, 4) if not np.isnan(net_benefit) else None,
        })

    return pd.DataFrame(rows)


def print_section2(df):
    print("\n" + "=" * W)
    print("SECTION 2 -- HINDSIGHT WINNER/LOSER SPLIT")
    print("=" * W)
    print(f"\n  W = would have won if held to 0   L = would have lost if held to 0")
    hdr = (f"  {'EXT_STOP':>8}  {'n_stp':>6}  {'pct_W':>7}  {'pct_L':>7}  "
           f"{'W_stopped':>10}  {'W_held':>10}  {'W_damage':>9}  "
           f"{'L_stopped':>10}  {'L_held':>10}  {'L_saving':>9}  {'net_benefit':>12}")
    print(hdr)
    print("  " + "-" * (W - 2))
    def fmt(v, w=10):
        return f"{v:>+{w}.4f}" if v is not None else f"{'n/a':>{w}}"
    for _, r in df.iterrows():
        print(f"  {r['EXT_STOP']:>8.2f}  {r['n_stopped']:>6}  {r['pct_W']:>6.1f}%  {r['pct_L']:>6.1f}%  "
              f"{fmt(r['avg_net_W_stopped'])}  {fmt(r['avg_net_W_held'])}  {fmt(r['W_damage'], 9)}  "
              f"{fmt(r['avg_net_L_stopped'])}  {fmt(r['avg_net_L_held'])}  {fmt(r['L_saving'], 9)}  "
              f"{fmt(r['net_benefit'], 12)}")


# ── Section 3 -- Max extension reconciliation ──────────────────────────────────

def section3(all_trades):
    mexts = np.array([t['max_ext']        for t in all_trades])
    nets  = np.array([t['net_final']      for t in all_trades])
    wins  = np.array([t['net_final'] > 0  for t in all_trades], dtype=float)
    holds = np.array([t['total_hold_cal'] for t in all_trades])

    buckets = [
        ('2.0-2.5', (mexts >= 2.0) & (mexts < 2.5)),
        ('2.5-3.0', (mexts >= 2.5) & (mexts < 3.0)),
        ('3.0-3.5', (mexts >= 3.0) & (mexts < 3.5)),
        ('3.5-4.0', (mexts >= 3.5) & (mexts < 4.0)),
        ('>4.0',    (mexts >= 4.0)),
    ]

    rows = []
    total = len(all_trades)
    for lbl, mask in buckets:
        if not mask.any():
            continue
        rows.append({
            'max_ext_range': lbl,
            'n_trades':      int(mask.sum()),
            'pct_total':     round(float(mask.mean() * 100), 1),
            'win_rate':      round(float(wins[mask].mean() * 100), 1),
            'avg_net':       round(float(nets[mask].mean()), 4),
            'avg_hold_cal':  round(float(holds[mask].mean()), 1),
        })
    return pd.DataFrame(rows)


def print_section3(df):
    print("\n" + "=" * W)
    print("SECTION 3 -- MAX EXTENSION DISTRIBUTION RECONCILIATION")
    print("=" * W)
    print(f"\n  {'max_ext_range':>12}  {'n_trades':>9}  {'pct_total':>10}  "
          f"{'win_rate':>9}  {'avg_net':>9}  {'avg_hold_cal':>13}")
    print("  " + "-" * 70)
    for _, r in df.iterrows():
        print(f"  {r['max_ext_range']:>12}  {r['n_trades']:>9}  "
              f"{r['pct_total']:>9.1f}%  {r['win_rate']:>8.1f}%  "
              f"{r['avg_net']:>+9.4f}  {r['avg_hold_cal']:>12.1f}d")

    print("""
  Extension stop targeting logic:
    EXT_STOP <= 2.50 -> fires on ALL buckets above 2.5 SD
    EXT_STOP = 3.00  -> targets 3.0-3.5 AND 3.5-4.0 AND >4.0 buckets
    EXT_STOP = 3.50  -> targets 3.5-4.0 AND >4.0 buckets only
    EXT_STOP = 4.00  -> targets >4.0 bucket only

  >3.5 SD anomaly: trades with max_ext >3.5 SD have BETTER avg_net than 3.0-3.5 SD
  bucket despite extending further -- faster snap-back offsets larger adverse move.
  A stop at 3.0 SD treats both identically; 3.5 SD separates them.
  Section 2 W/L split resolves which is optimal.""")


# ── Section 4 -- Velocity quintile interaction ─────────────────────────────────

def section4(all_trades, opt_ext):
    valid = [t for t in all_trades if not np.isnan(t['velocity_5'])]
    if not valid:
        return pd.DataFrame(), pd.DataFrame()

    vels    = np.array([t['velocity_5'] for t in valid])
    bounds  = np.percentile(vels, [20, 40, 60, 80])
    qlabels = np.digitize(vels, bounds)  # 0-indexed: 0=Q1 .. 4=Q5
    # Convert to 1-indexed for clarity
    quintiles = qlabels + 1             # 1=Q1 .. 5=Q5

    # Extension probability table
    ext_thresholds = [2.5, 3.0, 3.5, 4.0]
    prob_rows = []
    for q in range(1, 6):
        mask = quintiles == q
        qt   = [valid[i] for i in range(len(valid)) if mask[i]]
        if not qt:
            continue
        qv   = vels[mask]
        nets_bl = np.array([t['net_final'] for t in qt])
        mexts   = np.array([t['max_ext']   for t in qt])

        row = {
            'vel_quintile': f"Q{q}",
            'vel_range':    f"[{qv.min():.2f},{qv.max():.2f}]",
            'n_trades':     len(qt),
            'avg_net_bl':   round(float(nets_bl.mean()), 4),
        }
        for thr in ext_thresholds:
            row[f'pct_ext_{thr:.1f}'] = round(float((mexts >= thr).mean() * 100), 1)
        prob_rows.append(row)

    # Per-quintile improvement at optimal EXT_STOP
    impr_rows = []
    for q in range(1, 6):
        mask = quintiles == q
        qt   = [valid[i] for i in range(len(valid)) if mask[i]]
        if not qt:
            continue
        nets_bl   = np.array([t['net_final'] for t in qt])
        results   = [simulate_extension_stop(t, opt_ext) for t in qt]
        nets_with = np.array([r['net_with_stop'] for r in results])
        n_stopped = sum(1 for r in results if r['stopped'])
        impr_rows.append({
            'vel_quintile':      f"Q{q}",
            'avg_net_baseline':  round(float(nets_bl.mean()), 4),
            'avg_net_with_stop': round(float(nets_with.mean()), 4),
            'improvement':       round(float(nets_with.mean() - nets_bl.mean()), 4),
            'pct_stopped':       round(n_stopped / len(qt) * 100, 1),
        })

    return pd.DataFrame(prob_rows), pd.DataFrame(impr_rows)


def print_section4(prob_df, impr_df, opt_ext):
    print("\n" + "=" * W)
    print("SECTION 4 -- VELOCITY QUINTILE INTERACTION")
    print("=" * W)

    print(f"\n  Extension probability by velocity quintile (velocity_5, lookback=5):")
    print(f"  {'Q':>4}  {'vel_range':>18}  {'n':>6}  "
          f"{'pct>=2.5':>8}  {'pct>=3.0':>8}  {'pct>=3.5':>8}  {'pct>=4.0':>8}  {'avg_net_bl':>11}")
    print("  " + "-" * 80)
    for _, r in prob_df.iterrows():
        print(f"  {r['vel_quintile']:>4}  {r['vel_range']:>18}  {r['n_trades']:>6}  "
              f"{r['pct_ext_2.5']:>7.1f}%  {r['pct_ext_3.0']:>7.1f}%  "
              f"{r['pct_ext_3.5']:>7.1f}%  {r['pct_ext_4.0']:>7.1f}%  "
              f"{r['avg_net_bl']:>+11.4f}")

    print(f"\n  Per-quintile improvement at optimal EXT_STOP = {opt_ext:.2f}:")
    print(f"  {'Q':>4}  {'avg_net_bl':>12}  {'avg_net_stop':>13}  {'improvement':>12}  {'pct_stopped':>12}")
    print("  " + "-" * 60)
    for _, r in impr_df.iterrows():
        print(f"  {r['vel_quintile']:>4}  {r['avg_net_baseline']:>+12.4f}  "
              f"{r['avg_net_with_stop']:>+13.4f}  {r['improvement']:>+12.4f}  "
              f"{r['pct_stopped']:>11.1f}%")


# ── Section 5 -- Position sizing + extension stop ───────────────────────────────

def section5(all_trades, opt_ext):
    valid = [t for t in all_trades if not np.isnan(t['velocity_5'])]
    # Trades with no velocity get weight 1.0 (treated as Q3 equivalent)
    no_vel = [t for t in all_trades if np.isnan(t['velocity_5'])]

    vels    = np.array([t['velocity_5'] for t in valid])
    bounds  = np.percentile(vels, [20, 40, 60, 80])
    qlabels = np.digitize(vels, bounds)  # 0-indexed
    quintiles = qlabels + 1              # 1-indexed

    def get_weight(q):
        if q in (4, 5):  return 1.00
        if q == 3:        return 0.75
        return 0.50  # Q1, Q2

    # Simulate extension stop for all trades
    stop_results_valid  = [simulate_extension_stop(t, opt_ext) for t in valid]
    stop_results_nv     = [simulate_extension_stop(t, opt_ext) for t in no_vel]

    weights_valid = np.array([get_weight(int(quintiles[i])) for i in range(len(valid))])
    weights_nv    = np.full(len(no_vel), 1.0)

    nets_bl_valid    = np.array([t['net_final']           for t in valid])
    nets_stop_valid  = np.array([r['net_with_stop']       for r in stop_results_valid])
    nets_bl_nv       = np.array([t['net_final']           for t in no_vel])
    nets_stop_nv     = np.array([r['net_with_stop']       for r in stop_results_nv])

    all_weights  = np.concatenate([weights_valid, weights_nv])
    all_nets_bl  = np.concatenate([nets_bl_valid,   nets_bl_nv])
    all_nets_stp = np.concatenate([nets_stop_valid, nets_stop_nv])
    ones         = np.ones(len(all_trades))

    # All 4 scenarios
    bl_eq   = float(np.sum(ones       * all_nets_bl)  / np.sum(ones))
    stp_eq  = float(np.sum(ones       * all_nets_stp) / np.sum(ones))
    sz_only = float(np.sum(all_weights * all_nets_bl)  / np.sum(all_weights))
    comb    = float(np.sum(all_weights * all_nets_stp) / np.sum(all_weights))

    rows = [
        {'scenario': 'Baseline (equal weight, no stop)',    'weighted_avg_net': round(bl_eq,  4), 'vs_baseline': None},
        {'scenario': f'Extension stop only (equal weight, EXT={opt_ext})', 'weighted_avg_net': round(stp_eq, 4), 'vs_baseline': round(stp_eq - bl_eq, 4)},
        {'scenario': 'Position sizing only (no stop)',      'weighted_avg_net': round(sz_only, 4), 'vs_baseline': round(sz_only - bl_eq, 4)},
        {'scenario': f'Extension stop + position sizing',   'weighted_avg_net': round(comb, 4),   'vs_baseline': round(comb - bl_eq, 4)},
    ]
    df = pd.DataFrame(rows)

    stp_delta = stp_eq  - bl_eq
    sz_delta  = sz_only - bl_eq
    sum_ind   = stp_delta + sz_delta
    additive  = (comb - bl_eq) >= 0.9 * sum_ind if sum_ind > 0 else None

    return df, additive, sum_ind


def print_section5(df, additive, sum_ind, opt_ext):
    print("\n" + "=" * W)
    print("SECTION 5 -- POSITION SIZING + EXTENSION STOP COMBINED")
    print("=" * W)
    print(f"\n  Sizing: Q1/Q2 -> 0.50, Q3 -> 0.75, Q4/Q5 -> 1.00  |  EXT_STOP = {opt_ext:.2f}")
    print(f"\n  {'Scenario':55}  {'weighted_avg_net':>17}  {'vs_baseline':>12}")
    print("  " + "-" * 90)
    for _, r in df.iterrows():
        vs = f"{r['vs_baseline']:>+12.4f}" if r['vs_baseline'] is not None else f"{'--':>12}"
        print(f"  {r['scenario']:55}  {r['weighted_avg_net']:>+17.4f}  {vs}")
    print(f"\n  Sum of individual deltas: {sum_ind:+.4f}")
    if additive is not None:
        label = "ADDITIVE" if additive else "SUBSTITUTIVE"
        print(f"  Combined vs sum: {label}")


# ── Section 6 -- Per-pair analysis ────────────────────────────────────────────

def section6(all_trades):
    pairs = list(dict.fromkeys(t['pair'] for t in all_trades))
    rows  = []
    for lbl in pairs:
        pt = [t for t in all_trades if t['pair'] == lbl]
        if not pt:
            continue
        bl_avg = float(np.mean([t['net_final'] for t in pt]))

        best_ext = None
        best_vs  = -np.inf
        for ext in EXT_STOPS:
            results  = [simulate_extension_stop(t, ext) for t in pt]
            net_with = float(np.mean([r['net_with_stop'] for r in results]))
            vs_bl    = net_with - bl_avg
            if vs_bl > best_vs:
                best_vs  = vs_bl
                best_ext = ext
                best_n_stopped = sum(1 for r in results if r['stopped'])

        pct_stopped = best_n_stopped / len(pt) * 100
        net_with_best = bl_avg + best_vs

        if best_vs > 0.005 and pct_stopped < 40:
            flag = 'selective'
        elif pct_stopped > 60:
            flag = 'too-aggressive'
        elif best_vs < 0:
            flag = 'harmful'
        else:
            flag = 'marginal'

        rows.append({
            'pair':          lbl,
            'n_trades':      len(pt),
            'bl_avg_net':    round(bl_avg, 4),
            'opt_EXT_STOP':  best_ext,
            'pct_stopped':   round(pct_stopped, 1),
            'net_with_stop': round(net_with_best, 4),
            'improvement':   round(best_vs, 4),
            'flag':          flag,
        })

    return pd.DataFrame(rows)


def print_section6(df):
    print("\n" + "=" * W)
    print("SECTION 6 -- PER-PAIR OPTIMAL EXTENSION STOP")
    print("=" * W)
    print(f"\n  {'pair':>22}  {'n':>5}  {'bl_net':>8}  {'opt_EXT':>8}  "
          f"{'pct_stp':>8}  {'net_stop':>9}  {'impr':>8}  {'flag':>15}")
    print("  " + "-" * 95)
    for _, r in df.iterrows():
        print(f"  {r['pair']:>22}  {r['n_trades']:>5}  {r['bl_avg_net']:>+8.4f}  "
              f"{r['opt_EXT_STOP']:>8.2f}  {r['pct_stopped']:>7.1f}%  "
              f"{r['net_with_stop']:>+9.4f}  {r['improvement']:>+8.4f}  {r['flag']:>15}")

    n_sel  = (df['flag'] == 'selective').sum()
    n_agg  = (df['flag'] == 'too-aggressive').sum()
    n_harm = (df['flag'] == 'harmful').sum()
    n_marg = (df['flag'] == 'marginal').sum()
    print(f"\n  selective: {n_sel}  marginal: {n_marg}  "
          f"too-aggressive: {n_agg}  harmful: {n_harm}")


# ── Section 7 -- Recommendation ───────────────────────────────────────────────

def section7(s1_df, s2_df, s4_prob, s5_df, s6_df, opt_ext, opt_row):
    print("\n" + "=" * W)
    print("SECTION 7 -- RECOMMENDATION")
    print("=" * W)

    # 1. Is there an improvement?
    max_vs = s1_df['net_vs_baseline'].max()
    beneficial = max_vs > 0.0005

    # 2. Optimal threshold
    print(f"\n  1. Net improvement at optimal threshold: "
          f"{'YES' if beneficial else 'NO / marginal'}")
    print(f"     Optimal EXT_STOP: {opt_ext:.2f} SD  |  "
          f"net_vs_baseline: {opt_row['net_vs_baseline']:+.4f}")

    # 3. Fire rate
    print(f"  3. Fires on {opt_row['pct_stopped']:.1f}% of trades at optimal threshold.")

    # 4. W/L split at optimal
    opt_s2 = s2_df[s2_df['EXT_STOP'] == opt_ext]
    if len(opt_s2):
        r = opt_s2.iloc[0]
        print(f"  4. W/L split at {opt_ext:.2f}: {r['pct_W']:.1f}% winners cut, "
              f"{r['pct_L']:.1f}% losers cut  |  net_benefit: {r['net_benefit']:+.4f}")
        dominated = 'LOSERS' if r['pct_L'] > r['pct_W'] else 'WINNERS'
        print(f"     Stop is disproportionately cutting {dominated}.")

    # 5. >3.5 SD anomaly
    opt_s2_35 = s2_df[s2_df['EXT_STOP'] == 3.5]
    opt_s2_30 = s2_df[s2_df['EXT_STOP'] == 3.0]
    if len(opt_s2_35) and len(opt_s2_30):
        r35 = opt_s2_35.iloc[0]
        r30 = opt_s2_30.iloc[0]
        if r35['pct_W'] < r30['pct_W']:
            print(f"  5. >3.5 SD anomaly: pct_W drops {r30['pct_W']:.1f}% -> {r35['pct_W']:.1f}% "
                  f"when moving from 3.0 to 3.5. Raising threshold to 3.5 separates fast-reverting "
                  f"winners from slow losers -- consistent with anomaly.")
        else:
            print(f"  5. >3.5 SD anomaly: pct_W does not materially change between 3.0 and 3.5 "
                  f"thresholds -- anomaly trades behave similarly to 3.0-3.5 SD trades at stop.")

    # 6. Velocity interaction
    if len(s4_prob):
        q1_ext = s4_prob[s4_prob['vel_quintile'] == 'Q1']['pct_ext_3.0'].values
        q5_ext = s4_prob[s4_prob['vel_quintile'] == 'Q5']['pct_ext_3.0'].values
        if len(q1_ext) and len(q5_ext):
            gap = float(q1_ext[0]) - float(q5_ext[0])
            print(f"  6. Velocity -> extension probability (at 3.0 SD): "
                  f"Q1={q1_ext[0]:.1f}%  Q5={q5_ext[0]:.1f}%  gap={gap:+.1f}pp")
            print(f"     {'Slow entries more likely to extend -- right interaction.' if gap > 5 else 'No clear velocity->extension relationship.'}")

    # 7. Additivity
    s5_rows = s5_df.set_index('scenario')
    bl  = s5_df[s5_df['vs_baseline'].isna()]['weighted_avg_net'].values[0]
    stp = s5_df[s5_df['scenario'].str.contains('stop only')]['vs_baseline'].values
    sz  = s5_df[s5_df['scenario'].str.contains('sizing only')]['vs_baseline'].values
    cb  = s5_df[s5_df['scenario'].str.contains(r'stop \+ position')]['vs_baseline'].values
    if len(stp) and len(sz) and len(cb):
        print(f"  7. Position sizing additivity: stop={float(stp[0]):+.4f}, "
              f"sizing={float(sz[0]):+.4f}, combined={float(cb[0]):+.4f}, "
              f"sum_ind={float(stp[0]+sz[0]):+.4f} -> "
              f"{'ADDITIVE' if float(cb[0]) >= 0.9*(float(stp[0])+float(sz[0])) else 'SUBSTITUTIVE'}")

    # 8. Robustness
    n_impr  = (s6_df['improvement'] > 0).sum()
    n_total = len(s6_df)
    print(f"  8. Improvement across pairs: {n_impr}/{n_total} pairs show positive improvement.")
    n_sel = (s6_df['flag'] == 'selective').sum()
    print(f"     Genuinely selective (impr>0.005, pct_stopped<40%): {n_sel}/{n_total}")

    # 9. Production rule
    print(f"\n  9. OPTIMAL PRODUCTION RULE:")
    print(f"     Entry: ±{XING_SD:.1f} SD crossing")
    print(f"     Exit:  0 SD (mean reversion)  OR  |dist_SD| > {opt_ext:.2f} SD after entry")
    print(f"     MAX_HOLD: {int(MAX_HOLD)} calendar days")

    # 10. Research ladder
    print(f"\n  10. RESEARCH LADDER:")
    print(f"      Trailing stop:    −0.0011  (done)")
    print(f"      Partial exit:     +0.0003  (noise, done)")
    print(f"      Extension stop:   {opt_row['net_vs_baseline']:+.4f}  (this experiment)")

    print()


# ── Save CSVs ─────────────────────────────────────────────────────────────────

def save_csvs(s1_df, s2_df, s3_df, s4_prob, s4_impr, s5_df, s6_df):
    s1_df.to_csv(RESULTS_DIR / "extension_stop_sweep.csv",                index=False)
    s2_df.to_csv(RESULTS_DIR / "extension_stop_hindsight.csv",            index=False)
    s3_df.to_csv(RESULTS_DIR / "extension_stop_maxext_reconcile.csv",     index=False)

    vel_int = pd.concat([s4_prob, s4_impr], axis=1) if not s4_prob.empty and not s4_impr.empty else s4_prob
    s4_prob.to_csv(RESULTS_DIR / "extension_stop_velocity_interaction.csv", index=False)
    s4_impr.to_csv(RESULTS_DIR / "extension_stop_velocity_improvement.csv", index=False)

    s5_df.to_csv(RESULTS_DIR / "extension_stop_combined_sim.csv",         index=False)
    s6_df.to_csv(RESULTS_DIR / "extension_stop_per_pair.csv",             index=False)

    print(f"\nCSVs saved to {RESULTS_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Extension Stop Analysis")
    print("=" * 60)
    print(f"EXT_STOP thresholds: {EXT_STOPS}")
    print(f"Pairs: {len(ALL_PAIRS)}")
    print()

    print("Loading commodity prices...")
    prices, instruments = load_asset_prices(DATA_PATH)
    print(f"  {len(prices)} trading days, {len(instruments)} instruments")

    print(f"\nStep 0 -- Collecting trade trajectories ({len(ALL_PAIRS)} pairs)...")
    all_trades = collect_trades(prices)
    print(f"\n  Total trades collected: {len(all_trades)}")

    # Determine optimal EXT_STOP from Section 1
    print("\nSection 1 -- Threshold sweep...")
    s1_df, bl_avg, bl_hold, n_total = section1(all_trades)
    opt_idx = s1_df['net_vs_baseline'].idxmax()
    opt_ext = float(s1_df.loc[opt_idx, 'EXT_STOP'])
    opt_row = s1_df.loc[opt_idx]
    print_section1(s1_df, bl_avg, bl_hold)

    print("\nSection 2 -- Hindsight W/L split...")
    s2_df = section2(all_trades)
    print_section2(s2_df)

    print("\nSection 3 -- Max extension reconciliation...")
    s3_df = section3(all_trades)
    print_section3(s3_df)

    print(f"\nSection 4 -- Velocity quintile interaction (opt EXT_STOP={opt_ext:.2f})...")
    s4_prob, s4_impr = section4(all_trades, opt_ext)
    print_section4(s4_prob, s4_impr, opt_ext)

    print(f"\nSection 5 -- Combined simulation (opt EXT_STOP={opt_ext:.2f})...")
    s5_df, additive, sum_ind = section5(all_trades, opt_ext)
    print_section5(s5_df, additive, sum_ind, opt_ext)

    print("\nSection 6 -- Per-pair analysis...")
    s6_df = section6(all_trades)
    print_section6(s6_df)

    section7(s1_df, s2_df, s4_prob, s5_df, s6_df, opt_ext, opt_row)

    save_csvs(s1_df, s2_df, s3_df, s4_prob, s4_impr, s5_df, s6_df)


if __name__ == "__main__":
    main()
