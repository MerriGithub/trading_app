"""
partial_exit_research.py
========================
Time-based partial exit backtest on commodity spread pairs.

Tests closing fraction F of the position once BOTH gates are open:
  Conditional variant : N days elapsed since entry AND |dist_sd| <= sd_gate
  Time-only variant   : N days elapsed (no proximity gate — control case)

N is expressed as a fraction of each pair's own median holding period so the
time gate scales with the pair's reversion rhythm rather than an absolute count.

IMPLEMENTATION NOTE — partial_fired reset
  partial_fired tracks whether the partial exit has already triggered within
  the current open trade.  It MUST be reset to False each time a new trade
  opens (in the "not in_trade" branch of the state machine loop).  Without
  the reset, a fired=True state from the previous trade leaks into the next,
  causing the partial exit gate to be silently skipped for the entire next
  trade.  The reset sites are marked with a comment in run_partial_exit().

Usage (from project root):
  C:\\Users\\gordo\\AppData\\Local\\Python\\bin\\python.exe research/partial_exit_research.py

Outputs saved to results/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).parent
ROOT = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))           # engine.* imports
sys.path.insert(0, str(RESEARCH_DIR))   # trailing_stop_research import

from engine.backtest import load_asset_prices
from trailing_stop_research import (
    build_spread, run_baseline,
    SPREAD_COST, DAILY_FIN, DATA_PATH, RESULTS_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────

XING_SD  = 2.0
EXIT_SD  = 0.0
MAX_HOLD = 300

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

TIME_GATE_PCTS = [0.33, 0.50, 0.67]
SD_GATES       = [0.25, 0.50, 0.75, 1.00]
FRACTIONS      = [0.25, 0.50, 0.75]

# Representative combo used for Section 4 remainder analysis.
# Fixed at (sd_gate=0.50, F=0.50) to avoid double-counting trades that appear
# in multiple (sd_gate, fraction) combos in the main grid.
SEC4_SD_GATE  = 0.50
SEC4_FRACTION = 0.50


# ── Core state machine ───────────────────────────────────────────────────────

def run_partial_exit(cum, dist_sd, day_ints, N, F, sd_gate=None):
    """
    Simulate time-based partial exit for a single spread series.

    Parameters
    ----------
    cum, dist_sd, day_ints : arrays from build_spread()
    N        : minimum calendar days since entry before partial can fire
    F        : fraction of position closed at partial exit
    sd_gate  : max |dist_sd| for proximity gate; None = time-only variant

    Returns
    -------
    List of trade dicts.  One entry per trade whether or not partial fired.

    Keys per trade:
      net_total        — blended net return (partial F fraction + remainder)
      net_bl_equiv     — baseline net for the same trade (no partial exit)
      net_improvement  — net_total − net_bl_equiv
      fired            — bool: did partial exit trigger?
      r_partial        — gross return at partial exit from entry (nan if not fired)
      r_final          — gross return at final exit from entry
      partial_hold     — calendar days entry → partial exit (= final_hold if not fired)
      final_hold       — calendar days entry → final exit
      effective_hold   — F*partial_hold + (1−F)*final_hold if fired, else final_hold
      financing_saving — DAILY_FIN * F * (final_hold − partial_hold) if fired, else 0
      max_hold_exit    — bool: trade expired at MAX_HOLD rather than crossing 0 SD

    Financing formula (correct):
      Full position carries for partial_hold days, then (1−F) carries the
      REMAINING (final_hold − partial_hold) days — not final_hold in full.
      financing = DAILY_FIN * (F * partial_hold + (1−F) * final_hold)
    """
    T = len(cum)
    trades_out = []

    in_trade        = False
    entry_idx       = 0
    entry_cum       = 0.0
    side            = 0

    # partial_fired is reset to False each time a new trade opens — see the
    # marked reset sites below.  It must not carry over between trades.
    partial_fired   = False
    partial_idx     = 0
    partial_cum_val = 0.0

    for i in range(T):
        d = dist_sd[i]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > XING_SD:
                in_trade      = True
                entry_idx     = i
                entry_cum     = cum[i]
                side          = -1
                partial_fired = False          # ← reset: new trade starting
            elif d < -XING_SD:
                in_trade      = True
                entry_idx     = i
                entry_cum     = cum[i]
                side          = +1
                partial_fired = False          # ← reset: new trade starting
        else:
            # ── Check partial exit gates ──────────────────────────────────
            if not partial_fired:
                days_held  = day_ints[i] - day_ints[entry_idx]
                time_open  = days_held >= N
                prox_open  = (sd_gate is None) or (abs(d) <= sd_gate)
                if time_open and prox_open:
                    partial_fired   = True
                    partial_idx     = i
                    partial_cum_val = cum[i]

            # ── Check final exit ──────────────────────────────────────────
            max_hold_hit = (day_ints[i] - day_ints[entry_idx]) >= MAX_HOLD
            normal_exit  = (
                (side == -1 and d <= EXIT_SD) or
                (side == +1 and d >= -EXIT_SD)
            )

            if normal_exit or max_hold_hit:
                final_cum  = cum[i]
                final_hold = int(day_ints[i] - day_ints[entry_idx])

                if side == +1:
                    r_final = (final_cum - entry_cum) / entry_cum
                else:
                    r_final = (entry_cum - final_cum) / entry_cum

                if partial_fired:
                    partial_hold = int(day_ints[partial_idx] - day_ints[entry_idx])
                    r_partial    = (
                        (partial_cum_val - entry_cum) / entry_cum if side == +1
                        else (entry_cum - partial_cum_val) / entry_cum
                    )
                    gross_total      = F * r_partial + (1 - F) * r_final
                    financing        = DAILY_FIN * (F * partial_hold + (1 - F) * final_hold)
                    effective_hold   = F * partial_hold + (1 - F) * final_hold
                    financing_saving = DAILY_FIN * F * (final_hold - partial_hold)
                else:
                    partial_hold     = final_hold
                    r_partial        = float('nan')
                    gross_total      = r_final
                    financing        = DAILY_FIN * final_hold
                    effective_hold   = float(final_hold)
                    financing_saving = 0.0

                net_total  = gross_total - SPREAD_COST - financing
                net_bl_eq  = r_final     - SPREAD_COST - DAILY_FIN * final_hold

                trades_out.append(dict(
                    net_total        = net_total,
                    net_bl_equiv     = net_bl_eq,
                    net_improvement  = net_total - net_bl_eq,
                    fired            = bool(partial_fired),
                    r_partial        = r_partial,
                    r_final          = r_final,
                    partial_hold     = partial_hold,
                    final_hold       = final_hold,
                    effective_hold   = effective_hold,
                    financing_saving = financing_saving,
                    max_hold_exit    = bool(max_hold_hit and not normal_exit),
                ))
                in_trade = False

    return trades_out


# ── Aggregation helpers ──────────────────────────────────────────────────────

def agg_trades(trades):
    if not trades:
        return dict(n=0, net_wr=0.0, avg_net=0.0, avg_net_bl=0.0,
                    avg_net_imp=0.0, pct_fired=0.0, avg_hold=0.0,
                    avg_eff_hold=0.0, avg_fin_saving=0.0)
    nets  = np.array([t['net_total']        for t in trades])
    bl_n  = np.array([t['net_bl_equiv']     for t in trades])
    holds = np.array([t['final_hold']       for t in trades])
    efh   = np.array([t['effective_hold']   for t in trades])
    fins  = np.array([t['financing_saving'] for t in trades])
    fired = np.array([t['fired']            for t in trades], dtype=bool)
    return dict(
        n              = len(trades),
        net_wr         = float((nets > 0).mean()),
        avg_net        = float(nets.mean()),
        avg_net_bl     = float(bl_n.mean()),
        avg_net_imp    = float((nets - bl_n).mean()),
        pct_fired      = float(fired.mean() * 100),
        avg_hold       = float(holds.mean()),
        avg_eff_hold   = float(efh.mean()),
        avg_fin_saving = float(fins.mean()),
    )


def classify_remainders(trades):
    """
    Section 4 classification of fired-trade remainder outcomes.

    A — r_final > r_partial (remainder outperformed locked-in level)
    B — 0 <= r_final <= r_partial (profitable but <= locked-in level)
    C — r_final < 0 OR max_hold exit (spread deteriorated; partial exit protected)

    Note: bucket A is NOT evidence the partial exit was suboptimal — the closed
    fraction still eliminated carry for (final_hold − partial_hold) days.
    """
    fired_t = [t for t in trades if t['fired']]
    if not fired_t:
        return dict(n_fired=0, pct_A=0.0, pct_B=0.0, pct_C=0.0, avg_C_damage=0.0)

    A, B, C_damage = 0, 0, []
    for t in fired_t:
        rp, rf = t['r_partial'], t['r_final']
        if t['max_hold_exit'] or rf < 0:
            # C — "avg_C_damage" = how much worse the remainder did vs locking in
            C_damage.append(rp - rf)
        elif rf > rp:
            A += 1
        else:
            B += 1

    n_f = len(fired_t)
    n_C = n_f - A - B
    return dict(
        n_fired      = n_f,
        pct_A        = A / n_f * 100,
        pct_B        = B / n_f * 100,
        pct_C        = n_C / n_f * 100,
        avg_C_damage = float(np.mean(C_damage)) if C_damage else 0.0,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading prices...")
    prices, _ = load_asset_prices(DATA_PATH)

    # ── Step 0: per-pair baseline holding periods ────────────────────────────
    print("\nStep 0 — computing baseline holding periods for all 20 pairs...")
    holds_tbl  = []
    pair_holds = {}

    for long_inst, short_inst in ALL_PAIRS:
        lbl = f"{long_inst}/{short_inst}"
        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
        _, bl_hold, _ = run_baseline(cum, dist_sd, day_ints)
        median_h = float(np.median(bl_hold)) if len(bl_hold) else 0.0
        avg_h    = float(bl_hold.mean())     if len(bl_hold) else 0.0
        pair_holds[lbl] = median_h
        holds_tbl.append(dict(
            pair=lbl, n_trades=len(bl_hold),
            median_hold=round(median_h, 1), avg_hold=round(avg_h, 1),
        ))

    holds_df = pd.DataFrame(holds_tbl)

    # ── Conditional grid: 36 combos × 20 pairs ──────────────────────────────
    print("Running conditional grid (36 combos × 20 pairs)...")
    cond_rows     = []
    best_per_pair = {}

    for long_inst, short_inst in ALL_PAIRS:
        lbl         = f"{long_inst}/{short_inst}"
        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
        median_hold = pair_holds[lbl]
        bl_gross, bl_hold, _ = run_baseline(cum, dist_sd, day_ints)
        bl_avg_net = (
            float((bl_gross - SPREAD_COST - DAILY_FIN * bl_hold).mean())
            if len(bl_gross) else 0.0
        )
        pair_best_imp = -np.inf

        for tgp in TIME_GATE_PCTS:
            N = max(1, round(tgp * median_hold))
            for sdg in SD_GATES:
                for F in FRACTIONS:
                    trades = run_partial_exit(cum, dist_sd, day_ints, N, F, sd_gate=sdg)
                    st = agg_trades(trades)
                    row = dict(
                        pair            = lbl,
                        time_gate_pct   = tgp,
                        N_days          = N,
                        sd_gate         = sdg,
                        fraction        = F,
                        bl_avg_net      = round(bl_avg_net, 4),
                        pe_avg_net      = round(st['avg_net'], 4),
                        net_improvement = round(st['avg_net_imp'], 4),
                        pct_fired       = round(st['pct_fired'], 1),
                        pct_no_fire     = round(100 - st['pct_fired'], 1),
                        avg_hold_bl     = round(st['avg_hold'], 1),
                        avg_hold_pe     = round(st['avg_eff_hold'], 1),
                        avg_fin_saving  = round(st['avg_fin_saving'], 4),
                    )
                    cond_rows.append(row)
                    if st['avg_net_imp'] > pair_best_imp:
                        pair_best_imp       = st['avg_net_imp']
                        best_per_pair[lbl]  = row

    cond_df = pd.DataFrame(cond_rows)

    # Aggregate across pairs for Section 1
    cond_agg_rows = []
    for (tgp, sdg, F), grp in cond_df.groupby(['time_gate_pct', 'sd_gate', 'fraction']):
        cond_agg_rows.append(dict(
            time_gate_pct       = tgp,
            sd_gate             = sdg,
            fraction            = F,
            avg_N_days          = round(grp['N_days'].mean(), 0),
            avg_bl_net          = round(grp['bl_avg_net'].mean(), 4),
            avg_pe_net          = round(grp['pe_avg_net'].mean(), 4),
            avg_net_improvement = round(grp['net_improvement'].mean(), 4),
            avg_fire_pct        = round(grp['pct_fired'].mean(), 1),
            avg_hold_reduction  = round((grp['avg_hold_bl'] - grp['avg_hold_pe']).mean(), 1),
            avg_fin_saving      = round(grp['avg_fin_saving'].mean(), 4),
        ))
    cond_agg_df = (
        pd.DataFrame(cond_agg_rows)
        .sort_values('avg_net_improvement', ascending=False)
        .reset_index(drop=True)
    )

    sdg_tbl = (
        cond_df.groupby('sd_gate')
        .agg(avg_fire_pct=('pct_fired', 'mean'),
             avg_net_improvement=('net_improvement', 'mean'))
        .reset_index()
        .round({'avg_fire_pct': 1, 'avg_net_improvement': 4})
    )

    # ── Time-only grid: 9 combos × 20 pairs ──────────────────────────────────
    print("Running time-only grid (9 combos × 20 pairs)...")
    to_rows = []

    for long_inst, short_inst in ALL_PAIRS:
        lbl         = f"{long_inst}/{short_inst}"
        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
        median_hold = pair_holds[lbl]
        bl_gross, bl_hold, _ = run_baseline(cum, dist_sd, day_ints)
        bl_avg_net = (
            float((bl_gross - SPREAD_COST - DAILY_FIN * bl_hold).mean())
            if len(bl_gross) else 0.0
        )

        for tgp in TIME_GATE_PCTS:
            N = max(1, round(tgp * median_hold))
            for F in FRACTIONS:
                trades = run_partial_exit(cum, dist_sd, day_ints, N, F, sd_gate=None)
                st = agg_trades(trades)
                to_rows.append(dict(
                    pair            = lbl,
                    time_gate_pct   = tgp,
                    N_days          = N,
                    fraction        = F,
                    bl_avg_net      = round(bl_avg_net, 4),
                    pe_avg_net      = round(st['avg_net'], 4),
                    net_improvement = round(st['avg_net_imp'], 4),
                    pct_fired       = round(st['pct_fired'], 1),
                    avg_hold_bl     = round(st['avg_hold'], 1),
                    avg_hold_pe     = round(st['avg_eff_hold'], 1),
                    avg_fin_saving  = round(st['avg_fin_saving'], 4),
                ))

    to_df = pd.DataFrame(to_rows)

    to_agg_rows = []
    for (tgp, F), grp in to_df.groupby(['time_gate_pct', 'fraction']):
        to_agg_rows.append(dict(
            time_gate_pct       = tgp,
            fraction            = F,
            avg_N_days          = round(grp['N_days'].mean(), 0),
            avg_bl_net          = round(grp['bl_avg_net'].mean(), 4),
            avg_pe_net          = round(grp['pe_avg_net'].mean(), 4),
            avg_net_improvement = round(grp['net_improvement'].mean(), 4),
            avg_fire_pct        = round(grp['pct_fired'].mean(), 1),
            avg_hold_reduction  = round((grp['avg_hold_bl'] - grp['avg_hold_pe']).mean(), 1),
            avg_fin_saving      = round(grp['avg_fin_saving'].mean(), 4),
        ))
    to_agg_df = (
        pd.DataFrame(to_agg_rows)
        .sort_values('avg_net_improvement', ascending=False)
        .reset_index(drop=True)
    )

    # ── Section 4: remainder analysis ────────────────────────────────────────
    print("Running Section 4 remainder analysis (representative combo)...")
    sec4_rows = []
    for tgp in TIME_GATE_PCTS:
        all_trades = []
        for long_inst, short_inst in ALL_PAIRS:
            lbl         = f"{long_inst}/{short_inst}"
            _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
            N = max(1, round(tgp * pair_holds[lbl]))
            trades = run_partial_exit(
                cum, dist_sd, day_ints, N, SEC4_FRACTION, sd_gate=SEC4_SD_GATE
            )
            all_trades.extend(trades)

        cl = classify_remainders(all_trades)
        sec4_rows.append(dict(
            time_gate_pct = tgp,
            sd_gate_used  = SEC4_SD_GATE,
            fraction_used = SEC4_FRACTION,
            n_fired       = cl['n_fired'],
            pct_A         = round(cl['pct_A'], 1),
            pct_B         = round(cl['pct_B'], 1),
            pct_C         = round(cl['pct_C'], 1),
            avg_C_damage  = round(cl['avg_C_damage'], 4),
        ))
    sec4_df = pd.DataFrame(sec4_rows)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    holds_df.to_csv(RESULTS_DIR / "partial_exit_baseline_holds.csv",   index=False)
    cond_df.to_csv( RESULTS_DIR / "partial_exit_grid_conditional.csv", index=False)
    to_df.to_csv(   RESULTS_DIR / "partial_exit_grid_timeonly.csv",    index=False)
    pd.DataFrame(list(best_per_pair.values())).to_csv(
        RESULTS_DIR / "partial_exit_best_per_pair.csv", index=False)
    sec4_df.to_csv(RESULTS_DIR / "partial_exit_remainder_analysis.csv", index=False)

    # ── Print all sections ────────────────────────────────────────────────────
    _print_results(cond_agg_df, sdg_tbl, to_agg_df, best_per_pair, sec4_df)
    print(f"\nCSVs saved to {RESULTS_DIR}/")


# ── Terminal reporting ───────────────────────────────────────────────────────

def _print_results(cond_agg_df, sdg_tbl, to_agg_df, best_per_pair, sec4_df):
    W = 108

    print("\n" + "═" * W)
    print("SECTION 1 — CONDITIONAL GRID AGGREGATE  "
          "(time + proximity gate, 36 combos × 20 pairs)")
    print("═" * W)
    d1 = cond_agg_df.copy()
    d1['avg_net_improvement'] = d1['avg_net_improvement'].map('{:+.4f}'.format)
    d1['avg_bl_net']  = d1['avg_bl_net'].map('{:.4f}'.format)
    d1['avg_pe_net']  = d1['avg_pe_net'].map('{:.4f}'.format)
    print(d1.to_string(index=False))

    print("\n  sd_gate fire rate breakdown (aggregated across tgp and fraction):")
    s = sdg_tbl.copy()
    s['avg_net_improvement'] = s['avg_net_improvement'].map('{:+.4f}'.format)
    s['avg_fire_pct'] = s['avg_fire_pct'].map('{:.1f}%'.format)
    print(s.to_string(index=False))

    print("\n" + "═" * W)
    print("SECTION 2 — TIME-ONLY GRID AGGREGATE  "
          "(no proximity gate, 9 combos × 20 pairs)")
    print("═" * W)
    d2 = to_agg_df.copy()
    d2['avg_net_improvement'] = d2['avg_net_improvement'].map('{:+.4f}'.format)
    d2['avg_bl_net'] = d2['avg_bl_net'].map('{:.4f}'.format)
    d2['avg_pe_net'] = d2['avg_pe_net'].map('{:.4f}'.format)
    print(d2.to_string(index=False))

    bc = cond_agg_df.iloc[0]
    bt = to_agg_df.iloc[0]
    bc_imp = float(bc['avg_net_improvement'])
    bt_imp = float(bt['avg_net_improvement'])
    print(f"\n  Best conditional: tgp={bc['time_gate_pct']}, "
          f"sd_gate={bc['sd_gate']}, F={bc['fraction']}  →  {bc_imp:+.4f}")
    print(f"  Best time-only:   tgp={bt['time_gate_pct']}, "
          f"F={bt['fraction']}  →  {bt_imp:+.4f}")
    prox_value = "proximity gate ADDS value" if bc_imp > bt_imp + 0.0005 else \
                 "proximity gate adds negligible value" if abs(bc_imp - bt_imp) <= 0.0005 else \
                 "time-only OUTPERFORMS conditional"
    print(f"  Proximity gate verdict: {prox_value}")

    print("\n" + "═" * W)
    print("SECTION 3 — BEST COMBO PER PAIR  (conditional variant)")
    print("═" * W)
    best_df = pd.DataFrame(list(best_per_pair.values()))
    cols = ['pair', 'time_gate_pct', 'N_days', 'sd_gate', 'fraction',
            'bl_avg_net', 'pe_avg_net', 'net_improvement',
            'pct_fired', 'pct_no_fire', 'avg_hold_bl', 'avg_hold_pe']
    sub = best_df[cols].copy()
    sub['net_improvement'] = sub['net_improvement'].map('{:+.4f}'.format)
    sub['bl_avg_net'] = sub['bl_avg_net'].map('{:.4f}'.format)
    sub['pe_avg_net'] = sub['pe_avg_net'].map('{:.4f}'.format)
    print(sub.to_string(index=False))
    n_imp = (best_df['net_improvement'] > 0).sum()
    print(f"\n  Pairs improved: {n_imp}/{len(best_df)}")
    print(f"  Mean best improvement: {best_df['net_improvement'].mean():+.4f}")

    print("\n" + "═" * W)
    print("SECTION 4 — REMAINDER POSITION ANALYSIS")
    print(f"  Representative combo: sd_gate={SEC4_SD_GATE}, F={SEC4_FRACTION}  "
          f"(fixed to avoid double-counting across grid combos)")
    print("  A = r_final > r_partial   (remainder outperformed locked-in level)")
    print("  B = 0 ≤ r_final ≤ r_partial  (profitable but ≤ locked-in level)")
    print("  C = r_final < 0 OR max_hold exit  (deteriorated after partial)")
    print("  NOTE: bucket A does NOT mean partial exit was wrong — the closed")
    print("  fraction still saved DAILY_FIN × F × (final_hold − partial_hold) carry.")
    print("═" * W)
    print(sec4_df.to_string(index=False))

    print("\n" + "═" * W)
    print("SECTION 5 — RECOMMENDATION")
    print("═" * W)
    _recommendation(cond_agg_df, to_agg_df, best_df, sec4_df,
                    bc_imp, bt_imp, n_imp)
    print("═" * W + "\n")


def _recommendation(cond_agg_df, to_agg_df, best_df, sec4_df,
                    bc_imp, bt_imp, n_imp):
    bc        = cond_agg_df.iloc[0]
    bt        = to_agg_df.iloc[0]
    n_pairs   = len(best_df)
    trailing_best = -0.0011   # best trailing stop result for comparison

    print(f"\n  Best conditional combo : tgp={bc['time_gate_pct']}, "
          f"sd_gate={bc['sd_gate']}, F={bc['fraction']}  →  {bc_imp:+.4f}")
    print(f"  Best time-only combo   : tgp={bt['time_gate_pct']}, "
          f"F={bt['fraction']}  →  {bt_imp:+.4f}")
    print(f"  Trailing stop best     : {trailing_best:+.4f}")

    prox = "Proximity gate adds value." if bc_imp > bt_imp + 0.0005 else \
           "Time-only matches or outperforms conditional."
    print(f"\n  Proximity gate: {prox}")
    print(f"  Robustness: {n_imp}/{n_pairs} pairs improved by their best combo")

    # Section 4 summary (use row with most fired trades for most informative stats)
    row4 = sec4_df.loc[sec4_df['n_fired'].idxmax()]
    print(f"\n  Remainder analysis (at tgp={row4['time_gate_pct']}, "
          f"n_fired={row4['n_fired']}):")
    print(f"    A — outperformed locked-in : {row4['pct_A']:.1f}%")
    print(f"    B — completed fine          : {row4['pct_B']:.1f}%")
    print(f"    C — deteriorated            : {row4['pct_C']:.1f}%  "
          f"avg damage = {row4['avg_C_damage']:+.4f}")

    if bc_imp > trailing_best + 0.001:
        verdict = (
            f"VERDICT: Time-based partial exit ({bc_imp:+.4f}) beats the trailing "
            f"stop ({trailing_best:+.4f}). "
            f"{'Robust' if n_imp >= n_pairs * 0.6 else 'Not robust'} "
            f"across pairs ({n_imp}/{n_pairs} improved)."
        )
    elif bc_imp > 0:
        verdict = (
            f"VERDICT: Marginal positive improvement ({bc_imp:+.4f}) vs trailing "
            f"stop ({trailing_best:+.4f}), but not meaningfully better than "
            f"hold-to-zero."
        )
    else:
        verdict = (
            f"VERDICT: No partial exit configuration improves net expectancy. "
            f"Best: {bc_imp:+.4f}. Neither trailing stop nor time-based partial "
            f"exit beats hold-to-zero."
        )
    print(f"\n  {verdict}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
