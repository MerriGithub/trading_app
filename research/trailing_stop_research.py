"""
trailing_stop_research.py
=========================
Conditional profit-lock stop backtest on top commodity pairs.

Tests a stop that only arms once the spread has partially reverted,
then locks in partial profit if the spread re-extends.  Compared
against a no-stop baseline (exit at 0 SD).

Usage:
    python trailing_stop_research.py

Outputs:
    results/trailing_stop_grid.csv          — full param grid results
    results/trailing_stop_best_per_pair.csv — best combo per pair
    results/noise_analysis_<activation>.png — distribution charts
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROOT = Path(__file__).parent.parent   # project root (script lives in research/)
sys.path.insert(0, str(ROOT))

from engine.backtest import load_asset_prices
from engine.numba_core import rolling_mean_std, detect_trades, COL_GROSS_RETURN, COL_HOLDING_DAYS

# ── Config ──────────────────────────────────────────────────────────────────

DATA_PATH   = ROOT / "cache" / "commodity_prices.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VOL_WINDOW = 262
XING_SD    = 2.0
EXIT_SD    = 0.0
TARGET_VOL = 0.01
MAX_HOLD   = 300

# Costs applied directly — do not use aggregate_trades() multiplier
SPREAD_COST = 0.002                       # 0.10% entry RT + 0.10% exit RT
DAILY_FIN   = (0.0488 - 0.0088) / 365    # net carry 4.00% p.a. / 365

BASE_PAIRS = [
    ("NATGAS",   "COPPER"),
    ("NATGAS",   "BRENT"),
    ("NATGAS",   "COFFEE"),
    ("NATGAS",   "SUGAR"),
    ("NATGAS",   "SOYBEANS"),
    ("SILVER",   "COFFEE"),
    ("WHEAT",    "BRENT"),
    ("SOYBEANS", "WHEAT"),
    ("SOYBEANS", "PLATINUM"),
    ("GOLD",     "NATGAS"),
]

ACTIVATION_LEVELS = [1.5, 1.25, 1.0, 0.75, 0.5]
BUFFERS           = [0.1, 0.2, 0.3, 0.5, 0.75]

# Stepped lock: (activation_sd, new_stop_sd) steps applied in order
STEPPED_STEPS = [(1.5, 2.0), (1.0, 1.5), (0.5, 1.0)]


# ── Spread construction ──────────────────────────────────────────────────────

def build_spread(prices: pd.DataFrame, long_inst: str, short_inst: str):
    """
    Vol-scale each instrument, form spread returns, compute cumulative and dist_sd.

    Returns: spread_ret, day_ints, cum, dist_sd
    """
    df = prices[[long_inst, short_inst]].dropna()
    rets = df.pct_change().dropna()
    vols = rets.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std()
    scalings = (TARGET_VOL / vols.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    scaled = (rets * scalings).dropna(how='any')

    spread_ret = (scaled[long_inst] - scaled[short_inst]).values.astype(np.float64)
    idx = scaled.index
    day_ints = (
        (idx - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    ).values.astype(np.int64)

    cum = np.cumprod(1.0 + spread_ret)
    roll_mean, roll_std = rolling_mean_std(cum, VOL_WINDOW)
    dist_sd = np.where(
        (roll_std > 0) & ~np.isnan(roll_std),
        (cum - roll_mean) / roll_std,
        np.nan,
    )
    return spread_ret, day_ints, cum, dist_sd


# ── Net return calculation ───────────────────────────────────────────────────

def net_ret(gross: np.ndarray, holdings: np.ndarray) -> np.ndarray:
    return gross - SPREAD_COST - DAILY_FIN * holdings


def stats(gross: np.ndarray, holdings: np.ndarray) -> dict:
    if len(gross) == 0:
        return dict(n=0, gross_wr=0.0, net_wr=0.0,
                    avg_gross=0.0, avg_net=0.0, avg_hold=0.0)
    net = net_ret(gross, holdings)
    return dict(
        n         = int(len(gross)),
        gross_wr  = float((gross > 0).mean()),
        net_wr    = float((net > 0).mean()),
        avg_gross = float(gross.mean()),
        avg_net   = float(net.mean()),
        avg_hold  = float(holdings.mean()),
    )


# ── Baseline trade extraction ────────────────────────────────────────────────

def run_baseline(cum, dist_sd, day_ints):
    """Run standard crossing signal. Returns (gross, holdings, entry_indices)."""
    trades, n = detect_trades(cum, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)
    if n == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)
    t = trades[:n]
    return t[:, COL_GROSS_RETURN], t[:, COL_HOLDING_DAYS], t[:, 0].astype(int)


# ── Trailing stop state machine ──────────────────────────────────────────────

def run_trailing_stop(cum, dist_sd, day_ints, activation_sd, buffer_sd):
    """
    Single-activation profit-lock stop.

    After entry at ±XING_SD, when spread reverts to ±activation_sd the stop
    is armed at activation_sd + buffer_sd.  If spread re-extends to the stop
    level the trade closes early.  Otherwise exits at 0 SD.

    Returns: gross, holdings, stop_fired (bool array), entry_indices
    """
    T = len(cum)
    gross_l, hold_l, fired_l, eidx_l = [], [], [], []

    in_trade   = False
    entry_idx  = 0
    entry_cum  = 0.0
    side       = 0
    stop_armed = False
    stop_level = 0.0

    for i in range(T):
        d = dist_sd[i]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > XING_SD:
                in_trade = True; entry_idx = i; entry_cum = cum[i]
                side = -1; stop_armed = False
            elif d < -XING_SD:
                in_trade = True; entry_idx = i; entry_cum = cum[i]
                side = +1; stop_armed = False
        else:
            # Arm stop once activation threshold is first crossed
            if not stop_armed:
                if side == -1 and d <= activation_sd:
                    stop_armed = True
                    stop_level = activation_sd + buffer_sd
                elif side == +1 and d >= -activation_sd:
                    stop_armed = True
                    stop_level = -(activation_sd + buffer_sd)

            max_hold_hit = (day_ints[i] - day_ints[entry_idx]) >= MAX_HOLD
            normal_exit  = (side == -1 and d <= 0.0) or (side == +1 and d >= 0.0)
            stop_hit     = stop_armed and (
                (side == -1 and d >= stop_level) or
                (side == +1 and d <= stop_level)
            )

            if normal_exit or stop_hit or max_hold_hit:
                exit_cum = cum[i]
                gross = ((exit_cum - entry_cum) / entry_cum if side == +1
                         else (entry_cum - exit_cum) / entry_cum)
                hold  = float(day_ints[i] - day_ints[entry_idx])
                fired = bool(stop_hit and not normal_exit)
                gross_l.append(gross); hold_l.append(hold)
                fired_l.append(fired); eidx_l.append(entry_idx)
                in_trade = False

    if not gross_l:
        return (np.array([]), np.array([]),
                np.array([], dtype=bool), np.array([], dtype=int))
    return (np.array(gross_l), np.array(hold_l),
            np.array(fired_l, dtype=bool), np.array(eidx_l, dtype=int))


# ── Stepped lock state machine ───────────────────────────────────────────────

def run_stepped_lock(cum, dist_sd, day_ints, steps=None):
    """
    Stepped profit-lock stop.

    steps = [(activation_sd, stop_sd), ...] applied in order from largest to
    smallest activation_sd.  Each step ratchets the stop closer to 0 SD.

    Returns: gross, holdings, stop_fired, entry_indices
    """
    if steps is None:
        steps = STEPPED_STEPS

    T = len(cum)
    gross_l, hold_l, fired_l, eidx_l = [], [], [], []

    in_trade   = False
    entry_idx  = 0
    entry_cum  = 0.0
    side       = 0
    step_idx   = 0
    stop_level = None   # None = no stop yet

    for i in range(T):
        d = dist_sd[i]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > XING_SD:
                in_trade = True; entry_idx = i; entry_cum = cum[i]
                side = -1; step_idx = 0; stop_level = None
            elif d < -XING_SD:
                in_trade = True; entry_idx = i; entry_cum = cum[i]
                side = +1; step_idx = 0; stop_level = None
        else:
            # Advance through steps in order
            while step_idx < len(steps):
                act_sd, new_stop = steps[step_idx]
                triggered = (
                    (side == -1 and d <= act_sd) or
                    (side == +1 and d >= -act_sd)
                )
                if triggered:
                    stop_level = new_stop if side == -1 else -new_stop
                    step_idx += 1
                else:
                    break

            max_hold_hit = (day_ints[i] - day_ints[entry_idx]) >= MAX_HOLD
            normal_exit  = (side == -1 and d <= 0.0) or (side == +1 and d >= 0.0)
            stop_hit     = stop_level is not None and (
                (side == -1 and d >= stop_level) or
                (side == +1 and d <= stop_level)
            )

            if normal_exit or stop_hit or max_hold_hit:
                exit_cum = cum[i]
                gross = ((exit_cum - entry_cum) / entry_cum if side == +1
                         else (entry_cum - exit_cum) / entry_cum)
                hold  = float(day_ints[i] - day_ints[entry_idx])
                fired = bool(stop_hit and not normal_exit)
                gross_l.append(gross); hold_l.append(hold)
                fired_l.append(fired); eidx_l.append(entry_idx)
                in_trade = False

    if not gross_l:
        return (np.array([]), np.array([]),
                np.array([], dtype=bool), np.array([], dtype=int))
    return (np.array(gross_l), np.array(hold_l),
            np.array(fired_l, dtype=bool), np.array(eidx_l, dtype=int))


# ── Noise analysis ───────────────────────────────────────────────────────────

def noise_analysis(cum, dist_sd, day_ints, activation_sd):
    """
    For each baseline trade that crosses the activation threshold, record:
      peak_reext : max distance the spread bounced back above activation_sd
                   after first crossing it (0 = no bounce at all)
      outcome    : 'zero' if trade reached 0 SD, 'max_hold' otherwise
    """
    T = len(cum)
    records = []

    in_trade  = False
    side      = 0
    entry_idx = 0
    activated = False
    peak_reext = 0.0

    for i in range(T):
        d = dist_sd[i]
        if np.isnan(d):
            continue

        if not in_trade:
            if d > XING_SD:
                in_trade = True; side = -1; entry_idx = i
                activated = False; peak_reext = 0.0
            elif d < -XING_SD:
                in_trade = True; side = +1; entry_idx = i
                activated = False; peak_reext = 0.0
        else:
            if not activated:
                if (side == -1 and d <= activation_sd) or \
                   (side == +1 and d >= -activation_sd):
                    activated = True

            if activated:
                # Re-extension: how far did dist bounce back toward entry direction?
                reext = (d - activation_sd) if side == -1 else ((-d) - activation_sd)
                peak_reext = max(peak_reext, reext)

            max_hold_hit = (day_ints[i] - day_ints[entry_idx]) >= MAX_HOLD
            normal_exit  = (side == -1 and d <= 0.0) or (side == +1 and d >= 0.0)

            if normal_exit or max_hold_hit:
                if activated:
                    records.append({
                        'peak_reext': peak_reext,
                        'outcome': 'zero' if normal_exit else 'max_hold',
                    })
                in_trade = False

    return records


# ── Main analysis ────────────────────────────────────────────────────────────

def main():
    print("Loading commodity prices...")
    prices, instruments = load_asset_prices(DATA_PATH)
    print(f"  {len(prices)} trading days, instruments: {instruments}")

    # Build all pairs (forward + reversed)
    all_pairs = BASE_PAIRS + [(s, l) for l, s in BASE_PAIRS]
    print(f"\nRunning analysis on {len(all_pairs)} pairs "
          f"({len(ACTIVATION_LEVELS)} × {len(BUFFERS)} = "
          f"{len(ACTIVATION_LEVELS)*len(BUFFERS)} parameter combos)...\n")

    grid_rows  = []   # one row per (pair, activation, buffer)
    best_rows  = []   # one row per pair — best combo
    noise_data = {act: [] for act in ACTIVATION_LEVELS}  # for charts

    for pair_idx, (long_inst, short_inst) in enumerate(all_pairs):
        pair_label = f"{long_inst}/{short_inst}"
        print(f"[{pair_idx+1:02d}/{len(all_pairs)}] {pair_label}")

        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)

        # ── Baseline ──────────────────────────────────────────────────────
        bl_gross, bl_hold, bl_entry_idx = run_baseline(cum, dist_sd, day_ints)
        bl_stats = stats(bl_gross, bl_hold)
        bl_entry_set = set(bl_entry_idx.tolist())   # for counterfactual lookup

        # Map entry_idx → baseline outcome (True = reached 0, False = max_hold)
        _, bl_n = detect_trades(cum, dist_sd, XING_SD, EXIT_SD, day_ints, MAX_HOLD)
        # Rebuild with exit flag — detect_trades doesn't expose this directly,
        # so we infer: a baseline trade exited at 0 if holding < MAX_HOLD
        bl_reached_zero = {
            int(bl_entry_idx[k]): (bl_hold[k] < MAX_HOLD)
            for k in range(len(bl_entry_idx))
        }

        # ── Noise analysis (aggregate across activation levels) ────────────
        for act in ACTIVATION_LEVELS:
            recs = noise_analysis(cum, dist_sd, day_ints, act)
            noise_data[act].extend(recs)

        # ── Parameter grid ────────────────────────────────────────────────
        pair_best = None

        for act in ACTIVATION_LEVELS:
            for buf in BUFFERS:
                g, h, fired, eidx = run_trailing_stop(
                    cum, dist_sd, day_ints, act, buf
                )
                st = stats(g, h)

                n_fired = int(fired.sum())
                n_total = st['n']

                # Unnecessary fires: stop fired on trades that would have
                # eventually reached 0 SD in the baseline
                n_unnecessary = sum(
                    1 for ei in eidx[fired]
                    if bl_reached_zero.get(int(ei), False)
                )

                net_imp = st['avg_net'] - bl_stats['avg_net']

                row = dict(
                    pair         = pair_label,
                    long         = long_inst,
                    short        = short_inst,
                    activation   = act,
                    buffer       = buf,
                    bl_n         = bl_stats['n'],
                    bl_gross_wr  = round(bl_stats['gross_wr'], 4),
                    bl_net_wr    = round(bl_stats['net_wr'], 4),
                    bl_avg_gross = round(bl_stats['avg_gross'], 4),
                    bl_avg_net   = round(bl_stats['avg_net'], 4),
                    bl_avg_hold  = round(bl_stats['avg_hold'], 1),
                    st_n         = n_total,
                    st_gross_wr  = round(st['gross_wr'], 4),
                    st_net_wr    = round(st['net_wr'], 4),
                    st_avg_gross = round(st['avg_gross'], 4),
                    st_avg_net   = round(st['avg_net'], 4),
                    st_avg_hold  = round(st['avg_hold'], 1),
                    n_fired      = n_fired,
                    pct_fired    = round(n_fired / n_total * 100, 1) if n_total else 0,
                    n_unnecessary= n_unnecessary,
                    pct_unnecessary = round(n_unnecessary / n_fired * 100, 1) if n_fired else 0,
                    net_improvement = round(net_imp, 4),
                )
                grid_rows.append(row)

                if (pair_best is None or
                        row['net_improvement'] > pair_best['net_improvement']):
                    pair_best = row

        if pair_best:
            best_rows.append(pair_best)

    # ── Stepped lock pass ────────────────────────────────────────────────────
    print("\nRunning stepped lock variant...")
    stepped_rows = []

    for long_inst, short_inst in all_pairs:
        pair_label = f"{long_inst}/{short_inst}"
        _, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
        bl_gross, bl_hold, bl_entry_idx = run_baseline(cum, dist_sd, day_ints)
        bl_stats = stats(bl_gross, bl_hold)
        bl_reached_zero = {
            int(bl_entry_idx[k]): (bl_hold[k] < MAX_HOLD)
            for k in range(len(bl_entry_idx))
        }

        g, h, fired, eidx = run_stepped_lock(cum, dist_sd, day_ints)
        st = stats(g, h)
        n_fired = int(fired.sum())
        n_total = st['n']
        n_unnecessary = sum(
            1 for ei in eidx[fired]
            if bl_reached_zero.get(int(ei), False)
        )
        net_imp = st['avg_net'] - bl_stats['avg_net']

        stepped_rows.append(dict(
            pair            = pair_label,
            bl_n            = bl_stats['n'],
            bl_avg_net      = round(bl_stats['avg_net'], 4),
            st_n            = n_total,
            st_avg_net      = round(st['avg_net'], 4),
            st_net_wr       = round(st['net_wr'], 4),
            st_avg_hold     = round(st['avg_hold'], 1),
            n_fired         = n_fired,
            pct_fired       = round(n_fired / n_total * 100, 1) if n_total else 0,
            n_unnecessary   = n_unnecessary,
            pct_unnecessary = round(n_unnecessary / n_fired * 100, 1) if n_fired else 0,
            net_improvement = round(net_imp, 4),
        ))

    # ── Save CSVs ────────────────────────────────────────────────────────────
    grid_df   = pd.DataFrame(grid_rows)
    best_df   = pd.DataFrame(best_rows)
    stepped_df = pd.DataFrame(stepped_rows)

    grid_path    = RESULTS_DIR / "trailing_stop_grid.csv"
    best_path    = RESULTS_DIR / "trailing_stop_best_per_pair.csv"
    stepped_path = RESULTS_DIR / "trailing_stop_stepped_lock.csv"

    grid_df.to_csv(grid_path, index=False)
    best_df.to_csv(best_path, index=False)
    stepped_df.to_csv(stepped_path, index=False)

    # ── Noise analysis charts ────────────────────────────────────────────────
    if HAS_MPL:
        print("\nGenerating noise analysis charts...")
        for act, records in noise_data.items():
            if not records:
                continue
            df_n = pd.DataFrame(records)
            zero_reext    = df_n.loc[df_n['outcome'] == 'zero',     'peak_reext']
            maxhold_reext = df_n.loc[df_n['outcome'] == 'max_hold', 'peak_reext']

            fig, ax = plt.subplots(figsize=(10, 5))
            bins = np.linspace(-0.1, 2.0, 42)
            ax.hist(zero_reext, bins=bins, alpha=0.6,
                    label='Reached 0 SD (full reversion)', color='steelblue', density=True)
            ax.hist(maxhold_reext, bins=bins, alpha=0.6,
                    label='Never recovered (max hold)', color='firebrick', density=True)
            ax.axvline(0, color='k', linestyle='--', lw=0.8, label='Zero bounce threshold')
            ax.set_xlabel('Peak re-extension after activation (SD units)')
            ax.set_ylabel('Density')
            ax.set_title(
                f'Noise at activation = {act} SD — '
                f'{len(zero_reext)} reached 0 SD,  {len(maxhold_reext)} hit max hold'
            )
            ax.legend()
            fig.tight_layout()
            chart_path = RESULTS_DIR / f"noise_activation_{str(act).replace('.', '_')}.png"
            fig.savefig(chart_path, dpi=120)
            plt.close(fig)
            print(f"  Saved {chart_path.name}")
    else:
        print("\n  (matplotlib not installed — skipping charts; "
              "run: pip install matplotlib)")

    # ── Terminal output ──────────────────────────────────────────────────────
    _print_results(grid_df, best_df, stepped_df, noise_data)

    print(f"\nCSVs saved to {RESULTS_DIR}/")


# ── Terminal reporting ───────────────────────────────────────────────────────

def _print_results(grid_df, best_df, stepped_df, noise_data):

    print("\n" + "═" * 90)
    print("SECTION 1 — PARAMETER GRID AGGREGATE (all pairs combined)")
    print("═" * 90)
    agg = (
        grid_df
        .groupby(['activation', 'buffer'])
        .agg(
            avg_bl_net   = ('bl_avg_net',      'mean'),
            avg_st_net   = ('st_avg_net',       'mean'),
            avg_net_imp  = ('net_improvement',  'mean'),
            avg_fire_pct = ('pct_fired',        'mean'),
            avg_unnec_pct= ('pct_unnecessary',  'mean'),
            avg_hold_imp = ('st_avg_hold',      'mean'),
        )
        .reset_index()
        .sort_values('avg_net_imp', ascending=False)
    )
    agg['avg_bl_net']    = agg['avg_bl_net'].map('{:.4f}'.format)
    agg['avg_st_net']    = agg['avg_st_net'].map('{:.4f}'.format)
    agg['avg_net_imp']   = agg['avg_net_imp'].map('{:+.4f}'.format)
    agg['avg_fire_pct']  = agg['avg_fire_pct'].map('{:.1f}%'.format)
    agg['avg_unnec_pct'] = agg['avg_unnec_pct'].map('{:.1f}%'.format)
    print(agg.to_string(index=False))

    print("\n" + "═" * 90)
    print("SECTION 2 — BEST COMBO PER PAIR")
    print("═" * 90)
    cols = ['pair', 'activation', 'buffer', 'bl_avg_net', 'st_avg_net',
            'net_improvement', 'pct_fired', 'pct_unnecessary']
    sub = best_df[cols].copy()
    sub['net_improvement'] = sub['net_improvement'].map('{:+.4f}'.format)
    sub['bl_avg_net']      = sub['bl_avg_net'].map('{:.4f}'.format)
    sub['st_avg_net']      = sub['st_avg_net'].map('{:.4f}'.format)
    print(sub.to_string(index=False))

    n_improved = (best_df['net_improvement'] > 0).sum()
    print(f"\n  Pairs improved by best stop:  {n_improved} / {len(best_df)}")
    print(f"  Average best improvement:     {best_df['net_improvement'].mean():+.4f}")

    print("\n" + "═" * 90)
    print("SECTION 3 — STEPPED LOCK VARIANT")
    print("═" * 90)
    print(f"  Steps: {STEPPED_STEPS}")
    cols2 = ['pair', 'bl_avg_net', 'st_avg_net', 'net_improvement',
             'pct_fired', 'pct_unnecessary']
    sub2 = stepped_df[cols2].copy()
    sub2['net_improvement'] = sub2['net_improvement'].map('{:+.4f}'.format)
    sub2['bl_avg_net']      = sub2['bl_avg_net'].map('{:.4f}'.format)
    sub2['st_avg_net']      = sub2['st_avg_net'].map('{:.4f}'.format)
    print(sub2.to_string(index=False))
    sn_improved = (stepped_df['net_improvement'] > 0).sum()
    print(f"\n  Pairs improved by stepped lock: {sn_improved} / {len(stepped_df)}")
    print(f"  Average improvement:            {stepped_df['net_improvement'].mean():+.4f}")

    print("\n" + "═" * 90)
    print("SECTION 4 — NOISE ANALYSIS SUMMARY")
    print("═" * 90)
    print(f"  {'Activation':>12}  {'N zero':>8}  {'N maxhold':>9}  "
          f"{'Median reext (zero)':>21}  {'Median reext (fail)':>21}")
    for act in ACTIVATION_LEVELS:
        recs = noise_data[act]
        if not recs:
            continue
        df_n = pd.DataFrame(recs)
        z = df_n.loc[df_n['outcome'] == 'zero',     'peak_reext']
        f = df_n.loc[df_n['outcome'] == 'max_hold', 'peak_reext']
        print(f"  {act:>12.2f}  {len(z):>8}  {len(f):>9}  "
              f"{z.median():>21.3f}  {f.median():>21.3f}")

    print("\n" + "═" * 90)
    print("SECTION 5 — RECOMMENDATION")
    print("═" * 90)
    _print_recommendation(grid_df, best_df, stepped_df, noise_data)
    print("═" * 90 + "\n")


def _print_recommendation(grid_df, best_df, stepped_df, noise_data):
    # Best single parameter combo across all pairs
    agg_net = (
        grid_df.groupby(['activation', 'buffer'])['net_improvement']
        .mean().reset_index().sort_values('net_improvement', ascending=False)
    )
    best_act = agg_net.iloc[0]['activation']
    best_buf = agg_net.iloc[0]['buffer']
    best_imp = agg_net.iloc[0]['net_improvement']

    n_pairs_improved = (best_df['net_improvement'] > 0).sum()
    n_total_pairs    = len(best_df)

    stepped_mean_imp = stepped_df['net_improvement'].mean()
    single_mean_imp  = best_df['net_improvement'].mean()

    print(f"\n  Best single-activation combo: activation={best_act} SD, "
          f"buffer={best_buf} SD")
    print(f"  Mean net improvement (best combo, all pairs): {best_imp:+.4f}")
    print(f"  Pairs where ANY stop improves net expectancy: "
          f"{n_pairs_improved}/{n_total_pairs}")
    print(f"  Stepped lock mean improvement vs baseline:    {stepped_mean_imp:+.4f}")
    print(f"  Best single-activation mean improvement:      {single_mean_imp:+.4f}")

    if best_imp > 0.001:
        verdict = (
            f"  VERDICT: Activation={best_act}, buffer={best_buf} shows a "
            f"net improvement of {best_imp:+.4f} on average. "
            f"{'Robust' if n_pairs_improved >= n_total_pairs*0.6 else 'Concentrated'} "
            f"across pairs ({n_pairs_improved}/{n_total_pairs} improved)."
        )
    else:
        verdict = (
            "  VERDICT: No stop configuration meaningfully improves net expectancy "
            "vs hold-to-zero. The profit-lock stop is not beneficial for this strategy."
        )

    if stepped_mean_imp > single_mean_imp + 0.001:
        verdict += " Stepped lock outperforms single-activation."
    elif single_mean_imp > stepped_mean_imp + 0.001:
        verdict += " Single-activation outperforms stepped lock."
    else:
        verdict += " Single-activation and stepped lock perform similarly."

    print(verdict)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
