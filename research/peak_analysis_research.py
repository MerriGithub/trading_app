"""
peak_analysis_research.py
=========================
Peak characterisation analysis (Steps 1.1 + 1.2 of the Entry Strategy Research Plan).

Establishes observable characteristics of the peak extension event for each baseline
trade -- the point at which the spread reaches maximum dislocation before collapsing.

Usage (from project root):
  python research/peak_analysis_research.py

Outputs (all in results/):
  peak_analysis_summary_commodities.csv
  peak_by_velocity_quintile_commodities.csv
  peak_by_outcome_commodities.csv
  pre_peak_indicators_commodities.csv
  peak_per_pair_commodities.csv
  peak_trades_full_commodities.csv
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# -- Path setup ----------------------------------------------------------------
RESEARCH_DIR = Path(__file__).parent
ROOT         = RESEARCH_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH_DIR))

from trailing_stop_research import (
    build_spread, DAILY_FIN, SPREAD_COST, RESULTS_DIR, DATA_PATH,
    XING_SD, EXIT_SD, MAX_HOLD,
)

# -- Constants -----------------------------------------------------------------
VOL_WINDOW = 262

# -- Pair lists -- must be defined BEFORE DATA_CONFIGS -------------------------
COMMODITY_PAIRS = [
    ('NATGAS',   'COPPER'),  ('NATGAS',   'BRENT'),   ('NATGAS',   'COFFEE'),
    ('NATGAS',   'SUGAR'),   ('NATGAS',   'SOYBEANS'), ('SILVER',   'COFFEE'),
    ('WHEAT',    'BRENT'),   ('SOYBEANS', 'WHEAT'),   ('SOYBEANS', 'PLATINUM'),
    ('GOLD',     'NATGAS'),  ('COPPER',   'NATGAS'),  ('BRENT',    'NATGAS'),
    ('COFFEE',   'NATGAS'),  ('SUGAR',    'NATGAS'),  ('SOYBEANS', 'NATGAS'),
    ('COFFEE',   'SILVER'),  ('BRENT',    'WHEAT'),   ('WHEAT',    'SOYBEANS'),
    ('PLATINUM', 'SOYBEANS'),('NATGAS',   'GOLD'),
]

FX_PAIRS     = []
EQUITY_PAIRS = []

# -- Asset class configuration -- AFTER pair list definitions ------------------
DATA_CONFIGS = {
    'commodities': {
        'path':    DATA_PATH,
        'pairs':   COMMODITY_PAIRS,
        'bid_ask': SPREAD_COST,
        'label':   'Commodities',
    },
    'fx': {
        'path':    next((p for p in ['fx_prices.csv', 'cache/fx_prices.csv']
                         if os.path.exists(p)), None),
        'pairs':   FX_PAIRS,
        'bid_ask': 0.001,
        'label':   'FX',
    },
    'equities': {
        'path':    next((p for p in ['prices.csv', 'cache/prices.csv']
                         if os.path.exists(p)), None),
        'pairs':   EQUITY_PAIRS,
        'bid_ask': 0.0014,
        'label':   'Equities',
    },
}

W    = 78  # print width
SEP  = '=' * W
SEP2 = '-' * W


# ==============================================================================
# Core reusable functions
# ==============================================================================

def load_prices(asset_class):
    cfg = DATA_CONFIGS[asset_class]
    if cfg['path'] is None:
        raise FileNotFoundError(
            f"No data file found for asset_class='{asset_class}'. "
            f"Add the data file before running cross-asset analysis."
        )
    df = pd.read_csv(cfg['path'], parse_dates=['Date'], index_col='Date')
    return df


def run_baseline_with_paths(long_inst, short_inst, prices, asset_class='commodities'):
    """
    Run baseline backtest for one pair. Returns list of trade dicts containing the
    full dist_sd and cum paths from entry to exit, plus entry velocity pre-computed
    from the full dist_sd series.
    """
    spread_ret, day_ints, cum, dist_sd = build_spread(prices, long_inst, short_inst)
    trades   = []
    in_trade = False
    entry_idx = None
    side      = None

    for i in range(VOL_WINDOW, len(dist_sd)):
        if np.isnan(dist_sd[i]):
            continue

        if not in_trade:
            if dist_sd[i] > XING_SD:
                in_trade, entry_idx, side = True, i, -1
            elif dist_sd[i] < -XING_SD:
                in_trade, entry_idx, side = True, i, +1

            if in_trade:
                # Compute entry velocity from full dist_sd; dist_path has no pre-entry history
                entry_vels = {}
                for lb in [3, 5, 10]:
                    if entry_idx >= lb:
                        if side == -1:
                            v = dist_sd[entry_idx] - dist_sd[entry_idx - lb]
                        else:
                            v = dist_sd[entry_idx - lb] - dist_sd[entry_idx]
                    else:
                        v = np.nan
                    entry_vels[f'vel_{lb}_at_entry'] = v

        else:
            hold_cal    = int(day_ints[i] - day_ints[entry_idx])
            normal_exit = (side == -1 and dist_sd[i] <= EXIT_SD) or \
                          (side == +1 and dist_sd[i] >= EXIT_SD)
            max_hit     = hold_cal >= MAX_HOLD

            if normal_exit or max_hit:
                exit_idx = i
                dp  = dist_sd[entry_idx : exit_idx + 1].copy()
                cp  = cum[entry_idx : exit_idx + 1].copy()
                dip = (day_ints[entry_idx : exit_idx + 1]
                       - day_ints[entry_idx]).copy()

                gross = ((cp[-1] - cp[0]) / cp[0]) if side == +1 \
                        else ((cp[0] - cp[-1]) / cp[0])
                net   = gross \
                        - DATA_CONFIGS[asset_class]['bid_ask'] \
                        - DAILY_FIN * dip[-1]

                trade = {
                    'pair':           f"{long_inst}/{short_inst}",
                    'asset_class':    asset_class,
                    'side':           side,
                    'entry_idx':      entry_idx,
                    'exit_idx':       exit_idx,
                    'total_hold':     exit_idx - entry_idx,
                    'total_hold_cal': int(dip[-1]),
                    'gross_final':    gross,
                    'net_final':      net,
                    'is_winner':      net > 0,
                    'max_hold_exit':  max_hit,
                    'dist_path':      dp,
                    'cum_path':       cp,
                    'dayint_path':    dip,
                    **entry_vels,
                }
                trades.append(trade)
                in_trade = False

    return trades


def find_peak(trade):
    """
    Peak = time-step index of maximum |dist_sd| within dist_path.
    Returns (peak_step, peak_sd) where peak_sd = |dist_sd| at peak_step.
    """
    dp = trade['dist_path']
    if trade['side'] == -1:
        peak_step = int(np.argmax(dp))
    else:
        peak_step = int(np.argmin(dp))
    return peak_step, abs(dp[peak_step])


def compute_velocity_at_step(dist_sd_path, step, side, lookback):
    """Outward velocity at `step`. Positive = moving away from mean."""
    if step < lookback:
        return np.nan
    if side == -1:
        return dist_sd_path[step] - dist_sd_path[step - lookback]
    else:
        return dist_sd_path[step - lookback] - dist_sd_path[step]


def compute_acceleration_at_step(dist_sd_path, step, side, lookback=5):
    """Change in velocity over last `lookback` steps. Negative = decelerating."""
    v_now  = compute_velocity_at_step(dist_sd_path, step,           side, lookback)
    v_prev = compute_velocity_at_step(dist_sd_path, step - lookback, side, lookback)
    if np.isnan(v_now) or np.isnan(v_prev):
        return np.nan
    return v_now - v_prev


# ==============================================================================
# Step 0 -- Collect baseline trades
# ==============================================================================

def collect_all_trades(asset_class='commodities'):
    prices    = load_prices(asset_class)
    cfg       = DATA_CONFIGS[asset_class]
    all_trades = []
    for (long_inst, short_inst) in cfg['pairs']:
        if long_inst not in prices.columns or short_inst not in prices.columns:
            print(f"  Skipping {long_inst}/{short_inst} -- not in data")
            continue
        trades = run_baseline_with_paths(long_inst, short_inst, prices, asset_class)
        all_trades.extend(trades)
    print(f"\n  {asset_class}: {len(all_trades)} baseline trades collected")
    return all_trades


def print_baseline_summary(trades, asset_class):
    n          = len(trades)
    avg_net    = np.mean([t['net_final']      for t in trades])
    avg_hold   = np.mean([t['total_hold_cal'] for t in trades])
    win_rate   = np.mean([t['is_winner']      for t in trades])
    avg_gross  = np.mean([t['gross_final']    for t in trades])
    print(f"\nBaseline summary ({asset_class}):")
    print(f"  {'n_trades':>10} | {'avg_net':>8} | {'avg_hold_cal':>12} | "
          f"{'win_rate':>8} | {'avg_gross':>9}")
    print(f"  {n:>10} | {avg_net:>+8.4f} | {avg_hold:>10.0f} d | "
          f"  {win_rate:>6.1%} | {avg_gross:>+9.4f}")


# ==============================================================================
# Step 1 -- Augment each trade with peak signals
# ==============================================================================

def augment_with_peak_signals(trade):
    dp             = trade['dist_path']
    dip            = trade['dayint_path']
    side           = trade['side']
    total_hold_cal = trade['total_hold_cal']

    peak_step, peak_sd = find_peak(trade)
    peak_cal           = int(dip[peak_step])
    pct_hold_to_peak   = peak_cal / total_hold_cal if total_hold_cal > 0 else np.nan

    trade['peak_step']        = peak_step
    trade['peak_sd']          = peak_sd
    trade['peak_cal']         = peak_cal
    trade['pct_hold_to_peak'] = pct_hold_to_peak

    for lb in [3, 5, 10]:
        trade[f'vel_{lb}_at_peak'] = compute_velocity_at_step(dp, peak_step, side, lb)
    trade['accel_5_at_peak'] = compute_acceleration_at_step(dp, peak_step, side, 5)

    for n in [1, 3, 5, 10]:
        ref = peak_step - n
        if ref >= 0:
            trade[f'vel_5_at_peak_minus_{n}']   = compute_velocity_at_step(dp, ref, side, 5)
            trade[f'accel_5_at_peak_minus_{n}'] = compute_acceleration_at_step(dp, ref, side, 5)
        else:
            trade[f'vel_5_at_peak_minus_{n}']   = np.nan
            trade[f'accel_5_at_peak_minus_{n}'] = np.nan

    return trade


# ==============================================================================
# Output sections
# ==============================================================================

def _pct(arr, condition_fn):
    """Fraction of non-NaN values satisfying condition_fn."""
    valid = [x for x in arr if not np.isnan(x)]
    if not valid:
        return np.nan
    return sum(1 for x in valid if condition_fn(x)) / len(valid)


def section1_peak_distribution(trades, asset_class):
    label = DATA_CONFIGS[asset_class]['label']
    print(f"\n{SEP}")
    print(f"SECTION 1 -- PEAK SIGNAL INVENTORY  ({asset_class}, all pairs combined)")
    print(f"{SEP}")

    peak_sds   = [t['peak_sd']          for t in trades]
    pct_holds  = [t['pct_hold_to_peak'] for t in trades if not np.isnan(t['pct_hold_to_peak'])]
    peak_cals  = [t['peak_cal']         for t in trades]
    vel5_peaks = [t['vel_5_at_peak']    for t in trades if not np.isnan(t['vel_5_at_peak'])]
    accel5     = [t['accel_5_at_peak']  for t in trades if not np.isnan(t['accel_5_at_peak'])]

    print(f"\nPeak extension level:")
    print(f"  {'mean peak_sd':<30} | {np.mean(peak_sds):.2f} SD")
    print(f"  {'median peak_sd':<30} | {np.median(peak_sds):.2f} SD")
    print(f"  {'pct peak_sd < 2.5 SD':<30} | {_pct(peak_sds, lambda x: x < 2.5):.1%}")
    print(f"  {'pct peak_sd 2.5–3.0 SD':<30} | {_pct(peak_sds, lambda x: 2.5 <= x < 3.0):.1%}")
    print(f"  {'pct peak_sd 3.0–3.5 SD':<30} | {_pct(peak_sds, lambda x: 3.0 <= x < 3.5):.1%}")
    print(f"  {'pct peak_sd 3.5–4.0 SD':<30} | {_pct(peak_sds, lambda x: 3.5 <= x < 4.0):.1%}")
    print(f"  {'pct peak_sd > 4.0 SD':<30} | {_pct(peak_sds, lambda x: x > 4.0):.1%}")

    print(f"\nExtension phase duration:")
    print(f"  {'mean pct_hold_to_peak':<30} | {np.mean(pct_holds):.1%}   <-- KEY DECISION METRIC")
    print(f"  {'median pct_hold_to_peak':<30} | {np.median(pct_holds):.1%}")
    print(f"  {'mean peak_cal':<30} | {np.mean(peak_cals):.0f} d")
    print(f"  {'median peak_cal':<30} | {np.median(peak_cals):.0f} d")
    print(f"  {'pct pct_hold_to_peak > 50%':<30} | {_pct(pct_holds, lambda x: x > 0.5):.1%}")
    print(f"  {'pct pct_hold_to_peak > 75%':<30} | {_pct(pct_holds, lambda x: x > 0.75):.1%}")

    print(f"\nVelocity at peak (5-day):")
    print(f"  {'mean vel_5_at_peak':<30} | {np.mean(vel5_peaks):+.3f}")
    print(f"  {'median vel_5_at_peak':<30} | {np.median(vel5_peaks):+.3f}")
    print(f"  {'pct vel_5 < 0   at peak':<30} | {_pct(vel5_peaks, lambda x: x < 0.0):.1%}   <-- KEY DECISION METRIC")
    print(f"  {'pct vel_5 < 0.1 at peak':<30} | {_pct(vel5_peaks, lambda x: x < 0.1):.1%}")
    print(f"  {'pct vel_5 < 0.2 at peak':<30} | {_pct(vel5_peaks, lambda x: x < 0.2):.1%}")

    print(f"\nAcceleration at peak (5-day):")
    print(f"  {'mean accel_5_at_peak':<30} | {np.mean(accel5):+.3f}")
    print(f"  {'pct accel_5 < 0 at peak':<30} | {_pct(accel5, lambda x: x < 0):.1%}")

    # CSV
    summary_rows = [
        ('mean_peak_sd',             np.mean(peak_sds)),
        ('median_peak_sd',           np.median(peak_sds)),
        ('pct_peak_sd_lt_2.5',       _pct(peak_sds, lambda x: x < 2.5)),
        ('pct_peak_sd_2.5_3.0',      _pct(peak_sds, lambda x: 2.5 <= x < 3.0)),
        ('pct_peak_sd_3.0_3.5',      _pct(peak_sds, lambda x: 3.0 <= x < 3.5)),
        ('pct_peak_sd_3.5_4.0',      _pct(peak_sds, lambda x: 3.5 <= x < 4.0)),
        ('pct_peak_sd_gt_4.0',       _pct(peak_sds, lambda x: x > 4.0)),
        ('mean_pct_hold_to_peak',    np.mean(pct_holds)),
        ('median_pct_hold_to_peak',  np.median(pct_holds)),
        ('mean_peak_cal',            np.mean(peak_cals)),
        ('median_peak_cal',          np.median(peak_cals)),
        ('pct_hold_to_peak_gt_50',   _pct(pct_holds, lambda x: x > 0.5)),
        ('pct_hold_to_peak_gt_75',   _pct(pct_holds, lambda x: x > 0.75)),
        ('mean_vel_5_at_peak',       np.mean(vel5_peaks)),
        ('median_vel_5_at_peak',     np.median(vel5_peaks)),
        ('pct_vel5_lt_0_at_peak',    _pct(vel5_peaks, lambda x: x < 0.0)),
        ('pct_vel5_lt_0.1_at_peak',  _pct(vel5_peaks, lambda x: x < 0.1)),
        ('pct_vel5_lt_0.2_at_peak',  _pct(vel5_peaks, lambda x: x < 0.2)),
        ('mean_accel_5_at_peak',     np.mean(accel5)),
        ('pct_accel5_lt_0_at_peak',  _pct(accel5, lambda x: x < 0)),
    ]
    out = pd.DataFrame(summary_rows, columns=['metric', 'value'])
    path = RESULTS_DIR / f'peak_analysis_summary_{asset_class}.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}")


def section2_by_velocity_quintile(trades, asset_class):
    print(f"\n{SEP2}")
    print(f"SECTION 2 -- PEAK SIGNALS BY ENTRY VELOCITY QUINTILE")
    print(f"{'-'*W}")

    vel5_entry = [t['vel_5_at_entry'] for t in trades if not np.isnan(t.get('vel_5_at_entry', np.nan))]
    boundaries = np.percentile(vel5_entry, [20, 40, 60, 80])
    for t in trades:
        v = t.get('vel_5_at_entry', np.nan)
        t['vel_quintile'] = int(np.digitize(v, boundaries)) + 1 if not np.isnan(v) else np.nan

    hdr = (f"  {'quintile':<12} {'vel_range':>12}  {'n':>5}  {'peak_sd':>8}  "
           f"{'pct_hold_to_peak':>16}  {'vel_5_at_peak':>13}  {'pct_vel5_neg_peak':>17}")
    print(f"\nPeak characteristics by entry velocity quintile:")
    print(hdr)

    rows = []
    for q in range(1, 6):
        subset = [t for t in trades if t.get('vel_quintile') == q]
        if not subset:
            continue
        vels   = [t['vel_5_at_entry'] for t in subset if not np.isnan(t.get('vel_5_at_entry', np.nan))]
        label  = f"Q{q} ({'slow' if q==1 else 'fast' if q==5 else '':4})"
        vrange = f"[{min(vels):.2f},{max(vels):.2f}]" if vels else ''
        n      = len(subset)
        ps     = np.mean([t['peak_sd'] for t in subset])
        php    = np.mean([t['pct_hold_to_peak'] for t in subset
                          if not np.isnan(t['pct_hold_to_peak'])])
        v5     = np.mean([t['vel_5_at_peak'] for t in subset
                          if not np.isnan(t['vel_5_at_peak'])])
        pneg   = _pct([t['vel_5_at_peak'] for t in subset
                       if not np.isnan(t['vel_5_at_peak'])], lambda x: x < 0)
        print(f"  {label:<12} {vrange:>12}  {n:>5}  {ps:>7.2f} SD  "
              f"{php:>15.1%}  {v5:>+13.3f}  {pneg:>17.1%}")
        rows.append({'quintile': q, 'vel_range': vrange, 'n': n,
                     'mean_peak_sd': ps, 'mean_pct_hold_to_peak': php,
                     'mean_vel_5_at_peak': v5, 'pct_vel5_neg_at_peak': pneg})

    out  = pd.DataFrame(rows)
    path = RESULTS_DIR / f'peak_by_velocity_quintile_{asset_class}.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}")


def section3_by_outcome(trades, asset_class):
    print(f"\n{SEP2}")
    print(f"SECTION 3 -- PEAK SIGNALS BY TRADE OUTCOME")
    print(f"{'-'*W}")

    groups = {
        'Winners':        [t for t in trades if t['is_winner']],
        'Genuine losers': [t for t in trades if not t['is_winner'] and not t['max_hold_exit']],
        'Max-hold exits': [t for t in trades if t['max_hold_exit']],
    }

    hdr = (f"  {'outcome':<16} {'n':>5}  {'peak_sd':>8}  "
           f"{'pct_hold_to_peak':>16}  {'vel_5_at_peak':>13}  {'pct_vel5_neg':>12}")
    print(f"\nPeak characteristics by outcome:")
    print(hdr)

    rows = []
    for label, subset in groups.items():
        if not subset:
            print(f"  {label:<16} {'0':>5}")
            continue
        ps   = np.mean([t['peak_sd'] for t in subset])
        php  = np.mean([t['pct_hold_to_peak'] for t in subset
                        if not np.isnan(t['pct_hold_to_peak'])])
        v5   = np.mean([t['vel_5_at_peak'] for t in subset
                        if not np.isnan(t['vel_5_at_peak'])])
        pneg = _pct([t['vel_5_at_peak'] for t in subset
                     if not np.isnan(t['vel_5_at_peak'])], lambda x: x < 0)
        print(f"  {label:<16} {len(subset):>5}  {ps:>7.2f} SD  "
              f"{php:>15.1%}  {v5:>+13.3f}  {pneg:>12.1%}")
        rows.append({'outcome': label, 'n': len(subset), 'mean_peak_sd': ps,
                     'mean_pct_hold_to_peak': php, 'mean_vel_5_at_peak': v5,
                     'pct_vel5_neg_at_peak': pneg})

    out  = pd.DataFrame(rows)
    path = RESULTS_DIR / f'peak_by_outcome_{asset_class}.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}")


def section4_pre_peak_indicators(trades, asset_class):
    print(f"\n{SEP2}")
    print(f"SECTION 4 -- PRE-PEAK LEADING INDICATORS (Step 1.2)")
    print(f"{'-'*W}")

    print(f"\nPre-peak signal availability (5-day velocity lookback):")
    hdr = (f"  {'N days before peak':>18}  {'n_valid':>7}  "
           f"{'pct_vel5_neg':>12}  {'pct_accel5_neg':>14}  {'mean_vel5':>9}")
    print(hdr)

    rows = []
    for n in [10, 5, 3, 1, 0]:
        if n == 0:
            vel_key   = 'vel_5_at_peak'
            accel_key = 'accel_5_at_peak'
            label     = 'At peak (N=0)'
        else:
            vel_key   = f'vel_5_at_peak_minus_{n}'
            accel_key = f'accel_5_at_peak_minus_{n}'
            label     = str(n)

        vels   = [t[vel_key]   for t in trades if not np.isnan(t[vel_key])]
        accels = [t[accel_key] for t in trades if not np.isnan(t[accel_key])]
        nv     = len(vels)
        pvn    = _pct(vels,   lambda x: x < 0)
        pan    = _pct(accels, lambda x: x < 0)
        mv5    = np.mean(vels) if vels else np.nan

        if n == 0:
            print(f"  {'At peak (N=0)':>18}  {nv:>7}  {pvn:>12.1%}  {pan:>14.1%}  {mv5:>+9.3f}")
        else:
            print(f"  {n:>18}  {nv:>7}  {pvn:>12.1%}  {pan:>14.1%}  {mv5:>+9.3f}")
        rows.append({'n_days_before_peak': n, 'n_valid': nv,
                     'pct_vel5_neg': pvn, 'pct_accel5_neg': pan, 'mean_vel5': mv5})

    # Print interpretation
    rows_by_n = {r['n_days_before_peak']: r for r in rows}
    pv3 = rows_by_n.get(3, {}).get('pct_vel5_neg', np.nan)
    pv5 = rows_by_n.get(5, {}).get('pct_vel5_neg', np.nan)
    print()
    if not np.isnan(pv3):
        if pv3 > 0.5:
            print(f"  pct_vel5_neg at N=3 = {pv3:.1%} > 50% -> velocity turns negative ~3 days")
            print(f"  before peak on average; 3-day reversal signal provides real lead time.")
        else:
            print(f"  pct_vel5_neg at N=3 = {pv3:.1%} < 50% -> limited lead time at N=3.")
    if not np.isnan(pv5):
        if pv5 < 0.3:
            print(f"  pct_vel5_neg at N=5 = {pv5:.1%} < 30% -> velocity is concurrent, not leading;")
            print(f"  reversal fires at or after peak, not before.")

    out  = pd.DataFrame(rows)
    path = RESULTS_DIR / f'pre_peak_indicators_{asset_class}.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}")


def section5_per_pair(trades, asset_class):
    print(f"\n{SEP2}")
    print(f"SECTION 5 -- PER-PAIR PEAK SUMMARY")
    print(f"{'-'*W}")

    pairs = sorted(set(t['pair'] for t in trades))
    hdr   = (f"  {'pair':>22}  {'n':>5}  {'mean_peak_sd':>12}  "
             f"{'mean_pct_hold_to_peak':>21}  {'pct_vel5_neg_peak':>17}  {'bl_avg_net':>10}")
    print(f"\nPer-pair peak summary:")
    print(hdr)

    rows = []
    for pair in pairs:
        subset = [t for t in trades if t['pair'] == pair]
        ps     = np.mean([t['peak_sd'] for t in subset])
        php    = np.mean([t['pct_hold_to_peak'] for t in subset
                          if not np.isnan(t['pct_hold_to_peak'])])
        pneg   = _pct([t['vel_5_at_peak'] for t in subset
                       if not np.isnan(t['vel_5_at_peak'])], lambda x: x < 0)
        avg_net = np.mean([t['net_final'] for t in subset])

        flags = []
        if php > 0.6:   flags.append('SEVERE')
        if pneg > 0.7:  flags.append('CLEAN_SIG')
        if ps > 3.5:    flags.append('FAR_EXT')
        flag_str = ','.join(flags)

        print(f"  {pair:>22}  {len(subset):>5}  {ps:>10.2f} SD  "
              f"{php:>20.1%}  {pneg:>17.1%}  {avg_net:>+10.4f}  {flag_str}")
        rows.append({'pair': pair, 'n': len(subset), 'mean_peak_sd': ps,
                     'mean_pct_hold_to_peak': php, 'pct_vel5_neg_peak': pneg,
                     'bl_avg_net': avg_net, 'flags': flag_str})

    out  = pd.DataFrame(rows)
    path = RESULTS_DIR / f'peak_per_pair_{asset_class}.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}")


def section6_decision_gates(trades, asset_class):
    print(f"\n{SEP}")
    print(f"SECTION 6 -- DECISION GATES")
    print(f"{SEP}")

    pct_holds  = [t['pct_hold_to_peak'] for t in trades if not np.isnan(t['pct_hold_to_peak'])]
    vel5_peaks = [t['vel_5_at_peak']    for t in trades if not np.isnan(t['vel_5_at_peak'])]
    vel5_pm3   = [t['vel_5_at_peak_minus_3'] for t in trades
                  if not np.isnan(t['vel_5_at_peak_minus_3'])]

    mean_php    = np.mean(pct_holds)
    pv_lt0      = _pct(vel5_peaks, lambda x: x < 0.0)
    pv_lt01     = _pct(vel5_peaks, lambda x: x < 0.1)
    pv_lt0_pm3  = _pct(vel5_pm3,   lambda x: x < 0.0)

    winners = [t for t in trades if t['is_winner']]
    losers  = [t for t in trades if not t['is_winner'] and not t['max_hold_exit']]

    win_ps   = np.mean([t['peak_sd'] for t in winners]) if winners else np.nan
    los_ps   = np.mean([t['peak_sd'] for t in losers])  if losers  else np.nan
    win_pneg = _pct([t['vel_5_at_peak'] for t in winners
                     if not np.isnan(t['vel_5_at_peak'])], lambda x: x < 0)
    los_pneg = _pct([t['vel_5_at_peak'] for t in losers
                     if not np.isnan(t['vel_5_at_peak'])], lambda x: x < 0)

    # Gate 1
    if mean_php > 0.60:
        g1 = 'SEVERE >60%'
        g1_impl = ('Extension phase consumes more than 60% of hold time. '
                   'Entry timing is the primary lever. Phase 2 is high priority.')
    elif mean_php > 0.40:
        g1 = 'MODERATE 40–60%'
        g1_impl = ('Extension phase is substantial. Entry timing improvement '
                   'has meaningful but not dominant potential.')
    else:
        g1 = 'MILD <40%'
        g1_impl = ('Extension phase is short relative to hold. '
                   'Entry timing improvement is limited.')

    # Gate 2
    if pv_lt01 > 0.65:
        g2 = 'STRONG >65%'
        g2_impl = ('Velocity visibly stalls at peak. 5-day reversal signal is '
                   'viable. Step 2.1 (velocity reversal entry) is the first Phase 2 test.')
    elif pv_lt01 > 0.45:
        g2 = 'MODERATE 45–65%'
        g2_impl = ('Velocity decelerates at peak on most trades. Signal is usable '
                   'but noisy; combine with acceleration filter in Phase 2.')
    else:
        g2 = 'WEAK <45%'
        g2_impl = ('Velocity is too noisy to signal the peak reliably. '
                   'Consider price-level or time-based filters instead.')

    # Gate 3
    sd_gap   = abs(win_ps - los_ps)  if not (np.isnan(win_ps) or np.isnan(los_ps))  else np.nan
    pneg_gap = abs(win_pneg - los_pneg)
    if sd_gap > 0.3 or pneg_gap > 0.10:
        g3 = 'YES'
        g3_impl = ('Winner/loser peak characteristics are distinguishable. '
                   'Peak SD and velocity signal can be used as quality filters in Phase 2.')
    elif sd_gap > 0.1 or pneg_gap > 0.05:
        g3 = 'MARGINAL'
        g3_impl = ('Some separation between winners and losers but small. '
                   'Use as a secondary filter rather than primary signal.')
    else:
        g3 = 'NO'
        g3_impl = ('Winner and loser peak characteristics are similar. '
                   'Peak-level filters unlikely to add value in Phase 2.')

    print(f"\nGATE 1: How severe is the entry timing problem?")
    print(f"  mean_pct_hold_to_peak = {mean_php:.1%}")
    print(f"  -> [{g1}]")
    print(f"  Implication: {g1_impl}")

    print(f"\nGATE 2: Is velocity deceleration a reliable peak signal?")
    print(f"  pct_vel_5 < 0.1 at peak = {pv_lt01:.1%}")
    print(f"  pct_vel_5 < 0   at peak = {pv_lt0:.1%}")
    print(f"  Lead time (pct_vel_5 < 0 at peak_minus_3) = {pv_lt0_pm3:.1%}")
    print(f"  -> [{g2}]")
    print(f"  Implication: {g2_impl}")

    print(f"\nGATE 3: Do winners and losers have distinguishable peak characteristics?")
    print(f"  Winner peak_sd = {win_ps:.2f}   Loser peak_sd = {los_ps:.2f}   "
          f"Gap = {sd_gap:.2f} SD")
    print(f"  Winner pct_vel5_neg = {win_pneg:.1%}   Loser = {los_pneg:.1%}   "
          f"Gap = {pneg_gap:.1%} pp")
    print(f"  -> [{g3}]")
    print(f"  Implication: {g3_impl}")

    # Recommended next step
    print(f"\nRECOMMENDED NEXT STEP:")
    if 'STRONG' in g2 or 'MODERATE' in g2:
        rec = ('Test Step 2.1 (velocity reversal entry): enter on 5-day velocity '
               'turning negative within dist_path, using peak_trades_full_commodities.csv '
               'as input. Velocity decelerates reliably at peak, providing a real-time '
               'signal to delay entry until spread momentum reverses.')
    else:
        rec = ('Velocity signal is weak. Test Step 2.2 (time-based delay) instead: '
               'delay entry by a fixed number of days after the crossing and measure '
               'peak_sd improvement. Use peak_cal distribution to calibrate delay length.')
    print(f"  {rec}")
    print(f"{SEP}")


def save_full_trades(trades, asset_class):
    exclude = {'dist_path', 'cum_path', 'dayint_path'}
    rows    = [{k: v for k, v in t.items() if k not in exclude} for t in trades]
    out     = pd.DataFrame(rows)
    path    = RESULTS_DIR / f'peak_trades_full_{asset_class}.csv'
    out.to_csv(path, index=False)
    print(f"\n  -> {path.name}  ({len(out)} rows, {len(out.columns)} columns)")


# ==============================================================================
# Main
# ==============================================================================

def run_all(asset_class='commodities'):
    print(f"\n{SEP}")
    print(f"PEAK ANALYSIS -- {DATA_CONFIGS[asset_class]['label'].upper()}")
    print(f"{SEP}")

    trades = collect_all_trades(asset_class)
    if not trades:
        print("No trades collected -- check data and pair list.")
        return

    print_baseline_summary(trades, asset_class)

    print("\nAugmenting trades with peak signals...")
    trades = [augment_with_peak_signals(t) for t in trades]
    print(f"  Done -- {len(trades)} trades augmented.")

    section1_peak_distribution(trades, asset_class)
    section2_by_velocity_quintile(trades, asset_class)
    section3_by_outcome(trades, asset_class)
    section4_pre_peak_indicators(trades, asset_class)
    section5_per_pair(trades, asset_class)
    section6_decision_gates(trades, asset_class)
    save_full_trades(trades, asset_class)


if __name__ == '__main__':
    run_all(asset_class='commodities')
