"""
test_exit_sd_comparison.py — Exit Threshold Sensitivity Study (v2)
==================================================================

Re-runs the Q1 (equity), Q5 (FX), Q6 (commodity), and Q7 (fixed income)
backtests at exit_sd = 0.0, 0.5, and 1.0 to produce a like-for-like
comparison with the original research findings.

BUGS FIXED vs v1
----------------
1. DATA TRUNCATION: CEI instrument only starts 2007-03-30. Using
   dropna(how='any') silently cut 8 years of history for all instruments.
   Fix: run 11 full-history instruments from 1999 + separate 12-instrument
   run from 2007 so both are visible for comparison.

2. 3v3 FINANCING MULTIPLIER: v1 multiplied financing by n_legs (=6 for 3v3),
   giving 8.88% total financing vs Q1 paper's 1.47%. The correct model
   applies the net annual rate ONCE per spread position (n_legs=1).
   Economic rationale: vol-scaled basket return is the AVERAGE of n_leg
   returns, so the effective notional is per-instrument, not per-basket.
   Confirmed: Q1 3v3 at n_legs=1: 4.00%/365 * 135d = 1.48% (paper: 1.47%) OK
              Q5 FX 1v1 at n_legs=1: 3.60%/365 * 189d = 1.86% (paper: 1.89%) OK

3. FI YIELD SERIES: fi_prices.csv contains UST10Y, UST30Y, UST5Y which are
   yield series (values 0.5-6.8%), not price series. Including them corrupts
   spread return calculations. Fix: exclude them; use ETF price series only
   (matching Q7 which used 11 ETFs and excluded IBTM too).

METHODOLOGY (matching Q1/Q5/Q6/Q7 papers)
------------------------------------------
- All directional 1v1 pairs (i->j and j->i counted separately)
- Entry: +/-2.0 SD, Vol window: 262 days
- Financing: net_rate / 365 * holding_days (ONCE per spread, not * n_legs)
  Equity: (4.88% - 0.88%) = 4.00% net
  FX:     (1.80% + 1.80%) = 3.60% gross symmetric (Q5 methodology)
- Spread cost: 0.001 per leg * 2 legs * 2 directions = 0.4% round-trip (1v1)
- 3v3 equity section reproduces Q1 with correct n_legs=1 financing

PAPER BASELINES (exit_sd = 0.0)
--------------------------------
            Q1 equity 3v3   Q5 FX 1v1   Q6 cmd 1v1   Q7 FI 1v1
GWR           72.8%           72.6%        72.6%         70.7%
Gross exp     +0.36%          +0.49%       +4.83%        +0.39%
Net exp       -1.26%          -1.47%       +3.40%        -1.06%
Avg hold      135d            189d         121d          n/a
BE rate       ~0.5%           ~0.5%        ~3.0%         ~1.8%
Total cost    1.61%           1.96%        1.43%         n/a

RUN INSTRUCTIONS
----------------
cd to project root, then:
    C:\\Users\\gordo\\AppData\\Local\\Python\\bin\\python.exe test_exit_sd_comparison.py

OUTPUTS
-------
  exit_sd_comparison_summary.csv   -- aggregate table per asset/exit
  exit_sd_comparison_results.csv   -- per-pair detail
  exit_sd_comparison_report.txt    -- formatted report

EXPECTED RUNTIME: ~2-5 minutes (3v3 equity sample dominates)
"""

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).parent
_CACHE   = _PROJECT / 'cache'

# Code files live in project root; price CSVs live in cache/
def _csv(filename: str) -> Path:
    """Resolve a price CSV — cache/ first, project root as fallback."""
    p = _CACHE / filename
    if p.exists():
        return p
    p = _PROJECT / filename
    if p.exists():
        return p
    raise FileNotFoundError(
        f"{filename} not found.\n"
        f"  Looked in: {_CACHE}\n"
        f"         and: {_PROJECT}\n"
        "  prices.csv should be in the cache/ subdirectory."
    )

for _f in ['backtest.py', 'numba_core.py']:
    if not (_PROJECT / _f).exists():
        sys.exit(f"ERROR: {_f} not found in {_PROJECT}\nRun from the project root directory.")
try:
    _csv('prices.csv')
except FileNotFoundError as _e:
    sys.exit(f"ERROR: {_e}")

sys.path.insert(0, str(_PROJECT))

from asset_configs import get_spread_cost_lookup, basket_spread_cost
from numba_core import (
    batch_backtest, backtest_spread,
    BR_N_TRADES, BR_GROSS_WR, BR_AVG_GROSS, BR_AVG_HOLDING,
    COL_GROSS_RETURN, COL_HOLDING_DAYS,
)
from backtest import load_asset_prices, REGIMES

VOL_WINDOW   = 262
XING_SD      = 2.0
EXIT_TARGETS = [0.0, 0.5, 1.0]

# Financing rates (applied ONCE per spread position - not multiplied by n_legs)
FIN_ANNUAL = {
    'equity':    0.0488 - 0.0088,   # 4.00% net
    'fx':        0.018  + 0.018,    # 3.60% symmetric (Q5: both sides pay ~1.8%)
    'commodity': 0.0488 - 0.0088,   # 4.00% net
    'fi':        0.0488 - 0.0088,   # 4.00% net
}
# FI columns to exclude (yield series and IBTM - matches Q7 methodology)
FI_EXCLUDE = {'UST10Y', 'UST30Y', 'UST5Y', 'IBTM'}

PAPER = {
    'equity_3v3': {'gross_wr': 0.728, 'avg_gross': 0.00355, 'avg_net': -0.01258,
                   'avg_hold': 134.7, 'avg_cost': 0.01614, 'be_rate': 0.5},
    'fx':         {'gross_wr': 0.726, 'avg_gross': 0.0049,  'avg_net': -0.0147,
                   'avg_hold': 189.0, 'avg_cost': 0.0196,  'be_rate': 0.5},
    'commodity':  {'gross_wr': 0.726, 'avg_gross': 0.0483,  'avg_net':  0.0340,
                   'avg_hold': 121.0, 'avg_cost': 0.0143,  'be_rate': 3.0},
    'fi':         {'gross_wr': 0.707, 'avg_gross': 0.0039,  'avg_net': -0.0106,
                   'avg_hold': None,  'avg_cost': None,    'be_rate': 1.8},
}


def load_clean(filepath, exclude=None, start='1999-01-01', drop_any=True):
    """Load and vol-scale prices. drop_any=False fills NaN with 0 to preserve history."""
    prices, instruments = load_asset_prices(str(filepath), start_date=start)
    if exclude:
        instruments = [i for i in instruments if i not in exclude]
        prices = prices[instruments]
    rets = prices.pct_change().dropna(how='all')
    vols = rets.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std()
    scalings = (0.01 / vols.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    scaled_df = (rets * scalings)
    if drop_any:
        scaled_df = scaled_df.dropna(how='any')
    else:
        scaled_df = scaled_df.dropna(how='all').fillna(0.0)
    index = scaled_df.index
    day_ints = ((index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).values.astype(np.int64)
    return scaled_df.values.astype(np.float64), day_ints, index, instruments


def run_all_1v1(scaled, day_ints, instruments, exit_sd, fin_annual, spread_lookup=None):
    """All N*(N-1) directional pairs. Financing applied once per spread."""
    N = len(instruments)
    fin_daily = fin_annual / 365
    all_g, all_n, all_h, all_sc, recs = [], [], [], [], []

    for i in range(N):
        others = [j for j in range(N) if j != i]
        spread_mat = scaled[:, i:i+1] - scaled[:, others]
        results = batch_backtest(spread_mat, VOL_WINDOW, XING_SD, exit_sd, day_ints)
        for k, j in enumerate(others):
            br = results[k]
            nt = int(br[BR_N_TRADES])
            if nt == 0:
                continue
            gross = float(br[BR_AVG_GROSS])
            hold  = float(br[BR_AVG_HOLDING])
            gwr   = float(br[BR_GROSS_WR])
            fin   = fin_daily * hold               # once per spread, not * n_legs
            if spread_lookup is not None:
                pair_cost = basket_spread_cost((i,), (j,), instruments, spread_lookup)
            else:
                pair_cost = 0.001 * 2 * 2          # flat fallback: 0.1% per leg, round-trip
            net   = gross - pair_cost - fin
            recs.append({'long': instruments[i], 'short': instruments[j],
                         'exit_sd': exit_sd, 'n_trades': nt, 'gross_wr': gwr,
                         'avg_gross': gross, 'avg_net': net, 'avg_holding': hold,
                         'spread_cost': pair_cost, 'avg_fin_cost': fin,
                         'avg_total_cost': pair_cost + fin})
            all_g.extend([gross]     * nt)
            all_n.extend([net]       * nt)
            all_h.extend([hold]      * nt)
            all_sc.extend([pair_cost] * nt)

    if not all_g:
        return {}, []
    g, n, h, sc = np.array(all_g), np.array(all_n), np.array(all_h), np.array(all_sc)
    fin_daily_arr = fin_daily * h
    agg = {
        'n_pairs': len(recs), 'n_trades': len(g),
        'gross_wr': float((g > 0).mean()), 'net_wr': float((n > 0).mean()),
        'avg_gross': float(g.mean()), 'avg_net': float(n.mean()),
        'avg_hold': float(h.mean()), 'med_hold': float(np.median(h)),
        'avg_cost': float((sc + fin_daily_arr).mean()),
        'n_net_pos': sum(1 for r in recs if r['avg_net'] > 0),
        'pct_net_pos': sum(1 for r in recs if r['avg_net'] > 0) / max(len(recs), 1),
    }
    return agg, recs


def compute_be_rate(scaled, day_ints, instruments, exit_sd, spread_lookup=None):
    N = len(instruments)
    all_g, all_h, all_cost = [], [], []
    for i in range(N):
        others = [j for j in range(N) if j != i]
        spread_mat = scaled[:, i:i+1] - scaled[:, others]
        results = batch_backtest(spread_mat, VOL_WINDOW, XING_SD, exit_sd, day_ints)
        for k, j in enumerate(others):
            br = results[k]
            nt = int(br[BR_N_TRADES])
            if nt == 0:
                continue
            if spread_lookup is not None:
                pair_cost = basket_spread_cost((i,), (j,), instruments, spread_lookup)
            else:
                pair_cost = 0.001 * 2 * 2
            all_g.extend([float(br[BR_AVG_GROSS])]   * nt)
            all_h.extend([float(br[BR_AVG_HOLDING])] * nt)
            all_cost.extend([pair_cost]               * nt)
    if not all_g:
        return 0.0
    g, h, cost = np.array(all_g), np.array(all_h), np.array(all_cost)
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        avg_net = float((g - cost - (mid / 365) * h).mean())
        lo, hi = (mid, hi) if avg_net > 0 else (lo, mid)
        if hi - lo < 0.01:
            break
    return round((lo + hi) / 2, 2)


def print_table(results_by_exit, paper_bl=None):
    hdr = f"  {'Metric':<28} {'Exit 0.0':>12} {'Exit 0.5':>12} {'Exit 1.0':>12}"
    if paper_bl:
        hdr += f"  {'Paper (0.0)':>12}"
    print(hdr)
    print("  " + "-" * (68 if paper_bl else 56))

    ROWS = [
        ("Gross Win Rate",  'gross_wr',  100, '.1f',  '%', 'gross_wr',  100),
        ("Net Win Rate",    'net_wr',    100, '.1f',  '%', None,        100),
        ("Avg Gross/Trade", 'avg_gross', 100, '+.3f', '%', 'avg_gross', 100),
        ("Avg Net/Trade",   'avg_net',   100, '+.3f', '%', 'avg_net',   100),
        ("Avg Holding",     'avg_hold',    1, '.0f',  'd', 'avg_hold',    1),
        ("Median Holding",  'med_hold',    1, '.0f',  'd', None,          1),
        ("Avg Total Cost",  'avg_cost',  100, '.3f',  '%', 'avg_cost',  100),
        ("Total Trades",    'n_trades',    1, ',',    '',  None,          1),
        ("Net+ Pairs",      'n_net_pos',   1, 'd',    '',  None,          1),
        ("% Net+ Pairs",    'pct_net_pos', 100, '.1f', '%', None,        100),
    ]
    for label, key, scale, fmt, unit, pk, ps in ROWS:
        vals = []
        for ex in EXIT_TARGETS:
            v = results_by_exit.get(ex, {}).get(key)
            vals.append(f"{v*scale:{fmt}}{unit}" if v is not None else "n/a")
        line = f"  {label:<28} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}"
        if paper_bl and pk:
            pv = paper_bl.get(pk)
            line += f"  {pv*ps:{fmt}}{unit}" if pv is not None else "  n/a"
        print(line)

    be_vals = [f"{results_by_exit.get(ex, {}).get('be_rate', 0):.2f}%" for ex in EXIT_TARGETS]
    be_line = f"  {'Break-even Fin. Rate':<28} {be_vals[0]:>12} {be_vals[1]:>12} {be_vals[2]:>12}"
    if paper_bl:
        be_line += f"  {paper_bl.get('be_rate', 0):.1f}%"
    print(be_line)

    print()
    print("  Impact of 1.0 SD exit vs 0.0 SD exit:")
    for label, key, scale, unit in [
        ("Net expectancy",  'avg_net',     100, 'pp'),
        ("Avg holding",     'avg_hold',      1, 'd'),
        ("Gross win rate",  'gross_wr',    100, 'pp'),
        ("Net win rate",    'net_wr',      100, 'pp'),
        ("% Net+ pairs",    'pct_net_pos', 100, 'pp'),
    ]:
        v0 = results_by_exit.get(0.0, {}).get(key)
        v1 = results_by_exit.get(1.0, {}).get(key)
        if v0 is not None and v1 is not None:
            print(f"    {label:<26} {(v1-v0)*scale:+.2f}{unit}")


def main():
    t_start = time.time()
    all_summary, all_pairs, report = [], [], []

    def log(s=""):
        print(s)
        report.append(s)

    log("=" * 72)
    log("  EXIT THRESHOLD COMPARISON v2 -- Q1/Q5/Q6/Q7 REPLICATION")
    log(f"  Entry: +/-{XING_SD} SD  | Vol: {VOL_WINDOW}d  | Exits: {EXIT_TARGETS}")
    log("  Fixes: CEI truncation, n_legs financing, FI yield series")
    log("=" * 72)

    ASSETS = [
        # label            csv                     exclude      start         drop_any
        ('equity_11_1999', 'prices.csv',           {'CEI'},     '1999-01-01', False),
        ('equity_12_2007', 'prices.csv',           set(),       '1999-01-01', True ),
        ('fx',             'fx_prices.csv',        set(),       '1999-01-01', True ),
        ('commodity',      'commodity_prices.csv', set(),       '1999-01-01', True ),
        ('fi',             'fi_prices.csv',        FI_EXCLUDE,  '1999-01-01', True ),
    ]

    for label, csv_name, excl, start, drop_any in ASSETS:
        try:
            csv_path = _csv(csv_name)
        except FileNotFoundError as e:
            log(f"\n  SKIP {label}: {e}")
            continue

        asset_key  = label.split('_')[0]
        paper_bl   = PAPER.get(asset_key)
        fin_annual = FIN_ANNUAL.get(asset_key, 0.04)

        log()
        log("-" * 72)
        log(f"  {label.upper()}")
        log("-" * 72)
        print(f"  Loading {csv_name}...", end="", flush=True)

        scaled, day_ints, idx, instruments = load_clean(csv_path, excl, start, drop_any)
        N = len(instruments)
        print(f" {N} instruments, {len(scaled):,}d ({idx[0].date()} - {idx[-1].date()})")
        log(f"  Instruments ({N}): {', '.join(instruments)}")
        log(f"  Pairs: {N*(N-1)}  |  Financing: {fin_annual*100:.2f}% p.a. (once per spread)")

        _prices_df, _ = load_asset_prices(str(csv_path), start_date=start)
        _latest_px = _prices_df.iloc[-1].to_dict()
        spread_lookup = get_spread_cost_lookup(instruments, _latest_px, asset_key)
        _avg_sc = sum(spread_lookup.values()) / len(spread_lookup)
        log(f"  Avg spread cost: {_avg_sc*100:.4f}% per instrument one-way. "
            f"Estimated 1v1 basket cost: {_avg_sc*4*100:.4f}%")

        results_by_exit = {}
        for exit_sd in EXIT_TARGETS:
            t0 = time.time()
            print(f"  exit_sd={exit_sd}...", end="", flush=True)
            agg, recs = run_all_1v1(scaled, day_ints, instruments, exit_sd, fin_annual,
                                     spread_lookup=spread_lookup)
            be = compute_be_rate(scaled, day_ints, instruments, exit_sd,
                                  spread_lookup=spread_lookup)
            agg['be_rate'] = be
            results_by_exit[exit_sd] = agg
            for r in recs:
                r['asset_class'] = label
                all_pairs.append(r)
            all_summary.append({
                'asset_class': label, 'exit_sd': exit_sd,
                'n_instruments': N, 'n_pairs': agg.get('n_pairs', 0),
                'n_trades': agg.get('n_trades', 0),
                'gross_wr_pct': round(agg.get('gross_wr', 0)*100, 2),
                'net_wr_pct':   round(agg.get('net_wr', 0)*100, 2),
                'avg_gross_pct': round(agg.get('avg_gross', 0)*100, 4),
                'avg_net_pct':   round(agg.get('avg_net', 0)*100, 4),
                'avg_holding_days': round(agg.get('avg_hold', 0), 1),
                'median_holding_days': round(agg.get('med_hold', 0), 1),
                'avg_total_cost_pct': round(agg.get('avg_cost', 0)*100, 4),
                'n_net_pos_pairs': agg.get('n_net_pos', 0),
                'pct_net_pos': round(agg.get('pct_net_pos', 0)*100, 2),
                'breakeven_rate_pct': be,
            })
            print(f" {time.time()-t0:.1f}s | {agg.get('n_trades',0):,} trades"
                  f" | net {agg.get('avg_net',0)*100:+.2f}%"
                  f" | hold {agg.get('avg_hold',0):.0f}d | BE {be:.2f}%")

        print()
        print_table(results_by_exit, paper_bl)

    # -- Equity 3v3 Q1 like-for-like ----------------------------------------
    log()
    log("=" * 72)
    log("  EQUITY 3v3 -- Q1 LIKE-FOR-LIKE (2,000 baskets, seed=42)")
    log("  Financing: net_rate/365 * hold (n_legs=1). Spread cost: per-instrument lookup.")
    log("=" * 72)

    csv_eq = _csv('prices.csv')
    for eq_label, eq_excl, eq_drop in [
        ('equity_3v3_1999', {'CEI'}, False),   # 11 instruments, full history
        ('equity_3v3_2007', set(),   True),    # 12 instruments, from 2007
    ]:
        scaled_eq, day_ints_eq, idx_eq, instr_eq = load_clean(
            csv_eq, eq_excl, '1999-01-01', eq_drop
        )
        N_eq = len(instr_eq)
        log()
        log(f"  {eq_label}: {N_eq} instr, {len(scaled_eq):,}d "
            f"({idx_eq[0].date()} - {idx_eq[-1].date()})")

        longs  = list(combinations(range(N_eq), 3))
        shorts = list(combinations(range(N_eq), 3))
        valid  = [(l, s) for l in longs for s in shorts if not set(l) & set(s)]
        rng    = np.random.RandomState(42)
        samp   = [valid[i] for i in sorted(rng.choice(len(valid), min(2000, len(valid)), replace=False))]
        print(f"  {eq_label}: {len(valid):,} pop -> {len(samp)} sample (seed=42)")

        fin_daily_eq = FIN_ANNUAL['equity'] / 365   # once per spread

        _eq_prices_df, _ = load_asset_prices(str(csv_eq), start_date='1999-01-01')
        _eq_latest_px = _eq_prices_df.iloc[-1].to_dict()
        _eq_spread_lookup = get_spread_cost_lookup(instr_eq, _eq_latest_px, 'equity')
        _eq_avg_sc = sum(_eq_spread_lookup.values()) / len(_eq_spread_lookup)
        log(f"  Avg spread cost: {_eq_avg_sc*100:.4f}% per instrument. "
            f"Typical 3v3 basket cost: {_eq_avg_sc*4*100:.4f}%")

        for exit_sd in EXIT_TARGETS:
            t0 = time.time()
            print(f"  exit_sd={exit_sd}...", end="", flush=True)
            all_g, all_n, all_h, all_bc = [], [], [], []
            for lc, sc in samp:
                basket_cost = basket_spread_cost(lc, sc, instr_eq, _eq_spread_lookup)
                spread = scaled_eq[:, list(lc)].mean(1) - scaled_eq[:, list(sc)].mean(1)
                trades, nt, _, _ = backtest_spread(
                    spread, VOL_WINDOW, XING_SD, exit_sd, day_ints_eq
                )
                if nt == 0:
                    continue
                g = trades[:nt, COL_GROSS_RETURN]
                h = trades[:nt, COL_HOLDING_DAYS]
                n = g - basket_cost - fin_daily_eq * h
                all_g.extend(g.tolist())
                all_n.extend(n.tolist())
                all_h.extend(h.tolist())
                all_bc.extend([basket_cost] * nt)

            if all_g:
                g_a, n_a, h_a, bc_a = (np.array(all_g), np.array(all_n),
                                        np.array(all_h), np.array(all_bc))
                cost_a = bc_a + fin_daily_eq * h_a
                log(f"  exit={exit_sd}: GWR={float((g_a>0).mean())*100:.1f}%"
                    f"  NWR={float((n_a>0).mean())*100:.1f}%"
                    f"  Gross={float(g_a.mean())*100:+.3f}%"
                    f"  Net={float(n_a.mean())*100:+.3f}%"
                    f"  Hold={float(h_a.mean()):.0f}d (med {float(np.median(h_a)):.0f}d)"
                    f"  Cost={float(cost_a.mean())*100:.3f}%"
                    f"  N={len(g_a):,}")
                print(f" {time.time()-t0:.1f}s | {len(g_a):,} trades"
                      f" | net {float(n_a.mean())*100:+.2f}%"
                      f" | hold {float(h_a.mean()):.0f}d")
                all_summary.append({
                    'asset_class': eq_label, 'exit_sd': exit_sd,
                    'n_instruments': N_eq, 'n_pairs': len(samp),
                    'n_trades': len(g_a),
                    'gross_wr_pct': round(float((g_a>0).mean())*100, 2),
                    'net_wr_pct':   round(float((n_a>0).mean())*100, 2),
                    'avg_gross_pct': round(float(g_a.mean())*100, 4),
                    'avg_net_pct':   round(float(n_a.mean())*100, 4),
                    'avg_holding_days': round(float(h_a.mean()), 1),
                    'median_holding_days': round(float(np.median(h_a)), 1),
                    'avg_total_cost_pct': round(float(cost_a.mean())*100, 4),
                    'n_net_pos_pairs': None, 'pct_net_pos': None,
                    'breakeven_rate_pct': None,
                })
            else:
                print(" no trades")

    # -- Verification --------------------------------------------------------
    df_s = pd.DataFrame(all_summary)
    print()
    print("=" * 72)
    print("  VERIFICATION vs Q1 PAPER (equity_3v3_1999, exit=0.0)")
    print("=" * 72)
    row = df_s[(df_s['asset_class'] == 'equity_3v3_1999') & (df_s['exit_sd'] == 0.0)]
    if not row.empty:
        r = row.iloc[0]
        p = PAPER['equity_3v3']
        for metric, got, expected, tol in [
            ("Gross WR (%)",  r['gross_wr_pct'],  p['gross_wr']*100,  2.0),
            ("Avg Gross (%)", r['avg_gross_pct'], p['avg_gross']*100, 0.3),
            ("Avg Net (%)",   r['avg_net_pct'],   p['avg_net']*100,   0.5),
            ("Avg Hold (d)",  r['avg_holding_days'], p['avg_hold'],   20.0),
            ("Total Cost (%)", r['avg_total_cost_pct'], p['avg_cost']*100, 0.5),
        ]:
            ok = abs(got - expected) <= tol
            print(f"  {metric:<22} got {got:>8.3f}  paper {expected:>8.3f}  {'OK' if ok else 'MISMATCH'}")

    # -- Cross-asset summary -------------------------------------------------
    print()
    print("=" * 72)
    print("  CROSS-ASSET SUMMARY (exit 0.0 vs 1.0)")
    print("=" * 72)
    print(f"  {'Asset':<24} {'Net(0.0)':>10} {'Net(1.0)':>10} {'Delta':>9}"
          f" {'Hold(0)':>8} {'Hold(1)':>8} {'HoldD':>7} {'BE(0)':>7}")
    print("  " + "-" * 90)
    for asset in df_s['asset_class'].unique():
        sub = df_s[df_s['asset_class'] == asset]
        r0 = sub[sub['exit_sd'] == 0.0]
        r1 = sub[sub['exit_sd'] == 1.0]
        if r0.empty or r1.empty:
            continue
        r0, r1 = r0.iloc[0], r1.iloc[0]
        dn = r1['avg_net_pct'] - r0['avg_net_pct']
        dh = r1['avg_holding_days'] - r0['avg_holding_days']
        be = r0.get('breakeven_rate_pct')
        be_s = f"{be:.2f}%" if be is not None else "n/a"
        print(f"  {asset:<24} {r0['avg_net_pct']:>+9.3f}% {r1['avg_net_pct']:>+9.3f}%"
              f" {dn:>+8.3f}pp {r0['avg_holding_days']:>7.0f}d {r1['avg_holding_days']:>7.0f}d"
              f" {dh:>+6.0f}d {be_s:>7}")

    # -- Save ----------------------------------------------------------------
    print()
    pd.DataFrame(all_summary).to_csv(_PROJECT / 'exit_sd_comparison_summary.csv', index=False)
    pd.DataFrame(all_pairs).to_csv(_PROJECT / 'exit_sd_comparison_results.csv', index=False)
    (_PROJECT / 'exit_sd_comparison_report.txt').write_text('\n'.join(report), encoding='utf-8')
    print(f"  Saved 3 output files. Total runtime: {time.time()-t_start:.1f}s")
    print()
    print("  KEY CHECKS:")
    print("  - equity_3v3_1999 exit=0.0 should match Q1: GWR~72.8%, Net~-1.26%")
    print("  - GWR should be STABLE across exit thresholds (signal property, not exit)")
    print("  - Avg holding should DROP ~50-60% at exit=1.0 vs exit=0.0")
    print("  - Net expectancy should IMPROVE at exit=1.0 (may go positive for commodities)")
    print("  - Break-even rate should RISE as exit_sd increases")


if __name__ == '__main__':
    main()