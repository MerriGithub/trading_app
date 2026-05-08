import sys
import time
from itertools import combinations
from pathlib import Path
 
import numpy as np
import pandas as pd
 
# ── Verify we're in the right directory ────────────────────────────────────
_HERE = Path(__file__).parent
_PROJECT = _HERE  # script lives in project root; adjust if needed
for _f in ['backtest.py', 'numba_core.py']:
    if not (_PROJECT / _f).exists():
        sys.exit(
            f"ERROR: {_f} not found in {_PROJECT}\n"
            "Please run this script from the project root directory."
        )
for _f in ['prices.csv', 'fx_prices.csv', 'commodity_prices.csv', 'fi_prices.csv']:
    if not (_PROJECT / 'cache' / _f).exists():
        sys.exit(
            f"ERROR: cache/{_f} not found in {_PROJECT}\n"
            "Please ensure price files are in the cache/ directory."
        )
 
sys.path.insert(0, str(_PROJECT))
 
from backtest import (
    load_asset_prices, prepare_returns, aggregate_trades,
    regime_split, find_breakeven_financing, REGIMES,
)
from numba_core import batch_backtest, BR_N_TRADES, BR_GROSS_WR, BR_AVG_GROSS, BR_AVG_HOLDING
 
# ── Constants (matching all research papers) ─────────────────────────────
VOL_WINDOW      = 262
XING_SD         = 2.0
LONG_FIN_RATE   = 0.0488       # 4.88% p.a.
SHORT_REBATE    = 0.0088       # 0.88% p.a.
NET_FIN_DAILY   = (LONG_FIN_RATE - SHORT_REBATE) / 365   # ~0.01096 bps/day
SPREAD_COST_PCT = 0.001        # 0.1% per leg (round-trip = 0.2% for 1v1)
 
EXIT_THRESHOLDS = [0.0, 0.5, 1.0]   # The three conditions under test
 
# Q1/Q5/Q6/Q7 baseline values for comparison (from research papers)
PAPER_BASELINES = {
    'equity_1v1': None,       # not in papers — will be computed fresh
    'equity_3v3': {           # Q1 paper (3v3, exit=0.0, N=280,913 trades)
        'gross_wr': 0.728,  'avg_gross': 0.003551, 'avg_net': -0.012584,
        'avg_holding': 134.7, 'avg_total_cost': 0.016136, 'be_rate': 0.5,
    },
    'fx': {                   # Q5 paper (1v1, exit=0.0, N=1,474 trades)
        'gross_wr': 0.726,  'avg_gross': 0.0049, 'avg_net': -0.0147,
        'avg_holding': 189.0, 'avg_total_cost': 0.0196, 'be_rate': 0.5,
    },
    'commodity': {            # Q6 paper (1v1, exit=0.0, N=3,320 trades)
        'gross_wr': 0.726,  'avg_gross': 0.0483, 'avg_net': 0.0340,
        'avg_holding': 121.0, 'avg_total_cost': 0.0143, 'be_rate': 3.0,
    },
    'fi': {                   # Q7 paper (1v1, exit=0.0, n/a)
        'gross_wr': 0.707,  'avg_gross': 0.0039, 'avg_net': -0.0106,
        'avg_holding': None,  'avg_total_cost': None, 'be_rate': 1.8,
    },
}
 
# Price file paths (relative to project root)
ASSET_FILES = {
    'equity':    _PROJECT / 'cache' / 'prices.csv',
    'fx':        _PROJECT / 'cache' / 'fx_prices.csv',
    'commodity': _PROJECT / 'cache' / 'commodity_prices.csv',
    'fi':        _PROJECT / 'cache' / 'fi_prices.csv',
}
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════════════
 
def run_all_1v1_pairs(
    scaled: np.ndarray,
    day_ints: np.ndarray,
    instruments: list[str],
    exit_sd: float,
    n_legs: int = 2,
) -> dict:
    """
    Run batch backtest on all N*(N-1) directional 1v1 pairs.
 
    Returns aggregate dict pooling all trades across all pairs.
    Also returns per-pair records for detailed analysis.
    """
    N = len(instruments)
    pair_records = []
    all_gross = []
    all_net   = []
    all_hold  = []
 
    # Build all directional spread matrices in one vectorised pass per long leg
    for i in range(N):
        # Long instrument i vs all others
        others = [j for j in range(N) if j != i]
        if not others:
            continue
 
        spread_mat = scaled[:, i:i+1] - scaled[:, others]  # (T, N-1)
        results = batch_backtest(spread_mat, VOL_WINDOW, XING_SD, exit_sd, day_ints)
 
        for k, j in enumerate(others):
            br = results[k]
            n_trades = int(br[BR_N_TRADES])
            if n_trades == 0:
                continue
 
            avg_gross   = float(br[BR_AVG_GROSS])
            avg_holding = float(br[BR_AVG_HOLDING])
            gross_wr    = float(br[BR_GROSS_WR])
 
            spread_cost = SPREAD_COST_PCT * n_legs * 2          # round-trip
            avg_fin_cost = NET_FIN_DAILY * n_legs * avg_holding
            avg_net     = avg_gross - spread_cost - avg_fin_cost
 
            pair_records.append({
                'long':          instruments[i],
                'short':         instruments[j],
                'exit_sd':       exit_sd,
                'n_trades':      n_trades,
                'gross_wr':      gross_wr,
                'avg_gross':     avg_gross,
                'avg_net':       avg_net,
                'avg_holding':   avg_holding,
                'spread_cost':   spread_cost,
                'avg_fin_cost':  avg_fin_cost,
                'avg_total_cost': spread_cost + avg_fin_cost,
            })
 
            # Weight-by-trade-count to compute pooled aggregate
            all_gross.extend([avg_gross] * n_trades)
            all_net.extend([avg_net]   * n_trades)
            all_hold.extend([avg_holding] * n_trades)
 
    if not all_gross:
        return {'n_trades': 0, 'pair_records': [], 'agg': {}}
 
    g = np.array(all_gross)
    n = np.array(all_net)
    h = np.array(all_hold)
 
    agg = {
        'n_pairs':      len(pair_records),
        'n_trades':     len(g),
        'gross_wr':     float((g > 0).mean()),
        'net_wr':       float((n > 0).mean()),
        'avg_gross':    float(g.mean()),
        'avg_net':      float(n.mean()),
        'avg_holding':  float(h.mean()),
        'median_holding': float(np.median(h)),
        'avg_total_cost': float((h * NET_FIN_DAILY * n_legs + SPREAD_COST_PCT * n_legs * 2).mean()),
        'n_net_positive_pairs': sum(1 for r in pair_records if r['avg_net'] > 0),
        'pct_net_positive':     sum(1 for r in pair_records if r['avg_net'] > 0) / max(len(pair_records), 1),
    }
 
    return {'agg': agg, 'pair_records': pair_records}
 
 
def compute_breakeven_rate(
    scaled: np.ndarray,
    day_ints: np.ndarray,
    exit_sd: float,
    n_legs: int = 2,
) -> float:
    """
    Compute aggregate break-even financing rate across all 1v1 pairs
    by binary search on the pooled average net expectancy.
    """
    N = scaled.shape[1]
    all_gross = []
    all_hold  = []
 
    for i in range(N):
        others = [j for j in range(N) if j != i]
        spread_mat = scaled[:, i:i+1] - scaled[:, others]
        results = batch_backtest(spread_mat, VOL_WINDOW, XING_SD, exit_sd, day_ints)
 
        for k in range(len(others)):
            br = results[k]
            nt = int(br[BR_N_TRADES])
            if nt == 0:
                continue
            all_gross.extend([float(br[BR_AVG_GROSS])] * nt)
            all_hold.extend([float(br[BR_AVG_HOLDING])] * nt)
 
    if not all_gross:
        return -1.0
 
    g = np.array(all_gross)
    h = np.array(all_hold)
    spread_cost = SPREAD_COST_PCT * n_legs * 2
 
    # Binary search for annual rate where avg_net = 0
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        fin_daily = mid / 100 / 365
        avg_net = float((g - spread_cost - fin_daily * n_legs * h).mean())
        if avg_net > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.01:
            break
    return round((lo + hi) / 2, 2)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════
 
def pct(v):
    return f"{v * 100:+.2f}%" if v is not None else "n/a"
 
def days(v):
    return f"{v:.0f}d" if v is not None else "n/a"
 
def fmt_delta(new, baseline, scale=100):
    """Format change vs baseline with sign and absolute."""
    if baseline is None:
        return "(no baseline)"
    delta = (new - baseline) * scale
    return f"{'+' if delta >= 0 else ''}{delta:.2f}pp"
 
 
def print_section(title: str):
    print()
    print("═" * 72)
    print(f"  {title}")
    print("═" * 72)
 
 
def print_comparison_table(
    asset_label: str,
    results_by_exit: dict,
    paper_baseline: dict | None,
    baseline_label: str = "Paper (0.0 exit)",
):
    """Print the core comparison table for one asset class."""
 
    header = f"{'Metric':<28} {'Exit 0.0':>12} {'Exit 0.5':>12} {'Exit 1.0':>12}"
    if paper_baseline:
        header += f"  {'Paper baseline':>16}"
    print(header)
    print("-" * (72 if paper_baseline else 72))
 
    def row(label, key, formatter, paper_key=None, paper_formatter=None):
        vals = []
        for ex in EXIT_THRESHOLDS:
            agg = results_by_exit.get(ex, {}).get('agg', {})
            v = agg.get(key)
            vals.append(formatter(v) if v is not None else "n/a")
        line = f"  {label:<26} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}"
        if paper_baseline and paper_key:
            pv = paper_baseline.get(paper_key)
            pf = paper_formatter or formatter
            line += f"  {pf(pv):>16}"
        print(line)
 
    row("Gross Win Rate",     'gross_wr',    lambda v: f"{v*100:.1f}%",    'gross_wr',    lambda v: f"{v*100:.1f}%")
    row("Net Win Rate",       'net_wr',      lambda v: f"{v*100:.1f}%")
    row("Avg Gross/Trade",    'avg_gross',   lambda v: f"{v*100:+.3f}%",   'avg_gross',   lambda v: f"{v*100:+.3f}%")
    row("Avg Net/Trade",      'avg_net',     lambda v: f"{v*100:+.3f}%",   'avg_net',     lambda v: f"{v*100:+.3f}%")
    row("Avg Holding",        'avg_holding', lambda v: f"{v:.0f}d",        'avg_holding', lambda v: f"{v:.0f}d" if v else "n/a")
    row("Median Holding",     'median_holding', lambda v: f"{v:.0f}d")
    row("Avg Total Cost",     'avg_total_cost', lambda v: f"{v*100:.3f}%", 'avg_total_cost', lambda v: f"{v*100:.3f}%" if v else "n/a")
    row("Total Trades",       'n_trades',    lambda v: f"{int(v):,}")
    row("Net+ Pairs",         'n_net_positive_pairs', lambda v: f"{int(v)}")
    row("% Net+ Pairs",       'pct_net_positive', lambda v: f"{v*100:.1f}%")
 
    # Break-even rows (pre-computed)
    be_vals = []
    for ex in EXIT_THRESHOLDS:
        be = results_by_exit.get(ex, {}).get('breakeven_rate', -1.0)
        be_vals.append(f"{be:.2f}%" if be > 0 else ">10%" if be == -1 else "n/a")
    be_line = f"  {'Break-even Fin. Rate':<26} {be_vals[0]:>12} {be_vals[1]:>12} {be_vals[2]:>12}"
    if paper_baseline:
        pbe = paper_baseline.get('be_rate')
        be_line += f"  {f'{pbe:.1f}%' if pbe else 'n/a':>16}"
    print(be_line)
 
    # Delta rows (exit 1.0 vs exit 0.0)
    print()
    print(f"  Impact of 1.0 SD exit vs 0.0 SD exit:")
    agg_00 = results_by_exit.get(0.0, {}).get('agg', {})
    agg_10 = results_by_exit.get(1.0, {}).get('agg', {})
    for label, key, scale, unit in [
        ("Net expectancy",    'avg_net',     100, 'pp'),
        ("Avg holding",       'avg_holding',   1, 'd'),
        ("Gross win rate",    'gross_wr',    100, 'pp'),
        ("Net win rate",      'net_wr',      100, 'pp'),
        ("% Net+ pairs",      'pct_net_positive', 100, 'pp'),
    ]:
        v00 = agg_00.get(key)
        v10 = agg_10.get(key)
        if v00 is not None and v10 is not None:
            delta = (v10 - v00) * scale
            sign = '+' if delta >= 0 else ''
            print(f"    {label:<26} {sign}{delta:.2f}{unit}")
 
 
def print_regime_table(regime_rows_by_exit: dict):
    """Print regime breakdown comparing exit 0.0 vs 1.0."""
    regimes_seen = set()
    for rows in regime_rows_by_exit.values():
        for r in rows:
            regimes_seen.add(r['regime'])
 
    if not regimes_seen:
        print("  (no regime data)")
        return
 
    print(f"  {'Regime':<14} {'Exit 0.0 Net':>14} {'Exit 0.5 Net':>14} {'Exit 1.0 Net':>14} {'Exit 0.0 Hold':>14} {'Exit 1.0 Hold':>14}")
    print("  " + "-" * 86)
 
    for name, y0, y1 in REGIMES:
        if name not in regimes_seen:
            continue
        row_vals = {}
        for ex in EXIT_THRESHOLDS:
            matched = [r for r in regime_rows_by_exit.get(ex, []) if r['regime'] == name]
            row_vals[ex] = matched[0] if matched else None
 
        nets  = [f"{row_vals[ex]['avg_net']*100:+.2f}%" if row_vals[ex] else "n/a" for ex in EXIT_THRESHOLDS]
        holds = [f"{row_vals[ex]['avg_holding']:.0f}d"  if row_vals[ex] else "n/a" for ex in [0.0, 1.0]]
        print(f"  {name:<14} {nets[0]:>14} {nets[1]:>14} {nets[2]:>14} {holds[0]:>14} {holds[1]:>14}")
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
 
def main():
    t_start = time.time()
    all_summary_rows = []
    report_lines = []
 
    def log(line=""):
        print(line)
        report_lines.append(line)
 
    log("=" * 72)
    log("  EXIT THRESHOLD COMPARISON — Q1/Q5/Q6/Q7 REPLICATION STUDY")
    log(f"  Entry: ±{XING_SD} SD  |  Vol window: {VOL_WINDOW}d  |  Exit tested: {EXIT_THRESHOLDS}")
    log(f"  Financing: {LONG_FIN_RATE*100:.2f}% long / {SHORT_REBATE*100:.2f}% short")
    log(f"  Spread cost: {SPREAD_COST_PCT*100:.2f}% per leg")
    log("=" * 72)
 
    all_pair_records = []
 
    for asset_label, csv_path in ASSET_FILES.items():
        if not csv_path.exists():
            log(f"\n  SKIP {asset_label}: {csv_path} not found")
            continue
 
        paper_key = asset_label
        paper_baseline = PAPER_BASELINES.get(paper_key)
 
        log()
        log(f"{'─'*72}")
        log(f"  ASSET CLASS: {asset_label.upper()}")
        log(f"{'─'*72}")
        print(f"  Loading {csv_path.name}...", end="", flush=True)
 
        prices, instruments = load_asset_prices(str(csv_path))
        scaled, day_ints, idx = prepare_returns(prices, instruments, VOL_WINDOW)
        N = len(instruments)
        T = len(scaled)
 
        print(f" {N} instruments, {T} trading days ({idx[0].date()} – {idx[-1].date()})")
        log(f"  Instruments ({N}): {', '.join(instruments)}")
        log(f"  Directional 1v1 pairs: {N*(N-1)}")
        log()
 
        results_by_exit = {}
        regime_rows_by_exit = {}
 
        for exit_sd in EXIT_THRESHOLDS:
            t0 = time.time()
            print(f"  Running exit_sd={exit_sd:.1f}...", end="", flush=True)
 
            result = run_all_1v1_pairs(scaled, day_ints, instruments, exit_sd)
            agg = result.get('agg', {})
 
            # Compute break-even rate (fast enough for 1v1)
            be_rate = compute_breakeven_rate(scaled, day_ints, exit_sd)
            result['breakeven_rate'] = be_rate
 
            # Regime split: use aggregate spread (equal-weight mean across all pairs)
            # Build a single representative spread (mean of all 1v1 spread returns)
            all_spreads = []
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    all_spreads.append(scaled[:, i] - scaled[:, j])
            if all_spreads:
                mean_spread = np.mean(all_spreads, axis=0)
                from numba_core import backtest_spread
                trades_r, n_trades_r, _, _ = backtest_spread(
                    mean_spread, VOL_WINDOW, XING_SD, exit_sd, day_ints
                )
                fin_daily = NET_FIN_DAILY * 2   # 2 legs for 1v1
                regime_rows = regime_split(trades_r, n_trades_r, idx, SPREAD_COST_PCT, fin_daily, 2)
            else:
                regime_rows = []
 
            results_by_exit[exit_sd] = result
            regime_rows_by_exit[exit_sd] = regime_rows
 
            elapsed = time.time() - t0
            n_t = agg.get('n_trades', 0)
            net = agg.get('avg_net', 0.0)
            hold = agg.get('avg_holding', 0.0)
            print(f" done ({elapsed:.1f}s) | {n_t:,} trades | net {net*100:+.2f}% | avg hold {hold:.0f}d")
 
            # Collect per-pair records with exit_sd label
            for rec in result.get('pair_records', []):
                rec['asset_class'] = asset_label
                all_pair_records.append(rec)
 
            # Summary row for CSV
            all_summary_rows.append({
                'asset_class':          asset_label,
                'exit_sd':              exit_sd,
                'n_instruments':        N,
                'n_pairs':              agg.get('n_pairs', 0),
                'n_trades':             agg.get('n_trades', 0),
                'gross_wr':             agg.get('gross_wr', 0.0),
                'net_wr':               agg.get('net_wr', 0.0),
                'avg_gross_pct':        round(agg.get('avg_gross', 0.0) * 100, 4),
                'avg_net_pct':          round(agg.get('avg_net', 0.0) * 100, 4),
                'avg_holding_days':     round(agg.get('avg_holding', 0.0), 1),
                'median_holding_days':  round(agg.get('median_holding', 0.0), 1),
                'avg_total_cost_pct':   round(agg.get('avg_total_cost', 0.0) * 100, 4),
                'n_net_positive_pairs': agg.get('n_net_positive_pairs', 0),
                'pct_net_positive':     round(agg.get('pct_net_positive', 0.0) * 100, 2),
                'breakeven_rate_pct':   be_rate,
            })
 
        # Print formatted comparison table for this asset class
        print()
        print_comparison_table(asset_label, results_by_exit, paper_baseline)
        print()
        print(f"  Regime breakdown (mean of all 1v1 spread — indicative):")
        print_regime_table(regime_rows_by_exit)
 
    # ── 3v3 Equity Sample (reproduces Q1 methodology) ────────────────────
    equity_path = ASSET_FILES['equity']
    if equity_path.exists():
        print_section("EQUITY 3v3 SAMPLE — Q1 LIKE-FOR-LIKE (n=2,000 baskets)")
        log()
        log("  3v3 non-overlapping basket combinations, sampled N=2,000 (matches Q1 paper).")
        log("  Pooling all trades across all sampled baskets.")
        log()
 
        prices_eq, instruments_eq = load_asset_prices(str(equity_path))
        scaled_eq, day_ints_eq, idx_eq = prepare_returns(prices_eq, instruments_eq, VOL_WINDOW)
        N_eq = len(instruments_eq)
 
        # Build all 3v3 non-overlapping pairs (exhaustive population, then sample)
        long_combos  = list(combinations(range(N_eq), 3))
        short_combos = list(combinations(range(N_eq), 3))
        valid_pairs = [
            (lc, sc) for lc in long_combos for sc in short_combos
            if not set(lc) & set(sc)
        ]
        total_3v3 = len(valid_pairs)
        print(f"  Total 3v3 population: {total_3v3:,} non-overlapping pairs")
 
        rng = np.random.RandomState(42)          # fixed seed for reproducibility
        sample_n = min(2000, total_3v3)
        sampled_indices = rng.choice(total_3v3, sample_n, replace=False)
        sampled_pairs = [valid_pairs[i] for i in sorted(sampled_indices)]
        print(f"  Sample: {sample_n} pairs (seed=42, reproducible)")
        log(f"  3v3 population: {total_3v3:,}  |  Sample: {sample_n}  |  Seed: 42")
 
        for exit_sd in EXIT_THRESHOLDS:
            t0 = time.time()
            print(f"  Running 3v3 exit_sd={exit_sd:.1f}...", end="", flush=True)
 
            all_gross = []
            all_hold  = []
 
            for lc, sc in sampled_pairs:
                # Build spread: avg long basket - avg short basket
                lr = scaled_eq[:, list(lc)].mean(axis=1)
                sr = scaled_eq[:, list(sc)].mean(axis=1)
                spread = lr - sr
 
                from numba_core import backtest_spread
                trades_r, n_trades_r, _, _ = backtest_spread(
                    spread, VOL_WINDOW, XING_SD, exit_sd, day_ints_eq
                )
 
                if n_trades_r == 0:
                    continue
 
                t_arr = trades_r[:n_trades_r]
                g = t_arr[:, 3]    # COL_GROSS_RETURN = 3
                h = t_arr[:, 4]    # COL_HOLDING_DAYS = 4
                all_gross.extend(g.tolist())
                all_hold.extend(h.tolist())
 
            elapsed = time.time() - t0
 
            if all_gross:
                g_arr = np.array(all_gross)
                h_arr = np.array(all_hold)
                n_legs = 6   # 3v3
 
                spread_cost_rt = SPREAD_COST_PCT * n_legs * 2
                fin_costs      = NET_FIN_DAILY * n_legs * h_arr
                net_arr        = g_arr - spread_cost_rt - fin_costs
 
                gross_wr    = float((g_arr > 0).mean())
                net_wr      = float((net_arr > 0).mean())
                avg_gross   = float(g_arr.mean())
                avg_net     = float(net_arr.mean())
                avg_hold    = float(h_arr.mean())
                med_hold    = float(np.median(h_arr))
                avg_cost    = float((spread_cost_rt + fin_costs).mean())
                n_trades    = len(g_arr)
 
                print(f" done ({elapsed:.1f}s) | {n_trades:,} trades | net {avg_net*100:+.2f}% | avg hold {avg_hold:.0f}d")
 
                # Break-even search for 3v3
                lo, hi = 0.0, 20.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    fd = mid / 100 / 365
                    avg_n_be = float((g_arr - spread_cost_rt - fd * n_legs * h_arr).mean())
                    if avg_n_be > 0:
                        lo = mid
                    else:
                        hi = mid
                    if hi - lo < 0.01:
                        break
                be_3v3 = round((lo + hi) / 2, 2)
 
                paper_bl = PAPER_BASELINES['equity_3v3']
                log(f"  exit_sd={exit_sd:.1f}: GWR={gross_wr*100:.1f}%  NWR={net_wr*100:.1f}%  "
                    f"Gross={avg_gross*100:+.3f}%  Net={avg_net*100:+.3f}%  "
                    f"Hold={avg_hold:.0f}d (med {med_hold:.0f}d)  "
                    f"Cost={avg_cost*100:.3f}%  BE={be_3v3:.2f}%  N={n_trades:,}")
 
                all_summary_rows.append({
                    'asset_class':          'equity_3v3',
                    'exit_sd':              exit_sd,
                    'n_instruments':        N_eq,
                    'n_pairs':              sample_n,
                    'n_trades':             n_trades,
                    'gross_wr':             round(gross_wr, 4),
                    'net_wr':               round(net_wr, 4),
                    'avg_gross_pct':        round(avg_gross * 100, 4),
                    'avg_net_pct':          round(avg_net * 100, 4),
                    'avg_holding_days':     round(avg_hold, 1),
                    'median_holding_days':  round(med_hold, 1),
                    'avg_total_cost_pct':   round(avg_cost * 100, 4),
                    'n_net_positive_pairs': None,
                    'pct_net_positive':     None,
                    'breakeven_rate_pct':   be_3v3,
                })
            else:
                print(f" done ({elapsed:.1f}s) | no trades")
 
        # Print Q1 3v3 comparison table
        log()
        log("  3v3 Aggregate Results vs Q1 Baseline:")
        log(f"  {'Metric':<28} {'Exit 0.0':>12} {'Exit 0.5':>12} {'Exit 1.0':>12} {'Q1 Paper':>12}")
        log(f"  {'-'*72}")
        df_eq_3v3 = pd.DataFrame([r for r in all_summary_rows if r['asset_class'] == 'equity_3v3'])
        if not df_eq_3v3.empty:
            for col, label, fmt in [
                ('gross_wr',          'Gross Win Rate',  lambda v: f"{v*100:.1f}%"),
                ('net_wr',            'Net Win Rate',    lambda v: f"{v*100:.1f}%"),
                ('avg_gross_pct',     'Avg Gross/Trade', lambda v: f"{v:+.3f}%"),
                ('avg_net_pct',       'Avg Net/Trade',   lambda v: f"{v:+.3f}%"),
                ('avg_holding_days',  'Avg Holding',     lambda v: f"{v:.0f}d"),
                ('avg_total_cost_pct','Avg Total Cost',  lambda v: f"{v:.3f}%"),
                ('breakeven_rate_pct','Break-even Rate', lambda v: f"{v:.2f}%"),
            ]:
                paper_v = PAPER_BASELINES['equity_3v3'].get(col.replace('_pct', '').replace('_days', ''))
                row_vals = []
                for ex in EXIT_THRESHOLDS:
                    match = df_eq_3v3[df_eq_3v3['exit_sd'] == ex]
                    row_vals.append(fmt(match[col].iloc[0]) if not match.empty else "n/a")
                paper_str = fmt(paper_v) if paper_v is not None else "n/a"
                log(f"  {label:<28} {row_vals[0]:>12} {row_vals[1]:>12} {row_vals[2]:>12} {paper_str:>12}")
 
    # ── Save outputs ─────────────────────────────────────────────────────
    print_section("SAVING OUTPUTS")
 
    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = _PROJECT / 'exit_sd_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary CSV:  {summary_path}")
 
    detail_df = pd.DataFrame(all_pair_records)
    detail_path = _PROJECT / 'exit_sd_comparison_results.csv'
    detail_df.to_csv(detail_path, index=False)
    print(f"  Detail CSV:   {detail_path}")
 
    report_path = _PROJECT / 'exit_sd_comparison_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  Text report:  {report_path}")
 
    # ── Final summary across all asset classes ───────────────────────────
    print_section("CROSS-ASSET SUMMARY — Exit 0.0 vs Exit 1.0 Impact")
    print(f"  {'Asset':>14} {'Config':>8} {'Net (0.0)':>12} {'Net (1.0)':>12} "
          f"{'Delta':>10} {'Hold (0.0)':>11} {'Hold (1.0)':>11} {'Hold Δ':>9} "
          f"{'BE (0.0)':>9} {'BE (1.0)':>9}")
    print("  " + "-" * 110)
 
    for asset, conf in [('equity', '1v1'), ('equity_3v3', '3v3'),
                        ('fx', '1v1'), ('commodity', '1v1'), ('fi', '1v1')]:
        rows = [r for r in all_summary_rows if r['asset_class'] == asset]
        if not rows:
            continue
        r00 = next((r for r in rows if r['exit_sd'] == 0.0), None)
        r10 = next((r for r in rows if r['exit_sd'] == 1.0), None)
        if not r00 or not r10:
            continue
        delta_net  = r10['avg_net_pct'] - r00['avg_net_pct']
        delta_hold = r10['avg_holding_days'] - r00['avg_holding_days']
        be00 = r00.get('breakeven_rate_pct', 0.0)
        be10 = r10.get('breakeven_rate_pct', 0.0)
        print(
            f"  {asset:>14} {conf:>8} "
            f"{r00['avg_net_pct']:>+11.3f}% {r10['avg_net_pct']:>+11.3f}% "
            f"{delta_net:>+9.3f}pp "
            f"{r00['avg_holding_days']:>10.0f}d {r10['avg_holding_days']:>10.0f}d "
            f"{delta_hold:>+8.0f}d "
            f"{be00:>8.2f}% {be10:>8.2f}%"
        )
 
    elapsed_total = time.time() - t_start
    print()
    print(f"  Total runtime: {elapsed_total:.1f}s")
    print()
    print("  KEY VERIFICATION CHECKS:")
    print("  ✓ Gross WR should be stable across exit thresholds (~72% for all classes)")
    print("  ✓ Average holding should fall significantly at exit_sd=1.0")
    print("  ✓ Net expectancy should improve at exit_sd=1.0 (may go positive for some classes)")
    print("  ✓ Break-even financing rate should rise as exit_sd increases")
    print("  ✓ Q1 3v3 exit_sd=0.0 should match paper: GWR=72.8%, Net≈-1.26%")
    print()
 
 
if __name__ == '__main__':
    main()