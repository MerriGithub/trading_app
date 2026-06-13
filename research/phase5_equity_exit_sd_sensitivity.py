"""
phase5_equity_exit_sd_sensitivity.py
=====================================
Phase 5 research: EXIT_SD proximity-to-degenerate boundary test for equities.

Two tasks:
    Task A — Fine-grained EXIT_SD grid [1.6 … 2.5] across all 132 1v1 equity
             pairs.  Universe-mean avg_net, avg_holding, net_wr, and n_trades
             per EXIT_SD value.  Looking for a plateau (robust) vs a spike
             at exactly 2.0 (fragile).

    Task B — Walk-forward stability check: re-run WF at EXIT_SD ∈ [1.5, 1.8, 2.0]
             using the validated IS=3y/OOS=1y/step=1y/contrarian protocol.
             If ρ remains significantly positive across all three, the
             parameter region is stable.

Run from trading_app/:
    python research/phase5_equity_exit_sd_sensitivity.py

Outputs:
    data/phase5_equity_exit_sd_results.json   — machine-readable full results
    Stdout: formatted decision-log tables for copy-paste into Obsidian.

Methodology notes
-----------------
- Universe: all 12 equity indices (matches Tab 9 Q11 protocol exactly).
- Cost model: per-instrument spread_pct from asset_configs (equity financing
  net_daily = (0.0488 - 0.0088) / 365 per leg per day).
- XING_SD fixed at 2.0 (confirmed optimum, phase 2).
- Vol window: 262 (standard).
- Task A uses full history ('1999-01-01') for statistical power.
- Task B uses IS=3y, OOS=1y, step=1y — matching the validated WF params
  (same params that produced ρ=+0.208 at EXIT_SD=2.0).
- WF spread_cost_pct = 0.001 (uniform; matches validated benchmark run).

Register items referenced
-------------------------
H: Exit condition sign: -side * d <= EXIT_SD.  Enforced by numba_core canary.
I: Entry dislocation must be abs(d).  Enforced by numba_core canary.
K: Tab 10 uses engine/backtest.py pattern — this script follows the same.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap — allow running from trading_app/ without installation ──
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.backtest import (
    load_asset_prices,
    prepare_returns,
    run_backtest,
    aggregate_trades,
)
from engine.numba_core import backtest_spread, rolling_mean_std  # noqa: F401 — top-level import required (register O)
from engine.walkforward import run_walk_forward, summarise_walk_forward
from asset_configs import ASSET_CLASSES, get_tradeable_instruments

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_TRADING_APP_ROOT = Path(__file__).parent.parent
_CACHE_DIR        = _TRADING_APP_ROOT / 'cache'
_DATA_DIR         = _TRADING_APP_ROOT / 'data'
_RESULTS_PATH     = _DATA_DIR / 'phase5_equity_exit_sd_results.json'

# ── Fixed parameters ───────────────────────────────────────────────────────
XING_SD       = 2.0          # entry threshold — confirmed optimum
VOL_WINDOW    = 262          # rolling vol window
TARGET_VOL    = 0.01         # 1% daily vol scaling target
MAX_HOLD_DAYS = 300          # cap on trade duration

# Task A grid
TASK_A_EXIT_SDS: list[float] = [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

# Task B WF values
TASK_B_EXIT_SDS: list[float] = [1.5, 1.8, 2.0]

# WF parameters — must match validated benchmark (IS=3y, OOS=1y, step=1y)
WF_IS_YEARS   = 3
WF_OOS_YEARS  = 1
WF_STEP_YEARS = 1
WF_SCORING    = 'contrarian'
WF_SPREAD_COST_PCT = 0.001   # uniform, matching validated benchmark

# Equity cost model (from asset_configs.EQUITY)
_EQ_FIN = ASSET_CLASSES['equity']['financing']
FIN_NET_DAILY = _EQ_FIN['net_daily']          # (0.0488 - 0.0088) / 365 per leg per day
N_LEGS        = 2                              # 1v1 pair

# Average per-instrument spread cost across the equity universe
# Computed from asset_configs spread_pct values for all 12 instruments.
def _mean_equity_spread_cost() -> float:
    """Compute universe-mean one-way spread cost from asset_configs.

    Returns:
        Mean spread_pct across all 12 equity instruments, doubled for
        round-trip (one per leg × 2 legs × 2 directions = 4× one-way,
        but aggregate_trades expects round-trip per pair = 2×leg_mean_spread).
        Matches the cost model in Tab 10 / daily_scan.py.
    """
    eq_instruments = ASSET_CLASSES['equity']['instruments']
    costs = [cfg.get('spread_pct', 0.001) for cfg in eq_instruments.values()]
    mean_one_way = sum(costs) / len(costs)
    # Round-trip = 2 × (long one-way + short one-way) = 4 × mean_one_way.
    # aggregate_trades uses: per_trade_spread = spread_cost_pct × n_legs × 2
    # So we pass mean_one_way and let aggregate_trades apply the 2×n_legs factor.
    return mean_one_way

SPREAD_COST_PCT = _mean_equity_spread_cost()


def _setup_logging() -> None:
    """Configure root logger for standalone script use."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


# ═══════════════════════════════════════════════════════════════════════════
# TASK A — Fine-grained EXIT_SD grid, universe mean statistics
# ═══════════════════════════════════════════════════════════════════════════

def run_task_a(
    scaled: np.ndarray,
    day_ints: np.ndarray,
    instruments: list[str],
    exit_sds: list[float] = TASK_A_EXIT_SDS,
) -> pd.DataFrame:
    """Run fine-grained EXIT_SD grid across all 1v1 directional equity pairs.

    Generates all N×(N-1) = 132 directed pairs for 12 instruments, runs a
    full backtest at each EXIT_SD value, and aggregates universe-mean
    statistics across all pairs per EXIT_SD.

    This answers: is EXIT_SD=2.0 the peak of a broad plateau (robust) or a
    narrow spike that collapses at 2.1/1.9 (fragile)?

    Args:
        scaled: Vol-normalised return matrix (T, N) from prepare_returns.
        day_ints: Integer day index array (T,) from prepare_returns.
        instruments: Instrument codes matching columns of scaled.
        exit_sds: EXIT_SD values to evaluate.

    Returns:
        DataFrame with one row per EXIT_SD value, columns:
            exit_sd, n_pairs_with_trades, total_trades, avg_net,
            net_wr, avg_holding, median_holding, avg_gross, total_ev
        Universe mean is the average across all pairs that had ≥1 trade.
    """
    N = len(instruments)
    # All directed pairs: long i, short j, i≠j
    pairs: list[tuple[int, int]] = [
        (i, j) for i in range(N) for j in range(N) if i != j
    ]
    n_pairs = len(pairs)
    logger.info("Task A: %d directed pairs × %d EXIT_SD values", n_pairs, len(exit_sds))

    rows: list[dict] = []
    for exit_sd in exit_sds:
        pair_summaries: list[dict] = []
        for long_i, short_i in pairs:
            spread = scaled[:, long_i] - scaled[:, short_i]
            trades, n_trades, _cum, _dist = backtest_spread(
                spread, VOL_WINDOW, XING_SD, exit_sd, day_ints, MAX_HOLD_DAYS
            )
            if n_trades == 0:
                continue
            summary = aggregate_trades(
                trades, n_trades,
                spread_cost_pct=SPREAD_COST_PCT,
                financing_daily_pct=FIN_NET_DAILY,
                n_legs=N_LEGS,
            )
            pair_summaries.append(summary)

        if not pair_summaries:
            rows.append({
                'exit_sd': exit_sd,
                'n_pairs_with_trades': 0,
                'total_trades':        0,
                'avg_net':             float('nan'),
                'net_wr':              float('nan'),
                'avg_holding':         float('nan'),
                'median_holding':      float('nan'),
                'avg_gross':           float('nan'),
                'total_ev':            float('nan'),
            })
            continue

        total_trades = sum(s['n_trades'] for s in pair_summaries)
        avg_net      = float(np.mean([s['avg_net']       for s in pair_summaries]))
        net_wr       = float(np.mean([s['net_wr']        for s in pair_summaries]))
        avg_holding  = float(np.mean([s['avg_holding']   for s in pair_summaries]))
        med_holding  = float(np.mean([s['median_holding'] for s in pair_summaries]))
        avg_gross    = float(np.mean([s['avg_gross']     for s in pair_summaries]))
        total_ev     = float(np.sum([s['avg_net'] * s['n_trades'] for s in pair_summaries]))

        rows.append({
            'exit_sd':             exit_sd,
            'n_pairs_with_trades': len(pair_summaries),
            'total_trades':        total_trades,
            'avg_net':             avg_net,
            'net_wr':              net_wr,
            'avg_holding':         avg_holding,
            'median_holding':      med_holding,
            'avg_gross':           avg_gross,
            'total_ev':            total_ev,
        })
        logger.info(
            "  EXIT_SD=%.1f: %d pairs active, %d trades, avg_net=%.4f%%, "
            "net_wr=%.1f%%, avg_hold=%.1fd",
            exit_sd, len(pair_summaries), total_trades,
            avg_net * 100, net_wr * 100, avg_holding,
        )

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# TASK B — Walk-forward stability across EXIT_SD values
# ═══════════════════════════════════════════════════════════════════════════

def run_task_b(
    prices: pd.DataFrame,
    instruments: list[str],
    exit_sds: list[float] = TASK_B_EXIT_SDS,
) -> pd.DataFrame:
    """Rerun WF at multiple EXIT_SD values to test parameter region stability.

    Uses the identical protocol that produced the benchmark ρ=+0.208 result
    (IS=3y, OOS=1y, step=1y, contrarian, XING_SD=2.0).

    If ρ remains significantly positive at both 1.5 and 1.8, the 2.0 optimum
    sits on a stable plateau.  If ρ collapses below EXIT_SD=2.0, it is a
    narrow peak and should be flagged for paper trading caution.

    Args:
        prices: Raw price DataFrame loaded via load_asset_prices.
        instruments: Equity instrument codes to include.
        exit_sds: EXIT_SD values to validate.

    Returns:
        DataFrame with one row per EXIT_SD value, columns:
            exit_sd, rho, p_value, n_obs, q1_mean, q5_mean,
            t_stat, t_p, significant (p < 0.05)
    """
    rows: list[dict] = []
    for exit_sd in exit_sds:
        logger.info("Task B: running WF at EXIT_SD=%.1f …", exit_sd)
        t0 = time.time()

        results = run_walk_forward(
            prices=prices,
            instruments=instruments,
            is_years=WF_IS_YEARS,
            oos_years=WF_OOS_YEARS,
            step_years=WF_STEP_YEARS,
            scoring_mode=WF_SCORING,
            vol_window=VOL_WINDOW,
            target_vol=TARGET_VOL,
            xing_sd=XING_SD,
            exit_sd=exit_sd,
            spread_cost_pct=WF_SPREAD_COST_PCT,
        )

        elapsed = time.time() - t0

        if results.empty:
            logger.warning(
                "Task B: WF returned empty DataFrame at EXIT_SD=%.1f — "
                "insufficient price history or no trades.", exit_sd
            )
            rows.append({
                'exit_sd':    exit_sd,
                'rho':        float('nan'),
                'p_value':    float('nan'),
                'n_obs':      0,
                'q1_mean':    float('nan'),
                'q5_mean':    float('nan'),
                't_stat':     float('nan'),
                't_p':        float('nan'),
                'significant': False,
                'elapsed_s':  round(elapsed, 1),
            })
            continue

        summary = summarise_walk_forward(results)
        significant = summary['p_value'] < 0.05 and summary['rho'] > 0

        logger.info(
            "  EXIT_SD=%.1f: ρ=%.3f, p=%.4f, n_obs=%d, "
            "Q1_mean=%.4f, Q5_mean=%.4f, elapsed=%.1fs",
            exit_sd,
            summary['rho'], summary['p_value'], summary['n_obs'],
            summary['q1_mean'], summary['q5_mean'],
            elapsed,
        )

        rows.append({
            'exit_sd':    exit_sd,
            'rho':        round(summary['rho'], 4),
            'p_value':    round(summary['p_value'], 4),
            'n_obs':      summary['n_obs'],
            'q1_mean':    round(summary['q1_mean'], 4),
            'q5_mean':    round(summary['q5_mean'], 4),
            't_stat':     round(summary['t_stat'], 3),
            't_p':        round(summary['t_p'], 4),
            'significant': bool(significant),
            'elapsed_s':  round(elapsed, 1),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# INTERPRETATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _interpret_task_a(df: pd.DataFrame) -> str:
    """Return a one-line robustness verdict for the Task A grid.

    A plateau is defined as EXIT_SD 1.6–2.0 all being within 30% of the
    2.0 avg_net (i.e. no cliff-edge drop before 2.0).  A spike is anything
    else — avg_net at 2.0 is >50% higher than at 1.9 or 1.8.

    Args:
        df: Task A results DataFrame.

    Returns:
        Verdict string for the decision log.
    """
    if df.empty or df['avg_net'].isna().all():
        return "INCONCLUSIVE — no trades generated."

    at_2_0 = df.loc[df['exit_sd'] == 2.0, 'avg_net']
    if at_2_0.empty or np.isnan(at_2_0.iloc[0]):
        return "INCONCLUSIVE — EXIT_SD=2.0 missing from results."

    val_2_0 = at_2_0.iloc[0]
    lower_vals = df.loc[df['exit_sd'].isin([1.7, 1.8, 1.9]), 'avg_net'].dropna()

    if lower_vals.empty:
        return "INCONCLUSIVE — insufficient lower EXIT_SD values."

    min_lower = lower_vals.min()

    if val_2_0 <= 0:
        return "NEGATIVE at EXIT_SD=2.0 — strategy not viable in universe mean."

    # Plateau: all lower values ≥ 50% of peak value
    if min_lower >= 0.5 * val_2_0:
        return (
            f"PLATEAU CONFIRMED — avg_net stable from EXIT_SD 1.7–2.0 "
            f"(min lower = {min_lower*100:.3f}%, peak = {val_2_0*100:.3f}%). "
            f"EXIT_SD=2.0 is robust."
        )
    elif min_lower > 0:
        return (
            f"PARTIAL PLATEAU — avg_net positive across range but some degradation "
            f"below 2.0 (min lower = {min_lower*100:.3f}%, peak = {val_2_0*100:.3f}%). "
            f"Proceed with moderate caution."
        )
    else:
        return (
            f"SPIKE WARNING — avg_net turns negative or near-zero below EXIT_SD=2.0 "
            f"(min lower = {min_lower*100:.3f}%, peak = {val_2_0*100:.3f}%). "
            f"EXIT_SD=2.0 may be a fragile peak — flag for paper trading caution."
        )


def _interpret_task_b(df: pd.DataFrame) -> str:
    """Return a one-line WF stability verdict for the Task B results.

    Args:
        df: Task B results DataFrame.

    Returns:
        Verdict string for the decision log.
    """
    if df.empty:
        return "INCONCLUSIVE — no WF results generated."

    sig_count = int(df['significant'].sum())
    total     = len(df)

    at_2_0 = df.loc[df['exit_sd'] == 2.0]
    at_1_8 = df.loc[df['exit_sd'] == 1.8]
    at_1_5 = df.loc[df['exit_sd'] == 1.5]

    def _row_summary(row: pd.Series) -> str:
        if row.empty:
            return "missing"
        r = row.iloc[0]
        sig = "✓ sig" if r['significant'] else "✗ not sig"
        return f"ρ={r['rho']:.3f}, p={r['p_value']:.4f} ({sig})"

    s_2_0 = _row_summary(at_2_0)
    s_1_8 = _row_summary(at_1_8)
    s_1_5 = _row_summary(at_1_5)

    all_sig = all(df['significant'])
    none_sig_below = (
        not df.loc[df['exit_sd'] < 2.0, 'significant'].any()
        if not df.loc[df['exit_sd'] < 2.0].empty else False
    )

    if all_sig:
        return (
            f"STABLE PLATEAU — WF ρ significantly positive at all EXIT_SD values. "
            f"1.5: {s_1_5}  |  1.8: {s_1_8}  |  2.0: {s_2_0}. "
            f"Parameter region is robust."
        )
    elif none_sig_below and at_2_0['significant'].any():
        return (
            f"FRAGILE PEAK — ρ significant only at EXIT_SD=2.0, not below. "
            f"1.5: {s_1_5}  |  1.8: {s_1_8}  |  2.0: {s_2_0}. "
            f"⚠ Flag for paper trading caution — EXIT_SD=2.0 is at edge of viable region."
        )
    elif sig_count > 0:
        return (
            f"PARTIAL STABILITY — ρ significant at {sig_count}/{total} EXIT_SD values. "
            f"1.5: {s_1_5}  |  1.8: {s_1_8}  |  2.0: {s_2_0}. "
            f"Moderate confidence — review per-window ρ tables before trading."
        )
    else:
        return (
            f"WF SIGNAL ABSENT — ρ not significant at any EXIT_SD tested. "
            f"1.5: {s_1_5}  |  1.8: {s_1_8}  |  2.0: {s_2_0}. "
            f"⚠ Cannot confirm predictive validity — do not trade without further investigation."
        )


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════

def _print_task_a_table(df: pd.DataFrame) -> None:
    """Print a markdown-formatted Task A grid table to stdout."""
    print("\n## Task A — Fine-grained EXIT_SD sensitivity grid (equity universe mean)")
    print(f"Instruments: 12 equity indices, all 132 directed 1v1 pairs")
    print(f"Cost model:  spread={SPREAD_COST_PCT*100:.4f}% one-way, "
          f"financing={FIN_NET_DAILY*365*100:.2f}%p.a. net daily")
    print(f"Parameters:  XING_SD={XING_SD}, Vol={VOL_WINDOW}, MaxHold={MAX_HOLD_DAYS}d")
    print()
    header = (
        f"{'EXIT_SD':>8} | {'Pairs':>5} | {'Trades':>7} | "
        f"{'AvgNet%':>8} | {'NetWR%':>7} | "
        f"{'AvgHold':>8} | {'MedHold':>8} | {'AvgGross%':>10} | {'TotalEV':>9}"
    )
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        if np.isnan(row['avg_net']):
            print(f"{row['exit_sd']:>8.1f} | {'–':>5} | {'–':>7} | "
                  f"{'–':>8} | {'–':>7} | {'–':>8} | {'–':>8} | {'–':>10} | {'–':>9}")
            continue
        marker = " ← 2.0" if abs(row['exit_sd'] - 2.0) < 0.01 else ""
        marker = " ← 2.5 ⚠DEG" if abs(row['exit_sd'] - 2.5) < 0.01 else marker
        print(
            f"{row['exit_sd']:>8.1f} | {int(row['n_pairs_with_trades']):>5d} | "
            f"{int(row['total_trades']):>7d} | "
            f"{row['avg_net']*100:>8.3f} | {row['net_wr']*100:>7.1f} | "
            f"{row['avg_holding']:>8.1f} | {row['median_holding']:>8.1f} | "
            f"{row['avg_gross']*100:>10.3f} | {row['total_ev']:>9.2f}"
            f"{marker}"
        )
    print()


def _print_task_b_table(df: pd.DataFrame) -> None:
    """Print a markdown-formatted Task B WF stability table to stdout."""
    print("\n## Task B — Walk-forward stability check (IS=3y, OOS=1y, step=1y, contrarian)")
    print(f"Benchmark:   EXIT_SD=2.0 → ρ=+0.208, p≈0 (confirmed WF validation)")
    print()
    header = (
        f"{'EXIT_SD':>8} | {'ρ':>7} | {'p-value':>8} | {'n_obs':>6} | "
        f"{'Q1 mean':>8} | {'Q5 mean':>8} | {'t-stat':>7} | {'t-p':>7} | {'Sig?':>5} | {'Time':>6}"
    )
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        if np.isnan(row['rho']):
            print(f"{row['exit_sd']:>8.1f} | {'–':>7} | {'–':>8} | {'–':>6} | "
                  f"{'–':>8} | {'–':>8} | {'–':>7} | {'–':>7} | {'–':>5} | "
                  f"{row['elapsed_s']:>5.1f}s")
            continue
        marker = " ← ref" if abs(row['exit_sd'] - 2.0) < 0.01 else ""
        sig_str = "✓ Yes" if row['significant'] else "✗ No"
        print(
            f"{row['exit_sd']:>8.1f} | {row['rho']:>+7.3f} | {row['p_value']:>8.4f} | "
            f"{int(row['n_obs']):>6d} | {row['q1_mean']:>8.4f} | {row['q5_mean']:>8.4f} | "
            f"{row['t_stat']:>7.3f} | {row['t_p']:>7.4f} | {sig_str:>5} | "
            f"{row['elapsed_s']:>5.1f}s{marker}"
        )
    print()


def _print_decision_log(
    task_a_df: pd.DataFrame,
    task_b_df: pd.DataFrame,
    verdict_a: str,
    verdict_b: str,
    run_ts: str,
    elapsed_total: float,
) -> None:
    """Print full decision log entry to stdout, formatted for Obsidian paste."""
    print("\n" + "═" * 78)
    print(f"DECISION LOG — Phase 5: Equity EXIT_SD Robustness")
    print(f"Run timestamp: {run_ts}  |  Total elapsed: {elapsed_total:.0f}s")
    print("═" * 78)

    _print_task_a_table(task_a_df)
    _print_task_b_table(task_b_df)

    print("## Verdicts")
    print(f"Task A (grid shape): {verdict_a}")
    print(f"Task B (WF stability): {verdict_b}")
    print()

    # Overall recommendation
    a_plateau = "PLATEAU" in verdict_a
    b_stable  = "STABLE PLATEAU" in verdict_b
    b_partial = "PARTIAL" in verdict_b

    print("## Overall recommendation")
    if a_plateau and b_stable:
        print(
            "✅ EXIT_SD=2.0 CONFIRMED ROBUST — plateau shape confirmed and WF ρ "
            "stable across 1.5–2.0. Proceed to paper trading with EXIT_SD=2.0."
        )
    elif a_plateau and b_partial:
        print(
            "🟡 EXIT_SD=2.0 LIKELY ROBUST — plateau confirmed in grid but WF ρ "
            "degrades below 2.0. Use 2.0 but size smaller than commodity allocation. "
            "Monitor first 10 equity paper trades closely."
        )
    elif not a_plateau and not b_stable:
        print(
            "⚠ EXIT_SD=2.0 FRAGILE PEAK — grid and WF both show 2.0 is a narrow "
            "maximum. Do NOT trade equities at current parameter settings without "
            "further investigation. Consider EXIT_SD=1.8 as a more conservative choice "
            "if WF ρ is marginally positive there."
        )
    else:
        print(
            "🟡 MIXED SIGNALS — grid and WF results are not fully aligned. "
            "Review per-value tables above before committing to paper trading. "
            "Use the more conservative finding as the binding constraint."
        )
    print()


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run Phase 5 EXIT_SD sensitivity and WF stability tests, write results."""
    _setup_logging()
    t_start  = time.time()
    run_ts   = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Load equity price data ─────────────────────────────────────────────
    prices_path = _CACHE_DIR / 'prices.csv'
    if not prices_path.exists():
        raise FileNotFoundError(
            f"Equity price cache not found at {prices_path}. "
            f"Run the app or DataRegistry.refresh() first."
        )

    logger.info("Loading equity prices from %s", prices_path)
    prices, instruments = load_asset_prices(
        str(prices_path),
        start_date='1999-01-01',
        min_obs=VOL_WINDOW,
    )
    logger.info("Loaded %d instruments, %d trading days", len(instruments), len(prices))

    if len(instruments) < 2:
        raise ValueError(
            f"Fewer than 2 equity instruments loaded from {prices_path}. "
            f"Cannot run any pair tests."
        )

    # Task A uses vol-scaled returns for backtest_spread calls
    logger.info("Computing vol-scaled returns for Task A …")
    scaled, day_ints, _index = prepare_returns(
        prices,
        instruments,
        vol_window=VOL_WINDOW,
        target_vol=TARGET_VOL,
    )
    logger.info("Return matrix shape: %s", scaled.shape)

    # ── Task A ─────────────────────────────────────────────────────────────
    logger.info(
        "=== Task A: fine-grained EXIT_SD grid %s ===",
        TASK_A_EXIT_SDS,
    )
    t_a = time.time()
    task_a_df = run_task_a(scaled, day_ints, instruments, TASK_A_EXIT_SDS)
    logger.info("Task A complete in %.1fs", time.time() - t_a)

    # ── Task B ─────────────────────────────────────────────────────────────
    logger.info(
        "=== Task B: WF stability at EXIT_SD %s (IS=%dy, OOS=%dy) ===",
        TASK_B_EXIT_SDS, WF_IS_YEARS, WF_OOS_YEARS,
    )
    t_b = time.time()
    task_b_df = run_task_b(prices, instruments, TASK_B_EXIT_SDS)
    logger.info("Task B complete in %.1fs", time.time() - t_b)

    # ── Interpret ──────────────────────────────────────────────────────────
    verdict_a = _interpret_task_a(task_a_df)
    verdict_b = _interpret_task_b(task_b_df)

    # ── Print decision log ─────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    _print_decision_log(
        task_a_df, task_b_df,
        verdict_a, verdict_b,
        run_ts, elapsed_total,
    )

    # ── Persist results ────────────────────────────────────────────────────
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        'run_timestamp':       run_ts,
        'run_duration_seconds': round(elapsed_total, 1),
        'parameters': {
            'instruments':      instruments,
            'n_instruments':    len(instruments),
            'xing_sd':          XING_SD,
            'vol_window':       VOL_WINDOW,
            'target_vol':       TARGET_VOL,
            'max_hold_days':    MAX_HOLD_DAYS,
            'spread_cost_pct':  SPREAD_COST_PCT,
            'fin_net_daily':    FIN_NET_DAILY,
            'n_legs':           N_LEGS,
            'wf_is_years':      WF_IS_YEARS,
            'wf_oos_years':     WF_OOS_YEARS,
            'wf_step_years':    WF_STEP_YEARS,
            'wf_scoring':       WF_SCORING,
            'wf_spread_cost_pct': WF_SPREAD_COST_PCT,
        },
        'task_a': {
            'exit_sds':    TASK_A_EXIT_SDS,
            'results':     task_a_df.replace({float('nan'): None}).to_dict('records'),
            'verdict':     verdict_a,
        },
        'task_b': {
            'exit_sds':    TASK_B_EXIT_SDS,
            'results':     task_b_df.replace({float('nan'): None}).to_dict('records'),
            'verdict':     verdict_b,
        },
        'n_trading_days_loaded': len(prices),
    }

    _RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding='utf-8')
    logger.info("Results written to %s", _RESULTS_PATH)

    # Final stdout summary (print OK — standalone script entry point per daily_scan.py pattern)
    print(f"\nPhase 5 complete — {round(elapsed_total)}s")
    print(f"Results: {_RESULTS_PATH}")
    print(f"Task A verdict: {verdict_a}")
    print(f"Task B verdict: {verdict_b}")


if __name__ == '__main__':
    main()
