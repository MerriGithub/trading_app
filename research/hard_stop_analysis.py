"""
Hard Stop Analysis — Phase 5a
==============================
Computes MAE distributions and a stop-level grid search for equities and
commodities.  Uses backtest_spread_with_stop() from engine/numba_core.py.

Research optimums used (confirmed Phase 2–4b):
  Commodities — XING_SD=2.0, EXIT_SD=0.5
  Equities    — XING_SD=2.0, EXIT_SD=2.0

Run from trading_app/:
    python research/hard_stop_analysis.py

Outputs:
    Prints to stdout and saves to research/results/hard_stop_YYYYMMDD_HHMM.txt
"""
from __future__ import annotations

import itertools
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable regardless of launch directory
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from engine.numba_core import (
    COL_GROSS_RETURN,
    COL_MAE_SD,
    COL_STOPPED,
    backtest_spread_with_stop,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

ASSET_CONFIGS: dict[str, dict] = {
    'equities': {
        'price_file':      'cache/prices.csv',
        'asset_class_key': 'equity',
        'xing_sd':         2.0,
        'exit_sd':         2.0,    # Phase 4b confirmed optimum
        'vol_window':      262,
        'spread_cost_pct': 0.001,
        'avg_hold_days':   8,      # Phase 4b benchmark
        'exclude':         set(),
    },
    'commodities': {
        'price_file':      'cache/commodity_prices.csv',
        'asset_class_key': 'commodities',
        'xing_sd':         2.0,
        'exit_sd':         0.5,    # Phase 3 confirmed optimum
        'vol_window':      262,
        'spread_cost_pct': 0.001,
        'avg_hold_days':   104,    # Phase 3 benchmark
        'exclude':         {'WTI'},  # COMMODITY_EXCLUDE
    },
}

STOP_GRID: list[float] = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
BASELINE_STOP: float = 999.0  # effectively no stop

MAE_PERCENTILES: list[int] = [10, 25, 50, 75, 90, 95, 99]

# Stop trigger rate below which a stop level is operationally irrelevant
MIN_MEANINGFUL_STOP_RATE: float = 0.03  # 3 %

_W = 63
_SEP  = '═' * _W
_SEP2 = '─' * _W


# ── Helpers ────────────────────────────────────────────────────────────────

def _vol_scale_prices(
    prices: pd.DataFrame,
    instruments: list[str],
    vol_window: int,
    target_vol: float = 0.01,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute vol-scaled returns, identical to engine.walkforward._vol_scaled().

    Scales each instrument's daily returns so rolling volatility matches
    target_vol (capped at 1.0 — never lever up).  Drops any row with a NaN
    in any instrument column so the result is a clean rectangular matrix.

    Args:
        prices: Raw price DataFrame with DatetimeIndex.
        instruments: Subset of prices.columns to process.
        vol_window: Rolling window in trading days.
        target_vol: Target daily volatility fraction.  Defaults to 0.01.

    Returns:
        Tuple (scaled_df, day_ints) where scaled_df is the vol-normalised
        return DataFrame and day_ints is an int64 array of days since
        1970-01-01 for each row.
    """
    rets = prices[instruments].pct_change().dropna(how='all')
    vols = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    sc = (target_vol / vols.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    scaled_df = (rets * sc).dropna(how='any')
    day_ints = (
        (scaled_df.index - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    ).values.astype(np.int64)
    return scaled_df, day_ints


def _run_pairs(
    scaled_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    day_ints: np.ndarray,
    cfg: dict,
    stop_sd: float,
) -> tuple[np.ndarray, list[float]]:
    """Run all directional pairs with a given stop level.

    For each (A, B) pair, computes the vol-scaled spread A-B and runs
    backtest_spread_with_stop().  Aggregates individual trade arrays and
    collects per-pair avg-net values for equal-weighted EV calculation.

    Args:
        scaled_df: Vol-scaled returns DataFrame.
        pairs: Ordered (A, B) pairs to backtest.
        day_ints: Integer day indices aligned with scaled_df rows.
        cfg: Asset-class config dict (must contain xing_sd, exit_sd,
            vol_window, spread_cost_pct keys).
        stop_sd: Hard stop threshold in SD units.

    Returns:
        Tuple (all_trades, per_pair_avg_nets) where all_trades is the
        vertically stacked 8-column trade array (empty shape (0, 8) if no
        trades), and per_pair_avg_nets is a list of per-pair average net
        returns (as fractions) for pairs that produced at least one trade.
    """
    all_trade_arrays: list[np.ndarray] = []
    per_pair_avg_nets: list[float] = []
    cost = cfg['spread_cost_pct'] * 2  # round-trip spread cost

    for a, b in pairs:
        spread = (scaled_df[a] - scaled_df[b]).values.astype(np.float64)
        try:
            trades, n = backtest_spread_with_stop(
                spread,
                cfg['vol_window'],
                cfg['xing_sd'],
                cfg['exit_sd'],
                stop_sd,
                day_ints,
                max_hold_days=300,
            )
        except ValueError as exc:
            logger.warning("Skipping pair (%s, %s): %s", a, b, exc)
            continue

        if n > 0:
            all_trade_arrays.append(trades)
            avg_gross = float(trades[:, COL_GROSS_RETURN].mean())
            per_pair_avg_nets.append(avg_gross - cost)

    if all_trade_arrays:
        combined = np.vstack(all_trade_arrays)
    else:
        combined = np.zeros((0, 8), dtype=np.float64)

    return combined, per_pair_avg_nets


# ── Analysis ───────────────────────────────────────────────────────────────

def analyse_asset_class(
    name: str,
    cfg: dict,
    output_lines: list[str],
) -> None:
    """Run MAE distribution and stop-grid analysis for one asset class.

    Reads prices, vol-scales, generates all N×(N-1) ordered pairs, runs a
    baseline backtest (no stop), computes the MAE distribution for winners
    vs. losers, then re-runs each stop level in STOP_GRID.  All formatted
    output is appended to output_lines; nothing is printed here.

    Args:
        name: Asset class label used in section headers (e.g. 'equities').
        cfg: Asset-class config dict from ASSET_CONFIGS.
        output_lines: Mutable list to which formatted output lines are
            appended.  Caller is responsible for printing and saving.

    Raises:
        FileNotFoundError: If the configured price_file does not exist.
        ValueError: If fewer than 2 instruments remain after exclusion.
    """
    price_path = Path(cfg['price_file'])
    if not price_path.exists():
        raise FileNotFoundError(
            f"Price file not found: {price_path}. "
            f"Run from trading_app/ so relative paths resolve correctly."
        )

    prices = pd.read_csv(price_path, index_col='Date', parse_dates=True)
    prices = prices.ffill(limit=3).dropna(how='all')

    if prices.empty:
        raise ValueError(f"Price file {price_path} loaded empty after ffill/dropna")

    instruments = [c for c in prices.columns if c not in cfg['exclude']]
    if len(instruments) < 2:
        raise ValueError(
            f"Fewer than 2 instruments for {name} after exclusion "
            f"(excluded: {cfg['exclude']}). Check price file and exclude set."
        )

    scaled_df, day_ints = _vol_scale_prices(
        prices, instruments, cfg['vol_window']
    )

    pairs: list[tuple[str, str]] = list(itertools.permutations(instruments, 2))
    n_pairs = len(pairs)

    # Section header
    output_lines.append(_SEP)
    output_lines.append(f"HARD STOP ANALYSIS — {name.upper()}")
    output_lines.append(
        f"XING_SD={cfg['xing_sd']}  EXIT_SD={cfg['exit_sd']}  "
        f"VOL={cfg['vol_window']}  Pairs analysed: {n_pairs}"
    )
    output_lines.append(_SEP)
    output_lines.append('')

    # ── Baseline (no hard stop) ────────────────────────────────────────────
    baseline_trades, baseline_per_pair_nets = _run_pairs(
        scaled_df, pairs, day_ints, cfg, BASELINE_STOP
    )
    n_total = len(baseline_trades)

    if n_total == 0:
        output_lines.append("No trades generated — check price data and date range.")
        output_lines.append('')
        return

    cost = cfg['spread_cost_pct'] * 2
    gross_rets = baseline_trades[:, COL_GROSS_RETURN]
    n_wins = int((gross_rets > 0).sum())
    wr = n_wins / n_total
    avg_gross = float(gross_rets.mean())
    avg_net = avg_gross - cost
    total_ev = float(np.mean(baseline_per_pair_nets)) if baseline_per_pair_nets else 0.0

    output_lines.append("BASELINE (no hard stop):")
    output_lines.append(
        f"  Total pairs: {n_pairs}  |  Total trades: {n_total:,}  |  "
        f"Win rate: {wr:.1%}"
    )
    output_lines.append(
        f"  Avg gross/trade: {avg_gross * 100:+.3f}%  |  "
        f"Avg net/trade: {avg_net * 100:+.3f}%  |  "
        f"Total EV: {total_ev * 100:.2f}"
    )
    output_lines.append('')

    # ── MAE distribution ───────────────────────────────────────────────────
    mae_arr = baseline_trades[:, COL_MAE_SD]
    winners_mask = gross_rets > 0
    losers_mask = ~winners_mask
    mae_winners = mae_arr[winners_mask]
    mae_losers = mae_arr[losers_mask]
    n_w = len(mae_winners)
    n_l = len(mae_losers)

    output_lines.append("MAE DISTRIBUTION (SD units from entry):")
    output_lines.append(
        f"{'':22s}Winners (n={n_w:,})    Losers (n={n_l:,})"
    )

    for p in MAE_PERCENTILES:
        pw = float(np.percentile(mae_winners, p)) if n_w > 0 else float('nan')
        pl = float(np.percentile(mae_losers, p)) if n_l > 0 else float('nan')
        output_lines.append(
            f"  p{p:<3d}:              {pw:6.2f}             {pl:6.2f}"
        )

    output_lines.append('')
    output_lines.append("  % of WINNERS with MAE exceeding each stop level:")

    # Print in rows of 3
    row_parts: list[str] = []
    for idx, s in enumerate(STOP_GRID):
        pct = float((mae_winners > s).mean()) * 100 if n_w > 0 else 0.0
        row_parts.append(f"{s:.1f} SD: {pct:4.1f}%")
        if len(row_parts) == 3 or idx == len(STOP_GRID) - 1:
            output_lines.append("  " + "   ".join(row_parts))
            row_parts = []

    output_lines.append('')

    # ── Stop grid ──────────────────────────────────────────────────────────
    output_lines.append("STOP GRID RESULTS:")
    output_lines.append(
        f"  {'stop_sd':<8} {'trades':<7} {'WR%':<6} "
        f"{'AvgNet%':<10} {'TotalEV':<9} {'StopRate%':<11} {'EV_delta':<10} Flag"
    )

    stop_grid_rows: list[dict] = []

    for stop_sd in STOP_GRID:
        stop_trades, stop_per_pair_nets = _run_pairs(
            scaled_df, pairs, day_ints, cfg, stop_sd
        )
        n_t = len(stop_trades)
        if n_t == 0:
            continue

        stop_gross = stop_trades[:, COL_GROSS_RETURN]
        stop_wr = float((stop_gross > 0).sum()) / n_t
        stop_avg_gross = float(stop_gross.mean())
        stop_avg_net = stop_avg_gross - cost
        stop_total_ev = (
            float(np.mean(stop_per_pair_nets)) if stop_per_pair_nets else 0.0
        )
        stop_rate = float((stop_trades[:, COL_STOPPED] == 1.0).mean())
        ev_delta = stop_total_ev - total_ev

        flags: list[str] = []
        if ev_delta * 100 < -0.5:
            flags.append("⚠ cuts winners")
        if stop_rate < MIN_MEANINGFUL_STOP_RATE:
            flags.append("ℹ operationally irrelevant")
        flag_str = "  ".join(flags)

        stop_grid_rows.append({
            'stop_sd':   stop_sd,
            'n_trades':  n_t,
            'wr':        stop_wr,
            'avg_net':   stop_avg_net,
            'total_ev':  stop_total_ev,
            'stop_rate': stop_rate,
            'ev_delta':  ev_delta,
            'flag':      flag_str,
        })

        output_lines.append(
            f"  {stop_sd:<8.1f} {n_t:<7,d} {stop_wr * 100:<6.1f} "
            f"{stop_avg_net * 100:+.3f}     {stop_total_ev * 100:<9.2f} "
            f"{stop_rate * 100:<11.1f} {ev_delta * 100:<10.2f} {flag_str}"
        )

    output_lines.append('')

    # ── Recommendation ─────────────────────────────────────────────────────
    output_lines.append("RECOMMENDATION:")
    optimal: dict | None = None
    for row in stop_grid_rows:
        if row['ev_delta'] * 100 >= -0.1:
            optimal = row
            break

    if optimal:
        output_lines.append(
            f"  Optimal hard stop: {optimal['stop_sd']:.1f} SD  "
            f"(first level where EV_delta ≥ -0.10; "
            f"stop rate {optimal['stop_rate']:.1%})"
        )
    else:
        output_lines.append(
            "  No hard stop improves EV for this asset class. "
            "Use max_hold_days=300 as primary protection. "
            "Consider 4.5 SD as black swan protection only."
        )

    output_lines.append('')
    output_lines.append('')


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: run analysis for all configured asset classes."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = Path('research/results') / f'hard_stop_{timestamp}.txt'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f'Hard Stop Analysis — generated {datetime.now().isoformat()}',
        f'Research optimums: Commodities EXIT_SD=0.5, Equities EXIT_SD=2.0',
        '',
    ]

    for name, cfg in ASSET_CONFIGS.items():
        try:
            analyse_asset_class(name, cfg, lines)
        except FileNotFoundError as exc:
            lines.append(f"SKIPPED {name}: {exc}")
            lines.append('')
        except ValueError as exc:
            lines.append(f"ERROR {name}: {exc}")
            lines.append('')

    output = '\n'.join(lines)
    print(output)
    output_path.write_text(output, encoding='utf-8')
    print(f'\nSaved to {output_path}')


if __name__ == '__main__':
    main()
