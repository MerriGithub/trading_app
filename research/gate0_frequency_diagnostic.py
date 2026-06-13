"""
gate0_frequency_diagnostic.py — Gate #0: Trade Frequency Diagnostic
=====================================================================

PURPOSE
-------
Determines whether the central annual estimate is £4,000 or £2,400/yr by
comparing:

    (A) Full-history backtest trade counts per calendar year for CTN/UKX and
        UKX/CTN (what the live book will actually see year by year).

    (B) WF OOS trade counts for the same pairs per OOS window (what the WF
        validation captured).

If full-history counts ≈ WF OOS counts → central estimate is £4,000/yr.
If full-history counts are materially lower → central estimate is £2,400/yr.

USAGE
-----
Run from inside trading_app/:

    cd "C:\\Users\\gordo\\Documents\\trading_app"
    C:\\Users\\gordo\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe research\\gate0_frequency_diagnostic.py

OUTPUT
------
Prints a year-by-year comparison table to stdout and writes a summary to
data/gate0_results.json.

# pragma: research-script-stdout-intentional — print() calls below are the
# primary output mechanism for this standalone diagnostic. They are not bugs.

PARAMETERS (match validated production settings)
---------
    XING_SD  = 2.0   (confirmed optimum — all asset classes)
    EXIT_SD  = 2.0   (confirmed equity optimum — Decision 62, Phase 4b)
    VOL_WIN  = 262   (standard 1-year rolling window)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
# Script must be run from trading_app/ root so relative paths work.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.backtest import load_asset_prices, prepare_returns, run_backtest
from engine.numba_core import COL_ENTRY_IDX

# Standalone research script — configures its own handler so it can be run
# directly without the main app's logging_config.py.  Not imported by the app.
# pragma: research-script-stdout-intentional
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ── Parameters ───────────────────────────────────────────────────────────────
PAIRS = [
    ('CTN', 'UKX'),   # S&P 500 long, FTSE 100 short
    ('UKX', 'CTN'),   # FTSE 100 long, S&P 500 short
]
XING_SD   = 2.0
EXIT_SD   = 2.0   # equity scalp confirmed optimum
VOL_WIN   = 262
CACHE_CSV = ROOT / 'cache' / 'prices.csv'
OUT_JSON  = ROOT / 'data' / 'gate0_results.json'

WF_JSON = ROOT / 'data' / 'non_asx_wf_results.json'


def _load_wf_oos_counts(wf_json: Path) -> dict[str, dict[int, int]]:
    """Load WF OOS trade counts for CTN/UKX and UKX/CTN from the results JSON.

    Reads data/non_asx_wf_results.json at runtime so counts stay in sync with
    the most recent WF run rather than a frozen snapshot.

    Args:
        wf_json: Path to non_asx_wf_results.json.

    Returns:
        Dict keyed by pair label → {oos_end_year: oos_trade_count}.

    Raises:
        FileNotFoundError: If the WF results file does not exist.
        KeyError: If expected pair labels are absent from the JSON.
    """
    if not wf_json.exists():
        raise FileNotFoundError(
            f"WF results file not found: {wf_json}\n"
            "Re-run the walk-forward validation (Tab 9) to regenerate it."
        )
    with open(wf_json) as f:
        records = json.load(f)

    target_pairs = {'CTN / UKX', 'UKX / CTN'}
    counts: dict[str, dict[int, int]] = {p: {} for p in target_pairs}

    for rec in records:
        pair = rec.get('pair', '')
        if pair not in target_pairs:
            continue
        oos_year = int(rec['oos_end'])
        oos_n    = int(rec['OOS_Trades'])
        counts[pair][oos_year] = oos_n

    missing = [p for p in target_pairs if not counts[p]]
    if missing:
        raise KeyError(
            f"Pairs not found in WF results JSON: {missing}. "
            "Check that the non-ASX WF expansion has been run."
        )

    return counts


def _backtest_pair(
    prices: pd.DataFrame,
    long_code: str,
    short_code: str,
) -> tuple[np.ndarray, int, pd.DatetimeIndex]:
    """Run full-history backtest on one directional pair.

    Args:
        prices: Full price DataFrame (DatetimeIndex).
        long_code: Instrument code for the long leg.
        short_code: Instrument code for the short leg.

    Returns:
        Tuple of (trades_raw, n_trades, date_index) where:
            - trades_raw is the raw trade array from numba_core
            - n_trades is the count of completed trades
            - date_index is the DatetimeIndex of the return series
              (used to map entry array-index → calendar date)
    """
    scaled, day_ints, index = prepare_returns(
        prices,
        instruments=[long_code, short_code],
        vol_window=VOL_WIN,
    )

    # Spread: long leg minus short leg (col 0 minus col 1)
    spread_returns = scaled[:, 0] - scaled[:, 1]

    result = run_backtest(
        spread_returns,
        day_ints,
        vol_window=VOL_WIN,
        xing_sd=XING_SD,
        exit_sd=EXIT_SD,
        spread_cost_pct=0.0,   # cost-free: we only care about trade counts
        financing_daily_pct=0.0,
        n_legs=2,
    )

    return result['trades_raw'], result['n_trades'], index


def _trades_by_year(
    trades_raw: np.ndarray,
    n_trades: int,
    index: pd.DatetimeIndex,
) -> dict[int, int]:
    """Count trades per calendar year of entry.

    Args:
        trades_raw: Raw trade array (shape n x 5) from numba_core.
        n_trades: Number of valid trades in trades_raw.
        index: DatetimeIndex of the return series for mapping array idx → date.

    Returns:
        Dict mapping calendar year (int) → trade count (int).
    """
    if n_trades == 0:
        return {}

    counts: dict[int, int] = {}
    for trade in trades_raw[:n_trades]:
        entry_idx = int(trade[COL_ENTRY_IDX])
        if entry_idx < len(index):
            year = index[entry_idx].year
            counts[year] = counts.get(year, 0) + 1
    return counts


def _compare_table(
    pair_label: str,
    full_hist_by_year: dict[int, int],
    wf_oos_by_year: dict[int, int],
) -> pd.DataFrame:
    """Build year-by-year comparison DataFrame for one pair.

    Args:
        pair_label: Display string e.g. 'CTN / UKX'.
        full_hist_by_year: Full-history trade counts per year.
        wf_oos_by_year: WF OOS trade counts per oos_end year.

    Returns:
        DataFrame with columns: Year, FullHist, WF_OOS, Delta, Ratio.
    """
    all_years = sorted(
        set(full_hist_by_year.keys()) | set(wf_oos_by_year.keys())
    )
    rows = []
    for yr in all_years:
        fh  = full_hist_by_year.get(yr, 0)
        wf  = wf_oos_by_year.get(yr, None)
        if wf is None:
            delta = None
            ratio = None
        else:
            delta = fh - wf
            ratio = round(fh / wf, 2) if wf > 0 else None
        rows.append({
            'Year':     yr,
            'FullHist': fh,
            'WF_OOS':   wf if wf is not None else '—',
            'Delta':    delta if delta is not None else '—',
            'Ratio':    ratio if ratio is not None else '—',
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Run Gate #0 diagnostic and print results."""
    logger.info("Gate #0: Trade Frequency Diagnostic")
    logger.info("Parameters: XING_SD=%.1f  EXIT_SD=%.1f  VOL_WIN=%d",
                XING_SD, EXIT_SD, VOL_WIN)

    # ── Load WF OOS counts from JSON ─────────────────────────────────────────
    WF_OOS_COUNTS = _load_wf_oos_counts(WF_JSON)
    logger.info("WF OOS counts loaded from %s", WF_JSON.name)

    # ── Load prices ──────────────────────────────────────────────────────────
    if not CACHE_CSV.exists():
        logger.error("Price file not found: %s", CACHE_CSV)
        logger.error("Run the app once to populate the cache, then re-run this script.")
        sys.exit(1)

    prices, instruments = load_asset_prices(str(CACHE_CSV))
    logger.info("Loaded %d rows, %d instruments from %s",
                len(prices), len(instruments), CACHE_CSV.name)
    logger.info("Date range: %s → %s",
                prices.index[0].date(), prices.index[-1].date())

    for code in ('CTN', 'UKX'):
        if code not in instruments:
            logger.error("Instrument %s not found in %s", code, CACHE_CSV.name)
            sys.exit(1)

    # ── Run full-history backtests ────────────────────────────────────────────
    results: dict[str, dict] = {}
    all_tables: list[pd.DataFrame] = []

    for long_code, short_code in PAIRS:
        pair_label = f'{long_code} / {short_code}'
        logger.info("Running backtest: %s", pair_label)

        trades_raw, n_trades, index = _backtest_pair(prices, long_code, short_code)
        by_year = _trades_by_year(trades_raw, n_trades, index)

        logger.info("  Total trades (full history): %d", n_trades)

        wf_oos_ref = WF_OOS_COUNTS[pair_label]

        table = _compare_table(pair_label, by_year, wf_oos_ref)
        all_tables.append(table)

        # Compare only years covered by both full-hist and WF OOS
        overlap_years = [
            yr for yr in wf_oos_ref
            if yr in by_year
        ]
        if overlap_years:
            fh_vals  = [by_year[yr]        for yr in overlap_years]
            wf_vals  = [wf_oos_ref[yr]     for yr in overlap_years]
            fh_mean  = sum(fh_vals) / len(fh_vals)
            wf_mean  = sum(wf_vals) / len(wf_vals)
            ratio    = fh_mean / wf_mean if wf_mean > 0 else None
        else:
            fh_mean = wf_mean = ratio = None

        results[pair_label] = {
            'n_trades_total':           n_trades,
            'by_year':                  by_year,
            'fh_mean_per_year_overlap': round(fh_mean, 2) if fh_mean is not None else None,
            'wf_mean_per_year_overlap': round(wf_mean, 2) if wf_mean is not None else None,
            'fh_wf_ratio':              round(ratio, 3)   if ratio   is not None else None,
        }

        # ── Print pair table ─────────────────────────────────────────────────
        print()
        print("=" * 64)
        print(f"  {pair_label}")
        print("=" * 64)
        print(table.to_string(index=False))
        print()
        if fh_mean is not None:
            print(f"  Full-hist mean/yr (overlap period): {fh_mean:.1f}")
            print(f"  WF OOS   mean/yr (overlap period): {wf_mean:.1f}")
            ratio_str = f"{ratio:.2f}x" if ratio is not None else "N/A"
            print(f"  Ratio (FullHist / WF_OOS):         {ratio_str}")
            if ratio is not None and ratio >= 0.8:
                print("  ✅ CONSISTENT — full-history count aligns with WF OOS")
            elif ratio is not None and ratio >= 0.5:
                print("  ⚠️  MODERATE SHORTFALL — full-history count is lower than WF OOS")
            else:
                print("  ❌ MATERIAL SHORTFALL — full-history count is substantially lower")

    # ── Combined summary ──────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  COMBINED SUMMARY (CTN/UKX + UKX/CTN)")
    print("=" * 64)

    ctn_ukx_data = results.get('CTN / UKX', {})
    ukx_ctn_data = results.get('UKX / CTN', {})

    ctn_by_yr = ctn_ukx_data.get('by_year', {})
    ukx_by_yr = ukx_ctn_data.get('by_year', {})

    wf_ctn = WF_OOS_COUNTS['CTN / UKX']
    wf_ukx = WF_OOS_COUNTS['UKX / CTN']

    all_years = sorted(set(ctn_by_yr) | set(ukx_by_yr))
    combined_rows = []
    for yr in all_years:
        fh_total = ctn_by_yr.get(yr, 0) + ukx_by_yr.get(yr, 0)
        wf_total = wf_ctn.get(yr, 0) + wf_ukx.get(yr, 0)
        combined_rows.append({
            'Year':         yr,
            'FH_Combined':  fh_total,
            'WF_Combined':  wf_total if (yr in wf_ctn or yr in wf_ukx) else '—',
        })

    combined_df = pd.DataFrame(combined_rows)
    print(combined_df.to_string(index=False))

    # Overlap period combined means
    overlap_yrs = [yr for yr in wf_ctn if yr in ctn_by_yr and yr in ukx_by_yr]
    comb_ratio: float | None = None
    fh_comb_mean: float = 0.0
    wf_comb_mean: float = 0.0
    if overlap_yrs:
        fh_combined_vals = [ctn_by_yr.get(yr, 0) + ukx_by_yr.get(yr, 0) for yr in overlap_yrs]
        wf_combined_vals = [wf_ctn.get(yr, 0) + wf_ukx.get(yr, 0) for yr in overlap_yrs]
        fh_comb_mean = sum(fh_combined_vals) / len(fh_combined_vals)
        wf_comb_mean = sum(wf_combined_vals) / len(wf_combined_vals)
        comb_ratio   = fh_comb_mean / wf_comb_mean if wf_comb_mean > 0 else None

        print()
        print(f"  Full-hist combined mean/yr: {fh_comb_mean:.1f} trades")
        print(f"  WF OOS   combined mean/yr:  {wf_comb_mean:.1f} trades")
        print(f"  Ratio: {comb_ratio:.2f}x" if comb_ratio is not None else "  Ratio: N/A")

    # ── Annual estimate verdict ───────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  VERDICT")
    print("=" * 64)

    if overlap_yrs and comb_ratio is not None:
        if comb_ratio >= 0.8:
            central = 4000
            verdict = "£4,000/yr"
            reason  = "Full-history trade frequency is consistent with WF OOS."
        elif comb_ratio >= 0.5:
            central = 3000
            verdict = "~£3,000/yr (interpolated)"
            reason  = ("Full-history count is moderately below WF OOS. "
                       "The £4,000 central estimate should be discounted.")
        else:
            central = 2400
            verdict = "£2,400/yr (floor)"
            reason  = ("Full-history count is substantially below WF OOS. "
                       "Use the 7-pair floor estimate, not the central estimate.")
        print(f"  Central annual estimate: {verdict}")
        print(f"  Reason: {reason}")
        print(f"  (Decision 123 context: £4,000 = 6 trades/pair/yr; "
              f"£2,400 = floor based on 7-pair book)")
    else:
        central = None
        verdict = "unknown"
        reason  = "Could not compute — check pair outputs above."
        print(f"  {reason}")

    # ── Save results ──────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        'parameters': {
            'xing_sd': XING_SD,
            'exit_sd': EXIT_SD,
            'vol_window': VOL_WIN,
            'cache_file': str(CACHE_CSV),
        },
        'verdict': {
            'central_estimate_gbp': central,
            'verdict_label':        verdict,
            'reason':               reason,
            'fh_comb_mean':         round(fh_comb_mean, 2) if overlap_yrs else None,
            'wf_comb_mean':         round(wf_comb_mean, 2) if overlap_yrs else None,
            'fh_wf_ratio':          round(comb_ratio, 3)   if comb_ratio is not None else None,
        },
        'pairs': {
            label: {
                **data,
                'by_year': {str(k): v for k, v in data['by_year'].items()},
            }
            for label, data in results.items()
        },
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(save_data, f, indent=2)
    logger.info("Results saved to %s", OUT_JSON)


if __name__ == '__main__':
    main()
