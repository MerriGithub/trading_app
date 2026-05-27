"""
engine/unified_loader.py
========================
Multi-asset price loader for the MCP server.

Resolves a flat ticker list across any mix of asset classes, loads each
CSV file exactly once, inner-joins on the shared date range, and returns
columns in the requested order.
"""

from pathlib import Path

import pandas as pd

from asset_configs import ASSET_CLASSES
from engine.backtest import load_asset_prices


def _ticker_to_class(ticker: str) -> str | None:
    """Return the ASSET_CLASSES key for a ticker, or None if not found."""
    for cls, cfg in ASSET_CLASSES.items():
        if ticker in cfg.get('instruments', {}):
            return cls
    return None


def load_aligned_prices_unified(
    tickers: list[str],
    cache_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Load and date-align prices for an arbitrary mix of tickers.

    Parameters
    ----------
    tickers : list[str]
        Instrument codes. May span any mix of asset classes.
    cache_dir : str or Path
        Directory containing the price CSVs (e.g. "cache/").
    date_from : str or None
        Inclusive start date (e.g. "1999-01-04"). Passed to the CSV loader
        for efficiency; also applied after concatenation.
    date_to : str or None
        Inclusive end date (e.g. "2026-04-24").

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns in `tickers` order. Rows that are all-NaN
        are dropped; individual NaN cells are left for prepare_returns()
        to handle via dropna(how='any') on vol-scaled returns.

    Raises
    ------
    ValueError
        If any ticker is not found in ASSET_CLASSES, or is absent from its
        CSV file (instrument may have been excluded or not yet populated).
    FileNotFoundError
        If the required CSV file does not exist under cache_dir.
    KeyError
        If column reorder fails — indicates a bug in ticker resolution.
    """
    cache_dir = Path(cache_dir)

    # ── 1. Validate all tickers are known ─────────────────────────────────────
    unknown = [t for t in tickers if _ticker_to_class(t) is None]
    if unknown:
        raise ValueError(
            f"Tickers not found in any ASSET_CLASSES: {unknown}\n"
            f"Available classes: {list(ASSET_CLASSES.keys())}"
        )

    # ── 2. Group tickers by CSV file ──────────────────────────────────────────
    file_to_tickers: dict[str, list[str]] = {}
    for ticker in tickers:
        cls = _ticker_to_class(ticker)
        csv_file = ASSET_CLASSES[cls]['csv_file']
        file_to_tickers.setdefault(csv_file, []).append(ticker)

    # ── 3. Load each CSV once; validate requested tickers are present ─────────
    frames: list[pd.DataFrame] = []
    for csv_file, file_tickers in file_to_tickers.items():
        path = cache_dir / csv_file
        # min_obs=0: keep all columns — we validate presence ourselves
        df, available = load_asset_prices(
            str(path),
            start_date=date_from or '1999-01-01',
            min_obs=0,
        )
        missing = [t for t in file_tickers if t not in available]
        if missing:
            raise ValueError(
                f"Tickers absent from {csv_file} (no data or all-NaN): {missing}"
            )
        frames.append(df[file_tickers])

    # ── 4. Date-align via inner join ──────────────────────────────────────────
    if len(frames) == 1:
        combined = frames[0]
    else:
        combined = pd.concat(frames, axis=1, join='inner')

    # ── 5. Apply date range ───────────────────────────────────────────────────
    if date_from:
        combined = combined.loc[date_from:]
    if date_to:
        combined = combined.loc[:date_to]

    combined = combined.dropna(how='all')

    # ── 6. Enforce column order (KeyError here is informative) ────────────────
    return combined[tickers]
