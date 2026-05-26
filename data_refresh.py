"""
data_refresh.py — incremental CSV price data updater
No Streamlit imports; safe to call from tests and background processes.
"""
from __future__ import annotations

import datetime
import os

import pandas as pd
import yfinance as yf

# ── Ticker mappings: internal code → Yahoo Finance ticker ────────────────────

EQUITY_TICKERS: dict[str, str] = {
    'UKX': '^FTSE',
    'CBK': '^FCHI',
    'CEY': 'FTSEMIB.MI',
    'CFR': '^GDAXI',
    'CMD': '^IBEX',
    'CEI': '^STOXX50E',
    'COI': '^SSMI',
    'CRM': '^HSI',
    'CIL': '^AXJO',
    'CPH': '^NDX',
    'CTN': '^GSPC',
    'CTB': '^DJI',
}

FX_TICKERS: dict[str, str] = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X',
    'AUDUSD': 'AUDUSD=X',
    'NZDUSD': 'NZDUSD=X',
    'USDCAD': 'USDCAD=X',
    'EURGBP': 'EURGBP=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'EURCHF': 'EURCHF=X',
    'AUDNZD': 'AUDNZD=X',
}

COMMODITY_TICKERS: dict[str, str] = {
    'WTI':       'CL=F',
    'BRENT':     'BZ=F',
    'NATGAS':    'NG=F',
    'GOLD':      'GC=F',
    'SILVER':    'SI=F',
    'PLATINUM':  'PL=F',
    'COPPER':    'HG=F',
    'PALLADIUM': 'PA=F',
    'WHEAT':     'ZW=F',
    'CORN':      'ZC=F',
    'SOYBEANS':  'ZS=F',
    'COFFEE':    'KC=F',
    'SUGAR':     'SB=F',
}

FILE_CONFIG: dict[str, dict[str, str]] = {
    'prices.csv':           EQUITY_TICKERS,
    'fx_prices.csv':        FX_TICKERS,
    'commodity_prices.csv': COMMODITY_TICKERS,
}

_CACHE_DIR: str = os.path.join(os.path.dirname(__file__), 'cache')


# ── Staleness helpers ─────────────────────────────────────────────────────────

def _last_trading_date(
    _now: datetime.datetime | None = None,
) -> datetime.date:
    """
    Last date the market should have data for.
    Before 18:00 UTC → previous day (non-final close); rolls back over weekends.
    Accepts an optional _now for deterministic testing.
    """
    if _now is None:
        _now = datetime.datetime.now(datetime.timezone.utc)
    d = _now.date()
    if _now.hour < 18:
        d -= datetime.timedelta(days=1)
    while d.weekday() >= 5:  # Saturday=5, Sunday=6
        d -= datetime.timedelta(days=1)
    return d


def get_file_last_date(filename: str) -> datetime.date | None:
    """Return the most recent date in the cached CSV, or None if missing."""
    path = os.path.join(_CACHE_DIR, filename)
    if not os.path.exists(path):
        return None
    # usecols=[0] reads only the Date column as the index; check len not .empty
    # (.empty is True when there are no columns, even if there are rows)
    df = pd.read_csv(path, index_col=0, parse_dates=True, usecols=[0])
    if len(df.index) == 0:
        return None
    return df.index[-1].date()


def is_stale(filename: str, threshold_days: int = 1) -> bool:
    """True when the gap between last cached date and last trading date > threshold."""
    last = get_file_last_date(filename)
    if last is None:
        return True
    gap = (_last_trading_date() - last).days
    return gap > threshold_days


def any_file_stale() -> bool:
    return any(is_stale(f) for f in FILE_CONFIG)


def staleness_summary() -> dict[str, datetime.date | None]:
    return {f: get_file_last_date(f) for f in FILE_CONFIG}


# ── Fetch / update ────────────────────────────────────────────────────────────

def _fetch_new_rows(
    tickers: dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """
    Download close prices from Yahoo Finance for the given date range.
    end_date is passed directly to yfinance (which treats it as exclusive).
    Returns a DataFrame with internal-code columns, or None on failure.
    """
    yahoo_tickers = list(tickers.values())
    if not yahoo_tickers:
        return None
    try:
        raw = yf.download(
            yahoo_tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw['Close']
        elif 'Close' in raw.columns:
            raw = raw[['Close']]

        yahoo_to_code = {v: k for k, v in tickers.items()}
        df = raw.rename(columns=yahoo_to_code)

        cols = [c for c in tickers if c in df.columns]
        if not cols:
            return None
        df = df[cols].ffill(limit=3).dropna(how='all')

        # Normalise index to tz-naive date
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx.normalize()

        return df if not df.empty else None
    except Exception:
        return None


def update_file(filename: str) -> dict:
    """
    Append any missing rows to the cached CSV.
    Returns a result dict with keys: filename, status, rows_added, last_date.
    """
    tickers = FILE_CONFIG[filename]
    path = os.path.join(_CACHE_DIR, filename)

    cached: pd.DataFrame | None = None
    if os.path.exists(path):
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        start_date = (cached.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = '1999-01-01'

    target = _last_trading_date()
    target_str = target.strftime('%Y-%m-%d')

    if start_date > target_str:
        return {'filename': filename, 'status': 'up_to_date', 'rows_added': 0}

    # yfinance end is exclusive → add 1 day
    end_str = (target + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    new_rows = _fetch_new_rows(tickers, start_date, end_str)

    if new_rows is None or new_rows.empty:
        return {'filename': filename, 'status': 'no_new_data', 'rows_added': 0}

    if cached is not None:
        combined = pd.concat([cached, new_rows])
        combined = combined[~combined.index.duplicated(keep='last')]
        orig_cols = list(cached.columns)
        extra_cols = [c for c in new_rows.columns if c not in orig_cols]
        combined = combined[orig_cols + extra_cols]
    else:
        combined = new_rows
        cols = [c for c in tickers if c in combined.columns]
        combined = combined[cols]

    os.makedirs(_CACHE_DIR, exist_ok=True)
    combined.to_csv(path)

    return {
        'filename':  filename,
        'status':    'updated',
        'rows_added': len(new_rows),
        'last_date': str(combined.index[-1].date()),
    }


def refresh_all() -> list[dict]:
    """Update all three price files. Returns list of result dicts."""
    return [update_file(f) for f in FILE_CONFIG]
