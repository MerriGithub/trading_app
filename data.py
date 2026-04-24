from pathlib import Path
import pandas as pd
import yfinance as yf
import streamlit as st

from config import INSTRUMENTS, ACTIVE_INSTRUMENTS

# --- Cache Setup ---
# Daily price data is stored as a CSV next to this file to avoid re-downloading on every run
CACHE_DIR = Path(__file__).parent / 'cache'


def _cache_path() -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / 'prices.csv'


# --- Historical Daily Prices ---

@st.cache_data(ttl=3600)
def load_prices(start_date: str = '1999-01-01') -> pd.DataFrame:
    cache_file = _cache_path()
    today = pd.Timestamp.today().normalize()

    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        last = cached.index[-1]
        # Cache is fresh enough — skip the network call entirely
        if (today - last).days <= 3:
            return cached
        # Incremental update: only fetch the days we're missing
        new = _fetch((last + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        if new is not None and not new.empty:
            combined = pd.concat([cached, new])
            # Drop any accidental duplicate dates (e.g. if cache boundary overlaps)
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.to_csv(cache_file)
            return combined
        return cached

    # No cache at all — full download from start_date
    df = _fetch(start_date)
    if df is not None:
        df.to_csv(cache_file)
    return df


def _fetch(start_date: str) -> pd.DataFrame | None:
    """Download close prices from Yahoo Finance and rename columns to internal codes."""
    tickers = list(INSTRUMENTS.values())
    try:
        raw = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)
        # yfinance returns MultiIndex columns when multiple tickers are requested
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw['Close']
        else:
            raw = raw[['Close']] if 'Close' in raw.columns else raw

        ticker_to_label = {v: k for k, v in INSTRUMENTS.items()}
        df = raw.rename(columns=ticker_to_label)

        # Keep only known instruments in consistent order
        cols = [c for c in ACTIVE_INSTRUMENTS if c in df.columns]
        df = df[cols].ffill().dropna(how='all')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None


# --- Cache Management ---

def force_refresh() -> None:
    """Delete the local CSV cache and clear Streamlit's in-memory cache, forcing a full re-download."""
    for f in CACHE_DIR.glob('prices.*'):
        f.unlink()
    st.cache_data.clear()


# --- Intraday Prices ---

@st.cache_data(ttl=300)
def load_intraday_prices(interval: str = '5m') -> pd.DataFrame | None:
    """Today's intraday prices at the given interval (cached 5 minutes)."""
    tickers = list(INSTRUMENTS.values())
    try:
        # period='2d' ensures we catch pre-market and the previous session for pivot calculation
        raw = yf.download(tickers, period='2d', interval=interval,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw['Close']
        elif 'Close' in raw.columns:
            raw = raw[['Close']]
        ticker_to_label = {v: k for k, v in INSTRUMENTS.items()}
        df = raw.rename(columns=ticker_to_label)
        cols = [c for c in ACTIVE_INSTRUMENTS if c in df.columns]
        df = df[cols].ffill().dropna(how='all')
        return df
    except Exception:
        return None


def force_intraday_refresh() -> None:
    """Clear only the intraday cache, leaving daily prices intact."""
    load_intraday_prices.clear()


def get_latest_prices(prices: pd.DataFrame) -> pd.Series:
    return prices.iloc[-1]
