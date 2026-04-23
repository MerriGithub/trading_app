from pathlib import Path
import pandas as pd
import yfinance as yf
import streamlit as st

from config import INSTRUMENTS, ACTIVE_INSTRUMENTS

CACHE_DIR = Path(__file__).parent / 'cache'


def _cache_path() -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / 'prices.csv'


@st.cache_data(ttl=3600)
def load_prices(start_date: str = '1999-01-01') -> pd.DataFrame:
    cache_file = _cache_path()
    today = pd.Timestamp.today().normalize()

    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        last = cached.index[-1]
        if (today - last).days <= 3:
            return cached
        # Incremental update
        new = _fetch((last + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        if new is not None and not new.empty:
            combined = pd.concat([cached, new])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.to_csv(cache_file)
            return combined
        return cached

    df = _fetch(start_date)
    if df is not None:
        df.to_csv(cache_file)
    return df


def _fetch(start_date: str) -> pd.DataFrame | None:
    tickers = list(INSTRUMENTS.values())
    try:
        raw = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)
        # yfinance returns MultiIndex columns when multiple tickers
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


def force_refresh() -> None:
    for f in CACHE_DIR.glob('prices.*'):
        f.unlink()
    st.cache_data.clear()


@st.cache_data(ttl=300)
def load_intraday_prices(interval: str = '5m') -> pd.DataFrame | None:
    """Today's intraday prices at the given interval (cached 5 minutes)."""
    tickers = list(INSTRUMENTS.values())
    try:
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
    load_intraday_prices.clear()


def get_latest_prices(prices: pd.DataFrame) -> pd.Series:
    return prices.iloc[-1]
