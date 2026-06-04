"""
core/data_registry.py — DataRegistry: price data access layer.

Loads daily price data from the ``cache/`` CSVs and, for instruments with
an ``intraday_ticker``, fetches live intraday data from Yahoo Finance.

Singletons are constructed once via ``@st.cache_resource`` in tabs/shared.py.

Cache file conventions
----------------------
Prices are stored in asset-class-specific CSVs under ``cache/``:
    prices.csv           — equity indices (12 instruments)
    fx_prices.csv        — FX pairs (12 pairs)
    commodity_prices.csv — commodities (WTI excluded from pair gen)
    fi_prices.csv        — fixed income ETFs (IBTM excluded from pair gen)

Data quality notes
------------------
- Daily prices: ``ffill(limit=3)`` to bridge short holiday gaps.
- Intraday prices: no forward-fill (stale prices would cross market sessions).
- Inner-join on dates — only common trading days across instruments are kept.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataRegistry:
    """Price data access layer backed by cache CSVs and Yahoo Finance.

    Args:
        cache_dir: Directory containing the asset-class CSV files.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache = Path(cache_dir)

    def get_daily_prices(
        self,
        instruments: list[str],
        start_date: str = '1999-01-01',
    ) -> pd.DataFrame:
        """Return aligned daily closing prices for the given instruments.

        Routes each instrument to its data source (Yahoo Finance live fetch
        or CSV cache) based on ``asset_configs.get_data_source()``. The
        result is inner-joined on dates so only common trading days are
        returned.

        Args:
            instruments: List of instrument codes (e.g. ``['BRENT', 'GOLD']``).
                Must be non-empty.
            start_date: ISO date string; rows before this date are excluded.
                Defaults to ``'1999-01-01'``.

        Returns:
            DataFrame with a DatetimeIndex and one column per instrument.
            Returns an empty DataFrame if no price data is available.

        Raises:
            ValueError: If ``instruments`` is empty.
        """
        if not instruments:
            raise ValueError(
                "instruments must be a non-empty list; got empty list"
            )

        from asset_configs import get_data_source

        frames: dict[str, pd.Series] = {}

        for inst in instruments:
            source = get_data_source(inst)
            try:
                if source == 'yahoo':
                    s = self._load_yahoo_daily(inst, start_date)
                else:
                    s = self._load_csv_daily(inst, start_date)
                if s is not None and not s.empty:
                    frames[inst] = s
                else:
                    logger.warning("No price data available for %s (source=%s)", inst, source)
            except Exception as exc:
                logger.error("Failed to load prices for %s: %s", inst, exc)

        if not frames:
            logger.warning(
                "get_daily_prices: no data loaded for instruments %s", instruments
            )
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = df[df.index >= pd.Timestamp(start_date)]
        df = df.dropna(how='all')
        logger.debug(
            "get_daily_prices: %d rows × %d instruments from %s",
            len(df), len(df.columns), start_date,
        )
        return df

    def _load_csv_daily(self, instrument: str, start_date: str) -> Optional[pd.Series]:
        """Load a single instrument's prices from its asset-class CSV.

        Args:
            instrument: Instrument code (e.g. ``'BRENT'``).
            start_date: ISO date string for the earliest row to return.

        Returns:
            Price series indexed by date, or ``None`` if not found.
        """
        from asset_configs import ASSET_CLASSES
        for cfg in ASSET_CLASSES.values():
            if instrument in cfg.get('instruments', {}):
                csv_path = self._cache / cfg['csv_file']
                if not csv_path.exists():
                    logger.warning(
                        "_load_csv_daily: cache file not found: %s", csv_path
                    )
                    return None
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if instrument in df.columns:
                    return df[instrument].dropna()
                logger.warning(
                    "_load_csv_daily: %s not in columns of %s", instrument, csv_path
                )
                return None
        return None

    def _load_yahoo_daily(
        self, instrument: str, start_date: str
    ) -> Optional[pd.Series]:
        """Fetch daily close prices from Yahoo Finance for intraday-capable instruments.

        Args:
            instrument: Instrument code with an ``intraday_ticker`` configured
                in ``asset_configs``.
            start_date: ISO date string; earliest row to return.

        Returns:
            Price series indexed by date, or ``None`` if ticker is unconfigured
            or the download fails.
        """
        import yfinance as yf
        from asset_configs import ASSET_CLASSES

        ticker = None
        for cfg in ASSET_CLASSES.values():
            if instrument in cfg.get('instruments', {}):
                inst_cfg = cfg['instruments'][instrument]
                if isinstance(inst_cfg, dict):
                    ticker = inst_cfg.get('intraday_ticker')
                break

        if ticker is None:
            return None

        try:
            raw = yf.download(
                ticker,
                start=start_date,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                logger.warning("_load_yahoo_daily: empty response for ticker %s", ticker)
                return None
            close = raw['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = instrument
            return close
        except Exception as exc:
            logger.error("_load_yahoo_daily: download failed for %s: %s", ticker, exc)
            return None

    def get_latest_prices(self, instruments: list[str]) -> dict[str, float]:
        """Return the most recent close price per instrument.

        Required by ``Basket.spread_cost()``.

        Args:
            instruments: List of instrument codes.

        Returns:
            ``{instrument: latest_close}`` dict. Instruments with no data are
            absent from the result rather than mapped to NaN.
        """
        prices = self.get_daily_prices(instruments)
        if prices.empty:
            return {}
        latest = prices.iloc[-1].dropna()
        return latest.to_dict()

    def get_vols(
        self,
        instruments: list[str],
        window: int = 262,
    ) -> dict[str, float]:
        """Return the most recent rolling daily volatility per instrument.

        Args:
            instruments: List of instrument codes.
            window: Rolling window in trading days. Must be positive.
                Defaults to 262 (~1 year).

        Returns:
            ``{instrument: daily_vol}`` dict. Instruments with fewer than
            ``window // 2`` observations are absent (not enough data for a
            meaningful estimate).

        Raises:
            ValueError: If ``window`` is not positive.
        """
        if window <= 0:
            raise ValueError(
                f"window must be a positive integer; got {window!r}"
            )

        prices = self.get_daily_prices(instruments)
        if prices.empty:
            return {}
        rets = prices.pct_change().dropna(how='all')

        # Warn when any instrument has fewer rows than the full window —
        # min_periods=window//2 means a vol estimate is still returned, but
        # it is less reliable than a full-window estimate.
        short_instruments = [
            inst for inst in rets.columns if rets[inst].count() < window
        ]
        if short_instruments:
            logger.warning(
                "get_vols: instruments %s have fewer than window=%d observations; "
                "vol estimates may be unreliable",
                short_instruments, window,
            )

        vols = rets.rolling(window, min_periods=window // 2).std().iloc[-1]
        return vols.dropna().to_dict()

    def get_scalings(
        self,
        instruments: list[str],
        target_vol: float = 0.01,
        window: int = 262,
    ) -> dict[str, float]:
        """Return vol-scaling factors capped at 1.0.

        Scaling = min(1.0, target_vol / rolling_vol). A factor of 1.0 means
        the instrument is already at or below the target vol; no scaling up.

        Args:
            instruments: List of instrument codes.
            target_vol: Target daily volatility as a decimal (e.g. 0.01 = 1%).
                Must be positive; dividing by zero vol would produce inf.
            window: Rolling window passed to ``get_vols``.

        Returns:
            ``{instrument: scaling_factor}`` where 0 < factor <= 1.0.
            Instruments with zero vol get factor 1.0 (no data to scale on).

        Raises:
            ValueError: If ``target_vol`` is not positive.
        """
        if not (target_vol > 0):
            raise ValueError(
                f"target_vol must be positive; got {target_vol!r}. "
                f"Division by zero vol would produce infinite scaling factors."
            )

        vols = self.get_vols(instruments, window=window)
        return {
            inst: min(1.0, target_vol / v) if v > 0 else 1.0
            for inst, v in vols.items()
        }

    def get_intraday(
        self,
        instruments: list[str],
        interval: str = '5m',
    ) -> Optional[pd.DataFrame]:
        """Fetch intraday prices for instruments with an intraday_ticker.

        No forward-fill is applied — stale intraday prices would cross market
        sessions and produce misleading signals.

        Args:
            instruments: List of instrument codes.
            interval: Yahoo Finance interval string (e.g. ``'5m'``, ``'1h'``).

        Returns:
            DataFrame with intraday prices for the available subset of
            instruments, or ``None`` if no instruments have intraday data or
            the download fails. Partial results are valid — returns what is
            available.
        """
        try:
            import yfinance as yf
            from asset_configs import get_intraday_ticker

            tickers = {inst: get_intraday_ticker(inst) for inst in instruments}
            tradeable = {inst: t for inst, t in tickers.items() if t is not None}

            if not tradeable:
                return None

            raw = yf.download(
                list(tradeable.values()),
                period='1d',
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                return None

            ticker_to_inst = {v: k for k, v in tradeable.items()}
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw['Close'].rename(columns=ticker_to_inst)
            else:
                close = raw[['Close']].rename(columns={'Close': list(tradeable.keys())[0]})

            available = [c for c in close.columns if c in instruments]
            return close[available] if available else None
        except Exception as exc:
            logger.error("get_intraday failed: %s", exc)
            return None

    def refresh(self, instruments: Optional[list[str]] = None) -> None:
        """Append latest EOD data from Yahoo Finance to the cache CSVs.

        Appends new rows for all instruments (or the specified subset) since
        the last date already in each CSV. No-op if already up to date.
        Never raises — errors are logged and skipped per CSV file.

        Args:
            instruments: Instrument codes to refresh, or ``None`` to refresh
                all instruments in all asset classes.

        Returns:
            None
        """
        try:
            import yfinance as yf
            from asset_configs import ASSET_CLASSES

            # Group instruments by CSV file to minimise download calls.
            csv_groups: dict[str, dict] = {}
            for cfg in ASSET_CLASSES.values():
                csv_file = cfg.get('csv_file')
                if not csv_file:
                    continue
                for code, details in cfg.get('instruments', {}).items():
                    if instruments and code not in instruments:
                        continue
                    if not isinstance(details, dict):
                        continue
                    ticker = details.get('yahoo_ticker') or details.get('intraday_ticker')
                    if not ticker:
                        continue
                    csv_groups.setdefault(csv_file, {})[code] = ticker

            for csv_filename, ticker_map in csv_groups.items():
                csv_path = self._cache / csv_filename
                if not csv_path.exists():
                    logger.warning("refresh: cache file not found: %s", csv_path)
                    continue

                try:
                    existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    last_date = existing.index.max()
                    fetch_start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    today = pd.Timestamp.today().strftime('%Y-%m-%d')

                    if fetch_start >= today:
                        continue

                    tickers = list(ticker_map.values())
                    new_data = yf.download(
                        tickers,
                        start=fetch_start,
                        end=today,
                        progress=False,
                        auto_adjust=True,
                    )
                    if new_data.empty:
                        continue

                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_close = new_data['Close'].copy()
                    else:
                        new_close = new_data[['Close']].copy()

                    reverse_map = {v: k for k, v in ticker_map.items()}
                    new_close.rename(columns=reverse_map, inplace=True)
                    new_close.index.name = 'Date'

                    combined = pd.concat([existing, new_close])
                    combined = combined[~combined.index.duplicated(keep='first')]
                    combined.sort_index(inplace=True)
                    combined.to_csv(csv_path)
                    logger.info("refresh: updated %s with data from %s", csv_path, fetch_start)
                except Exception as exc:
                    logger.error("refresh: failed to update %s: %s", csv_path, exc)
        except Exception as exc:
            logger.error("refresh: unexpected error: %s", exc)
