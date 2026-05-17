from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataRegistry:
    def __init__(self, cache_dir: Path):
        self._cache = Path(cache_dir)

    def get_daily_prices(
        self,
        instruments: list[str],
        start_date: str = '1999-01-01',
    ) -> pd.DataFrame:
        """
        Aligned daily prices for instruments.
        Routes each instrument by data_source from asset_configs.
        Inner-joins on dates — only common trading days are kept.
        """
        from asset_configs import ASSET_CLASSES, _KEY_ALIASES, get_data_source

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
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = df[df.index >= pd.Timestamp(start_date)]
        df = df.dropna(how='all')
        return df

    def _load_csv_daily(self, instrument: str, start_date: str) -> pd.Series | None:
        """Load a single instrument from its asset class CSV."""
        from asset_configs import ASSET_CLASSES
        for cfg in ASSET_CLASSES.values():
            if instrument in cfg.get('instruments', {}):
                csv_path = self._cache / cfg['csv_file']
                if not csv_path.exists():
                    return None
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if instrument in df.columns:
                    return df[instrument].dropna()
                return None
        return None

    def _load_yahoo_daily(self, instrument: str, start_date: str) -> pd.Series | None:
        """Fetch daily close prices from Yahoo Finance."""
        import yfinance as yf
        from asset_configs import get_intraday_ticker, ASSET_CLASSES

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
                return None
            close = raw['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = instrument
            return close
        except Exception:
            return None

    def get_latest_prices(self, instruments: list[str]) -> dict[str, float]:
        """Most recent close per instrument. Required by Basket.spread_cost()."""
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
        prices = self.get_daily_prices(instruments)
        if prices.empty:
            return {}
        rets = prices.pct_change().dropna(how='all')
        vols = rets.rolling(window, min_periods=window // 2).std().iloc[-1]
        return vols.dropna().to_dict()

    def get_scalings(
        self,
        instruments: list[str],
        target_vol: float = 0.01,
    ) -> dict[str, float]:
        vols = self.get_vols(instruments)
        return {
            inst: (target_vol / v if v > 0 else 1.0)
            for inst, v in vols.items()
        }

    def get_intraday(
        self,
        instruments: list[str],
        interval: str = '5m',
    ) -> pd.DataFrame | None:
        """
        Fetch intraday prices for instruments that have an intraday_ticker.
        Returns None if no instruments have intraday data available.
        Partial results valid — returns what is available.
        """
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

        return close

    def refresh(self, instruments: list[str] | None = None) -> None:
        raise NotImplementedError('Deferred to Sprint 2')
