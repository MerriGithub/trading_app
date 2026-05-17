from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from core.basket import Basket


@dataclass
class SpreadSignal:
    basket:     Basket
    prices:     pd.DataFrame
    vol_window: int   = 262
    xing_sd:    float = 2.0
    exit_sd:    float = 1.0

    # Computed in __post_init__
    spread_ret:  pd.Series = field(init=False, repr=False)
    cum_spread:  pd.Series = field(init=False, repr=False)
    distance_sd: pd.Series = field(init=False, repr=False)
    velocity:    pd.Series = field(init=False, repr=False)
    tvr:         float     = field(init=False)

    def __post_init__(self):
        missing = [i for i in self.basket.all_instruments
                   if i not in self.prices.columns]
        if missing:
            raise ValueError(f'Missing price columns: {missing}')

        self.spread_ret  = self._compute_spread_ret()
        self.cum_spread  = (1 + self.spread_ret).cumprod()

        from engine.calculations import crossing_signals, velocity_acceleration
        xing = crossing_signals(
            self.spread_ret,
            tolerance_sd=self.xing_sd,
            window=self.vol_window,
        )
        self.distance_sd = xing['distance_sd']

        va = velocity_acceleration(self.spread_ret)
        self.velocity = va['velocity']

        self.tvr = self._compute_tvr()

    def _compute_spread_ret(self) -> pd.Series:
        from engine.backtest import prepare_returns
        instr  = self.basket.all_instruments
        scaled, _, index = prepare_returns(
            self.prices[instr], instr, vol_window=self.vol_window
        )
        col       = {inst: i for i, inst in enumerate(instr)}
        long_ret  = scaled[:, [col[i] for i in self.basket.long_legs]].mean(axis=1)
        short_ret = scaled[:, [col[i] for i in self.basket.short_legs]].mean(axis=1)
        return pd.Series(long_ret - short_ret, index=index)

    def _compute_tvr(self) -> float:
        """
        |OLS slope of cumulative spread| / daily vol.
        Consistent with _batch_scores() in engine/search.py.
        """
        cum = self.cum_spread.values
        T = len(cum)
        if T < 2:
            return 0.0
        x = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
        xx = (x ** 2).sum()
        cum_mean = cum.mean()
        slope = (x * (cum - cum_mean)).sum() / xx
        std = self.spread_ret.std()
        return float(abs(slope) / std) if std > 0 else 0.0

    @property
    def current_sd(self) -> float:
        return float(self.distance_sd.iloc[-1])

    @property
    def is_long_signal(self) -> bool:
        return self.current_sd < -self.xing_sd

    @property
    def is_short_signal(self) -> bool:
        return self.current_sd > self.xing_sd

    @property
    def is_exit_signal(self) -> bool:
        return abs(self.current_sd) < self.exit_sd

    @property
    def signal_state(self) -> str:
        """'LONG_ENTRY' | 'SHORT_ENTRY' | 'EXIT' | 'NONE'"""
        if self.current_sd < -self.xing_sd:
            return 'LONG_ENTRY'
        if self.current_sd > self.xing_sd:
            return 'SHORT_ENTRY'
        if abs(self.current_sd) < self.exit_sd:
            return 'EXIT'
        return 'NONE'

    def signal_history(self, n_days: int = 262) -> pd.DataFrame:
        """Last n_days of signal metrics as a DataFrame."""
        return pd.DataFrame({
            'spread_ret':  self.spread_ret,
            'cum_spread':  self.cum_spread,
            'distance_sd': self.distance_sd,
            'velocity':    self.velocity,
        }).tail(n_days)

    def chart_data(self) -> dict:
        """Serialisable data for the spread chart."""
        hist = self.signal_history()
        return {
            'dates':       hist.index.strftime('%Y-%m-%d').tolist(),
            'cum_spread':  hist['cum_spread'].tolist(),
            'distance_sd': hist['distance_sd'].tolist(),
            'velocity':    hist['velocity'].tolist(),
            'current_sd':  self.current_sd,
            'signal_state': self.signal_state,
            'tvr':         self.tvr,
        }
