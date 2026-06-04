"""
core/signal.py — SpreadSignal: live signal state for a basket.

Wraps a Basket + price history and exposes current z-score, entry/exit
state, velocity, and trend volatility ratio (TVR) for the UI and journal.

Session state: none (stateless; re-constructed on each Streamlit interaction).

Register notes
--------------
Register item H — exit sign inversion.
  The LIVE exit check here (`abs(current_sd) < exit_sd`) is symmetric:
  it fires when the spread is within exit_sd of zero on either side.
  The BACKTEST exit check in engine/numba_core.py is sign-aware:
      side == -1 → exits when d <=  exit_sd
      side == +1 → exits when d >= -exit_sd
  Equivalently: `-side * d <= exit_sd` (see register item H in CLAUDE.md).
  A module-level assertion canary in engine/numba_core.py enforces the
  correct backtest formula on import. Do NOT change the backtest formula
  to match the symmetric live form — they serve different purposes.

Register item I — entry dislocation must use abs(d).
  If entry dislocation is ever added to production trade records, use
  `abs(d)` not `d`. Signed values from long/short trades cancel when
  averaged, producing a near-zero mean. See engine/numba_core.py canary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.basket import Basket

logger = logging.getLogger(__name__)


@dataclass
class SpreadSignal:
    """Live signal state for a spread basket over a price history.

    Computes the vol-normalised spread return series, z-score distance from
    rolling mean, velocity, and TVR. All derived series are built once in
    ``__post_init__`` and cached as instance attributes.

    Attributes:
        basket: The long/short leg definition.
        prices: Daily closing prices (DatetimeIndex, instrument columns).
            Must contain all instruments in ``basket.all_instruments``.
        vol_window: Rolling window in trading days for mean/std calculation.
            Defaults to 262 (~1 trading year).
        xing_sd: Entry threshold in standard deviations. Defaults to 2.0.
        exit_sd: Exit threshold in standard deviations. Defaults to 1.0.
            Confirmed optimums: commodities=0.5, equities=2.0.
    """

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

    def __post_init__(self) -> None:
        """Validate inputs and compute all derived signal series.

        Raises:
            TypeError: If ``prices`` is not a pandas DataFrame.
            ValueError: If ``prices`` is empty, or contains fewer rows than
                ``vol_window``, or does not contain all basket instruments.
        """
        if not isinstance(self.prices, pd.DataFrame):
            raise TypeError(
                f"prices must be a pandas DataFrame; got {type(self.prices).__name__}"
            )
        if self.prices.empty:
            raise ValueError("prices must be a non-empty DataFrame")
        if len(self.prices) < self.vol_window:
            logger.warning(
                "prices has %d rows but vol_window=%d; vol estimate may be unreliable",
                len(self.prices),
                self.vol_window,
            )

        missing = [i for i in self.basket.all_instruments
                   if i not in self.prices.columns]
        if missing:
            raise ValueError(f"Missing price columns: {missing}")

        logger.debug(
            "SpreadSignal: building %s, %d rows, vol_window=%d",
            self.basket,
            len(self.prices),
            self.vol_window,
        )

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
        """Compute the vol-normalised spread return series.

        Returns:
            Daily spread returns as a pd.Series indexed by date. Positive
            means the long basket outperformed the short basket.
        """
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
        """Compute the trend-volatility ratio (TVR).

        TVR = |OLS slope of cumulative spread| / daily vol.
        Consistent with _batch_scores() in engine/search.py.

        Returns:
            Non-negative float. Higher values indicate a stronger directional
            trend relative to spread volatility. Returns 0.0 if fewer than 2
            data points or if daily vol is zero.
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
        """Latest z-score distance of the spread from its rolling mean.

        Returns:
            Signed float in standard deviation units. Positive means the long
            leg is expensive relative to the short leg.
        """
        return float(self.distance_sd.iloc[-1])

    @property
    def is_long_signal(self) -> bool:
        """True when current z-score is below the long-entry threshold.

        Returns:
            True if ``current_sd < -xing_sd`` (spread is unusually cheap).
        """
        return self.current_sd < -self.xing_sd

    @property
    def is_short_signal(self) -> bool:
        """True when current z-score is above the short-entry threshold.

        Returns:
            True if ``current_sd > xing_sd`` (spread is unusually expensive).
        """
        return self.current_sd > self.xing_sd

    @property
    def is_exit_signal(self) -> bool:
        """True when the spread has reverted to within exit_sd of zero.

        Note: this is the symmetric live-signal check. The backtest uses a
        sign-aware form; see register item H in CLAUDE.md and the module
        docstring above.

        Returns:
            True if ``abs(current_sd) < exit_sd``.
        """
        return abs(self.current_sd) < self.exit_sd

    @property
    def signal_state(self) -> str:
        """Current entry/exit signal as a string label.

        Evaluated in priority order: entry signals take precedence over exit.

        Returns:
            One of ``'LONG_ENTRY'``, ``'SHORT_ENTRY'``, ``'EXIT'``, or ``'NONE'``.
        """
        if self.current_sd < -self.xing_sd:
            return 'LONG_ENTRY'
        if self.current_sd > self.xing_sd:
            return 'SHORT_ENTRY'
        if abs(self.current_sd) < self.exit_sd:
            return 'EXIT'
        return 'NONE'

    def signal_history(self, n_days: int = 262) -> pd.DataFrame:
        """Return the last ``n_days`` of signal metrics as a DataFrame.

        Args:
            n_days: Number of most-recent trading days to return.
                Defaults to 262 (~1 year).

        Returns:
            DataFrame with columns ``spread_ret``, ``cum_spread``,
            ``distance_sd``, and ``velocity``, indexed by date.
        """
        return pd.DataFrame({
            'spread_ret':  self.spread_ret,
            'cum_spread':  self.cum_spread,
            'distance_sd': self.distance_sd,
            'velocity':    self.velocity,
        }).tail(n_days)

    def chart_data(self) -> dict:
        """Return serialisable data for the spread chart widget.

        Returns:
            Dict with keys ``dates``, ``cum_spread``, ``distance_sd``,
            ``velocity``, ``current_sd``, ``signal_state``, and ``tvr``.
            All list values are JSON-safe Python scalars.
        """
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
