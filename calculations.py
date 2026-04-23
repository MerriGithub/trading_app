import pandas as pd
import numpy as np
from scipy import stats

from config import PARAMS, ACTIVE_INSTRUMENTS

_TDY = PARAMS['trading_days_per_year']
_VOL_WIN = PARAMS['vol_calc_days']
_TARGET_VOL = PARAMS['target_daily_vol']


# ── Returns & Volatility ─────────────────────────────────────────────────────

def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how='all')


def rolling_volatility(rets: pd.DataFrame, window: int = _VOL_WIN) -> pd.DataFrame:
    """Rolling daily standard deviation of returns (not annualised)."""
    return rets.rolling(window, min_periods=window // 2).std()


def scaling_vectors(prices: pd.DataFrame, rets: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Volatility-normalised position scaling.
    scaling[i] = target_daily_vol / rolling_daily_vol[i], capped at 1.0.

    Matches the spreadsheet's 'Scaling Used' row:
    e.g. UKX daily vol ≈ 1.15% → scaling = 1%/1.15% ≈ 87%.
    """
    if rets is None:
        rets = returns(prices)
    vol = rolling_volatility(rets)
    scaling = (_TARGET_VOL / vol.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    return scaling


# ── Portfolio Returns ────────────────────────────────────────────────────────

def portfolio_returns(
    rets: pd.DataFrame,
    scaling: pd.DataFrame,
    long_flags: dict,
    short_flags: dict,
) -> pd.Series:
    """
    Daily spread return = long basket return - short basket return.
    Each leg is equally weighted across its selected instruments,
    with position sizes scaled by the vol-normalised scaling vector.
    """
    instruments = [i for i in ACTIVE_INSTRUMENTS if i in rets.columns]

    def leg(flags: dict) -> pd.Series:
        active = [k for k in instruments if flags.get(k, 0)]
        if not active:
            return pd.Series(0.0, index=rets.index)
        r = rets[active]
        s = scaling[active]
        weighted = (r * s).mean(axis=1)   # equal weight, vol-scaled
        return weighted

    return leg(long_flags) - leg(short_flags)


def rolling_nd_returns(spread: pd.Series, max_n: int = 5) -> pd.DataFrame:
    """Cumulative rolling returns over 1–5 days."""
    return pd.DataFrame({f'{n}D': spread.rolling(n).sum() for n in range(1, max_n + 1)})


# ── Trend Analysis ───────────────────────────────────────────────────────────

def linear_trend(series: pd.Series, n_points: int | None = None) -> dict:
    """
    OLS fit over the last n_points of series.
    Returns slope (velocity), r², and a Series of fitted values.
    """
    if n_points is None:
        n_points = PARAMS['linear_fit_points']
    tail = series.dropna().tail(n_points)
    if len(tail) < 3:
        return {'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'fitted': None}
    x = np.arange(len(tail), dtype=float)
    slope, intercept, r, _, _ = stats.linregress(x, tail.values)
    fitted = pd.Series(intercept + slope * x, index=tail.index)
    return {'slope': slope, 'intercept': intercept, 'r2': r ** 2, 'fitted': fitted}


def roc(series: pd.Series, days: int | None = None) -> pd.Series:
    """Rate of change over N days."""
    if days is None:
        days = PARAMS['roc_days']
    return series.pct_change(days)


def velocity_acceleration(spread_ret: pd.Series, roc_days: int | None = None) -> pd.DataFrame:
    """
    Velocity  = ROC of cumulative spread over roc_days.
    Acceleration = day-on-day change in velocity.
    """
    if roc_days is None:
        roc_days = PARAMS['roc_days']
    cum = (1 + spread_ret).cumprod()
    vel = roc(cum, roc_days)
    return pd.DataFrame({'velocity': vel, 'acceleration': vel.diff()})


# ── Crossing Signals ─────────────────────────────────────────────────────────

def crossing_signals(
    spread_ret: pd.Series,
    tolerance_sd: float | None = None,
    window: int = _VOL_WIN,
) -> pd.DataFrame:
    """
    Tracks where the cumulative spread is relative to its rolling mean,
    expressed in standard deviations.  Signal fires when |dist| > tolerance_sd.
    """
    if tolerance_sd is None:
        tolerance_sd = PARAMS['xing_tolerance_sd']
    cum = (1 + spread_ret).cumprod()
    roll_mean = cum.rolling(window, min_periods=window // 2).mean()
    roll_std = cum.rolling(window, min_periods=window // 2).std()
    dist = (cum - roll_mean) / roll_std.replace(0, np.nan)
    return pd.DataFrame({
        'cumulative': cum,
        'distance_sd': dist,
        'signal': dist.abs() > tolerance_sd,
    })


# ── Portfolio Statistics ──────────────────────────────────────────────────────

def correlation_matrix(rets: pd.DataFrame, window: int = _VOL_WIN) -> pd.DataFrame:
    return rets.tail(window).corr()


def intraday_spread(
    intraday_prices: pd.DataFrame,
    pivot_prices: pd.Series,
    scalings_latest: pd.Series,
    long_flags: dict,
    short_flags: dict,
) -> pd.Series:
    """
    Cumulative intraday spread return from pivot_prices (yesterday's close).
    Returns a Series indexed by timestamp.
    """
    instruments = [
        i for i in intraday_prices.columns
        if i in pivot_prices.index and pd.notna(pivot_prices[i]) and pivot_prices[i] != 0
    ]
    n_long  = sum(long_flags.get(i, 0)  for i in instruments)
    n_short = sum(short_flags.get(i, 0) for i in instruments)
    if n_long == 0 or n_short == 0:
        return pd.Series(dtype=float)

    intra_rets = (intraday_prices[instruments] - pivot_prices[instruments]) / pivot_prices[instruments]

    long_w  = pd.Series({i: scalings_latest.get(i, 1.0) * long_flags.get(i, 0)  / n_long  for i in instruments})
    short_w = pd.Series({i: scalings_latest.get(i, 1.0) * short_flags.get(i, 0) / n_short for i in instruments})

    return (intra_rets * long_w).sum(axis=1) - (intra_rets * short_w).sum(axis=1)


def portfolio_stats(port_ret: pd.Series) -> dict:
    if port_ret.empty or port_ret.isna().all():
        nan = float('nan')
        return {k: nan for k in ('total_return', 'max_return', 'min_return',
                                 'mean_daily', 'vol_daily', 'sharpe', 'last')}
    cum = (1 + port_ret).cumprod() - 1
    vol = port_ret.std()
    return {
        'total_return': cum.iloc[-1],
        'max_return':   cum.max(),
        'min_return':   cum.min(),
        'mean_daily':   port_ret.mean(),
        'vol_daily':    vol,
        'sharpe':       port_ret.mean() / vol * np.sqrt(_TDY) if vol > 0 else float('nan'),
        'last':         port_ret.iloc[-1],
    }
