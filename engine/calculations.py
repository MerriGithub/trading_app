import pandas as pd
import numpy as np
from scipy import stats

from config import PARAMS, ACTIVE_INSTRUMENTS

# Shorthand aliases for frequently-used params
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
    # Replace zero vol with NaN to avoid divide-by-zero, then clip so we never over-size
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
        weighted = (r * s).mean(axis=1)   # equal weight across selected instruments, vol-scaled
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
    # Normalise distance so it's comparable across different spread levels
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
    # Only use instruments that have a valid pivot price to calculate returns from
    instruments = [
        i for i in intraday_prices.columns
        if i in pivot_prices.index and pd.notna(pivot_prices[i]) and pivot_prices[i] != 0
    ]
    n_long  = sum(long_flags.get(i, 0)  for i in instruments)
    n_short = sum(short_flags.get(i, 0) for i in instruments)
    if n_long == 0 or n_short == 0:
        return pd.Series(dtype=float)

    # Return from yesterday's close to each intraday bar
    intra_rets = (intraday_prices[instruments] - pivot_prices[instruments]) / pivot_prices[instruments]

    # Equal-weight each side, scaled by latest vol scaling
    long_w  = pd.Series({i: scalings_latest.get(i, 1.0) * long_flags.get(i, 0)  / n_long  for i in instruments})
    short_w = pd.Series({i: scalings_latest.get(i, 1.0) * short_flags.get(i, 0) / n_short for i in instruments})

    return (intra_rets * long_w).sum(axis=1) - (intra_rets * short_w).sum(axis=1)


def compute_contraction_betas(rets: pd.DataFrame, window: int = _VOL_WIN) -> pd.Series:
    """
    Beta of each instrument vs the equal-weight market return.
    Used as a contraction factor indicator — values > 1 mean the instrument
    amplifies market moves, < 1 means it dampens them.
    """
    instruments = [i for i in ACTIVE_INSTRUMENTS if i in rets.columns]
    if not instruments:
        return pd.Series(dtype=float)
    recent = rets[instruments].tail(window)
    mkt = recent.mean(axis=1)           # equal-weight proxy for the market
    mkt_var = float(mkt.var())
    if mkt_var == 0:
        return pd.Series(1.0, index=instruments)
    return pd.Series({i: float(recent[i].cov(mkt)) / mkt_var for i in instruments})


def crossing_signal_backtest(
    spread_ret: pd.Series,
    tolerance_sd: float | None = None,
    exit_sd: float = 0.0,
    window: int = _VOL_WIN,
) -> pd.DataFrame:
    """
    Run the crossing signal backtest on a spread return series.

    Returns a DataFrame with one row per trade.  Does not modify
    crossing_signals() — this is an additive wrapper only.
    """
    from numba_core import backtest_spread, COL_ENTRY_IDX, COL_EXIT_IDX, \
        COL_SIDE, COL_GROSS_RETURN, COL_HOLDING_DAYS

    if tolerance_sd is None:
        tolerance_sd = float(PARAMS['xing_tolerance_sd'])

    arr      = spread_ret.dropna().values.astype(np.float64)
    idx      = spread_ret.dropna().index
    day_ints = ((idx - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).values.astype(np.int64)

    trades, n_trades, cum, dist_sd = backtest_spread(
        arr, window, tolerance_sd, exit_sd, day_ints
    )

    if n_trades == 0:
        return pd.DataFrame(columns=[
            'entry_date', 'exit_date', 'side', 'gross_return', 'holding_days',
        ])

    t = trades[:n_trades]
    return pd.DataFrame({
        'entry_date':   [idx[int(t[i, COL_ENTRY_IDX])] for i in range(n_trades)],
        'exit_date':    [idx[int(t[i, COL_EXIT_IDX])]  for i in range(n_trades)],
        'side':         ['long' if t[i, COL_SIDE] == 1 else 'short' for i in range(n_trades)],
        'gross_return': t[:, COL_GROSS_RETURN],
        'holding_days': t[:, COL_HOLDING_DAYS].astype(int),
    })


def portfolio_stats(port_ret: pd.Series) -> dict:
    """Summary statistics for a daily return series, including annualised Sharpe."""
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
