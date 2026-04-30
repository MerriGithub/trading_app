"""
Portfolio search engine.

Enumerates all (long, short) instrument combinations within the specified
leg-count range, scores each on 5 metrics from the original spreadsheet's
SearchEngine sheet, and returns the top-ranked results.

Vectorised with numpy — 3-4 legs (~511K pairs) runs in ~2-5 seconds.
"""
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from config import ACTIVE_INSTRUMENTS, DISPLAY_NAMES, PARAMS
from numba_core import (
    batch_backtest,
    BR_N_TRADES, BR_GROSS_WR, BR_AVG_GROSS, BR_AVG_HOLDING, BR_PAYOFF_RATIO,
)
from scoring import apply_scoring, estimate_trade_cost

_TDY = PARAMS['trading_days_per_year']

# --- Metric Definitions ---
# Each tuple: (name, higher_is_better, spreadsheet_default_filter_limit)
METRICS = [
    ('ReturnSD',        False, -2.0),   # annualised Sharpe — filter out very negative
    ('TrendVolRatio',   True,   0.35),  # trend strength vs noise — higher = more trending
    ('ReturnTopology',  False, -3.0),   # skewness of returns — filter out severe negative skew
    ('FitDataMinMaxSD', True,  12.0),   # cumulative price range in SDs — higher = more movement
    ('LastSD',          True,   3.0),   # current distance from rolling mean — higher = further from mean
]
METRIC_NAMES = [m[0] for m in METRICS]


# ── Internals ────────────────────────────────────────────────────────────────

def _combo_matrix(n: int, k: int) -> tuple[np.ndarray, list[tuple]]:
    """All C(n,k) combos as an (M, n) equal-weight float matrix."""
    combos = list(combinations(range(n), k))
    mat = np.zeros((len(combos), n), dtype=np.float64)
    for i, idx in enumerate(combos):
        # Each row sums to 1.0 (equal weight across k legs)
        mat[i, list(idx)] = 1.0 / k
    return mat, combos


def _batch_scores(spread_mat: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute all 5 scoring metrics for a (T, M) matrix of spread return series.
    Returns dict of metric_name -> (M,) float array.
    """
    T, M = spread_mat.shape

    # ReturnSD — annualised Sharpe (mean / std * sqrt(trading_days))
    means = spread_mat.mean(axis=0)
    stds  = spread_mat.std(axis=0)
    return_sd = np.where(stds > 0, means / stds * np.sqrt(_TDY), 0.0)

    # Cumulative return series — used as the basis for several metrics below
    cum = np.cumprod(1.0 + spread_mat, axis=0)  # (T, M)

    # TrendVolRatio — |OLS slope of cumulative spread| / daily vol
    # High TVR means the spread is trending strongly relative to its noise
    x = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
    xx = (x ** 2).sum()
    cum_mean = cum.mean(axis=0)
    xy = (x[:, None] * (cum - cum_mean)).sum(axis=0)
    slopes = xy / xx
    tvr = np.where(stds > 0, np.abs(slopes) / stds, 0.0)

    # ReturnTopology — skewness of daily returns
    # Positive skew = more large upside moves; negative = fat left tail
    c  = spread_mat - means
    m2 = (c ** 2).mean(axis=0)
    m3 = (c ** 3).mean(axis=0)
    topology = np.where(m2 > 0, m3 / m2 ** 1.5, 0.0)

    # FitDataMinMaxSD — cumulative price range / trailing vol
    # Measures how much the spread has moved relative to recent noise
    price_range = cum.max(axis=0) - cum.min(axis=0)
    n_tail = min(T, 20)
    tail_std = spread_mat[-n_tail:].std(axis=0)
    fit_minmax_sd = np.where(
        tail_std > 0, price_range / (tail_std * np.sqrt(n_tail)), 0.0
    )

    # LastSD — current cumulative level vs rolling mean, expressed in SDs
    # High absolute value means the spread is far from its historical centre
    win = min(T, _TDY)
    roll_mean = cum[-win:].mean(axis=0)
    roll_std  = cum[-win:].std(axis=0)
    last_sd = np.where(roll_std > 0, (cum[-1] - roll_mean) / roll_std, 0.0)

    return {
        'ReturnSD':        return_sd,
        'TrendVolRatio':   tvr,
        'ReturnTopology':  topology,
        'FitDataMinMaxSD': fit_minmax_sd,
        'LastSD':          last_sd,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def estimate_combinations(
    min_long: int = 3,
    max_long: int = 4,
    min_short: int = 3,
    max_short: int = 4,
    n: int = 12,
    min_legs: int | None = None,
    max_legs: int | None = None,
) -> int:
    """Total (long, short) pairs for given leg ranges and instrument count."""
    if min_legs is not None:
        min_long = min_short = min_legs
    if max_legs is not None:
        max_long = max_short = max_legs
    m_long  = sum(comb(n, k) for k in range(min_long,  max_long  + 1))
    m_short = sum(comb(n, k) for k in range(min_short, max_short + 1))
    return m_long * m_short


def run_search(
    rets: pd.DataFrame,
    scalings: pd.DataFrame,
    min_long_legs: int = 3,
    max_long_legs: int = 4,
    min_short_legs: int = 3,
    max_short_legs: int = 4,
    window_days: int = _TDY,
    filters: dict | None = None,
    top_n: int = 30,
    progress_cb=None,
    xing_sd: float | None = None,
    exit_sd: float = 0.0,
    scoring_mode: str = 'composite',
    # Backward compatibility
    min_legs: int | None = None,
    max_legs: int | None = None,
) -> pd.DataFrame:
    """
    Enumerate and score all (long, short) combinations.

    Parameters
    ----------
    filters : dict of {metric: (direction, limit)}
        direction =  1 → include if metric >= limit
        direction = -1 → include if metric <= limit
    progress_cb : callable(float) | None
        Called with fraction complete (0–1) periodically.

    Returns
    -------
    DataFrame ranked by composite score, top_n rows.
    Columns: Config, Long, Short, ReturnSD, TrendVolRatio, ReturnTopology,
             FitDataMinMaxSD, LastSD, Trades, WinRate, Expectancy,
             AvgHolding, PayoffRatio, plus hidden _long_flags / _short_flags.
    """
    if min_legs is not None:
        min_long_legs = min_short_legs = min_legs
    if max_legs is not None:
        max_long_legs = max_short_legs = max_legs

    if xing_sd is None:
        xing_sd = float(PARAMS['xing_tolerance_sd'])

    instruments = [i for i in ACTIVE_INSTRUMENTS if i in rets.columns]
    N = len(instruments)

    # Align both series to the same window and pre-multiply returns by scaling
    window_df = rets[instruments].tail(window_days).dropna(how='any')
    r = window_df.to_numpy(dtype=np.float64)
    s = scalings[instruments].tail(window_days).dropna(how='any').to_numpy(dtype=np.float64)
    T = min(r.shape[0], s.shape[0])
    scaled = r[-T:] * s[-T:]  # (T, N) — vol-normalised returns

    # Day integers for batch_backtest (pandas-version-agnostic day count)
    day_ints = (
        (window_df.index[-T:] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    ).values.astype(np.int64)

    # Pre-compute vol-scaled leg returns for every combo size needed
    all_k = set(range(min_long_legs, max_long_legs + 1)) | set(range(min_short_legs, max_short_legs + 1))
    leg_cache: dict[int, tuple[np.ndarray, list]] = {}
    for k in all_k:
        mat, combos = _combo_matrix(N, k)
        leg_cache[k] = (scaled @ mat.T, combos)  # (T, M_k), list of tuples

    total = estimate_combinations(min_long_legs, max_long_legs, min_short_legs, max_short_legs, N)
    records: list[dict] = []
    done = 0

    # Outer loops: all long-side combo sizes × short-side combo sizes
    for n_long in range(min_long_legs, max_long_legs + 1):
        long_rets, long_combos = leg_cache[n_long]   # (T, M_long)

        for n_short in range(min_short_legs, max_short_legs + 1):
            short_rets, short_combos = leg_cache[n_short]  # (T, M_short)

            for l_i, long_combo in enumerate(long_combos):
                lr = long_rets[:, l_i]                   # (T,) — single long basket
                # Compute spread vs ALL short combos at once (vectorised batch)
                spread_mat = lr[:, None] - short_rets    # (T, M_short)
                batch = _batch_scores(spread_mat)
                bt_results = batch_backtest(spread_mat, window_days, xing_sd, exit_sd, day_ints)

                for s_i, short_combo in enumerate(short_combos):
                    done += 1

                    # Skip if any instrument appears on both sides — partial overlap
                    # causes those legs to cancel, inflating metrics for the remainder
                    if set(long_combo) & set(short_combo):
                        continue

                    sc = {k: float(v[s_i]) for k, v in batch.items()}

                    # Apply any active metric filters
                    if filters:
                        fail = False
                        for metric, (direction, limit) in filters.items():
                            val = sc.get(metric, 0.0)
                            if direction == 1 and val < limit:
                                fail = True; break
                            if direction == -1 and val > limit:
                                fail = True; break
                        if fail:
                            continue

                    long_names  = [DISPLAY_NAMES.get(instruments[i], instruments[i]) for i in long_combo]
                    short_names = [DISPLAY_NAMES.get(instruments[i], instruments[i]) for i in short_combo]

                    bt = bt_results[s_i]
                    avg_hold = float(bt[BR_AVG_HOLDING])
                    est_cost = estimate_trade_cost(avg_hold, n_long, n_short)
                    records.append({
                        'Config':         f'{n_long}v{n_short}',
                        'Long':           ' | '.join(long_names),
                        'Short':          ' | '.join(short_names),
                        '_long_flags':    {instruments[i]: 1 for i in long_combo},
                        '_short_flags':   {instruments[i]: 1 for i in short_combo},
                        'Trades':         int(bt[BR_N_TRADES]),
                        'WinRate':        float(bt[BR_GROSS_WR]),
                        'Expectancy':     float(bt[BR_AVG_GROSS]),
                        'NetExpectancy':  float(bt[BR_AVG_GROSS]) - est_cost,
                        'EstCost':        est_cost,
                        'AvgHolding':     avg_hold,
                        'PayoffRatio':    float(bt[BR_PAYOFF_RATIO]),
                        **sc,
                    })

                # Fire every ~2% of total combinations so the bar is responsive on small searches
                _tick = max(1, total // 50)
                if progress_cb and (done % _tick == 0 or done >= total):
                    progress_cb(min(done / total, 1.0))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = apply_scoring(df, scoring_mode)
    df = (df.sort_values('_score', ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    df.index += 1  # 1-based rank for display
    return df.drop(columns=['_score'])
