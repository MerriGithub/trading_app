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

_TDY = PARAMS['trading_days_per_year']

# Metric definitions: (label, higher_is_better, spreadsheet_default_limit)
METRICS = [
    ('ReturnSD',        False, -2.0),
    ('TrendVolRatio',   True,   0.35),
    ('ReturnTopology',  False, -3.0),
    ('FitDataMinMaxSD', True,  12.0),
    ('LastSD',          True,   3.0),
]
METRIC_NAMES = [m[0] for m in METRICS]


# ── Internals ────────────────────────────────────────────────────────────────

def _combo_matrix(n: int, k: int) -> tuple[np.ndarray, list[tuple]]:
    """All C(n,k) combos as an (M, n) equal-weight float matrix."""
    combos = list(combinations(range(n), k))
    mat = np.zeros((len(combos), n), dtype=np.float64)
    for i, idx in enumerate(combos):
        mat[i, list(idx)] = 1.0 / k
    return mat, combos


def _batch_scores(spread_mat: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute all 5 scoring metrics for a (T, M) matrix of spread return series.
    Returns dict of metric_name -> (M,) float array.
    """
    T, M = spread_mat.shape

    # ReturnSD — annualised Sharpe
    means = spread_mat.mean(axis=0)
    stds  = spread_mat.std(axis=0)
    return_sd = np.where(stds > 0, means / stds * np.sqrt(_TDY), 0.0)

    # Cumulative return series
    cum = np.cumprod(1.0 + spread_mat, axis=0)  # (T, M)

    # TrendVolRatio — |OLS slope of cum| / daily vol
    x = np.arange(T, dtype=np.float64) - (T - 1) / 2.0
    xx = (x ** 2).sum()
    cum_mean = cum.mean(axis=0)
    xy = (x[:, None] * (cum - cum_mean)).sum(axis=0)
    slopes = xy / xx
    tvr = np.where(stds > 0, np.abs(slopes) / stds, 0.0)

    # ReturnTopology — skewness of daily returns
    c  = spread_mat - means
    m2 = (c ** 2).mean(axis=0)
    m3 = (c ** 3).mean(axis=0)
    topology = np.where(m2 > 0, m3 / m2 ** 1.5, 0.0)

    # FitDataMinMaxSD — cumulative price range / trailing vol
    price_range = cum.max(axis=0) - cum.min(axis=0)
    n_tail = min(T, 20)
    tail_std = spread_mat[-n_tail:].std(axis=0)
    fit_minmax_sd = np.where(
        tail_std > 0, price_range / (tail_std * np.sqrt(n_tail)), 0.0
    )

    # LastSD — current level vs rolling mean, in SDs
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

def estimate_combinations(min_legs: int, max_legs: int, n: int = 12) -> int:
    """Total (long, short) pairs for given leg range and instrument count."""
    m = sum(comb(n, k) for k in range(min_legs, max_legs + 1))
    return m * m


def run_search(
    rets: pd.DataFrame,
    scalings: pd.DataFrame,
    min_legs: int = 3,
    max_legs: int = 4,
    window_days: int = _TDY,
    filters: dict | None = None,
    top_n: int = 30,
    progress_cb=None,
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
    Columns: Long, Short, ReturnSD, TrendVolRatio, ReturnTopology,
             FitDataMinMaxSD, LastSD, plus hidden _long_flags / _short_flags.
    """
    instruments = [i for i in ACTIVE_INSTRUMENTS if i in rets.columns]
    N = len(instruments)

    # Align and clip to window
    r = rets[instruments].tail(window_days).dropna(how='any').to_numpy(dtype=np.float64)
    s = scalings[instruments].tail(window_days).dropna(how='any').to_numpy(dtype=np.float64)
    T = min(r.shape[0], s.shape[0])
    scaled = r[-T:] * s[-T:]  # (T, N)

    # Pre-compute vol-scaled leg returns for every combo size
    leg_cache: dict[int, tuple[np.ndarray, list]] = {}
    for k in range(min_legs, max_legs + 1):
        mat, combos = _combo_matrix(N, k)
        leg_cache[k] = (scaled @ mat.T, combos)  # (T, M_k), list of tuples

    total = estimate_combinations(min_legs, max_legs, N)
    records: list[dict] = []
    done = 0

    for n_long in range(min_legs, max_legs + 1):
        long_rets, long_combos = leg_cache[n_long]   # (T, M_long)

        for n_short in range(min_legs, max_legs + 1):
            short_rets, short_combos = leg_cache[n_short]  # (T, M_short)

            for l_i, long_combo in enumerate(long_combos):
                lr = long_rets[:, l_i]                  # (T,)
                spread_mat = lr[:, None] - short_rets   # (T, M_short)
                batch = _batch_scores(spread_mat)

                for s_i, short_combo in enumerate(short_combos):
                    done += 1

                    if set(long_combo) == set(short_combo):
                        continue

                    sc = {k: float(v[s_i]) for k, v in batch.items()}

                    # Apply metric filters
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

                    records.append({
                        'Long':         ' | '.join(long_names),
                        'Short':        ' | '.join(short_names),
                        '_long_flags':  {instruments[i]: 1 for i in long_combo},
                        '_short_flags': {instruments[i]: 1 for i in short_combo},
                        **sc,
                    })

                if progress_cb and (done % 5000 == 0 or done >= total):
                    progress_cb(min(done / total, 1.0))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['_score'] = df['LastSD'].abs() + df['TrendVolRatio'] + df['FitDataMinMaxSD'] * 0.05
    df = (df.sort_values('_score', ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    df.index += 1  # 1-based rank
    return df.drop(columns=['_score'])
