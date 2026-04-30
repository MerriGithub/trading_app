"""
scoring.py — Centralised pair scoring
======================================

Single source of truth for all three scoring modes used by search.py,
backtest.py, and walkforward.py.  Removes the previous inconsistency where
the Search and Backtest tabs ranked by different formulas.

Scoring modes
-------------
composite  : z(|LastSD|) + z(TrendVolRatio) + z(WinRate) + z(Expectancy)
             FitDataMinMaxSD removed per Q11 walk-forward analysis
             (Spearman ρ = -0.129 vs OOS gross return, p < 0.001).
cost_rank  : -EstCost + 0.01 × WinRate  — structural predictor, cannot be
             data-mined because it ranks cheapest pairs to hold.
contrarian : inverts the composite score — exploits the weak negative
             IS→OOS rank correlation found in Q11.

Usage
-----
    from scoring import apply_scoring, estimate_trade_cost, SCORING_MODES

    df = apply_scoring(df, scoring_mode='composite')
    df = df.sort_values('_score', ascending=False).head(top_n)
    df = df.drop(columns=['_score'])
"""
import pandas as pd

from account import load_account

# Metrics used in composite and contrarian modes.
# NOTE: FitDataMinMaxSD excluded per Q11 (ρ = -0.129 vs OOS gross, p < 0.001).
#       It is still computed and displayed; it just does not drive ranking.
_SCORE_COMPONENTS = ('LastSD', 'TrendVolRatio', 'WinRate', 'Expectancy')

SCORING_MODES = {
    'composite':  'Composite Score (Q11 improved)',
    'cost_rank':  'Cost-Based (lowest trading cost)',
    'contrarian': 'Contrarian (IS underperformers)',
}


def estimate_trade_cost(
    avg_holding_days: float,
    n_long: int,
    n_short: int,
    spread_cost_pct: float = 0.001,
) -> float:
    """
    Estimate total round-trip cost per trade.

    Components
    ----------
    spread  = spread_cost_pct × (n_long + n_short) × 2  (entry + exit, both legs)
    finance = (long_rate × n_long + short_rate × n_short) / 365 × avg_holding_days

    Rates are read from account.json so they reflect the user's actual broker
    terms rather than hardcoded defaults.
    """
    acct = load_account()
    long_rate  = acct.get('long_rate',  0.0488)
    short_rate = acct.get('short_rate', 0.0088)
    n_legs = n_long + n_short
    spread  = spread_cost_pct * n_legs * 2
    finance = (long_rate * n_long + short_rate * n_short) / 365 * avg_holding_days
    return spread + finance


def _z_scores(df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    """Add _z_{col} columns to df for each metric in cols."""
    for m in cols:
        std  = df[m].std()
        mean = df[m].mean()
        df[f'_z_{m}'] = (df[m] - mean) / std if std > 0 else 0.0
    return df


def apply_scoring(df: pd.DataFrame, scoring_mode: str = 'composite') -> pd.DataFrame:
    """
    Add '_score' column to df.  Caller handles sort, head(top_n), and drop.

    Required columns depend on mode:
        composite / contrarian : LastSD, TrendVolRatio, WinRate, Expectancy
        cost_rank              : EstCost, WinRate
    z-score helper columns (_z_*) are added then removed internally.
    """
    if scoring_mode == 'cost_rank':
        df['_score'] = -df['EstCost'] + 0.01 * df['WinRate']

    elif scoring_mode == 'contrarian':
        df = _z_scores(df, _SCORE_COMPONENTS)
        df['_score'] = -(
            df['_z_LastSD'].abs() + df['_z_TrendVolRatio'] +
            df['_z_WinRate'] + df['_z_Expectancy']
        )

    else:  # 'composite' — default
        df = _z_scores(df, _SCORE_COMPONENTS)
        df['_score'] = (
            df['_z_LastSD'].abs() + df['_z_TrendVolRatio'] +
            df['_z_WinRate'] + df['_z_Expectancy']
        )

    return df.drop(columns=[c for c in df.columns if c.startswith('_z_')], errors='ignore')
