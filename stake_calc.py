import numpy as np
import pandas as pd

from config import ACTIVE_INSTRUMENTS, SPREADS, POINT_SIZES


def compute_stakes(
    prices: dict,
    daily_vols: dict,
    scalings: dict,
    long_flags: dict,
    short_flags: dict,
    target_exposure: float = 500.0,
) -> pd.DataFrame:
    """
    Size positions so that a 1 SD price move delivers target_exposure P&L.

    stake = (target_exposure / (price × daily_vol × point_size)) × scaling

    Returns a DataFrame with one row per instrument showing both long and short
    legs, stakes, expected 1 SD P&L, and spread cost.
    """
    rows = []
    for label in ACTIVE_INSTRUMENTS:
        price = prices.get(label, np.nan)
        vol = daily_vols.get(label, np.nan)       # daily vol as a fraction (e.g. 0.0115)
        scaling = scalings.get(label, 0.0)
        is_long = int(long_flags.get(label, 0))
        is_short = int(short_flags.get(label, 0))
        spread_pts = SPREADS.get(label, 0.0)
        pt_size = POINT_SIZES.get(label, 1.0)

        stake = 0.0
        if not (np.isnan(price) or np.isnan(vol) or vol == 0 or price == 0):
            one_sd_pts = price * vol                         # 1 SD move in index points
            raw_stake = target_exposure / (one_sd_pts * pt_size)
            stake = round(raw_stake * scaling, 2)

        one_sd_pnl = round(stake * price * vol * pt_size, 0) if stake else 0.0
        cost = round(stake * spread_pts * pt_size, 2) if stake else 0.0

        rows.append({
            'Instrument': label,
            'Price': round(float(price), 1) if not np.isnan(price) else np.nan,
            'Spread pts': spread_pts,
            'Daily Vol': f"{vol*100:.2f}%" if not np.isnan(vol) else 'N/A',
            'Scaling': f"{scaling*100:.0f}%",
            'Long': is_long,
            'Short': is_short,
            'Stake': stake,
            '1 SD P&L': one_sd_pnl,
            'Cost': cost,
        })

    return pd.DataFrame(rows)


def pnl_scenario(stakes_df: pd.DataFrame, price_changes_pct: dict) -> float:
    """
    Compute total P&L given a dict of {instrument: % price change}.
    Direction is +1 for long, -1 for short, 0 if not in trade.
    """
    total = 0.0
    for _, row in stakes_df.iterrows():
        label = row['Instrument']
        direction = row['Long'] - row['Short']
        if direction == 0:
            continue
        price = row['Price']
        stake = row['Stake']
        change = price_changes_pct.get(label, 0.0) / 100.0
        pt_size = POINT_SIZES.get(label, 1.0)
        if not np.isnan(price):
            total += direction * stake * price * change * pt_size
    return total
