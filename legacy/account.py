import json
from pathlib import Path

from config import POINT_SIZES, SPREADS

# --- Persistence ---
# account.json lives next to this file; stores user-edited settings across sessions
ACCOUNT_PATH = Path(__file__).parent / 'account.json'

# Fallback values used when account.json is absent or corrupt
_DEFAULTS = {
    'starting_capital': 10000.0,
    'long_rate':        0.0488,   # annual financing rate charged on long positions
    'short_rate':       0.0088,   # annual rebate/cost on short positions
    'margin':           0.0,
}


# --- Load / Save ---

def load_account() -> dict:
    # Return defaults if no file exists yet
    if not ACCOUNT_PATH.exists():
        return dict(_DEFAULTS)
    try:
        # Merge: file values override defaults, so new default keys are always present
        return {**_DEFAULTS, **json.loads(ACCOUNT_PATH.read_text())}
    except Exception:
        return dict(_DEFAULTS)


def save_account(data: dict) -> None:
    ACCOUNT_PATH.write_text(json.dumps(data, indent=2))


# --- Daily Financing Cost ---

def compute_daily_funding(
    open_trades: list,
    current_prices: dict,
    long_rate: float,
    short_rate: float,
) -> float:
    # Accumulate notional exposure for long and short sides separately
    long_val = short_val = 0.0
    for trade in open_trades:
        for leg in trade.get('legs', []):
            pct = leg.get('pct_open', 0.0)
            if pct < 1e-6:
                continue  # leg is fully closed, skip
            bi, si = leg['buy_instrument'], leg['sell_instrument']
            # Use live price where available, fall back to entry price
            bp = current_prices.get(bi, leg['buy_entry_price'])
            sp = current_prices.get(si, leg['sell_entry_price'])
            # Notional = stake * fraction_open * price * point_size
            long_val  += leg['buy_stake']  * pct * bp * POINT_SIZES.get(bi, 1.0)
            short_val += leg['sell_stake'] * pct * sp * POINT_SIZES.get(si, 1.0)
    # Net daily charge: longs pay the long rate, shorts receive the short rate
    return long_val * long_rate / 365 - short_val * short_rate / 365


# --- Spread Cost (one-off entry cost) ---

def compute_spread_costs(open_trades: list) -> float:
    # Sum the bid/ask spread cost for every leg across all open trades
    total = 0.0
    for trade in open_trades:
        for leg in trade.get('legs', []):
            bi = leg['buy_instrument']
            si = leg['sell_instrument']
            # Cost = stake * spread_in_points * point_size (converts to £/$ value)
            total += leg['buy_stake']  * SPREADS.get(bi, 0.0) * POINT_SIZES.get(bi, 1.0)
            total += leg['sell_stake'] * SPREADS.get(si, 0.0) * POINT_SIZES.get(si, 1.0)
    return total
