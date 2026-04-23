import json
from pathlib import Path

from config import POINT_SIZES, SPREADS

ACCOUNT_PATH = Path(__file__).parent / 'account.json'
_DEFAULTS = {
    'starting_capital': 10000.0,
    'long_rate':        0.0488,
    'short_rate':       0.0088,
    'margin':           0.0,
}


def load_account() -> dict:
    if not ACCOUNT_PATH.exists():
        return dict(_DEFAULTS)
    try:
        return {**_DEFAULTS, **json.loads(ACCOUNT_PATH.read_text())}
    except Exception:
        return dict(_DEFAULTS)


def save_account(data: dict) -> None:
    ACCOUNT_PATH.write_text(json.dumps(data, indent=2))


def compute_daily_funding(
    open_trades: list,
    current_prices: dict,
    long_rate: float,
    short_rate: float,
) -> float:
    long_val = short_val = 0.0
    for trade in open_trades:
        for leg in trade.get('legs', []):
            pct = leg.get('pct_open', 0.0)
            if pct < 1e-6:
                continue
            bi, si = leg['buy_instrument'], leg['sell_instrument']
            bp = current_prices.get(bi, leg['buy_entry_price'])
            sp = current_prices.get(si, leg['sell_entry_price'])
            long_val  += leg['buy_stake']  * pct * bp * POINT_SIZES.get(bi, 1.0)
            short_val += leg['sell_stake'] * pct * sp * POINT_SIZES.get(si, 1.0)
    return long_val * long_rate / 365 - short_val * short_rate / 365


def compute_spread_costs(open_trades: list) -> float:
    total = 0.0
    for trade in open_trades:
        for leg in trade.get('legs', []):
            bi = leg['buy_instrument']
            si = leg['sell_instrument']
            total += leg['buy_stake']  * SPREADS.get(bi, 0.0) * POINT_SIZES.get(bi, 1.0)
            total += leg['sell_stake'] * SPREADS.get(si, 0.0) * POINT_SIZES.get(si, 1.0)
    return total
