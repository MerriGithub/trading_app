"""
account.py — Account settings and financing rate access.

Reads from data/account.json which stores per-asset-class financing rates,
spread cost fallback, starting capital, and margin. This replaces the legacy
flat account.json format (long_rate / short_rate as top-level keys).
"""
import json
from pathlib import Path

ACCOUNT_PATH = Path(__file__).parent / 'data' / 'account.json'

_DEFAULTS = {
    'starting_capital': 10000.0,
    'margin': 0.10,
    'spread_cost_fallback': 0.001,
    'financing': {
        'equity':       {'long_rate': 0.0488, 'short_rate': 0.0088},
        'fx':           {'long_rate': 0.018,  'short_rate': -0.018},
        'commodities':  {'long_rate': 0.0488, 'short_rate': 0.0088},
        'fixed_income': {'long_rate': 0.0488, 'short_rate': 0.0088},
    },
}


def load_account() -> dict:
    """
    Load account settings. Returns defaults if file missing or malformed.

    Backward compatibility: also injects flat keys ('long_rate', 'short_rate')
    so legacy callers (scoring.py estimate_trade_cost) continue to work.
    The flat keys reflect equity rates — the most common legacy use case.
    """
    try:
        data = json.loads(ACCOUNT_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = dict(_DEFAULTS)

    if 'long_rate' not in data:
        eq = data.get('financing', {}).get('equity', {})
        data['long_rate']  = eq.get('long_rate',  0.0488)
        data['short_rate'] = eq.get('short_rate', 0.0088)

    return data


def get_financing_rates(asset_class: str) -> tuple[float, float]:
    """
    Return (long_rate, short_rate) for an asset class.

    Parameters
    ----------
    asset_class : str
        One of 'equity', 'fx', 'commodities', 'fixed_income'.

    Returns
    -------
    (long_rate, short_rate) as annual fractions.
    short_rate is positive for a rebate, negative if both sides pay.
    """
    acct = load_account()
    financing = acct.get('financing', _DEFAULTS['financing'])
    rates = financing.get(asset_class, financing.get('equity', {}))
    return (
        rates.get('long_rate',  _DEFAULTS['financing']['equity']['long_rate']),
        rates.get('short_rate', _DEFAULTS['financing']['equity']['short_rate']),
    )


def get_spread_cost_fallback() -> float:
    """Return the fallback spread cost fraction for unknown instruments."""
    return load_account().get('spread_cost_fallback',
                              _DEFAULTS['spread_cost_fallback'])


def get_starting_capital() -> float:
    return load_account().get('starting_capital', _DEFAULTS['starting_capital'])


def get_margin() -> float:
    return load_account().get('margin', _DEFAULTS['margin'])


def save_account(data: dict) -> None:
    """Persist account settings atomically."""
    tmp = ACCOUNT_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(ACCOUNT_PATH)
