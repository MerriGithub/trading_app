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
    'margin_rates': {
        'fx':           0.0333,   # IG SB/CFD: 1:30 leverage
        'equity':       0.0500,   # IG SB/CFD: 1:20 leverage
        'commodities':  0.1000,   # IG SB/CFD: 1:10 leverage
        'fixed_income': 0.0500,   # IG SB/CFD: 1:20 leverage
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


def get_margin_rate(asset_class: str) -> float:
    """
    Return the IG margin rate for a given asset class.
    Falls back to get_margin() (flat rate) if asset class not found.
    """
    acct = load_account()
    rates = acct.get('margin_rates', _DEFAULTS['margin_rates'])
    return float(rates.get(asset_class.lower(), get_margin()))


def save_account(data: dict) -> None:
    """Persist account settings atomically."""
    tmp = ACCOUNT_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(ACCOUNT_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# BROKER-AWARE COST MODEL
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_BROKER_PROFILES = {
    'ig_spreadbet': {
        'label': 'IG Spread Bet',
        'commodity_admin_pct': 0.034,
        'commodity_divisor':   360,
        'fx_admin_pct':        0.015,
        'fx_divisor':          360,
        'index_admin_pct':     0.034,
        'index_divisor':       365,
    },
    'ig_cfd': {
        'label': 'IG CFD',
        'commodity_admin_pct': 0.030,
        'commodity_divisor':   360,
        'fx_admin_pct':        0.003,
        'fx_divisor':          360,
        'index_admin_pct':     0.030,
        'index_divisor':       365,
    },
}

_DEFAULT_BENCHMARK_RATES = {
    'GBP': 0.0376, 'USD': 0.0367, 'EUR': 0.0198, 'CHF': -0.0011,
    'JPY': 0.0070,  'AUD': 0.0430, 'NZD': 0.0247, 'CAD': 0.0225,
}

_DEFAULT_COMMODITY_BASIS = {
    'GOLD': 0.040, 'SILVER': 0.035, 'PLATINUM': 0.030, 'PALLADIUM': 0.035,
    'COPPER': 0.030, 'WTI': 0.050, 'BRENT': 0.050, 'NATGAS': 0.080,
    'WHEAT': 0.020, 'CORN': 0.020, 'SOYBEANS': 0.020,
    'COFFEE': 0.025, 'SUGAR': 0.025,
}

_DEFAULT_FX_CURRENCY_MAP: dict[str, tuple[str, str]] = {
    'EURUSD': ('EUR', 'USD'), 'GBPUSD': ('GBP', 'USD'), 'USDJPY': ('USD', 'JPY'),
    'USDCHF': ('USD', 'CHF'), 'AUDUSD': ('AUD', 'USD'), 'NZDUSD': ('NZD', 'USD'),
    'USDCAD': ('USD', 'CAD'), 'EURGBP': ('EUR', 'GBP'), 'EURJPY': ('EUR', 'JPY'),
    'GBPJPY': ('GBP', 'JPY'), 'EURCHF': ('EUR', 'CHF'), 'AUDNZD': ('AUD', 'NZD'),
}

BROKER_PROFILE_LABELS = {
    'ig_spreadbet': 'IG Spread Bet',
    'ig_cfd':       'IG CFD',
}


def get_active_broker_profile(profile_key: str | None = None) -> dict:
    """
    Return the broker profile dict for the given key.
    If profile_key is None, uses active_broker_profile from account.json.
    Falls back to _DEFAULT_BROKER_PROFILES if the key is missing.
    """
    acct = load_account()
    key = profile_key or acct.get('active_broker_profile', 'ig_spreadbet')
    profiles = acct.get('broker_profiles', {})
    profile = profiles.get(key) or _DEFAULT_BROKER_PROFILES.get(key)
    if profile is None:
        profile = _DEFAULT_BROKER_PROFILES['ig_spreadbet']
    return profile


def get_commodity_daily_rate(
    instrument_code: str,
    direction: str = 'long',
    profile_key: str | None = None,
) -> float:
    """
    Daily financing rate for a commodity spread bet/CFD as a fraction of notional.

    Formula:
        basis_daily = commodity_basis_estimates[code] / divisor
        admin_daily = admin_pct / divisor
        long:  basis_daily + admin_daily
        short: -basis_daily + admin_daily  (contango basis credits short)

    Returns positive = cost, negative = credit.
    Note: holding_days in the backtest engine uses calendar days, so this
    daily rate is already correct without any weekend multiplier.
    """
    acct = load_account()
    profile = get_active_broker_profile(profile_key)
    admin_pct = profile['commodity_admin_pct']
    divisor   = float(profile['commodity_divisor'])
    basis_map = acct.get('commodity_basis_estimates', _DEFAULT_COMMODITY_BASIS)
    basis = basis_map.get(instrument_code.upper(), 0.030)

    if direction == 'long':
        return (basis + admin_pct) / divisor
    else:
        return (admin_pct - basis) / divisor


def get_fx_daily_rate(
    instrument_code: str,
    direction: str = 'long',
    price: float | None = None,
    profile_key: str | None = None,
) -> float:
    """
    Daily financing rate for an FX spread bet/CFD as a fraction of notional.

    Derivation (spread bet, long base/short quote):
        long  daily cost = (admin + quote_rate - base_rate) / divisor
        short daily cost = (admin + base_rate - quote_rate) / divisor

    This correctly models:
        long  USDCHF → (1.5 - 0.11 - 3.67)%/360 = -2.28%/360  (credit from USD carry)
        short USDCHF → (1.5 + 3.67 + 0.11)%/360 = +5.28%/360  (cost verified at ~5.3%)

    The `price` parameter is accepted for interface consistency but is not
    needed as it cancels in the fraction-of-notional formula.

    Returns positive = cost, negative = credit.
    """
    acct = load_account()
    profile   = get_active_broker_profile(profile_key)
    admin_pct = profile['fx_admin_pct']
    divisor   = float(profile['fx_divisor'])

    benchmark_rates = acct.get('benchmark_rates', _DEFAULT_BENCHMARK_RATES)

    raw_map = acct.get('fx_currency_map', {})
    fx_map: dict[str, tuple[str, str]] = {
        k: tuple(v) for k, v in raw_map.items()  # type: ignore[misc]
    } if raw_map else _DEFAULT_FX_CURRENCY_MAP

    if instrument_code not in fx_map:
        # Fallback: use legacy flat rate
        from account import get_financing_rates
        long_rate, short_rate = get_financing_rates('fx')
        return (long_rate if direction == 'long' else abs(short_rate)) / divisor

    base_ccy, quote_ccy = fx_map[instrument_code]
    base_rate  = benchmark_rates.get(base_ccy,  0.035)
    quote_rate = benchmark_rates.get(quote_ccy, 0.035)

    if direction == 'long':
        return (admin_pct + quote_rate - base_rate) / divisor
    else:
        return (admin_pct + base_rate - quote_rate) / divisor


def get_index_daily_rate(
    instrument_code: str,
    currency: str = 'GBP',
    direction: str = 'long',
    profile_key: str | None = None,
) -> float:
    """
    Daily financing rate for an index/ETF spread bet/CFD as a fraction of notional.

    Formula:
        long:  (admin + benchmark_rate) / divisor
        short: (admin - benchmark_rate) / divisor

    Returns positive = cost, negative = credit.
    """
    acct = load_account()
    profile   = get_active_broker_profile(profile_key)
    admin_pct = profile['index_admin_pct']
    divisor   = float(profile['index_divisor'])
    benchmark_rates = acct.get('benchmark_rates', _DEFAULT_BENCHMARK_RATES)
    benchmark = benchmark_rates.get(currency.upper(), 0.035)

    if direction == 'long':
        return (admin_pct + benchmark) / divisor
    else:
        return (admin_pct - benchmark) / divisor


def get_financing_daily_rate(
    instrument_code: str,
    asset_class: str,
    direction: str = 'long',
    price: float | None = None,
    broker_profile: str | None = None,
) -> float:
    """
    Unified entry point for broker-aware per-instrument daily financing rate.

    Routes to the correct function based on asset_class.
    Returns daily rate as fraction of notional (positive = cost, negative = credit).

    Parameters
    ----------
    instrument_code : str
        E.g. 'GOLD', 'USDCHF', 'UKX'.
    asset_class : str
        One of 'commodities', 'fx', 'equity', 'fixed_income'.
    direction : str
        'long' or 'short'.
    price : float | None
        Current price (accepted for interface consistency; not used in formulas).
    broker_profile : str | None
        'ig_spreadbet', 'ig_cfd', or None (uses account.json default).
    """
    ac = asset_class.lower()
    if ac == 'commodities':
        return get_commodity_daily_rate(instrument_code, direction, broker_profile)
    elif ac == 'fx':
        return get_fx_daily_rate(instrument_code, direction, price, broker_profile)
    else:
        return get_index_daily_rate(instrument_code, 'GBP', direction, broker_profile)
