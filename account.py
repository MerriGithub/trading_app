"""
account.py — Account settings and broker-aware financing rate access.

Reads from ``data/account.json`` which stores per-asset-class financing
rates, spread cost fallback, starting capital, margin, and broker profiles.

Financing rate model
--------------------
Annual rate ÷ 365 per calendar day, applied to each leg separately:
    long leg:  pays long_rate / 365
    short leg: receives short_rate / 365

Where short_rate is positive, the short side earns a rebate. Where it is
negative (e.g. FX at −1.8%), both sides pay a swap cost.

Net daily drag = (long_rate − short_rate) / 365, compounding over hold period.

Broker-aware cost model
-----------------------
For more accurate per-instrument rates (used in Tab 3 Stake Calculator):
    - ``get_commodity_daily_rate()``  — basis + admin, per IG SB/CFD profile
    - ``get_fx_daily_rate()``         — carry differential, per IG SB/CFD profile
    - ``get_index_daily_rate()``      — benchmark rate + admin

These are routed through ``get_financing_daily_rate()`` based on asset_class.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ACCOUNT_PATH = Path(__file__).parent / 'data' / 'account.json'

# Known asset class keys — validated in get_financing_rates and related funcs.
_KNOWN_ASSET_CLASSES = frozenset({'equity', 'fx', 'commodities', 'fixed_income'})
_KNOWN_DIRECTIONS = frozenset({'long', 'short'})

_DEFAULTS: dict = {
    'starting_capital': 10000.0,
    'margin': 0.10,
    'spread_cost_fallback': 0.001,
    'financing': {
        # Annual rates as decimal fractions.
        # Equity/Commodity/FI: long=4.88%, short rebate=0.88% → net drag=4.00%/yr
        # FX: both sides pay 1.8% swap (no rebate) → short_rate is negative
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

# Required top-level keys for save_account validation.
_REQUIRED_ACCOUNT_KEYS = frozenset({'starting_capital', 'margin', 'financing'})


def load_account() -> dict:
    """Load account settings from ``data/account.json``.

    Returns defaults if the file is missing or contains invalid JSON.
    Also injects flat ``long_rate`` / ``short_rate`` keys for backward
    compatibility with legacy callers (e.g. ``scoring.py``).  The flat
    keys reflect equity rates — the most common legacy use case.

    Returns:
        Account settings dict. Always contains at least the keys from
        ``_DEFAULTS``.
    """
    try:
        data = json.loads(ACCOUNT_PATH.read_text())
    except FileNotFoundError:
        logger.warning("account.json not found at %s; using defaults", ACCOUNT_PATH)
        data = dict(_DEFAULTS)
    except json.JSONDecodeError as exc:
        logger.error("account.json is malformed: %s; using defaults", exc)
        data = dict(_DEFAULTS)

    # Backward compat: inject flat equity rates for legacy callers.
    if 'long_rate' not in data:
        eq = data.get('financing', {}).get('equity', {})
        data['long_rate']  = eq.get('long_rate',  0.0488)
        data['short_rate'] = eq.get('short_rate', 0.0088)

    return data


def get_financing_rates(asset_class: str) -> tuple[float, float]:
    """Return annual financing rates ``(long_rate, short_rate)`` for an asset class.

    Rates are annual fractions (e.g. 0.0488 = 4.88% per year).
    ``short_rate`` is positive when the short side earns a rebate, negative
    when both sides pay (FX swap cost model).

    Args:
        asset_class: One of ``'equity'``, ``'fx'``, ``'commodities'``,
            ``'fixed_income'``. Case-sensitive.

    Returns:
        ``(long_rate, short_rate)`` as decimal fractions of notional per year.

    Raises:
        ValueError: If ``asset_class`` is not one of the known values.
    """
    if asset_class not in _KNOWN_ASSET_CLASSES:
        raise ValueError(
            f"asset_class must be one of {sorted(_KNOWN_ASSET_CLASSES)}; "
            f"got {asset_class!r}"
        )

    acct = load_account()
    financing = acct.get('financing', _DEFAULTS['financing'])
    rates = financing.get(asset_class, financing.get('equity', {}))
    return (
        rates.get('long_rate',  _DEFAULTS['financing']['equity']['long_rate']),
        rates.get('short_rate', _DEFAULTS['financing']['equity']['short_rate']),
    )


def get_spread_cost_fallback() -> float:
    """Return the fallback bid-ask spread cost fraction for unknown instruments.

    Returns:
        One-way spread cost as a fraction (e.g. 0.001 = 0.1%). Used by
        ``asset_configs.get_spread_cost_lookup()`` when an instrument has no
        configured spread_pct.
    """
    return load_account().get('spread_cost_fallback',
                              _DEFAULTS['spread_cost_fallback'])


def get_starting_capital() -> float:
    """Return the configured starting capital in account currency.

    Returns:
        Starting capital (e.g. 10000.0 for £10,000).
    """
    return load_account().get('starting_capital', _DEFAULTS['starting_capital'])


def get_margin() -> float:
    """Return the flat margin rate used for hypothetical margin estimates.

    Returns:
        Margin fraction (e.g. 0.10 = 10%). Used as a fallback when no
        per-asset-class margin rate is configured.
    """
    return load_account().get('margin', _DEFAULTS['margin'])


def get_margin_rate(asset_class: str) -> float:
    """Return the IG margin rate for a given asset class.

    Falls back to ``get_margin()`` (flat rate) if the asset class is not
    found in the margin_rates dict.

    Args:
        asset_class: Asset class key (case-insensitive).

    Returns:
        Margin fraction (e.g. 0.0333 = 1:30 leverage for FX).
    """
    acct = load_account()
    rates = acct.get('margin_rates', _DEFAULTS['margin_rates'])
    return float(rates.get(asset_class.lower(), get_margin()))


def save_account(data: dict) -> None:
    """Persist account settings atomically to ``data/account.json``.

    Validates that required top-level keys are present before writing to
    avoid producing a corrupt account.json that breaks the whole app on
    the next load.

    Args:
        data: Account settings dict. Must contain at least ``starting_capital``,
            ``margin``, and ``financing`` keys.

    Raises:
        ValueError: If any required key is absent from ``data``.
    """
    missing = _REQUIRED_ACCOUNT_KEYS - set(data.keys())
    if missing:
        raise ValueError(
            f"save_account: missing required keys {sorted(missing)}. "
            f"A partial write would corrupt account.json and break app startup."
        )

    tmp = ACCOUNT_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(ACCOUNT_PATH)
    logger.info("account.json saved (capital=%.0f)", data.get('starting_capital', 0))


# ═══════════════════════════════════════════════════════════════════════════
# BROKER-AWARE COST MODEL
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_BROKER_PROFILES: dict[str, dict] = {
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

_DEFAULT_BENCHMARK_RATES: dict[str, float] = {
    'GBP': 0.0376, 'USD': 0.0367, 'EUR': 0.0198, 'CHF': -0.0011,
    'JPY': 0.0070,  'AUD': 0.0430, 'NZD': 0.0247, 'CAD': 0.0225,
}

_DEFAULT_COMMODITY_BASIS: dict[str, float] = {
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

BROKER_PROFILE_LABELS: dict[str, str] = {
    'ig_spreadbet': 'IG Spread Bet',
    'ig_cfd':       'IG CFD',
}


def get_active_broker_profile(profile_key: Optional[str] = None) -> dict:
    """Return the broker profile dict for the given key.

    Args:
        profile_key: ``'ig_spreadbet'``, ``'ig_cfd'``, or ``None`` to use the
            ``active_broker_profile`` value from ``account.json``.

    Returns:
        Broker profile dict with keys ``label``, ``*_admin_pct``,
        ``*_divisor`` for each instrument type. Falls back to
        ``ig_spreadbet`` if the key is not found in either the account
        file or the built-in defaults.
    """
    acct = load_account()
    key = profile_key or acct.get('active_broker_profile', 'ig_spreadbet')
    profiles = acct.get('broker_profiles', {})
    profile = profiles.get(key) or _DEFAULT_BROKER_PROFILES.get(key)
    if profile is None:
        logger.warning(
            "Broker profile %r not found; falling back to ig_spreadbet", key
        )
        profile = _DEFAULT_BROKER_PROFILES['ig_spreadbet']
    return profile


def get_commodity_daily_rate(
    instrument_code: str,
    direction: str = 'long',
    profile_key: Optional[str] = None,
) -> float:
    """Daily financing rate for a commodity spread bet/CFD as a fraction of notional.

    Formula (per calendar day, using IG's 360-day convention):
        basis_daily = commodity_basis[code] / divisor
        admin_daily = admin_pct / divisor
        long:  basis_daily + admin_daily   (contango basis costs the long)
        short: admin_daily - basis_daily   (contango basis credits the short)

    Returns positive = cost, negative = credit.

    Args:
        instrument_code: Commodity code (e.g. ``'GOLD'``, ``'BRENT'``).
        direction: ``'long'`` or ``'short'``.
        profile_key: Broker profile key, or ``None`` for account default.

    Returns:
        Daily rate as a fraction of notional. Positive = cost.

    Raises:
        ValueError: If ``direction`` is not ``'long'`` or ``'short'``.
    """
    if direction not in _KNOWN_DIRECTIONS:
        raise ValueError(
            f"direction must be 'long' or 'short'; got {direction!r}"
        )

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
    price: Optional[float] = None,
    profile_key: Optional[str] = None,
) -> float:
    """Daily financing rate for an FX spread bet/CFD as a fraction of notional.

    Derivation (spread bet, long base/short quote):
        long  daily cost = (admin + quote_rate - base_rate) / divisor
        short daily cost = (admin + base_rate - quote_rate) / divisor

    Example (USDCHF long):
        (1.5% + 0.11% − 3.67%) / 360 = −2.28%/360  (credit from USD carry)
    Example (USDCHF short):
        (1.5% + 3.67% + 0.11%) / 360 = +5.28%/360  (cost)

    The ``price`` parameter is accepted for interface consistency but is not
    needed — it cancels in the fraction-of-notional formula.

    Args:
        instrument_code: FX pair code (e.g. ``'USDCHF'``).
        direction: ``'long'`` or ``'short'``.
        price: Current price (unused, kept for interface compatibility).
        profile_key: Broker profile key, or ``None`` for account default.

    Returns:
        Daily rate as a fraction of notional. Positive = cost, negative = credit.

    Raises:
        ValueError: If ``direction`` is not ``'long'`` or ``'short'``.
    """
    if direction not in _KNOWN_DIRECTIONS:
        raise ValueError(
            f"direction must be 'long' or 'short'; got {direction!r}"
        )

    acct = load_account()
    profile   = get_active_broker_profile(profile_key)
    admin_pct = profile['fx_admin_pct']
    divisor   = float(profile['fx_divisor'])

    benchmark_rates = acct.get('benchmark_rates', _DEFAULT_BENCHMARK_RATES)

    raw_map = acct.get('fx_currency_map', {})
    fx_map: dict[str, tuple[str, str]] = (
        {k: tuple(v) for k, v in raw_map.items()}  # type: ignore[misc]
        if raw_map else _DEFAULT_FX_CURRENCY_MAP
    )

    if instrument_code not in fx_map:
        # Fallback: use legacy flat rate for unrecognised pairs.
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
    profile_key: Optional[str] = None,
) -> float:
    """Daily financing rate for an index/ETF spread bet/CFD.

    Formula:
        long:  (admin + benchmark_rate) / divisor
        short: (admin - benchmark_rate) / divisor

    Args:
        instrument_code: Index code (e.g. ``'UKX'``). Used for logging only.
        currency: Currency of the index (default ``'GBP'``). Determines which
            benchmark rate to apply.
        direction: ``'long'`` or ``'short'``.
        profile_key: Broker profile key, or ``None`` for account default.

    Returns:
        Daily rate as a fraction of notional. Positive = cost, negative = credit.

    Raises:
        ValueError: If ``direction`` is not ``'long'`` or ``'short'``.
    """
    if direction not in _KNOWN_DIRECTIONS:
        raise ValueError(
            f"direction must be 'long' or 'short'; got {direction!r}"
        )

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
    price: Optional[float] = None,
    broker_profile: Optional[str] = None,
) -> float:
    """Unified entry point for broker-aware per-instrument daily financing rate.

    Routes to the correct function based on ``asset_class``.
    Returns daily rate as fraction of notional (positive = cost, negative = credit).

    Annual rate ÷ 365 per day for equity/commodity/FI; ÷ 360 for FX/commodities
    per IG's actual day-count convention in the broker profile.

    Args:
        instrument_code: E.g. ``'GOLD'``, ``'USDCHF'``, ``'UKX'``.
        asset_class: One of ``'commodities'``, ``'fx'``, ``'equity'``,
            ``'fixed_income'``.
        direction: ``'long'`` or ``'short'``.
        price: Current price (accepted for interface consistency; unused).
        broker_profile: ``'ig_spreadbet'``, ``'ig_cfd'``, or ``None``
            (uses account.json default).

    Returns:
        Daily financing rate as a fraction of notional.

    Raises:
        ValueError: If ``direction`` is not ``'long'`` or ``'short'``.
    """
    if direction not in _KNOWN_DIRECTIONS:
        raise ValueError(
            f"direction must be 'long' or 'short'; got {direction!r}"
        )

    ac = asset_class.lower()
    if ac == 'commodities':
        return get_commodity_daily_rate(instrument_code, direction, broker_profile)
    elif ac == 'fx':
        return get_fx_daily_rate(instrument_code, direction, price, broker_profile)
    else:
        return get_index_daily_rate(instrument_code, 'GBP', direction, broker_profile)
