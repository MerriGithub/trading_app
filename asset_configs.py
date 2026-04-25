"""
asset_configs.py — Instrument definitions and cost models per asset class
=========================================================================

Each asset class dict follows the same structure as config.py's equity
definitions, so the backtest engine and search can treat them uniformly.

Keys per asset class:
    instruments  : dict  code → display name
    spreads_pct  : dict  code → round-trip bid-ask spread as a fraction
    financing    : dict  with 'long_rate', 'short_rate' (annual, as fraction)
    point_sizes  : dict  code → P&L multiplier per point (1.0 for most)

Usage in the backtest tab:
    from asset_configs import ASSET_CLASSES
    cfg = ASSET_CLASSES['fx']
    instruments = list(cfg['instruments'].keys())
    spread_cost = sum(cfg['spreads_pct'][i] for i in basket) * 2  # round-trip
"""

# ═══════════════════════════════════════════════════════════════════════════
# EQUITY INDICES (mirrors config.py — canonical source stays there)
# ═══════════════════════════════════════════════════════════════════════════

EQUITY = {
    'label': 'Equity Indices',
    'csv_file': 'prices.csv',
    'instruments': {
        'UKX': 'FTSE', 'CBK': 'CAC', 'CEY': 'MIB', 'CFR': 'DAX',
        'CMD': 'IBEX', 'CEI': 'STOXX50', 'COI': 'SMI', 'CRM': 'HSI',
        'CIL': 'ASX', 'CPH': 'NDX', 'CTN': 'SPX', 'CTB': 'DJI',
    },
    # Spread as fraction of price (from config.py SPREADS / typical prices)
    'spreads_pct': {
        'UKX': 0.00038, 'CBK': 0.00050, 'CEY': 0.00135, 'CFR': 0.00022,
        'CMD': 0.00055, 'CEI': 0.00080, 'COI': 0.00042, 'CRM': 0.00050,
        'CIL': 0.00050, 'CPH': 0.00010, 'CTN': 0.00009, 'CTB': 0.00013,
    },
    'financing': {
        'long_rate': 0.0488,    # annual, charged on long notional
        'short_rate': 0.0088,   # annual, rebate on short notional
        'net_daily': (0.0488 - 0.0088) / 365,  # ≈ 0.01096% per day
    },
    'point_sizes': {k: 1.0 for k in [
        'UKX', 'CBK', 'CEY', 'CFR', 'CMD', 'CEI', 'COI',
        'CRM', 'CIL', 'CPH', 'CTN', 'CTB',
    ]},
}


# ═══════════════════════════════════════════════════════════════════════════
# FX PAIRS
# ═══════════════════════════════════════════════════════════════════════════

FX = {
    'label': 'FX Pairs',
    'csv_file': 'fx_prices.csv',
    'instruments': {
        'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD', 'USDJPY': 'USD/JPY',
        'USDCHF': 'USD/CHF', 'AUDUSD': 'AUD/USD', 'NZDUSD': 'NZD/USD',
        'USDCAD': 'USD/CAD', 'EURGBP': 'EUR/GBP', 'EURJPY': 'EUR/JPY',
        'GBPJPY': 'GBP/JPY', 'EURCHF': 'EUR/CHF', 'AUDNZD': 'AUD/NZD',
    },
    # Round-trip spread as fraction (based on typical retail CFD spreads)
    'spreads_pct': {
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.0001,
        'USDCHF': 0.00015, 'AUDUSD': 0.00015, 'NZDUSD': 0.0002,
        'USDCAD': 0.00015, 'EURGBP': 0.0002, 'EURJPY': 0.0002,
        'GBPJPY': 0.0003, 'EURCHF': 0.0002, 'AUDNZD': 0.0003,
    },
    'financing': {
        'long_rate': 0.018,     # ~1.8% p.a. average swap cost (retail)
        'short_rate': 0.018,    # symmetric for FX (both sides pay swap)
        'net_daily': 0.018 / 365,   # ≈ 0.00493% per day per leg
    },
    'point_sizes': {k: 1.0 for k in [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD',
        'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY', 'EURCHF', 'AUDNZD',
    ]},
}


# ═══════════════════════════════════════════════════════════════════════════
# COMMODITIES
# ═══════════════════════════════════════════════════════════════════════════

COMMODITIES = {
    'label': 'Commodities',
    'csv_file': 'commodity_prices.csv',
    'instruments': {
        # Energy
        'WTI': 'WTI Crude', 'BRENT': 'Brent Crude', 'NATGAS': 'Natural Gas',
        # Precious metals
        'GOLD': 'Gold', 'SILVER': 'Silver', 'PLATINUM': 'Platinum',
        # Industrial metals
        'COPPER': 'Copper', 'PALLADIUM': 'Palladium',
        # Agriculture
        'WHEAT': 'Wheat', 'CORN': 'Corn', 'SOYBEANS': 'Soybeans',
        'COFFEE': 'Coffee', 'SUGAR': 'Sugar',
    },
    # Round-trip spread as fraction (based on typical CFD commodity spreads)
    # Energy spreads are wider due to volatility; ags vary by exchange
    'spreads_pct': {
        'WTI': 0.0005, 'BRENT': 0.0005, 'NATGAS': 0.0020,
        'GOLD': 0.0003, 'SILVER': 0.0005, 'PLATINUM': 0.0010,
        'COPPER': 0.0005, 'PALLADIUM': 0.0015,
        'WHEAT': 0.0010, 'CORN': 0.0008, 'SOYBEANS': 0.0008,
        'COFFEE': 0.0012, 'SUGAR': 0.0012,
    },
    'financing': {
        'long_rate': 0.0488,    # same as equity CFDs (broker markup)
        'short_rate': 0.0088,   # same as equity CFDs
        'net_daily': (0.0488 - 0.0088) / 365,
    },
    'point_sizes': {k: 1.0 for k in [
        'WTI', 'BRENT', 'NATGAS', 'GOLD', 'SILVER', 'PLATINUM',
        'COPPER', 'PALLADIUM', 'WHEAT', 'CORN', 'SOYBEANS',
        'COFFEE', 'SUGAR',
    ]},
    # Sector groupings for analysis
    'sectors': {
        'Energy': ['WTI', 'BRENT', 'NATGAS'],
        'Precious': ['GOLD', 'SILVER', 'PLATINUM'],
        'Industrial': ['COPPER', 'PALLADIUM'],
        'Agriculture': ['WHEAT', 'CORN', 'SOYBEANS', 'COFFEE', 'SUGAR'],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FIXED INCOME
# ═══════════════════════════════════════════════════════════════════════════

FIXED_INCOME = {
    'label': 'Fixed Income',
    'csv_file': 'fi_prices.csv',
    'instruments': {
        # US duration ladder
        'SHY': 'US 1-3Y', 'IEI': 'US 3-7Y', 'IEF': 'US 7-10Y', 'TLT': 'US 20+Y',
        # US credit
        'LQD': 'US IG Corp', 'HYG': 'US HY Corp',
        # US inflation
        'TIP': 'US TIPS',
        # International
        'BWX': 'Intl Treasury', 'EMB': 'EM Bonds',
        # UK/Europe
        'IGLT': 'UK Gilts', 'IEAG': 'EUR Govt', 'IBTM': 'US Trsy 7-10 (L)',
    },
    # Reference yield indices — NOT tradeable, excluded from basket search
    'reference_only': ['UST10Y', 'UST30Y', 'UST5Y'],
    'reference_instruments': {
        'UST10Y': 'US 10Y Yield', 'UST30Y': 'US 30Y Yield', 'UST5Y': 'US 5Y Yield',
    },
    # ETF spreads are very tight
    'spreads_pct': {
        'SHY': 0.0001, 'IEI': 0.0001, 'IEF': 0.0001, 'TLT': 0.0002,
        'LQD': 0.0002, 'HYG': 0.0003,
        'TIP': 0.0002,
        'BWX': 0.0003, 'EMB': 0.0003,
        'IGLT': 0.0003, 'IEAG': 0.0003, 'IBTM': 0.0003,
    },
    'financing': {
        'long_rate': 0.0488,    # same as equity CFDs
        'short_rate': 0.0088,
        'net_daily': (0.0488 - 0.0088) / 365,
    },
    'point_sizes': {k: 1.0 for k in [
        'SHY', 'IEI', 'IEF', 'TLT', 'LQD', 'HYG', 'TIP',
        'BWX', 'EMB', 'IGLT', 'IEAG', 'IBTM',
    ]},
    # Duration groupings for analysis
    'sectors': {
        'Short Duration': ['SHY'],
        'Medium Duration': ['IEI', 'IBTM'],
        'Long Duration': ['IEF', 'TLT'],
        'Credit': ['LQD', 'HYG'],
        'Inflation': ['TIP'],
        'International': ['BWX', 'EMB', 'IGLT', 'IEAG'],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY — single lookup for the backtest tab
# ═══════════════════════════════════════════════════════════════════════════

ASSET_CLASSES = {
    'equity': EQUITY,
    'fx': FX,
    'commodities': COMMODITIES,
    'fixed_income': FIXED_INCOME,
}

# Display order for the UI dropdown
ASSET_CLASS_OPTIONS = [
    ('equity', 'Equity Indices'),
    ('fx', 'FX Pairs'),
    ('commodities', 'Commodities'),
    ('fixed_income', 'Fixed Income'),
]


def get_tradeable_instruments(asset_class_key: str) -> list[str]:
    """Return the list of tradeable instrument codes (excludes reference-only)."""
    cfg = ASSET_CLASSES[asset_class_key]
    exclude = set(cfg.get('reference_only', []))
    return [k for k in cfg['instruments'] if k not in exclude]


def get_display_name(asset_class_key: str, code: str) -> str:
    """Look up display name for an instrument code."""
    cfg = ASSET_CLASSES[asset_class_key]
    return cfg['instruments'].get(code, cfg.get('reference_instruments', {}).get(code, code))
