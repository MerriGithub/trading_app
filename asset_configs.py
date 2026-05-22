"""
asset_configs.py — Instrument definitions and cost models per asset class
=========================================================================

Instrument dict structure (per instrument):
    display         : str   human-readable name
    data_source     : str   'yahoo' | 'csv' | 'broker'
    intraday_ticker : str | None  Yahoo Finance ticker for intraday data
    spread_pct      : float one-way bid-ask spread as fraction
    point_size      : float P&L multiplier per point
    sector          : str   grouping label

Class-level keys (unchanged for backward compat with engine/):
    label           : str
    csv_file        : str   filename under cache/
    financing       : dict  long_rate, short_rate, net_daily
    point_sizes     : dict  code → float  (kept for engine backward compat)
"""


# ═══════════════════════════════════════════════════════════════════════════
# EQUITY INDICES
# ═══════════════════════════════════════════════════════════════════════════

EQUITY = {
    'label':    'Equity Indices',
    'csv_file': 'prices.csv',
    'financing': {
        'long_rate':  0.0488,
        'short_rate': 0.0088,
        'net_daily':  (0.0488 - 0.0088) / 365,
    },
    'point_sizes': {k: 1.0 for k in [
        'UKX', 'CBK', 'CEY', 'CFR', 'CMD', 'CEI', 'COI',
        'CRM', 'CIL', 'CPH', 'CTN', 'CTB',
    ]},
    'instruments': {
        'UKX': {'display': 'FTSE 100',   'data_source': 'yahoo', 'intraday_ticker': '^FTSE',    'spread_pct': 0.00038, 'point_size': 1.0, 'sector': 'Europe', 'cfd_contract_size':  10.0, 'cfd_currency': 'GBP'},
        'CBK': {'display': 'CAC 40',     'data_source': 'yahoo', 'intraday_ticker': '^FCHI',    'spread_pct': 0.00050, 'point_size': 1.0, 'sector': 'Europe', 'cfd_contract_size':  10.0, 'cfd_currency': 'EUR'},
        'CEY': {'display': 'FTSE MIB',   'data_source': 'yahoo', 'intraday_ticker': None,       'spread_pct': 0.00135, 'point_size': 1.0, 'sector': 'Europe', 'cfd_contract_size':   5.0, 'cfd_currency': 'EUR'},
        'CFR': {'display': 'DAX',        'data_source': 'yahoo', 'intraday_ticker': '^GDAXI',   'spread_pct': 0.00022, 'point_size': 1.0, 'sector': 'Europe', 'cfd_contract_size':  25.0, 'cfd_currency': 'EUR'},
        'CMD': {'display': 'IBEX 35',    'data_source': 'yahoo', 'intraday_ticker': '^IBEX',    'spread_pct': 0.00055, 'point_size': 1.0, 'sector': 'Europe', 'cfd_contract_size':  10.0, 'cfd_currency': 'EUR'},
        'CEI': {'display': 'STOXX 50',   'data_source': 'yahoo', 'intraday_ticker': '^STOXX50E','spread_pct': 0.00080, 'point_size': 1.0, 'sector': 'Europe', 'cfd_contract_size':  10.0, 'cfd_currency': 'EUR'},
        'COI': {'display': 'SMI',        'data_source': 'yahoo', 'intraday_ticker': '^SSMI',    'spread_pct': 0.00042, 'point_size': 1.0, 'sector': 'Europe', 'cfd_contract_size':  10.0, 'cfd_currency': 'CHF'},
        'CRM': {'display': 'HSI',        'data_source': 'yahoo', 'intraday_ticker': '^HSI',     'spread_pct': 0.00050, 'point_size': 1.0, 'sector': 'Asia',   'cfd_contract_size':  50.0, 'cfd_currency': 'HKD'},
        'CIL': {'display': 'ASX 200',    'data_source': 'yahoo', 'intraday_ticker': '^AXJO',    'spread_pct': 0.00050, 'point_size': 1.0, 'sector': 'Asia',   'cfd_contract_size':  25.0, 'cfd_currency': 'AUD'},
        'CPH': {'display': 'NASDAQ 100', 'data_source': 'yahoo', 'intraday_ticker': '^NDX',     'spread_pct': 0.00010, 'point_size': 1.0, 'sector': 'US',     'cfd_contract_size':  20.0, 'cfd_currency': 'USD'},
        'CTN': {'display': 'S&P 500',    'data_source': 'yahoo', 'intraday_ticker': '^GSPC',    'spread_pct': 0.00009, 'point_size': 1.0, 'sector': 'US',     'cfd_contract_size':  50.0, 'cfd_currency': 'USD'},
        'CTB': {'display': 'Dow Jones',  'data_source': 'yahoo', 'intraday_ticker': '^DJI',     'spread_pct': 0.00013, 'point_size': 1.0, 'sector': 'US',     'cfd_contract_size':   5.0, 'cfd_currency': 'USD'},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FX PAIRS
# ═══════════════════════════════════════════════════════════════════════════

FX = {
    'label':    'FX Pairs',
    'csv_file': 'fx_prices.csv',
    'financing': {
        'long_rate':  0.018,
        'short_rate': -0.018,   # negative = both sides pay swap cost
        'net_daily':  (0.018 + 0.018) / 365,
    },
    'point_sizes': {k: 1.0 for k in [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD',
        'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY', 'EURCHF', 'AUDNZD',
    ]},
    'instruments': {
        'EURUSD': {'display': 'EUR/USD', 'data_source': 'csv', 'intraday_ticker': 'EURUSD=X', 'spread_pct': 0.0001,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'USD'},
        'GBPUSD': {'display': 'GBP/USD', 'data_source': 'csv', 'intraday_ticker': 'GBPUSD=X', 'spread_pct': 0.0001,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'USD'},
        'USDJPY': {'display': 'USD/JPY', 'data_source': 'csv', 'intraday_ticker': 'USDJPY=X', 'spread_pct': 0.0001,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'JPY'},
        'USDCHF': {'display': 'USD/CHF', 'data_source': 'csv', 'intraday_ticker': 'USDCHF=X', 'spread_pct': 0.00015, 'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'CHF'},
        'AUDUSD': {'display': 'AUD/USD', 'data_source': 'csv', 'intraday_ticker': 'AUDUSD=X', 'spread_pct': 0.00015, 'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'USD'},
        'NZDUSD': {'display': 'NZD/USD', 'data_source': 'csv', 'intraday_ticker': 'NZDUSD=X', 'spread_pct': 0.0002,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'USD'},
        'USDCAD': {'display': 'USD/CAD', 'data_source': 'csv', 'intraday_ticker': 'USDCAD=X', 'spread_pct': 0.00015, 'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'CAD'},
        'EURGBP': {'display': 'EUR/GBP', 'data_source': 'csv', 'intraday_ticker': 'EURGBP=X', 'spread_pct': 0.0002,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'GBP'},
        'EURJPY': {'display': 'EUR/JPY', 'data_source': 'csv', 'intraday_ticker': 'EURJPY=X', 'spread_pct': 0.0002,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'JPY'},
        'GBPJPY': {'display': 'GBP/JPY', 'data_source': 'csv', 'intraday_ticker': 'GBPJPY=X', 'spread_pct': 0.0003,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'JPY'},
        'EURCHF': {'display': 'EUR/CHF', 'data_source': 'csv', 'intraday_ticker': 'EURCHF=X', 'spread_pct': 0.0002,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'CHF'},
        'AUDNZD': {'display': 'AUD/NZD', 'data_source': 'csv', 'intraday_ticker': 'AUDNZD=X', 'spread_pct': 0.0003,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'NZD'},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# COMMODITIES
# ═══════════════════════════════════════════════════════════════════════════

COMMODITIES = {
    'label':    'Commodities',
    'csv_file': 'commodity_prices.csv',
    'financing': {
        'long_rate':  0.0488,
        'short_rate': 0.0088,
        'net_daily':  (0.0488 - 0.0088) / 365,
    },
    'point_sizes': {k: 1.0 for k in [
        'WTI', 'BRENT', 'NATGAS', 'GOLD', 'SILVER', 'PLATINUM',
        'COPPER', 'PALLADIUM', 'WHEAT', 'CORN', 'SOYBEANS',
        'COFFEE', 'SUGAR',
    ]},
    'instruments': {
        # Energy
        'WTI':       {'display': 'WTI Crude',    'data_source': 'csv', 'intraday_ticker': 'CL=F',  'spread_pct': 0.0005,  'point_size': 1.0, 'sector': 'Energy',      'cfd_contract_size':    1000.0, 'cfd_currency': 'USD'},
        'BRENT':     {'display': 'Brent Crude',  'data_source': 'csv', 'intraday_ticker': 'BZ=F',  'spread_pct': 0.0005,  'point_size': 1.0, 'sector': 'Energy',      'cfd_contract_size':    1000.0, 'cfd_currency': 'USD'},
        'NATGAS':    {'display': 'Natural Gas',  'data_source': 'csv', 'intraday_ticker': 'NG=F',  'spread_pct': 0.0020,  'point_size': 1.0, 'sector': 'Energy',      'cfd_contract_size':   10000.0, 'cfd_currency': 'USD'},
        # Precious metals
        'GOLD':      {'display': 'Gold',         'data_source': 'csv', 'intraday_ticker': 'GC=F',  'spread_pct': 0.0003,  'point_size': 1.0, 'sector': 'Precious',    'cfd_contract_size':     100.0, 'cfd_currency': 'USD'},
        'SILVER':    {'display': 'Silver',       'data_source': 'csv', 'intraday_ticker': 'SI=F',  'spread_pct': 0.0005,  'point_size': 1.0, 'sector': 'Precious',    'cfd_contract_size':    5000.0, 'cfd_currency': 'USD'},
        'PLATINUM':  {'display': 'Platinum',     'data_source': 'csv', 'intraday_ticker': 'PL=F',  'spread_pct': 0.0010,  'point_size': 1.0, 'sector': 'Precious',    'cfd_contract_size':      50.0, 'cfd_currency': 'USD'},
        # Industrial metals
        'COPPER':    {'display': 'Copper',       'data_source': 'csv', 'intraday_ticker': 'HG=F',  'spread_pct': 0.0005,  'point_size': 1.0, 'sector': 'Industrial',  'cfd_contract_size':   25000.0, 'cfd_currency': 'USD'},
        'PALLADIUM': {'display': 'Palladium',    'data_source': 'csv', 'intraday_ticker': 'PA=F',  'spread_pct': 0.0015,  'point_size': 1.0, 'sector': 'Industrial',  'cfd_contract_size':     100.0, 'cfd_currency': 'USD'},
        # Agriculture
        'WHEAT':     {'display': 'Wheat',        'data_source': 'csv', 'intraday_ticker': 'ZW=F',  'spread_pct': 0.0010,  'point_size': 1.0, 'sector': 'Agriculture', 'cfd_contract_size':    5000.0, 'cfd_currency': 'USD'},
        'CORN':      {'display': 'Corn',         'data_source': 'csv', 'intraday_ticker': 'ZC=F',  'spread_pct': 0.0008,  'point_size': 1.0, 'sector': 'Agriculture', 'cfd_contract_size':    5000.0, 'cfd_currency': 'USD'},
        'SOYBEANS':  {'display': 'Soybeans',     'data_source': 'csv', 'intraday_ticker': 'ZS=F',  'spread_pct': 0.0008,  'point_size': 1.0, 'sector': 'Agriculture', 'cfd_contract_size':    5000.0, 'cfd_currency': 'USD'},
        'COFFEE':    {'display': 'Coffee',       'data_source': 'csv', 'intraday_ticker': 'KC=F',  'spread_pct': 0.0012,  'point_size': 1.0, 'sector': 'Agriculture', 'cfd_contract_size':   37500.0, 'cfd_currency': 'USD'},
        'SUGAR':     {'display': 'Sugar',        'data_source': 'csv', 'intraday_ticker': 'SB=F',  'spread_pct': 0.0012,  'point_size': 1.0, 'sector': 'Agriculture', 'cfd_contract_size':  112000.0, 'cfd_currency': 'USD'},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FIXED INCOME
# ═══════════════════════════════════════════════════════════════════════════

FIXED_INCOME = {
    'label':    'Fixed Income',
    'csv_file': 'fi_prices.csv',
    'financing': {
        'long_rate':  0.0488,
        'short_rate': 0.0088,
        'net_daily':  (0.0488 - 0.0088) / 365,
    },
    'point_sizes': {k: 1.0 for k in [
        'SHY', 'IEI', 'IEF', 'TLT', 'LQD', 'HYG', 'TIP',
        'BWX', 'EMB', 'IGLT', 'IEAG', 'IBTM',
    ]},
    # Reference yield indices — NOT tradeable, excluded from basket search
    'reference_only': ['UST10Y', 'UST30Y', 'UST5Y'],
    'reference_instruments': {
        'UST10Y': 'US 10Y Yield', 'UST30Y': 'US 30Y Yield', 'UST5Y': 'US 5Y Yield',
    },
    'instruments': {
        # US duration ladder
        'SHY':  {'display': 'US 1-3Y',          'data_source': 'csv', 'intraday_ticker': 'SHY',  'spread_pct': 0.0001, 'point_size': 1.0, 'sector': 'Short Duration',  'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        'IEI':  {'display': 'US 3-7Y',          'data_source': 'csv', 'intraday_ticker': 'IEI',  'spread_pct': 0.0001, 'point_size': 1.0, 'sector': 'Medium Duration', 'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        'IEF':  {'display': 'US 7-10Y',         'data_source': 'csv', 'intraday_ticker': 'IEF',  'spread_pct': 0.0001, 'point_size': 1.0, 'sector': 'Long Duration',   'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        'TLT':  {'display': 'US 20+Y',          'data_source': 'csv', 'intraday_ticker': 'TLT',  'spread_pct': 0.0002, 'point_size': 1.0, 'sector': 'Long Duration',   'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        # US credit
        'LQD':  {'display': 'US IG Corp',       'data_source': 'csv', 'intraday_ticker': 'LQD',  'spread_pct': 0.0002, 'point_size': 1.0, 'sector': 'Credit',          'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        'HYG':  {'display': 'US HY Corp',       'data_source': 'csv', 'intraday_ticker': 'HYG',  'spread_pct': 0.0003, 'point_size': 1.0, 'sector': 'Credit',          'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        # US inflation
        'TIP':  {'display': 'US TIPS',          'data_source': 'csv', 'intraday_ticker': 'TIP',  'spread_pct': 0.0002, 'point_size': 1.0, 'sector': 'Inflation',       'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        # International
        'BWX':  {'display': 'Intl Treasury',    'data_source': 'csv', 'intraday_ticker': 'BWX',  'spread_pct': 0.0003, 'point_size': 1.0, 'sector': 'International',   'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        'EMB':  {'display': 'EM Bonds',         'data_source': 'csv', 'intraday_ticker': 'EMB',  'spread_pct': 0.0003, 'point_size': 1.0, 'sector': 'International',   'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
        # UK/Europe
        'IGLT': {'display': 'UK Gilts',         'data_source': 'csv', 'intraday_ticker': 'IGLT', 'spread_pct': 0.0003, 'point_size': 1.0, 'sector': 'International',   'cfd_contract_size': 1.0, 'cfd_currency': 'GBP'},
        'IEAG': {'display': 'EUR Govt',         'data_source': 'csv', 'intraday_ticker': 'IEAG', 'spread_pct': 0.0003, 'point_size': 1.0, 'sector': 'International',   'cfd_contract_size': 1.0, 'cfd_currency': 'EUR'},
        'IBTM': {'display': 'US Trsy 7-10 (L)', 'data_source': 'csv', 'intraday_ticker': 'IBTM', 'spread_pct': 0.0003, 'point_size': 1.0, 'sector': 'Medium Duration', 'cfd_contract_size': 1.0, 'cfd_currency': 'USD'},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FI EXCLUSION LIST
# ═══════════════════════════════════════════════════════════════════════════

FI_EXCLUDE = frozenset({'UST10Y', 'UST30Y', 'UST5Y', 'IBTM'})


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

ASSET_CLASSES = {
    'equity':       EQUITY,
    'fx':           FX,
    'commodities':  COMMODITIES,
    'fixed_income': FIXED_INCOME,
}

ASSET_CLASS_OPTIONS = [
    ('equity',       'Equity Indices'),
    ('fx',           'FX Pairs'),
    ('commodities',  'Commodities'),
    ('fixed_income', 'Fixed Income'),
]

_KEY_ALIASES: dict[str, str] = {'commodity': 'commodities', 'fi': 'fixed_income'}


# ═══════════════════════════════════════════════════════════════════════════
# LOOKUP FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_spread_cost_lookup(
    instruments: list[str],
    latest_prices: dict[str, float],
    asset_key: str = 'equity',
) -> dict[str, float]:
    """
    Return per-instrument spread cost (one-way %, as fraction).
    Reads spread_pct from nested instrument dict in ASSET_CLASSES.
    Falls back to account.get_spread_cost_fallback() for unknowns.
    """
    from account import get_spread_cost_fallback
    fallback = get_spread_cost_fallback()

    lookup: dict[str, float] = {}
    for inst in instruments:
        found = False
        for cfg in ASSET_CLASSES.values():
            if inst in cfg.get('instruments', {}):
                inst_cfg = cfg['instruments'][inst]
                if isinstance(inst_cfg, dict):
                    lookup[inst] = inst_cfg.get('spread_pct', fallback)
                else:
                    lookup[inst] = fallback
                found = True
                break
        if not found:
            lookup[inst] = fallback
    return lookup


def basket_spread_cost(
    long_combo: tuple[int, ...],
    short_combo: tuple[int, ...],
    instruments: list[str],
    spread_cost_lookup: dict[str, float],
) -> float:
    """
    Compute the round-trip basket spread cost as a fraction.

    Formula: 4 × mean(spread_cost_pct_i for i in all combo instruments)
    Factor of 4 = ×2 round-trip, ×2 long + short sides.
    """
    all_instr = [instruments[i] for i in long_combo] + [instruments[i] for i in short_combo]
    if not all_instr:
        return 0.0
    mean_cost = sum(spread_cost_lookup.get(inst, 0.001) for inst in all_instr) / len(all_instr)
    return 4.0 * mean_cost


def get_tradeable_instruments(asset_class_key: str) -> list[str]:
    """Return tradeable instrument codes (excludes reference-only)."""
    cfg = ASSET_CLASSES[asset_class_key]
    exclude = set(cfg.get('reference_only', []))
    return [k for k in cfg['instruments'] if k not in exclude]


def get_display_name(asset_class_key: str, code: str) -> str:
    """Look up display name for an instrument code."""
    cfg = ASSET_CLASSES[asset_class_key]
    inst_cfg = cfg['instruments'].get(code)
    if inst_cfg is not None:
        if isinstance(inst_cfg, dict):
            return inst_cfg.get('display', code)
        return inst_cfg  # legacy string fallback
    return cfg.get('reference_instruments', {}).get(code, code)


def get_data_source(code: str) -> str:
    """Return the data source key for an instrument ('yahoo', 'csv', 'broker')."""
    for cfg in ASSET_CLASSES.values():
        if code in cfg.get('instruments', {}):
            inst_cfg = cfg['instruments'][code]
            if isinstance(inst_cfg, dict):
                return inst_cfg.get('data_source', 'csv')
    return 'csv'


def get_intraday_ticker(code: str) -> str | None:
    """Return the Yahoo Finance intraday ticker for an instrument, or None."""
    for cfg in ASSET_CLASSES.values():
        if code in cfg.get('instruments', {}):
            inst_cfg = cfg['instruments'][code]
            if isinstance(inst_cfg, dict):
                return inst_cfg.get('intraday_ticker')
    return None


def get_cfd_contract_size(code: str) -> tuple[float, str]:
    """
    Return (cfd_contract_size, cfd_currency) for an instrument code.
    Falls back to (1.0, 'USD') if not found.
    """
    for cfg in ASSET_CLASSES.values():
        if code in cfg.get('instruments', {}):
            inst_cfg = cfg['instruments'][code]
            if isinstance(inst_cfg, dict):
                return (
                    inst_cfg.get('cfd_contract_size', 1.0),
                    inst_cfg.get('cfd_currency', 'USD'),
                )
    return (1.0, 'USD')


def get_cross_asset_label(long_class: str, short_class: str) -> str:
    """Human-readable label for a cross-asset category pair."""
    labels = {
        'equity':       'Equity',
        'fx':           'FX',
        'commodity':    'Commodity',
        'commodities':  'Commodity',
        'fi':           'Fixed Income',
        'fixed_income': 'Fixed Income',
    }
    return f"{labels.get(long_class, long_class)} vs {labels.get(short_class, short_class)}"
