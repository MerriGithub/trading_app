"""
asset_configs.py — Instrument definitions and cost models per asset class
=========================================================================

This is the **authoritative source** for all instrument definitions across
equity indices, FX pairs, commodities, and fixed income.  Multi-asset changes
must go here, NOT in ``config.py`` (which holds algorithm parameters only).

Relation to config.py
---------------------
- ``asset_configs.py`` — instrument definitions, spread costs, CFD specs,
  financing structures, exclusion lists, and WF-validated scoring modes.
- ``config.py`` — algorithm parameters (vol_window, xing_sd, etc.) and
  legacy equity mappings imported by older engine code.

Per-instrument dict structure
-----------------------------
    display         : str   human-readable name for the UI
    data_source     : str   'yahoo' | 'csv' — determines how DataRegistry loads prices
    intraday_ticker : str | None   Yahoo Finance ticker for live/intraday data
    spread_pct      : float one-way bid-ask spread as a fraction of price
    point_size      : float P&L multiplier per point move (typically 1.0)
    sector          : str   grouping label for search/display

Per-class-level dict keys (unchanged for backward compat with engine/)
----------------------------------------------------------------------
    label           : str
    csv_file        : str   filename under cache/ (e.g. 'prices.csv')
    financing       : dict  long_rate, short_rate, net_daily (annual fractions)
    point_sizes     : dict  code → float  (flat dict kept for engine backward compat)

Exclusion lists
---------------
``COMMODITY_EXCLUDE``: instruments in the CSV but excluded from all pair
generation, backtests, and walk-forward.  Reason: WTI had a catastrophic
negative price event in April 2020 (daily return ~ −300%) that contaminates
exhaustive search results.

``FI_EXCLUDE``: fixed income instruments excluded from pair generation due
to data quality issues.  IBTM had irregular pricing that skews signals.
Reference-only tickers (UST10Y etc.) are also in this set.

Scoring mode constants
----------------------
Walk-forward validated defaults (see CLAUDE.md and Obsidian Project Reference):
    equities:       contrarian  ρ=+0.208, p~0,     EXIT_SD=2.0 scalp regime
    commodities:    contrarian  ρ=+0.122, p=0.0009
    equity × FX:    contrarian  ρ=+0.053, p=0.0030
    commodities×FI: composite   ρ=+0.069, p=0.0016
    FX, FI:         composite   ρ≈0 — no validated predictor
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
        # IG FX spreadbet convention: price expressed in pips for margin calculation.
        # 4-decimal pairs (non-JPY): IG price = Yahoo rate × 10000
        # JPY pairs: IG price = Yahoo rate × 100
        #
        # IG CFD contract specs (confirmed 2026-05-23):
        # Non-JPY majors: 1 contract = $10/point at 5dp pricing → cfd_contract_size = 10, cfd_currency = USD/GBP
        # JPY pairs: contract size unverified — left at 100000 (lot size) pending confirmation.
        # Source: IG platform deal ticket.
        'EURUSD': {'display': 'EUR/USD', 'data_source': 'csv', 'intraday_ticker': 'EURUSD=X', 'spread_pct': 0.0001,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'USD', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price': 10850,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
        'GBPUSD': {'display': 'GBP/USD', 'data_source': 'csv', 'intraday_ticker': 'GBPUSD=X', 'spread_pct': 0.0001,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'USD', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price': 12700,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
        'USDJPY': {'display': 'USD/JPY', 'data_source': 'csv', 'intraday_ticker': 'USDJPY=X', 'spread_pct': 0.0001,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'JPY', 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price': 14500,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 100,   1)},
        'USDCHF': {'display': 'USD/CHF', 'data_source': 'csv', 'intraday_ticker': 'USDCHF=X', 'spread_pct': 0.00015, 'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'CHF', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price':  8900,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
        'AUDUSD': {'display': 'AUD/USD', 'data_source': 'csv', 'intraday_ticker': 'AUDUSD=X', 'spread_pct': 0.00015, 'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'USD', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price':  6400,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
        'NZDUSD': {'display': 'NZD/USD', 'data_source': 'csv', 'intraday_ticker': 'NZDUSD=X', 'spread_pct': 0.0002,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'USD', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price':  5900,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
        'USDCAD': {'display': 'USD/CAD', 'data_source': 'csv', 'intraday_ticker': 'USDCAD=X', 'spread_pct': 0.00015, 'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'CAD', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price': 13600,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
        'EURGBP': {'display': 'EUR/GBP', 'data_source': 'csv', 'intraday_ticker': 'EURGBP=X', 'spread_pct': 0.0002,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'GBP', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price':  8550,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
        'EURJPY': {'display': 'EUR/JPY', 'data_source': 'csv', 'intraday_ticker': 'EURJPY=X', 'spread_pct': 0.0002,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'JPY', 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price': 11750,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 100,   1)},
        'GBPJPY': {'display': 'GBP/JPY', 'data_source': 'csv', 'intraday_ticker': 'GBPJPY=X', 'spread_pct': 0.0003,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 100000.0, 'cfd_currency': 'JPY', 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price': 16150,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 100,   1)},
        'EURCHF': {'display': 'EUR/CHF', 'data_source': 'csv', 'intraday_ticker': 'EURCHF=X', 'spread_pct': 0.0002,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'CHF', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price':  9500,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
        'AUDNZD': {'display': 'AUD/NZD', 'data_source': 'csv', 'intraday_ticker': 'AUDNZD=X', 'spread_pct': 0.0003,  'point_size': 1.0, 'sector': 'G10', 'cfd_contract_size': 10, 'cfd_currency': 'NZD', 'cfd_price_dp': 5, 'cfd_min_contracts': 0.5, 'ig_price_override': True, 'ig_default_price': 10850,  'ig_price_unit': 'pips', 'ig_price_conversion': lambda p: round(p * 10000, 1)},
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
        'NATGAS':    {
            'display': 'Natural Gas',  'data_source': 'csv', 'intraday_ticker': 'NG=F',
            'spread_pct': 0.0020,  'point_size': 1.0, 'sector': 'Energy',
            'cfd_contract_size': 10000.0, 'cfd_currency': 'USD',
            'ig_price_override': True,
            'ig_default_price':  80.0,
            'ig_market_name':    'Natural Gas (pence per therm)',
            'ig_price_unit':     'pence_per_therm',
        },
        # Precious metals
        'GOLD':      {
            'display': 'Gold',         'data_source': 'csv', 'intraday_ticker': 'GC=F',
            'spread_pct': 0.0003,  'point_size': 1.0, 'sector': 'Precious',
            'cfd_contract_size': 100.0, 'cfd_currency': 'USD',
            'ig_price_override': True,
            'ig_default_price':  3500.0,
            'ig_market_name':    'Gold (£ per troy oz)',
            'ig_price_unit':     'GBP_per_oz',
        },
        'SILVER':    {
            'display': 'Silver',       'data_source': 'csv', 'intraday_ticker': 'SI=F',
            'spread_pct': 0.0005,  'point_size': 1.0, 'sector': 'Precious',
            'cfd_contract_size': 5000.0, 'cfd_currency': 'USD',
            'ig_price_override': True,
            'ig_default_price':  5500.0,
            'ig_market_name':    'Silver (pence per troy oz)',
            'ig_price_unit':     'pence_per_oz',
        },
        'PLATINUM':  {
            'display': 'Platinum',     'data_source': 'csv', 'intraday_ticker': 'PL=F',
            'spread_pct': 0.0010,  'point_size': 1.0, 'sector': 'Precious',
            'cfd_contract_size': 50.0, 'cfd_currency': 'USD',
            'ig_price_override': True,
            'ig_default_price':  1500.0,
            'ig_market_name':    'Platinum (£ per troy oz)',
            'ig_price_unit':     'GBP_per_oz',
        },
        # Industrial metals
        'COPPER':    {
            'display': 'Copper',       'data_source': 'csv', 'intraday_ticker': 'HG=F',
            'spread_pct': 0.0005,  'point_size': 1.0, 'sector': 'Industrial',
            'cfd_contract_size': 25000.0, 'cfd_currency': 'USD',
            'spreadbet_min_stake': 0.04,
            'margin_rate':         0.05,
            'ig_price_override':   True,
            'ig_default_price':    63975.0,
            'ig_market_name':      'Copper (pence per tonne)',
            'ig_price_unit':       'pence_per_tonne',
        },
        'PALLADIUM': {
            'display': 'Palladium',    'data_source': 'csv', 'intraday_ticker': 'PA=F',
            'spread_pct': 0.0015,  'point_size': 1.0, 'sector': 'Industrial',
            'cfd_contract_size': 100.0, 'cfd_currency': 'USD',
            'ig_price_override': True,
            'ig_default_price':  1100.0,
            'ig_market_name':    'Palladium (£ per troy oz)',
            'ig_price_unit':     'GBP_per_oz',
        },
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

# ── Exclusion lists ───────────────────────────────────────────────────────

# IBTM excluded: pricing data has irregular gaps that produce spurious signals.
# UST*Y excluded: reference-only yield indices, not tradeable instruments.
FI_EXCLUDE = frozenset({'UST10Y', 'UST30Y', 'UST5Y', 'IBTM'})

# WTI excluded: April 2020 negative price event produced a daily return of
# approximately −300%, which corrupts rolling vol, z-scores, and exhaustive
# search results for any commodity pair containing WTI.
# Excluded from: Tab 8 commodity backtest, Tab 9 walk-forward, Tab 10 scanner.
# Individual WTI monitoring (Tabs 1 and 2) is intentionally unaffected.
COMMODITY_EXCLUDE = frozenset({'WTI'})

# ── WF-validated scoring mode defaults (per-asset-class) ─────────────────
# Tabs 8 and 9 auto-default to these; warn users on deviation.
# Tabs 10 and 11 do not use a scoring mode selector — see CLAUDE.md reg. F/G.
_DEFAULT_SCORING_MODE = {
    'equity':       'contrarian',   # WF: ρ=+0.208, p~0    — scalp regime EXIT_SD=2.0
    'fx':           'composite',    # WF: ρ≈0 — no validated predictor
    'fixed_income': 'composite',    # WF: ρ≈0 — no validated predictor
    'commodities':  'contrarian',   # WF: ρ=+0.122, p=0.0009
}

# Combination-specific scoring mode defaults for cross-asset pairs.
# Keys are (long_asset_class_key, short_asset_class_key) tuples.
# Both orderings are included so callers don't need to normalise order.
# Evidence basis: Walk-forward ρ tests, 2026-05-25.
CROSS_ASSET_SCORING_MODE: dict[tuple[str, str], str] = {
    ('commodities', 'fx'):           'composite',   # ρ≈0, p=0.331 — mode irrelevant
    ('fx', 'commodities'):           'composite',
    ('commodities', 'fixed_income'): 'composite',   # ρ=+0.069, p=0.0016 — composite positive predictor
    ('fixed_income', 'commodities'): 'composite',
    ('equity', 'fx'):                'contrarian',  # ρ=+0.053, p=0.0030 — contrarian validated
    ('fx', 'equity'):                'contrarian',
    # All untested combinations → 'composite' as neutral default
}


def get_cross_asset_scoring_default(long_ac: str, short_ac: str) -> str:
    """Return the WF-validated scoring mode for a cross-asset pair combination.

    Args:
        long_ac: Asset class key for the long leg (e.g. ``'equity'``).
        short_ac: Asset class key for the short leg (e.g. ``'fx'``).

    Returns:
        ``'contrarian'`` or ``'composite'`` per ``CROSS_ASSET_SCORING_MODE``.
        Defaults to ``'composite'`` for untested combinations.
    """
    return CROSS_ASSET_SCORING_MODE.get((long_ac, short_ac), 'composite')

CROSS_ASSET_COMBINATIONS = [
    ('commodities', 'fx'),           # Primary — Commodity × FX
    ('commodities', 'fixed_income'), # Secondary — future
    ('equity', 'commodities'),       # Secondary — future
]


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
    ('cross_asset',  'Cross-Asset'),
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
    """Return per-instrument one-way bid-ask spread cost as a fraction.

    Reads ``spread_pct`` from the nested instrument dict in ASSET_CLASSES.
    Falls back to ``account.get_spread_cost_fallback()`` for instruments
    not found in any asset class.

    Args:
        instruments: Instrument codes to look up.
        latest_prices: Latest ``{instrument: price}`` dict (accepted for
            interface consistency; not used in the current implementation).
        asset_key: Hint for the primary asset class (unused; kept for
            backward compatibility).

    Returns:
        ``{instrument: spread_fraction}`` dict where spread_fraction is
        the one-way cost (e.g. 0.0003 = 0.03%). Round-trip = 4× this value.
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
    """Compute the round-trip basket spread cost as a fraction of notional.

    Formula: ``4 × mean(spread_cost_pct_i)`` over all instruments in the basket.
    Factor of 4 = ×2 round-trip (enter + exit) × ×2 (long leg + short leg).

    Args:
        long_combo: Index positions in ``instruments`` for the long legs.
        short_combo: Index positions in ``instruments`` for the short legs.
        instruments: Full instrument list (indices are into this list).
        spread_cost_lookup: ``{instrument: one_way_spread_fraction}`` dict
            from ``get_spread_cost_lookup()``.

    Returns:
        Round-trip spread cost as a fraction (e.g. 0.004 = 0.4%).
        Returns 0.0 for an empty basket.
    """
    all_instr = [instruments[i] for i in long_combo] + [instruments[i] for i in short_combo]
    if not all_instr:
        return 0.0
    mean_cost = sum(spread_cost_lookup.get(inst, 0.001) for inst in all_instr) / len(all_instr)
    return 4.0 * mean_cost


def get_tradeable_instruments(asset_class_key: str) -> list[str]:
    """Return tradeable instrument codes for an asset class.

    Excludes instruments listed in the class-level ``reference_only`` list
    (e.g. UST10Y yield indices in fixed income).

    Args:
        asset_class_key: One of ``'equity'``, ``'fx'``, ``'commodities'``,
            ``'fixed_income'``.

    Returns:
        List of tradeable instrument code strings.
    """
    cfg = ASSET_CLASSES[asset_class_key]
    exclude = set(cfg.get('reference_only', []))
    return [k for k in cfg['instruments'] if k not in exclude]


def get_display_name(asset_class_key: str, code: str) -> str:
    """Return the human-readable display name for an instrument code.

    Args:
        asset_class_key: Asset class to search (e.g. ``'equity'``).
        code: Instrument code (e.g. ``'UKX'``).

    Returns:
        Display name string (e.g. ``'FTSE 100'``), or ``code`` itself if
        not found.
    """
    cfg = ASSET_CLASSES[asset_class_key]
    inst_cfg = cfg['instruments'].get(code)
    if inst_cfg is not None:
        if isinstance(inst_cfg, dict):
            return inst_cfg.get('display', code)
        return inst_cfg  # legacy string fallback
    return cfg.get('reference_instruments', {}).get(code, code)


def get_data_source(code: str) -> str:
    """Return the data source key for an instrument.

    Args:
        code: Instrument code.

    Returns:
        ``'yahoo'`` for instruments with live Yahoo Finance daily data,
        ``'csv'`` for instruments loaded from the cache CSV only.
        Defaults to ``'csv'`` if the instrument is not found.
    """
    for cfg in ASSET_CLASSES.values():
        if code in cfg.get('instruments', {}):
            inst_cfg = cfg['instruments'][code]
            if isinstance(inst_cfg, dict):
                return inst_cfg.get('data_source', 'csv')
    return 'csv'


def get_intraday_ticker(code: str) -> str | None:
    """Return the Yahoo Finance intraday ticker for an instrument.

    Args:
        code: Instrument code (e.g. ``'BRENT'``).

    Returns:
        Yahoo Finance ticker string (e.g. ``'BZ=F'``), or ``None`` if the
        instrument has no configured intraday ticker.
    """
    for cfg in ASSET_CLASSES.values():
        if code in cfg.get('instruments', {}):
            inst_cfg = cfg['instruments'][code]
            if isinstance(inst_cfg, dict):
                return inst_cfg.get('intraday_ticker')
    return None


def get_cfd_contract_size(code: str) -> tuple[float, str]:
    """Return CFD contract size and currency for an instrument.

    Args:
        code: Instrument code (e.g. ``'GOLD'``).

    Returns:
        ``(cfd_contract_size, cfd_currency)`` tuple. Falls back to
        ``(1.0, 'USD')`` if the instrument is not found.
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
    """Return a human-readable label for a cross-asset category pair.

    Args:
        long_class: Asset class key for the long universe.
        short_class: Asset class key for the short universe.

    Returns:
        Label string e.g. ``'Equity vs FX'``.
    """
    labels = {
        'equity':       'Equity',
        'fx':           'FX',
        'commodity':    'Commodity',
        'commodities':  'Commodity',
        'fi':           'Fixed Income',
        'fixed_income': 'Fixed Income',
    }
    return f"{labels.get(long_class, long_class)} vs {labels.get(short_class, short_class)}"
