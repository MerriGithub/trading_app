# --- Instrument Mapping ---
# Maps internal trading codes to Yahoo Finance tickers used for data fetching
INSTRUMENTS = {
    'UKX': '^FTSE',
    'CBK': '^FCHI',
    'CEY': 'FTSEMIB.MI',   # Italian MIB (^MIB30 is discontinued in Yahoo)
    'CFR': '^GDAXI',
    'CMD': '^IBEX',
    'CEI': '^STOXX50E',
    'COI': '^SSMI',
    'CRM': '^HSI',
    'CIL': '^AXJO',
    'CPH': '^NDX',
    'CTN': '^GSPC',         # S&P 500 (^SPX not reliable in yfinance)
    'CTB': '^DJI',
}

# --- Bid/Ask Spreads ---
# Bid-ask spreads in index points (from spreadsheet row 8)
# Used to calculate round-trip entry cost for each instrument
SPREADS = {
    'UKX': 3.0,
    'CBK': 4.0,
    'CEY': 50.0,
    'CFR': 4.0,
    'CMD': 6.0,
    'CEI': 4.0,
    'COI': 5.0,
    'CRM': 10.0,
    'CIL': 4.0,
    'CPH': 2.0,
    'CTN': 0.5,
    'CTB': 5.0,
}

# --- Point Sizes ---
# Multiplier converting index points to currency P&L (all 1.0 for cash-settled indices)
POINT_SIZES = {label: 1.0 for label in INSTRUMENTS}

# All active instrument codes in insertion order — used as the canonical list throughout the app
ACTIVE_INSTRUMENTS = list(INSTRUMENTS.keys())

# --- Display Names ---
# Standard display names used in the GUI (internal codes kept for calculations)
DISPLAY_NAMES = {
    'UKX': 'FTSE',
    'CBK': 'CAC',
    'CEY': 'MIB',
    'CFR': 'DAX',
    'CMD': 'IBEX',
    'CEI': 'STOXX50',
    'COI': 'SMI',
    'CRM': 'HSI',
    'CIL': 'ASX',
    'CPH': 'NDX',
    'CTN': 'SPX',
    'CTB': 'DJI',
}
# Reverse lookup: display name → internal code (used when loading saved portfolios)
DISPLAY_NAMES_INV = {v: k for k, v in DISPLAY_NAMES.items()}

# --- Algorithm Parameters ---
# Central config for all calculation constants; changing these affects the whole app
PARAMS = {
    'trading_days_per_year': 262,
    'vol_calc_days': 262,       # Rolling window for volatility (1 year)
    'target_daily_vol': 0.01,   # 1% daily vol target for position scaling
    'roc_days': 12,             # Rate-of-change look-back period
    'linear_fit_points': 10,    # Points used for linear trend fit
    'xing_tolerance_sd': 2.0,   # Crossing signal threshold in standard deviations
    'end_of_range_offset': 7,   # Days offset for range end
    'correl_smoothing': 3,      # Smoothing periods for correlation display
    'margin_rate': 0.10,        # Broker margin rate used for hypothetical margin estimates
}

# --- Normal Trading Range ---
# NTR in index points (from spreadsheet) — used in the pre-trade scanner to contextualise moves
NTR = {
    'UKX': 50, 'CBK': 30, 'CEY': 300, 'CFR': 40, 'CMD': 70,
    'CEI': 30, 'COI': 50, 'CRM': 100, 'CIL': 30,
    'CPH': 20, 'CTN': 100, 'CTB': 100,
}
