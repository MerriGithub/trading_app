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

# Bid-ask spreads in index points (from spreadsheet row 8)
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

POINT_SIZES = {label: 1.0 for label in INSTRUMENTS}

ACTIVE_INSTRUMENTS = list(INSTRUMENTS.keys())

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
DISPLAY_NAMES_INV = {v: k for k, v in DISPLAY_NAMES.items()}

PARAMS = {
    'trading_days_per_year': 262,
    'vol_calc_days': 262,       # Rolling window for volatility (1 year)
    'target_daily_vol': 0.01,   # 1% daily vol target for position scaling
    'roc_days': 12,             # Rate-of-change look-back period
    'linear_fit_points': 10,    # Points used for linear trend fit
    'xing_tolerance_sd': 2.0,   # Crossing signal threshold in standard deviations
    'end_of_range_offset': 7,   # Days offset for range end
    'correl_smoothing': 3,      # Smoothing periods for correlation display
}
