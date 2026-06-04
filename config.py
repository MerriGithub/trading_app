"""
config.py — Algorithm parameters and legacy equity instrument mappings.

Role vs asset_configs.py
-------------------------
This file holds two things that do not belong in ``asset_configs.py``:

1. **Algorithm parameters** (``PARAMS``) — constants that control signal
   computation and position sizing.  These are the knobs the backtest and
   signal engine read.  Changing them affects the whole app.

2. **Legacy equity mappings** (``INSTRUMENTS``, ``SPREADS``, etc.) — the
   original flat-dict format used before ``asset_configs.py`` introduced the
   multi-asset nested structure.  Older engine code still imports from here
   directly (e.g. ``from config import INSTRUMENTS``).  Do not remove these
   symbols without auditing all importers.

Do NOT add new instrument definitions here — use ``asset_configs.py`` instead.
"""

# ── Legacy instrument mapping (equity indices only) ───────────────────────
# Maps internal trading codes to Yahoo Finance tickers for data fetching.
# These are replicated in asset_configs.EQUITY for the multi-asset engine;
# this dict is kept for backward compatibility with legacy importers.
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

# ── Bid/Ask Spreads ────────────────────────────────────────────────────────
# Bid-ask spreads in index points (from original spreadsheet, row 8).
# Used to calculate round-trip entry cost for equity index instruments.
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

# ── Point Sizes ────────────────────────────────────────────────────────────
# Multiplier converting index points to account-currency P&L.
# All equity indices use 1.0 (cash-settled, GBP/USD/EUR point value = 1).
POINT_SIZES = {label: 1.0 for label in INSTRUMENTS}

# All active equity instrument codes in insertion order — canonical list for
# legacy backtest and display code.
ACTIVE_INSTRUMENTS = list(INSTRUMENTS.keys())

# ── Display Names ──────────────────────────────────────────────────────────
# Standard display names used in the GUI; internal codes are kept for all
# calculations so price data alignment is unambiguous.
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
# Reverse lookup: display name → internal code (used when loading saved portfolios).
DISPLAY_NAMES_INV = {v: k for k, v in DISPLAY_NAMES.items()}

# ── Algorithm Parameters ───────────────────────────────────────────────────
# Central config for all calculation constants. Changing these affects the
# whole app — update CLAUDE.md benchmarks if you change any default.
PARAMS = {
    # Number of trading days assumed in one calendar year.
    # 262 is the standard for European/US markets (approx. 252–262 depending
    # on year and exchange).  Used for annualised vol scaling.
    'trading_days_per_year': 262,

    # Rolling window for volatility calculation.
    # 262 ≈ 1 trading year — a full year of returns gives a stable vol estimate.
    # Shorter windows react faster but produce noisier position sizing.
    'vol_calc_days': 262,

    # Target daily volatility for position scaling (fraction of notional).
    # 0.01 = 1% daily — positions are scaled so each instrument contributes
    # approximately 1% daily vol to the spread.
    'target_daily_vol': 0.01,

    # Rate-of-change look-back period in trading days.
    # Used in momentum / trend filters where applicable.
    'roc_days': 12,

    # Number of points used for the linear trend fit in the UI charts.
    'linear_fit_points': 10,

    # Crossing signal threshold in standard deviations.
    # Entry fires when the spread z-score exceeds this threshold.
    # Confirmed optimum from WF research: 2.0 SD for all asset classes.
    'xing_tolerance_sd': 2.0,

    # Days offset used to define the end of the display range in charts.
    'end_of_range_offset': 7,

    # Smoothing periods for the correlation display in the portfolio tab.
    'correl_smoothing': 3,

    # Flat broker margin rate used for hypothetical margin estimates in the UI.
    # This is the legacy default; per-asset margin rates are in account.json.
    'margin_rate': 0.10,
}

# ── Normal Trading Range ───────────────────────────────────────────────────
# NTR in index points (from original spreadsheet) — used in the pre-trade
# scanner to contextualise intraday moves relative to typical range.
NTR = {
    'UKX': 50, 'CBK': 30, 'CEY': 300, 'CFR': 40, 'CMD': 70,
    'CEI': 30, 'COI': 50, 'CRM': 100, 'CIL': 30,
    'CPH': 20, 'CTN': 100, 'CTB': 100,
}
