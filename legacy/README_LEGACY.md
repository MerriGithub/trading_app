# Legacy App — Equity Index Spread Trading Monitor

**Status: FROZEN as of 2026-05-17 — do not modify**

Equity index spreads only (12 instruments from config.py).

## To run (bash/WSL/Mac):
    cd legacy
    streamlit run app_legacy.py --server.port 8501

## To run (Windows PowerShell):
    cd legacy
    .\run_legacy.bat

## Benchmark results:
- Equity 3v3, exit=0.0: Gross +0.353%, Net -1.687% (2,000 baskets, seed=42)
- FI walk-forward GWR: ~70% (507/1,430 active pairs, 11 instruments)

## Data locations:
- Price CSVs: ./cache/  (symlinked to ../cache — see setup below)
- Account state: ./account.json  (own copy — independent of new app)
- Trade journal: ./trade_journal.json

## First-time setup — cache symlink required:
Tabs 8 (Backtest) and 9 (Walk-Forward) resolve _CACHE_DIR relative to
app_legacy.py, so they look for legacy/cache/. Create a symlink:

    # bash/WSL/Mac:
    cd legacy && ln -s ../cache cache

    # Windows (run as Administrator):
    cd legacy && mklink /D cache .\..\cache

The test scripts (test_cross_asset_replication.py etc.) use the same
path and also require this symlink.
