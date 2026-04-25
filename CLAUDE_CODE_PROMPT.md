# Claude Code Task: Performance Optimisation, Asymmetric Search & Backtest Tab

## Context

This is a Streamlit application for trading long/short spreads across global equity indices using CFDs. The core files are:

- `config.py` — instrument definitions, parameters, display names
- `calculations.py` — returns, volatility, scaling, crossing signals, portfolio stats
- `search.py` — exhaustive basket combination search engine (the main performance bottleneck)
- `app.py` — Streamlit UI with 7 tabs
- `data.py` — Yahoo Finance data loading and caching
- `account.py`, `journal.py`, `saved.py`, `stake_calc.py` — supporting modules

The strategy's crossing signal has been backtested across equity indices (Q1), FX pairs (Q5), and will be extended to commodities (Q6), fixed income (Q7), and cross-asset baskets (Q8). Price data CSVs exist for each asset class: `prices.csv` (equities), `fx_prices.csv`, `commodity_prices.csv`, `fi_prices.csv`.
These can be found in the cache directory (prices.csv = Equity indices)
Read `STRATEGY.md` for the full strategy description before starting.

---

## Pre-Built Library Files

The following files have already been written and tested.  They are the foundation for this task — **do not rewrite them from scratch**.  Read each file, understand its API, then integrate it into the existing application.

| File | Purpose | Status |
|------|---------|--------|
| `numba_core.py` | Numba JIT functions + Python reference implementations | **Ready** — drop into project root |
| `asset_configs.py` | Instrument definitions and cost models for FX, commodities, fixed income | **Ready** — drop into project root |
| `backtest.py` | High-level backtest engine: data loading, exhaustive search, regime analysis, sensitivity grid | **Ready** — drop into project root |
| `test_numba_parity.py` | Parity test verifying Numba matches Python reference | **Ready** — run FIRST before any integration |

### Step 0: Run the parity test

Before making ANY changes to existing files:

```bash
cp numba_core.py asset_configs.py backtest.py test_numba_parity.py <project_root>/
cd <project_root>
pip install numba>=0.59
python test_numba_parity.py
```

All tests must pass.  If they don't, fix `numba_core.py` until they do.  Do not proceed to integration until the test suite reports `All tests passed`.

---

## Task 1: Integrate numba_core.py into search.py

### Current state of search.py

- `estimate_combinations(min_legs, max_legs, n=12)` — symmetric only
- `run_search()` — takes `min_legs` / `max_legs` (single values for both sides)
- Inner loop is already `n_long x n_short` but the interface doesn't expose this
- Computes 5 proxy metrics via `_batch_scores()` — no actual crossing signal backtest
- Composite score: `z(|LastSD|) + z(TVR) + z(FitDataMinMaxSD)` (search.py line 215)
- No `Config` column (e.g. "3v3") in results

### Required changes

1. **Add `numba_core` import** at the top of `search.py`:
   ```python
   from numba_core import batch_backtest
   ```

2. **Update `estimate_combinations()`** to accept separate long/short params:
   ```python
   def estimate_combinations(
       min_long: int, max_long: int,
       min_short: int, max_short: int,
       n: int = 12,
       min_legs: int | None = None,
       max_legs: int | None = None,
   ) -> int:
   ```
   Use the implementation from `backtest.py` as reference.

3. **Update `run_search()` signature** to accept separate long/short leg counts:
   ```python
   def run_search(
       rets, scalings,
       min_long_legs=3, max_long_legs=4,
       min_short_legs=3, max_short_legs=4,
       ...
       min_legs=None, max_legs=None,  # backward compat
   ):
   ```
   If `min_legs`/`max_legs` are passed, map them to the new params.

4. **Add batch backtest** alongside existing `_batch_scores()`.  Keep `_batch_scores()` — its 5 metrics are displayed in the results table.  Also run:
   ```python
   day_ints = (rets.tail(window_days).dropna(how='any')
               .index.astype(np.int64) // 10**9 // 86400).values.astype(np.int64)
   bt_results = batch_backtest(spread_mat, vol_window, xing_sd, exit_sd, day_ints)
   ```
   Add backtest columns to each record: `Trades`, `WinRate`, `Expectancy`, `AvgHolding`, `PayoffRatio`.

5. **Add `Config` column** showing basket shape (e.g. "3v3", "4v3"):
   ```python
   records.append({ ..., 'Config': f'{n_long}v{n_short}', ... })
   ```

6. **Update composite score** to incorporate backtest metrics:
   ```python
   df['_score'] = (df['_z_LastSD'].abs() + df['_z_TrendVolRatio'] +
                   df['_z_FitDataMinMaxSD'] + df['_z_WinRate'] + df['_z_Expectancy'])
   ```

7. **Update app.py line 688** where `estimate_combinations()` is called — pass the new params.

### Verification

After integration, run the search with 3v3 equity indices (the default).  The scoring-metric columns (ReturnSD, TrendVolRatio, etc.) should be identical to before.  The new backtest columns are additive.

---

## Task 2: Asymmetric Search UI in app.py Tab 5

### Current state (app.py, tab5)

- Single "Legs per side" control: min/max number inputs
- `estimate_combinations(min_legs, max_legs)` called at line 688
- Results table has no Config column

### Required changes

Replace the single "Legs per side" block with:

```python
pcol1, pcol2, pcol3 = st.columns(3)

with pcol1:
    st.markdown("**Long legs**")
    min_long = st.number_input("Min", 2, 6, 3, key='s_min_long')
    max_long = st.number_input("Max", 2, 6, 4, key='s_max_long')

with pcol2:
    st.markdown("**Short legs**")
    symmetric = st.checkbox("Same as long", value=True, key='s_symmetric')
    if symmetric:
        min_short, max_short = min_long, max_long
    else:
        min_short = st.number_input("Min", 2, 6, 3, key='s_min_short')
        max_short = st.number_input("Max", 2, 6, 4, key='s_max_short')

with pcol3:
    st.markdown("**Scale**")
    n_combos = estimate_combinations(min_long, max_long, min_short, max_short)
    st.metric("Combinations", f"{n_combos:,}")
```

Update the `run_search()` call to pass the new params.  Add `Config` to the display columns list.

---

## Task 3: Backtest Tab in app.py

### Goal

Add a new Streamlit tab (tab 8, after Journal).  This is a self-contained backtesting environment for any asset class, replacing the need for standalone Python scripts.

### Implementation

The backtest engine is already built in `backtest.py`.  This task is purely UI — wiring `backtest.py` functions to Streamlit widgets and charts.

#### Imports at the top of app.py

```python
from asset_configs import ASSET_CLASSES, ASSET_CLASS_OPTIONS, get_tradeable_instruments
from backtest import (
    load_asset_prices, prepare_returns, run_backtest,
    run_exhaustive_search, regime_split, sensitivity_grid,
    find_breakeven_financing, aggregate_trades,
)
```

#### Tab definition

Change the tabs line to include the new tab:
```python
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Dashboard", "Analysis", "Stake Calculator", "Portfolio",
    "Search", "Live", "Journal", "Backtest",
])
```

#### Data source selector

```python
with tab8:
    st.header("Backtest")

    asset_key = st.selectbox(
        "Data source",
        [k for k, _ in ASSET_CLASS_OPTIONS],
        format_func=lambda k: dict(ASSET_CLASS_OPTIONS)[k],
        key='bt_asset',
    )
    cfg = ASSET_CLASSES[asset_key]

    uploaded = st.file_uploader("Or upload custom CSV", type=['csv'], key='bt_upload')
```

#### Signal parameters (expander)

```python
    with st.expander("Signal Parameters", expanded=True):
        bc1, bc2, bc3, bc4 = st.columns(4)
        bt_vol_window = bc1.number_input("Vol window", 50, 500, 262, key='bt_vol')
        bt_xing_sd = bc2.number_input("Entry SD", 0.5, 5.0, 2.0, 0.5, key='bt_xing')
        bt_exit_sd = bc3.number_input("Exit SD", 0.0, 2.0, 0.0, 0.5, key='bt_exit',
                                       help="0.0=full reversion. 0.5=partial.")
        bt_fin_rate = bc4.number_input("Financing %pa", 0.0, 10.0,
                                        cfg['financing']['long_rate']*100, 0.1, key='bt_fin')
```

The `exit_sd` parameter is already implemented in `numba_core._detect_trades()`.  `exit_sd=0.0` gives the original behaviour.  `exit_sd=0.5` exits earlier, producing shorter holds and lower financing costs.

#### Basket configuration

Same pattern as Task 2 (separate long/short leg controls with symmetric checkbox).  Add a sample size control:
```python
    bt_sample = st.number_input("Sample size (0=exhaustive)", 0, 50000, 2000, key='bt_sample')
```

#### Run button and results

When "Run Backtest" is clicked:
1. Load prices via `load_asset_prices(cfg['csv_file'])`
2. Prepare returns via `prepare_returns()`
3. Run `run_exhaustive_search()` with all parameters
4. Display summary metrics as `st.metric()` columns
5. Display results table (same format as Search tab + backtest columns)
6. Show charts using Plotly (matching existing app style):
   - Trade return distribution (histogram with gross/net)
   - Holding period distribution (histogram with percentile lines)
   - Regime analysis (bar chart from `regime_split()`)
7. Sensitivity analysis in a collapsible expander using `sensitivity_grid()`

#### "Backtest this" button in Search tab

In tab5, after "Load to Analysis tabs", add:
```python
if st.button("Backtest this", key=f's_bt_{rank}'):
    st.session_state['bt_pending'] = {
        'long_flags': row['_long_flags'],
        'short_flags': row['_short_flags'],
    }
    st.rerun()
```

At the top of tab8, check for `bt_pending` and pre-populate the basket.

#### Export

Add an export button that saves summary JSON and trade list CSV to `backtest_exports/` with timestamped filenames.

---

## Task 4: Wire calculations.py to numba_core

### Current state

- `crossing_signals()` computes dist_sd but does NOT detect trades
- No `crossing_signal_backtest()` function

### Required changes

Add a wrapper that provides a pandas-friendly interface:

```python
def crossing_signal_backtest(
    spread_ret: pd.Series,
    tolerance_sd: float | None = None,
    exit_sd: float = 0.0,
    window: int = _VOL_WIN,
) -> pd.DataFrame:
    from numba_core import backtest_spread, COL_ENTRY_IDX, COL_EXIT_IDX, \
        COL_SIDE, COL_GROSS_RETURN, COL_HOLDING_DAYS

    if tolerance_sd is None:
        tolerance_sd = PARAMS['xing_tolerance_sd']

    arr = spread_ret.dropna().values.astype(np.float64)
    idx = spread_ret.dropna().index
    day_ints = (idx.astype(np.int64) // 10**9 // 86400).values.astype(np.int64)

    trades, n_trades, cum, dist_sd = backtest_spread(
        arr, window, tolerance_sd, exit_sd, day_ints
    )

    if n_trades == 0:
        return pd.DataFrame(columns=['entry_date','exit_date','side',
                                      'gross_return','holding_days'])

    t = trades[:n_trades]
    return pd.DataFrame({
        'entry_date': [idx[int(t[i, COL_ENTRY_IDX])] for i in range(n_trades)],
        'exit_date': [idx[int(t[i, COL_EXIT_IDX])] for i in range(n_trades)],
        'side': ['long' if t[i, COL_SIDE]==1 else 'short' for i in range(n_trades)],
        'gross_return': t[:, COL_GROSS_RETURN],
        'holding_days': t[:, COL_HOLDING_DAYS],
    })
```

**Do not modify `crossing_signals()`** — it is used by dashboard and analysis tabs.

---

## Task 5: Update data.py

Add external price loader:

```python
def load_external_prices(filepath: str, start_date: str = '1999-01-01') -> pd.DataFrame | None:
    from pathlib import Path
    p = Path(filepath)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.ffill(limit=3)
    if start_date:
        df = df.loc[start_date:]
    return df.dropna(how='all')
```

---

## Task 6: Update requirements.txt

Add:
```
numba>=0.59
```

---

## File change summary

| File | Action | What changes |
|------|--------|--------------|
| `numba_core.py` | **ADD** (pre-built) | Drop into project root |
| `asset_configs.py` | **ADD** (pre-built) | Drop into project root |
| `backtest.py` | **ADD** (pre-built) | Drop into project root |
| `test_numba_parity.py` | **ADD** (pre-built) | Run first, keep for regression |
| `search.py` | **MODIFY** | Import numba_core, asymmetric params, backtest columns, Config column |
| `calculations.py` | **MODIFY** | Add `crossing_signal_backtest()` wrapper |
| `data.py` | **MODIFY** | Add `load_external_prices()` |
| `config.py` | **NO CHANGE** | Asset class configs live in `asset_configs.py` |
| `app.py` | **MODIFY** | Asymmetric search UI (tab5), new Backtest tab (tab8) |
| `requirements.txt` | **MODIFY** | Add `numba>=0.59` |

---

## Constraints

1. **Run `test_numba_parity.py` FIRST.** Do not modify existing files until all parity tests pass.

2. **Do not rewrite the pre-built files.** Fix bugs in-place if found — don't start over.

3. **Do not change the signal logic.** Entry at +/-N SD, exit when distance crosses exit_sd.

4. **Do not break existing tabs.** Tabs 1-7 must work exactly as before.

5. **Preserve backward compatibility.** `run_search()` must accept old `min_legs`/`max_legs`.

6. **Match existing code style.** Comments, type hints, descriptive names.  Widget keys: `bt_` for backtest, `s_` for search.

7. **Performance targets.** 3v3 symmetric equity with backtest: < 5s.  Full 2-5 asymmetric (316K combos): < 30s.  Sensitivity grid (90 params): < 60s.


Claude Code run  on 25th May2026 tasks and results

Task 1: Integrate numba_core into search.py

Task 2: Asymmetric search UI in app.py tab5

Task 3: Add Backtest tab (tab8) to app.py

Task 4: Add crossing_signal_backtest() to calculations.py

Task 5: Add load_external_prices() to data.py

Task 6: Update requirements.txt
All syntax-clean. Here's what was done:

Task 1 — search.py: Added numba_core import, updated estimate_combinations() to accept separate min_long/max_long/min_short/max_short with backward compat, updated run_search() with asymmetric leg params + batch_backtest call per long basket, added Config, Trades, WinRate, Expectancy, AvgHolding, PayoffRatio columns, updated composite score to include _z_WinRate and _z_Expectancy.

Task 2 — app.py tab5: Replaced single "Legs per side" block with 4-column layout (Long legs | Short legs | History window | Scale), symmetric checkbox mirrors short to long, updated estimate_combinations() and run_search() calls, added Config and backtest columns to results table, added "🔬 Backtest this" button that stages bt_pending in session state.

Task 3 — app.py tab8: Full Backtest tab — asset selector + CSV upload, signal params expander, asymmetric basket config, run button that calls run_exhaustive_search(), summary metrics, results table, trade distribution histogram, holding period histogram with p50/p90 lines, regime bar chart, sensitivity grid expander, export to backtest_exports/.

Task 4 — calculations.py: Added crossing_signal_backtest() wrapper that calls backtest_spread from numba_core and returns a per-trade DataFrame. Does not touch crossing_signals().

Task 5 — data.py: Added load_external_prices().

Task 6 — requirements.txt: Added numba>=0.59.