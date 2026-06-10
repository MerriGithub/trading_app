# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.
The **authoritative knowledge base** for this project is the Obsidian Project Reference.

At the start of any non-trivial session, locate and read the latest version:

```bash
# The file lives in the Obsidian vault root and is date-prefixed.
# Always use the most recent one — do not hardcode the date.
ls -1 "$OBSIDIAN_VAULT"/*Trading-App-Rebuild_PROJECT_REFERENCE.md 2>/dev/null | sort | tail -1
```

If `$OBSIDIAN_VAULT` is not set, the vault is typically at:
`C:/Users/gordo/Google Drive Streaming/.shortcut-targets-by-id/1lIIeHMZyWhQmYyhjk4AacJUByZClFbiv/Merri-Obsidian-KB/`

The filename pattern is: `YYYY-MM-DD-Trading-App-Rebuild_PROJECT_REFERENCE.md`
Pick the **highest date** in the vault root — that is the active file.
Files in `Archive/` are superseded versions; ignore them.

The Project Reference contains design decisions, walk-forward results, research
findings, and the full prompt correction register.
This file intentionally stays thin — it covers only what Claude Code needs
without Obsidian access.

---

## ⚠️ Python interpreter — always use the full path

**Dev PC interpreter:**
```
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

Use this explicit path for every `python`, `pytest`, and `pip` invocation. Do not
rely on PATH-resolved `python` or `pytest` shims — they may resolve to the wrong
interpreter and waste time on environment debugging.

```bash
# Correct form for all commands on the dev PC:
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest tests/ -q
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app.py --server.port 8502
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install -r requirements.txt
```

VS Code is configured to use this interpreter. The integrated terminal uses it
automatically. For standalone Claude Code tool calls, always use the full path.

---

## Project scope

**Project root:** `trading_app/`
(Path varies by machine. Confirm with `pwd` before exploring. On the primary
dev machine it lives under the Google Drive sync root.)

**Strict file scope:** Read and write only within `trading_app/`.
Do not explore sibling directories, the Obsidian vault, memory files, or logs.

**Legacy app:** `trading_app/legacy/` is frozen. Treat as read-only.
Never modify anything under `legacy/` — it is the benchmark.

---

## Commands

All commands must be run from inside `trading_app/` — `app.py` uses relative
paths for cache and data files and will fail if run from the workspace root.

```bash
cd trading_app

# Run the rebuild app (port 8502)
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run app.py --server.port 8502

# Run the legacy benchmark (port 8501 — read-only, do not modify)
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run legacy/app_legacy.py --server.port 8501

# Run all tests
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest tests/

# Run a single test file
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest tests/test_signal.py

# Install dependencies
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pip install -r requirements.txt
```

Always use the full interpreter path — never bare `python`, `pytest`, or `pip`.
See the **Python interpreter** section above.

---

## ⚠️ Prompt correction register

> Read this before writing or modifying any code. These are confirmed
> discrepancies where prior prompts differed from the actual implementation.
> **Implementation is always authoritative.**

| # | What a prompt might say | What the code actually does | Confirmed |
|---|-------------------------|-----------------------------|-----------|
| A | Journal = Tab 6, `tab6_journal.py`, `tab6_pending_entry` | Journal = **Tab 7**, `tab7_journal.py`, **`tab7_pending_entry`** | 2026-05-23 |
| B | Navigate via `sidebar_nav` direct write | Navigate via **`sidebar_nav_pending`** — avoids "cannot modify after instantiation" crash | 2026-05-23 |
| C | Save journal entry with `_save_trade()` to `trade_journal.json` | Confirm step calls **`portfolio.open_position()`** directly — no separate JSON | 2026-05-23 |
| D | Tab 10 session state keys: `pa_vol_window`, `pa_entry_sd`, `pa_exit_sd` | Actual Tab 2 widget keys: **`pa_vol`, `pa_xing`, `pa_exit`, `pa_trend_window`** | 2026-05-22 |
| E | CFD capital needed: reconstruct from `cfd_size × price / gbpusd / vol` | **`(cfd_min / contracts_raw) × target_1sd`** — ratio pattern; currency-agnostic | 2026-05-24 |
| F | Scoring mode selectors in Tab 10 and Tab 11 | Scoring mode selectors are in **Tab 8** (Backtest) and **Tab 9** (Trade Validation). Tab 10 sorts by AvgNet_WT directly. Tab 11 has no scoring mode. | 2026-05-25 |
| G | Q11 protocol lives in Tab 11 | Q11 protocol lives in **Tab 9** (Trade Validation). Tab 11 = single-pair rolling WF validator. | 2026-05-25 |
| H | Exit formula: `side * d <= EXIT_SD` | Correct form: **`-side * d <= EXIT_SD`** (sign inverted). Expanded: `(side==-1 and d<=EXIT_SD) or (side==+1 and d>=-EXIT_SD)`. The wrong form fires at entry. **A module-level assertion in `engine/numba_core.py` enforces this at import time — do not remove it.** | 2026-06-02 |
| I | `dist_sd_at_entry = d` (signed) | Must be **`abs(d)`** — signed values cancel across long/short sides and produce near-zero averages. **A module-level assertion in `engine/numba_core.py` enforces this at import time — do not remove it.** | 2026-06-02 |
| J | Tab 6 = Journal | Tab 6 = **Live Monitor** (`tab6_live_monitor.py`). Journal = Tab 7. Tab 6 shows signal alerts and live signal state for watchlist pairs. | 2026-06-04 |
| K | Tab 10 scan uses `engine/search.py` | Tab 10 uses **`engine/backtest.py`** (`prepare_returns`, `run_backtest`, `aggregate_trades`). `engine/search.py` is the 3v3/4v4 multi-leg basket scanner for Tab 8 only. Daily scan batch must follow Tab 10 pattern. | 2026-06-05 |
| L | Tab 3 margin accurate for all commodities | **Tab 3 margin was 100× too low for BRENT, WTI, COFFEE, SUGAR** — Yahoo delivers USD/USc, IG quotes in pence. `ig_price_override` added to all four in `asset_configs.py`. CORN, WHEAT, SOYBEANS unaffected. | 2026-06-05 |
| M | `get_financing_daily_rate` takes 2 args | **Actual signature is `(instrument_code, asset_class, direction, price, broker_profile)`** — 5 args. ARCHITECTURE.md shows a simplified version. Always use the full signature from `account.py` directly. | 2026-06-06 |
| N | `registry.refresh()` raises `NotImplementedError` | **`DataRegistry.refresh()` is fully implemented** — fetches missing rows only, appends, deduplicates. ARCHITECTURE.md note is outdated. | 2026-06-06 |
| O | `rolling_mean_std` can be imported inside a function | **Must be imported at module level** — local import inside `_compute_current_sd()` fails. Add to top-level `from engine.numba_core import ... rolling_mean_std` block. | 2026-06-06 |
| P | `wf_summary['n_obs']` = OOS trade count | **`n_obs` is the number of WF windows**, not OOS trades. Use `_oos_stats_from_wf(wf_df, long_i)` to read `OOS_Trades`, `OOS_WinRate`, `OOS_Gross` from the WF DataFrame directly. | 2026-06-06 |
| Q | positions.json instruments at top-level keys | Instruments are **nested**: `pos['basket']['long_legs'][0]` and `pos['basket']['short_legs'][0]`. Keys `long_instrument`, `long`, `short`, `short_instrument` do not exist — silent `''` return. | 2026-06-06 |
| R | `sidebar_nav_pending` value omits emoji | Sidebar entry strings include emojis: **`'📈 Pair Analysis'`** not `'Pair Analysis'`. Plain string silently fails to match — navigation never fires. Always use the full string with emoji. | 2026-06-06 |
| S | Benchmark "Commodity 1v1 exhaustive, AvgNet ~5.40%, WR ~87%, AvgHold ~121d" is a universe average | These figures are **NATGAS/BRENT single-pair counter-trend gross** (vol-scaled units), not a universe average — see benchmark table correction below. **All Tab 8 AvgNet/Expectancy figures are vol-scaled (~1% daily vol/leg), not real % on capital.** Vol scaling varies by instrument (NATGAS ~0.264 → ~3× raw). Net for NATGAS/BRENT CT = **+1.60%**. Universe contrarian-top mean at exit=0.0 = **−6.78%** (3/30 positive). See Decision 108 in Project Reference. | 2026-06-07 |

---

## Architecture

Streamlit-based spread trading platform — a modernised replacement for a
20-year-old spreadsheet. Trades long/short spread baskets across equity indices,
FX pairs, commodities, and fixed income.

> **Method signatures, parameters, and return types live in `ARCHITECTURE.md`
> at the repo root. Read that file before writing any code that calls domain
> objects (`Basket`, `SpreadSignal`, `Position`, `Portfolio`, `DataRegistry`,
> `account.py`). Do not infer signatures from this file.**

### Layer overview

```
app.py                    # Streamlit entry point; imports tabs/
logging_config.py         # Centralised logging — configure_logging() called once from app.py
tabs/tab1.py … tab13.py   # One file per UI tab
tabs/shared.py            # Singleton helpers (portfolio, registry, account)
engine/                   # Backtest, search, scoring, walk-forward
  numba_core.py           # JIT-compiled loops + pure-Python fallback (keep both in sync)
  walkforward.py          # run_walk_forward() + run_cross_asset_walkforward()
core/                     # Domain objects — see ARCHITECTURE.md for all signatures
asset_configs.py          # Instrument definitions — authoritative for all asset classes
config.py                 # Algorithm parameters + legacy equity mappings
data.py                   # Yahoo Finance fetch + CSV cache
account.py                # Financing rates, margin; loads from data/account.json
research/daily_scan.py    # Overnight batch scan — runs without Streamlit
scripts/                  # Task Scheduler wrappers (run_daily_scan.ps1 / .bat)
data/scan_config.json     # Named search registry (algo + WF params per search)
data/daily_scan_results.json  # Latest overnight scan output (read by Tab 13)
data/scan_history/        # Per-run archived results (YYYY-MM-DD_HHMM.json)
```

### Tab map

| Tab file | Purpose | Key session state |
|----------|---------|-------------------|
| `tab2_pair_analysis.py` | Pair Analysis | `pa_vol`, `pa_xing`, `pa_exit`, `pa_trend_window`, `pa_wl_id`, `pa_long_pending`, `pa_short_pending`, `pa_pair_pending` |
| `tab3_stake_calc.py` | Stake Calculator | `tab3_direction`, `_pricing_broker`, `tab3_broker_profile`, `sc_long_pending`, `sc_short_pending` |
| `tab6_live_monitor.py` | Live Monitor | signal alerts, watchlist signal states |
| `tab7_journal.py` | Trade Journal | `tab7_pending_entry` |
| `tab8_backtest.py` | Backtest | `bt_scoring_mode`, `bt_ca_scoring` |
| `tab9_walkforward.py` | Trade Validation (Q11) | `wf_scoring`, `wf_ca_long`, `wf_ca_short` |
| `tab10_scenario.py` | Scenario Scanner | `tab10_min_wt_trades`, `tab10_broker_profile` |
| `tab11_walkforward.py` | Single-pair rolling WF | `wf_pair`, `wf11_long_pending`, `wf11_short_pending` |
| `tab12_watchlist.py` | Watchlist | `tab12_selected_id`, `tab12_selected_entry`, `tab12_dedup` |
| `tab13_daily_scan.py` | Daily Scan — overnight batch results + AI review | `tab13_dedup` |

**Navigation:** always use `st.session_state['sidebar_nav_pending']`, never write
`sidebar_nav` directly — crashes on rerun (register item B).

### Config split — important

Two config layers; they are **not** interchangeable:

- **`config.py`** — algorithm parameters (`vol_calc_days`, `xing_tolerance_sd`,
  `roc_days`, etc.) and legacy equity mappings that older engine code imports.
- **`asset_configs.py`** — authoritative source for all instrument definitions
  across asset classes: ticker → Yahoo Finance symbol, `spread_pct`, `point_size`,
  financing rates, CFD specs. **Multi-asset changes go here, not in `config.py`.**

### Key constants in `asset_configs.py`

| Constant | Purpose |
|----------|---------|
| `COMMODITY_EXCLUDE` | `frozenset({'WTI'})` — excluded from pair generation (April 2020 negative price event) |
| `FI_EXCLUDE` | `frozenset({'IBTM'})` — excluded due to data quality issues |
| `_DEFAULT_SCORING_MODE` | `{'commodities': 'contrarian', ...}` — auto-default per asset class |
| `CROSS_ASSET_SCORING_MODE` | Dict keyed by `(long_ac, short_ac)` tuples — validated mode per combination |
| `get_cross_asset_scoring_default()` | Helper used by Tab 9 to resolve cross-asset scoring mode |
| `CROSS_ASSET_COMBINATIONS` | All valid cross-asset pair type combinations |

### Scoring mode rules

| Asset class / combination | Mode | Basis |
|---------------------------|------|-------|
| Commodities | **Contrarian** | ρ=+0.122, p=0.0009 |
| Equities | **Contrarian** | ρ=+0.208, p≈0 |
| Equity × FX | **Contrarian** | ρ=+0.053, p=0.0030 |
| Commodity × FI | Composite | ρ=+0.069, p=0.0016 |
| Commodity × FX | Composite | ρ≈0 |

Tab 8 and Tab 9 auto-default to the correct mode and warn on deviation.
Tab 10 has no scoring mode selector. Tab 11 has none at all (register item F/G).

### Data flow

1. `data.py` fetches daily prices from Yahoo Finance, caches to:
   - `cache/prices.csv` — equity indices
   - `cache/fx_prices.csv` — FX pairs
   - `cache/commodity_prices.csv` — commodities (WTI in file; excluded from pair gen)
   - `cache/fi_prices.csv` — fixed income
2. `DataRegistry` loads cache; exposes prices, rolling vol, and scaling factors.
3. `SpreadSignal` computes crossing signals: deviation from rolling mean in SD units.
   Signal fires at ±`XING_SD` (default 2.0); exit at ±`EXIT_SD`.
4. `Portfolio` aggregates open positions, live P&L, and correlation.
5. Singletons (`portfolio`, `registry`, `account`) initialised once via
   `@st.cache_resource` in `tabs/shared.py`.

### Key formulae

**Position sizing:**
```python
stake = (target_exposure / (price × daily_vol × point_size)) × scaling_factor
scaling_factor = min(1.0, target_daily_vol / rolling_daily_vol)
```

**Exit condition** (register item H — sign matters):
```python
-side * d <= EXIT_SD          # CORRECT — fires on reversion back through threshold
side * d <= EXIT_SD           # WRONG   — fires immediately at entry
```

**Entry dislocation** (register item I — always abs):
```python
entry_dist_sd = abs(d)        # CORRECT — signed values cancel across long/short sides
entry_dist_sd = d             # WRONG   — produces near-zero averages
```

**CFD capital needed** (register item E):
```python
capital = (cfd_min / contracts_raw) × target_1sd   # CORRECT — currency-agnostic ratio
```

### Persistence

| Path | Contents |
|------|----------|
| `data/account.json` | Capital, financing rates, margin — editable in Dashboard tab |
| `data/positions.json` | Open and closed trade history |
| `data/walkforward_cache.json` | Walk-forward result cache |
| `data/watchlist.json` | Saved watchlists; includes `scoring_mode` field |

### Price data conventions

- Daily prices: `ffill(limit=3)` to bridge short holiday gaps.
- Intraday prices: no forward-fill (stale prices would cross market sessions).
- Financing: annual rate ÷ 365 per day; long and short legs calculated separately.
- Spread (bid-ask) cost is one-way; round-trip is 2×.

---

## Ticker maps

| Cache file | Columns |
|------------|---------|
| `prices.csv` | UKX, CBK, CEY, CFR, CMD, CEI, COI, CRM, CIL, CPH, CTN, CTB |
| `fx_prices.csv` | EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD, EURGBP, EURJPY, GBPJPY, EURCHF, AUDNZD |
| `commodity_prices.csv` | WTI⚠, BRENT, NATGAS, GOLD, SILVER, PLATINUM, COPPER⚠, PALLADIUM, WHEAT, CORN, SOYBEANS, COFFEE, SUGAR |
| `fi_prices.csv` | SHY, IEI, IEF, TLT, LQD, HYG, TIP, BWX, EMB, IGLT, IEAG |

⚠ **WTI** — present in CSV, excluded from pair generation via `COMMODITY_EXCLUDE`.
⚠ **COPPER** — CSV ticker is HG=F (USD/lb, CME). IG spread-bet price is pence/tonne (LME).
Use `ig_price_override` for margin calculations.

---

## Benchmark reference values

Always use **Full** history window, **Vol=262**, **Exit SD=0.0** when running
commodity benchmarks for comparison.

| Test | Expected result |
|------|----------------|
| Equity 3v3 exhaustive (SD=2.0/0.0, Vol=262, 4.88%) | Net −0.122, WR 86.4%, AvgHold 169d |
| Commodity 1v1 exhaustive (Vol=262, exit=0.0) | ~~AvgNet ~5.40%, WR ~87%, AvgHold ~121d~~ ⚠️ **Mislabelled — register S.** These are NATGAS/BRENT single-pair CT **gross** (vol-scaled), not a universe average. Universe contrarian-top mean at exit=0.0 = **−6.78%** (3/30 positive). Net NATGAS/BRENT CT = **+1.60%**. |
| NATGAS/BRENT counter-trend (Vol=262, exit=0) | Gross WR 86.7%, Avg net +1.60%, AvgHold 121d |
| WF equity contrarian scalp (IS=3y, OOS=1y, EXIT=2.0) | ρ=+0.208, p≈0 |
| WF commodity contrarian (IS=3y, OOS=1y) | ρ=+0.122, p=0.0009 |
| WF Equity×FX contrarian (IS=5y, OOS=3y) | ρ=+0.053, p=0.0030 |

---

## Research optimums (Phase 2–4b, confirmed June 2026)

| Asset class | XING_SD | EXIT_SD | Avg net/trade | Avg hold | Status |
|-------------|---------|---------|---------------|----------|--------|
| Commodities | 2.0 | **0.5** | +1.137% | 104d | ✅ Confirmed optimum |
| Equities (scalp regime) | 2.0 | **2.0** | +0.380% | 8d | ✅ Confirmed; WF validated ρ=+0.208 |
| FX | 2.0 | 1.5 (best of bad) | −0.160% | 35d | ❌ Net negative at all EXIT_SD |
| Fixed Income | 2.0 | 1.5 (best of bad) | −0.174% | 35d | ❌ Net negative at all EXIT_SD |

**Equity scalp note:** EXIT_SD=2.5 is the degenerate boundary (win rate 50.4%,
median hold 1d — exits on noise). Do not go above 2.0 for equities.

---

## Code quality standard

> **Every file you touch must be left compliant with this standard.**
> Full specification is in §17 of the Obsidian Project Reference.
> The rules below are the mandatory minimum — read §17 for examples and patterns.

1. **Type hints** — all function signatures; `from __future__ import annotations` for
   forward references; no `Any` without a comment.

2. **Google-style docstrings** — every non-trivial function and class; Args / Returns /
   Raises blocks; explain the *why*, not just the *what*; reference register items
   by letter where relevant.

3. **Logging, not print** — `import logging; logger = logging.getLogger(__name__)` at
   the top of every module. Never call `logging.basicConfig()` or add handlers inside
   module files — only `logging_config.py` and `app.py` do that.

4. **Fail-fast input guards** — validate at the function boundary before any computation.
   Use explicit `if / raise` with informative messages. Common checks: None/empty,
   minimum row count, instrument membership, side ∈ {1, −1}, numeric range.

5. **Specific exception types** — no bare `except:`, no `except Exception:` without a
   comment explaining what is caught and why.

6. **Register item canaries** — the module-level assertions for items H and I in
   `engine/numba_core.py` are permanent. Do not remove or suppress them. They make
   the application unrunnable if either invariant is violated.

### Compliance check — run before ending any session

```bash
C:\Users\gordo\AppData\Local\Python\pythoncore-3.14-64\python.exe -m pytest tests/ -q
grep -r "print("        trading_app/ --include="*.py" --exclude-dir=legacy
grep -r "except:"       trading_app/ --include="*.py" --exclude-dir=legacy
grep -r "except Exception:" trading_app/ --include="*.py" --exclude-dir=legacy
```

The first command must show the same pass count as your session baseline.
The remaining three must return only expected results (UI-rendering prints,
zero bare excepts, zero unjustified broad excepts).

---

## For further context

The Obsidian Project Reference contains the full design decisions log (decisions
1–108), walk-forward pipeline architecture diagram, paper trading log, all Phase
2–4b experiment tables, open items list, Sprint 3 scope, and the complete code
quality standard (§17). Consult it before making architectural decisions.
