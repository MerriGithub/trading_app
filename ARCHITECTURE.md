# Architecture — Global Spread Trading Platform

_Sprint 1 complete. Signatures reflect actual implementation._

> This file is the authoritative reference for method signatures, parameters,
> and return types. For project scope, commands, tab map, config split, scoring
> rules, and research optimums, see `CLAUDE.md`.

---

## Domain Layer

### Basket (core/basket.py)
- `financing_cost_daily(self) -> float` — no args; reads rates from account.py
- `spread_cost(self, registry) -> float`
- `pair(long, short) -> Basket` — classmethod, 1v1 shorthand
- `validate(self) -> None` — raises ValueError on self-dealing/empty legs

### SpreadSignal (core/signal.py)
- `__init__(basket, prices, vol_window=262, xing_sd=2.0, exit_sd=1.0)`
- `current_sd -> float` — latest distance_sd value
- `signal_state -> str` — `'LONG_ENTRY'` | `'SHORT_ENTRY'` | `'EXIT'` | `'NONE'`
- `signal_history(n_days=262) -> pd.DataFrame`

### Position (core/position.py)
- `live_pnl(current_prices: dict) -> float`
- `financing_cost_to_date(self) -> float` — no args; uses basket.financing_cost_daily()
- `net_pnl(current_prices: dict) -> float` — no account param
- `to_summary_row(current_prices: dict) -> dict` — no account param
- `partial_close(pct, exit_prices, exit_date) -> float`
- `close(exit_prices, exit_date) -> float`
- `from_legacy_trade(trade: dict) -> Position` — classmethod

### Portfolio (core/portfolio.py)
- `open_position(basket, direction, entry_prices, stakes, target_exposure, name, comments='') -> Position`
- `close_position(position_id, exit_prices, exit_date) -> float`
- `partial_close(position_id, pct, exit_prices, exit_date) -> float`
- `open_positions -> list[Position]`
- `closed_positions -> list[Position]`
- `total_unrealised_pnl(current_prices) -> float`
- `total_realised_pnl() -> float`
- `cross_pair_correlation(self, registry) -> pd.DataFrame` — no `lookback` param; returns empty DataFrame (not None) when <2 positions
- `capital_at_risk(current_prices) -> float` — raises `NotImplementedError('Deferred to Sprint 3')`
- `position_summary(self, current_prices, account=None) -> pd.DataFrame` — `account` unused

### DataRegistry (core/data_registry.py)
- `get_daily_prices(instruments, start_date='1999-01-01') -> pd.DataFrame`
- `get_latest_prices(instruments) -> dict[str, float]`
- `get_vols(instruments, window=262) -> dict[str, float]`
- `get_scalings(instruments, target_vol=0.01) -> dict[str, float]`
- `get_intraday(instruments, interval='5m') -> pd.DataFrame | None`
- `refresh(instruments=None) -> None` — raises `NotImplementedError` (deferred to Sprint 3)

---

## Account (account.py)
- `get_financing_rates(asset_class: str) -> tuple[float, float]` — (long_rate, short_rate)
- `get_spread_cost_fallback() -> float`
- `get_starting_capital() -> float`
- `get_margin() -> float`
- `get_margin_rate(asset_class: str) -> float` — per-asset margin rate
- `get_financing_daily_rate(asset_class: str, side: str) -> float`
- `save_account(data: dict) -> None`

---

## Financing Rates (data/account.json)

| Asset class | Long rate | Short rebate |
|-------------|-----------|-------------|
| Equity | 4.88% p.a. | 0.88% p.a. |
| FX | 1.80% p.a. | −1.80% p.a. (both sides pay swap) |
| Commodities | 4.88% p.a. | 0.88% p.a. |
| Fixed Income | 4.88% p.a. | 0.88% p.a. |

Net daily drag = (long_rate − short_rebate) / 365, compounding over holding period.
