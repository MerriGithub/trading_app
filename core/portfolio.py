from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from core.basket import Basket
from core.position import Position


class Portfolio:
    def __init__(self, positions_file: Path, account_file: Path):
        self._path    = Path(positions_file)
        self._acct    = Path(account_file)
        self._positions: list[Position] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> list[Position]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            return [Position.from_dict(d) for d in data]
        except Exception:
            return []

    def _save(self) -> None:
        data = [p.to_dict() for p in self._positions]
        tmp  = self._path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2, default=str))
        os.replace(tmp, self._path)

    # ── Mutators ──────────────────────────────────────────────────────────

    def open_position(
        self,
        basket: Basket,
        direction: str,
        entry_prices: dict[str, float],
        stakes: dict[str, float],
        target_exposure: float,
        name: str,
        comments: str = '',
    ) -> Position:
        from datetime import datetime
        pos_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        pos = Position(
            id              = pos_id,
            name            = name,
            basket          = basket,
            direction       = direction,
            entry_date      = date.today(),
            entry_prices    = entry_prices,
            stakes          = stakes,
            target_exposure = target_exposure,
            comments        = comments,
        )
        self._positions.append(pos)
        self._save()
        return pos

    def close_position(
        self,
        position_id: str,
        exit_prices: dict[str, float],
        exit_date: date,
    ) -> float:
        pos = self._get(position_id)
        realised = pos.close(exit_prices, exit_date)
        self._save()
        return realised

    def partial_close(
        self,
        position_id: str,
        pct: float,
        exit_prices: dict[str, float],
        exit_date: date,
    ) -> float:
        pos = self._get(position_id)
        realised = pos.partial_close(pct, exit_prices, exit_date)
        self._save()
        return realised

    def _get(self, position_id: str) -> Position:
        for p in self._positions:
            if p.id == position_id:
                return p
        raise KeyError(f'Position not found: {position_id}')

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self._positions if p.status == 'open']

    @property
    def closed_positions(self) -> list[Position]:
        return [p for p in self._positions if p.status == 'closed']

    def total_unrealised_pnl(self, current_prices: dict[str, float]) -> float:
        return sum(p.live_pnl(current_prices) for p in self.open_positions)

    def total_realised_pnl(self) -> float:
        return sum(p.realised_pnl for p in self._positions)

    def position_summary(
        self,
        current_prices: dict[str, float],
        account: dict | None = None,
    ) -> pd.DataFrame:
        rows = [p.to_summary_row(current_prices) for p in self.open_positions]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def cross_pair_correlation(self, registry) -> pd.DataFrame:
        """Correlation matrix of spread returns across open positions."""
        if not self.open_positions:
            return pd.DataFrame()
        series = {}
        for pos in self.open_positions:
            instr = pos.basket.all_instruments
            try:
                prices = registry.get_daily_prices(instr)
                from engine.backtest import prepare_returns
                scaled, _, idx = prepare_returns(prices, instr)
                col = {inst: i for i, inst in enumerate(instr)}
                long_r  = scaled[:, [col[i] for i in pos.basket.long_legs]].mean(axis=1)
                short_r = scaled[:, [col[i] for i in pos.basket.short_legs]].mean(axis=1)
                import pandas as pd
                series[pos.name] = pd.Series(long_r - short_r, index=idx)
            except Exception:
                pass
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series).corr()

    def capital_at_risk(self, current_prices: dict) -> float:
        """
        Parametric VaR estimate: sum of 2-SD 10-day loss per open position.
        Uses position stakes × instrument vol × sqrt(10) × 2.
        Not a full portfolio VaR (does not account for cross-position correlation).
        """
        from core.data_registry import DataRegistry

        registry = DataRegistry(Path(__file__).parent.parent / 'cache')

        total_car = 0.0
        for pos in self.open_positions:
            try:
                vols = registry.get_vols(pos.basket.all_instruments)
                for inst, stake in pos.stakes.items():
                    price     = current_prices.get(inst, pos.entry_prices.get(inst, 0.0))
                    daily_vol = vols.get(inst, 0.02)
                    notional  = abs(stake) * price
                    car_10d   = notional * daily_vol * (10 ** 0.5) * 2
                    total_car += car_10d * pos.pct_open
            except Exception:
                pass
        return total_car
