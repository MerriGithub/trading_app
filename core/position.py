from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from core.basket import Basket


@dataclass
class Position:
    id:              str
    name:            str
    basket:          Basket
    direction:       str            # 'long_spread' | 'short_spread'
    entry_date:      date
    entry_prices:    dict[str, float]
    stakes:          dict[str, float]   # signed: + long, - short
    target_exposure: float
    comments:        str = ''
    status:          str = 'open'
    exit_date:       date | None = None
    exit_prices:     dict[str, float] = field(default_factory=dict)
    realised_pnl:    float = 0.0
    pct_open:        float = 1.0

    @property
    def days_held(self) -> int:
        end = self.exit_date or date.today()
        return max(0, (end - self.entry_date).days)

    def live_pnl(self, current_prices: dict[str, float]) -> float:
        """Gross unrealised P&L at current prices."""
        from asset_configs import ASSET_CLASSES
        pnl = 0.0
        for inst, stake in self.stakes.items():
            entry   = self.entry_prices.get(inst, 0.0)
            current = current_prices.get(inst, entry)
            pt_size = 1.0
            for cfg in ASSET_CLASSES.values():
                if inst in cfg.get('point_sizes', {}):
                    pt_size = cfg['point_sizes'][inst]
                    break
            pnl += stake * (current - entry) * pt_size
        return pnl * self.pct_open

    def financing_cost_to_date(self) -> float:
        """Cumulative financing drag since entry, in account currency."""
        return self.basket.financing_cost_daily() * self.days_held

    def net_pnl(self, current_prices: dict[str, float]) -> float:
        """Live P&L minus financing drag (entry spread cost excluded — already paid)."""
        return self.live_pnl(current_prices) - self.financing_cost_to_date()

    def partial_close(
        self,
        pct: float,
        exit_prices: dict[str, float],
        exit_date: date,
    ) -> float:
        """Partially close position. Returns realised P&L for the closed fraction."""
        close_fraction = self.pct_open * pct
        realised = self.live_pnl(exit_prices) * pct
        self.realised_pnl += realised
        self.pct_open     -= close_fraction
        self.exit_prices   = exit_prices
        if self.pct_open < 1e-6:
            self.pct_open  = 0.0
            self.status    = 'closed'
            self.exit_date = exit_date
        return realised

    def close(self, exit_prices: dict[str, float], exit_date: date) -> float:
        """Fully close the position. Returns total realised P&L."""
        realised = self.live_pnl(exit_prices)
        self.realised_pnl += realised
        self.pct_open    = 0.0
        self.status      = 'closed'
        self.exit_date   = exit_date
        self.exit_prices = exit_prices
        return realised

    def to_summary_row(self, current_prices: dict[str, float]) -> dict:
        """Single-row dict for display in a DataFrame."""
        return {
            'id':             self.id,
            'name':           self.name,
            'direction':      self.direction,
            'entry_date':     str(self.entry_date),
            'days_held':      self.days_held,
            'pct_open':       self.pct_open,
            'live_pnl':       self.live_pnl(current_prices),
            'financing_cost': self.financing_cost_to_date(),
            'net_pnl':        self.net_pnl(current_prices),
            'status':         self.status,
        }

    def to_dict(self) -> dict:
        return {
            'id':              self.id,
            'name':            self.name,
            'basket':          self.basket.to_dict(),
            'direction':       self.direction,
            'entry_date':      str(self.entry_date),
            'entry_prices':    self.entry_prices,
            'stakes':          self.stakes,
            'target_exposure': self.target_exposure,
            'comments':        self.comments,
            'status':          self.status,
            'exit_date':       str(self.exit_date) if self.exit_date else None,
            'exit_prices':     self.exit_prices,
            'realised_pnl':    self.realised_pnl,
            'pct_open':        self.pct_open,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Position':
        def _parse_date(v) -> date | None:
            if v is None:
                return None
            if isinstance(v, date):
                return v
            return date.fromisoformat(str(v))

        return cls(
            id              = d['id'],
            name            = d['name'],
            basket          = Basket.from_dict(d['basket']),
            direction       = d.get('direction', 'long_spread'),
            entry_date      = _parse_date(d['entry_date']),
            entry_prices    = d['entry_prices'],
            stakes          = d['stakes'],
            target_exposure = d.get('target_exposure', 0.0),
            comments        = d.get('comments', ''),
            status          = d.get('status', 'open'),
            exit_date       = _parse_date(d.get('exit_date')),
            exit_prices     = d.get('exit_prices', {}),
            realised_pnl    = d.get('realised_pnl', 0.0),
            pct_open        = d.get('pct_open', 1.0),
        )

    @classmethod
    def from_legacy_trade(cls, trade: dict) -> 'Position':
        """
        Convert a legacy journal.py trade dict to a Position.

        All legacy trades → direction = 'long_spread'.
        Multi-leg: all legs merged into one signed stakes dict.
        pct_open = min(leg['pct_open']) across all legs.
        """
        long_legs, short_legs = [], []
        entry_prices: dict[str, float] = {}
        stakes: dict[str, float] = {}
        pct_opens: list[float] = []

        for leg in trade.get('legs', []):
            bi = leg['buy_instrument']
            si = leg['sell_instrument']
            long_legs.append(bi)
            short_legs.append(si)
            entry_prices[bi] = leg['buy_entry_price']
            entry_prices[si] = leg['sell_entry_price']
            stakes[bi] = +leg['buy_stake']
            stakes[si] = -leg['sell_stake']
            pct_opens.append(leg.get('pct_open', 1.0))

        def _parse_date(v) -> date | None:
            if v is None:
                return None
            if isinstance(v, date):
                return v
            return date.fromisoformat(str(v))

        return cls(
            id              = trade['id'],
            name            = trade['name'],
            basket          = Basket(long_legs=long_legs, short_legs=short_legs),
            direction       = 'long_spread',
            entry_date      = _parse_date(trade.get('entry_date', '2000-01-01')),
            entry_prices    = entry_prices,
            stakes          = stakes,
            target_exposure = trade.get('target_exposure', 0.0),
            comments        = trade.get('comments', ''),
            status          = trade.get('status', 'open'),
            exit_date       = _parse_date(trade.get('exit_date')),
            exit_prices     = trade.get('exit_prices', {}),
            realised_pnl    = trade.get('realised_pnl', 0.0),
            pct_open        = min(pct_opens) if pct_opens else 1.0,
        )
