"""
core/position.py — Position: a single open or closed spread trade.

A Position binds a Basket to entry metadata (prices, stakes, date) and
tracks realised P&L through close/partial-close operations.

Persistence: serialised to/from data/positions.json via ``to_dict`` /
``from_dict``. The legacy importer ``from_legacy_trade`` reads the old
journal.py trade dict format.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from core.basket import Basket

logger = logging.getLogger(__name__)

# Valid position directions stored in positions.json.
_VALID_DIRECTIONS = frozenset({'long_spread', 'short_spread'})
# Valid lifecycle statuses.
_VALID_STATUSES = frozenset({'open', 'closed'})


@dataclass
class Position:
    """A single open or closed spread trade.

    Attributes:
        id: Unique identifier; typically a ``YYYYMMDD_HHMMSS`` timestamp.
        name: Human-readable label shown in the journal (e.g. ``"FTSE/DAX"``).
        basket: Long/short leg definition.
        direction: ``'long_spread'`` or ``'short_spread'``.
        entry_date: Calendar date the position was opened.
        entry_prices: ``{instrument: price}`` dict at entry.
        stakes: Signed contracts per instrument. Positive = long, negative = short.
        target_exposure: Target notional exposure in account currency.
        comments: Free-text notes; defaults to ``''``.
        status: ``'open'`` or ``'closed'``; defaults to ``'open'``.
        exit_date: Date fully closed; ``None`` while open.
        exit_prices: ``{instrument: price}`` dict at exit; empty while open.
        realised_pnl: Cumulative P&L locked in through close operations.
        pct_open: Fraction still open (1.0 = fully open, 0.0 = fully closed).
    """

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
    exit_date:       Optional[date] = None
    exit_prices:     dict[str, float] = field(default_factory=dict)
    realised_pnl:    float = 0.0
    pct_open:        float = 1.0

    @property
    def days_held(self) -> int:
        """Calendar days from entry to today (or to exit date if closed).

        Returns:
            Non-negative integer day count.
        """
        end = self.exit_date or date.today()
        return max(0, (end - self.entry_date).days)

    def live_pnl(self, current_prices: dict[str, float]) -> float:
        """Gross unrealised P&L at current prices, scaled by open fraction.

        Does not deduct financing or spread entry cost.

        Args:
            current_prices: Latest ``{instrument: price}`` dict. Instruments
                absent from this dict fall back to their entry price (P&L = 0).

        Returns:
            Gross P&L in account currency. Positive = profitable.
        """
        if not current_prices:
            logger.warning("live_pnl called with empty current_prices; returning 0.0")
            return 0.0

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
        """Cumulative financing drag since entry, in account currency.

        Returns:
            Daily financing rate × days held. Positive = net cost to holder.
        """
        return self.basket.financing_cost_daily() * self.days_held

    def net_pnl(self, current_prices: dict[str, float]) -> float:
        """Live P&L minus cumulative financing drag.

        Entry spread cost is excluded — it was paid at open and is captured
        in the position sizing, not tracked here.

        Args:
            current_prices: Latest ``{instrument: price}`` dict.

        Returns:
            Net P&L in account currency.
        """
        return self.live_pnl(current_prices) - self.financing_cost_to_date()

    def partial_close(
        self,
        pct: float,
        exit_prices: dict[str, float],
        exit_date: date,
    ) -> float:
        """Partially close the position.

        Args:
            pct: Fraction of the *currently open* portion to close. Must be
                in the range ``(0, 1]``. Passing 0 closes nothing; passing
                1 closes the entire open portion.
            exit_prices: ``{instrument: price}`` dict at exit.
            exit_date: Calendar date of the close.

        Returns:
            Realised P&L for the closed fraction.

        Raises:
            ValueError: If ``pct`` is not in ``(0, 1]``.
            ValueError: If the position is already closed.
        """
        if not (0.0 < pct <= 1.0):
            raise ValueError(
                f"pct must be in (0, 1]; got {pct!r}. "
                f"pct=0 closes nothing; pct=1 closes the full open portion."
            )
        if self.status == 'closed':
            raise ValueError(
                f"Position {self.id!r} is already closed; cannot partial_close again."
            )

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
        """Fully close the position.

        Args:
            exit_prices: ``{instrument: price}`` dict at exit.
            exit_date: Calendar date of the close.

        Returns:
            Total realised P&L for the full position.

        Raises:
            ValueError: If the position is already closed (idempotent close
                would silently corrupt cumulative P&L by double-counting).
        """
        if self.status == 'closed':
            raise ValueError(
                f"Position {self.id!r} is already closed (exit_date={self.exit_date}). "
                f"Calling close() again would double-count realised P&L."
            )

        realised = self.live_pnl(exit_prices)
        self.realised_pnl += realised
        self.pct_open    = 0.0
        self.status      = 'closed'
        self.exit_date   = exit_date
        self.exit_prices = exit_prices
        logger.info(
            "Position %s (%s) closed on %s; realised P&L: %.2f",
            self.id, self.name, exit_date, realised,
        )
        return realised

    def to_summary_row(self, current_prices: dict[str, float]) -> dict:
        """Build a single-row dict for display in a portfolio DataFrame.

        Args:
            current_prices: Latest ``{instrument: price}`` dict.

        Returns:
            Dict with keys ``id``, ``name``, ``direction``, ``entry_date``,
            ``days_held``, ``pct_open``, ``live_pnl``, ``financing_cost``,
            ``net_pnl``, and ``status``.
        """
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
        """Serialise to a JSON-safe dict for persistence in positions.json.

        Returns:
            Dict with all Position fields; dates serialised as ISO strings.
        """
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
    def from_dict(cls, d: dict) -> Position:
        """Deserialise from a dict produced by ``to_dict``.

        Args:
            d: Dict loaded from positions.json. Required keys: ``id``,
                ``name``, ``basket``, ``entry_date``, ``entry_prices``,
                ``stakes``. Optional keys fall back to safe defaults.

        Returns:
            A new Position instance.

        Raises:
            KeyError: If a required key is absent from ``d``.
        """
        def _parse_date(v) -> Optional[date]:
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
    def from_legacy_trade(cls, trade: dict) -> Position:
        """Convert a legacy journal.py trade dict to a Position.

        The legacy format used a ``legs`` list where each leg described one
        buy/sell pair. This converter merges all legs into a single Basket
        and signed stakes dict, which is the current format.

        Args:
            trade: Legacy trade dict. Required keys: ``id``, ``name``,
                ``legs``. Each leg must have ``buy_instrument``,
                ``sell_instrument``, ``buy_entry_price``,
                ``sell_entry_price``, ``buy_stake``, ``sell_stake``.

        Returns:
            A Position with ``direction='long_spread'`` and ``pct_open``
            set to ``min(leg['pct_open'])`` across all legs.

        Raises:
            KeyError: If required top-level or per-leg keys are absent.
            ValueError: If ``trade['legs']`` is empty.
        """
        required_top = ('id', 'name', 'legs')
        missing_top = [k for k in required_top if k not in trade]
        if missing_top:
            raise KeyError(
                f"from_legacy_trade: missing required keys {missing_top} in trade dict"
            )
        if not trade['legs']:
            raise ValueError(
                f"from_legacy_trade: trade {trade.get('id')!r} has no legs"
            )

        long_legs: list[str] = []
        short_legs: list[str] = []
        entry_prices: dict[str, float] = {}
        stakes: dict[str, float] = {}
        pct_opens: list[float] = []

        required_leg_keys = (
            'buy_instrument', 'sell_instrument',
            'buy_entry_price', 'sell_entry_price',
            'buy_stake', 'sell_stake',
        )
        for idx, leg in enumerate(trade.get('legs', [])):
            missing_leg = [k for k in required_leg_keys if k not in leg]
            if missing_leg:
                raise KeyError(
                    f"from_legacy_trade: leg {idx} missing keys {missing_leg}"
                )
            bi = leg['buy_instrument']
            si = leg['sell_instrument']
            long_legs.append(bi)
            short_legs.append(si)
            entry_prices[bi] = leg['buy_entry_price']
            entry_prices[si] = leg['sell_entry_price']
            stakes[bi] = +leg['buy_stake']
            stakes[si] = -leg['sell_stake']
            pct_opens.append(leg.get('pct_open', 1.0))

        def _parse_date(v) -> Optional[date]:
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
