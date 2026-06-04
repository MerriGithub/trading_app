"""
core/portfolio.py — Portfolio: aggregate open and closed positions.

Owns the positions.json persistence layer. All mutations go through this
class so that the file is always in a consistent state (atomic write via a
temp file + os.replace).

Singletons are constructed once via ``@st.cache_resource`` in tabs/shared.py.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from core.basket import Basket
from core.position import Position

logger = logging.getLogger(__name__)

# Valid direction strings accepted by open_position.
_VALID_DIRECTIONS = frozenset({'long_spread', 'short_spread'})


class Portfolio:
    """Aggregate store for open and closed spread positions.

    Loads positions from a JSON file on construction. All mutations
    (open, close, partial-close) are persisted atomically after each
    operation.

    Args:
        positions_file: Path to ``data/positions.json``.
        account_file: Path to ``data/account.json`` (reserved for future use).
    """

    def __init__(self, positions_file: Path, account_file: Path) -> None:
        self._path    = Path(positions_file)
        self._acct    = Path(account_file)
        self._positions: list[Position] = self._load()
        logger.info(
            "Portfolio loaded: %d positions (%d open) from %s",
            len(self._positions),
            len(self.open_positions),
            self._path,
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> list[Position]:
        """Load positions from disk; return empty list if file is absent or corrupt."""
        if not self._path.exists():
            logger.debug("positions file not found at %s; starting empty", self._path)
            return []
        try:
            data = json.loads(self._path.read_text())
            return [Position.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            # File corrupt or from an incompatible schema version.
            # Log and return empty rather than crashing the app.
            logger.error(
                "Failed to load positions from %s: %s. Starting with empty portfolio.",
                self._path, exc,
            )
            return []

    def _save(self) -> None:
        """Atomically write all positions to disk via a temp file."""
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
        """Open a new spread position and persist it.

        Args:
            basket: Long/short leg definition.
            direction: ``'long_spread'`` or ``'short_spread'``.
            entry_prices: ``{instrument: price}`` at entry. Must be non-empty.
            stakes: Signed contracts per instrument (positive = long).
                Must be non-empty.
            target_exposure: Target notional in account currency. Must be
                positive.
            name: Human-readable label (e.g. ``'FTSE/DAX'``).
            comments: Optional free-text notes.

        Returns:
            The newly created ``Position`` instance.

        Raises:
            ValueError: If ``direction`` is not ``'long_spread'`` or
                ``'short_spread'``, ``entry_prices`` is empty, ``stakes``
                is empty, or ``target_exposure`` is not positive.
        """
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_VALID_DIRECTIONS)}; "
                f"got {direction!r}"
            )
        if not entry_prices:
            raise ValueError(
                "entry_prices must be a non-empty dict; got empty dict. "
                "Provide at least one {instrument: price} entry."
            )
        if not stakes:
            raise ValueError(
                "stakes must be a non-empty dict; got empty dict. "
                "Provide at least one {instrument: signed_contracts} entry."
            )
        if not (target_exposure > 0):
            raise ValueError(
                f"target_exposure must be positive; got {target_exposure!r}"
            )

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
        logger.info(
            "Opened position %s (%s %s); target_exposure=%.0f",
            pos_id, direction, name, target_exposure,
        )
        return pos

    def close_position(
        self,
        position_id: str,
        exit_prices: dict[str, float],
        exit_date: date,
    ) -> float:
        """Fully close an open position.

        Args:
            position_id: The ``id`` of the position to close.
            exit_prices: ``{instrument: price}`` at exit.
            exit_date: Calendar date of the close.

        Returns:
            Total realised P&L for the position.

        Raises:
            KeyError: If ``position_id`` is not found in open positions.
        """
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
        """Partially close an open position.

        Args:
            position_id: The ``id`` of the position to partially close.
            pct: Fraction of the currently open portion to close. Must be
                in ``(0, 1]``.
            exit_prices: ``{instrument: price}`` at exit.
            exit_date: Calendar date of the close.

        Returns:
            Realised P&L for the closed fraction.

        Raises:
            KeyError: If ``position_id`` is not found.
        """
        pos = self._get(position_id)
        realised = pos.partial_close(pct, exit_prices, exit_date)
        self._save()
        return realised

    def _get(self, position_id: str) -> Position:
        """Look up a position by id.

        Args:
            position_id: The target position id.

        Returns:
            The matching ``Position`` instance.

        Raises:
            KeyError: With a descriptive message if the id is not found.
        """
        for p in self._positions:
            if p.id == position_id:
                return p
        raise KeyError(
            f"Position not found: {position_id!r}. "
            f"Open position ids: {[p.id for p in self.open_positions]}"
        )

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> list[Position]:
        """All positions with status ``'open'``."""
        return [p for p in self._positions if p.status == 'open']

    @property
    def closed_positions(self) -> list[Position]:
        """All positions with status ``'closed'``."""
        return [p for p in self._positions if p.status == 'closed']

    def total_unrealised_pnl(self, current_prices: dict[str, float]) -> float:
        """Sum of live P&L across all open positions.

        Args:
            current_prices: Latest ``{instrument: price}`` dict.

        Returns:
            Aggregate gross unrealised P&L in account currency.
        """
        return sum(p.live_pnl(current_prices) for p in self.open_positions)

    def total_realised_pnl(self) -> float:
        """Sum of realised P&L across all positions (open and closed).

        Returns:
            Aggregate realised P&L. Partial closes contribute incrementally.
        """
        return sum(p.realised_pnl for p in self._positions)

    def position_summary(
        self,
        current_prices: dict[str, float],
        account: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Build a display DataFrame of all open positions.

        Args:
            current_prices: Latest ``{instrument: price}`` dict.
            account: Unused; accepted for interface compatibility.

        Returns:
            DataFrame with one row per open position. Empty DataFrame if
            there are no open positions.
        """
        rows = [p.to_summary_row(current_prices) for p in self.open_positions]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def cross_pair_correlation(self, registry) -> pd.DataFrame:
        """Correlation matrix of spread returns across open positions.

        Returns an empty DataFrame (not None) when fewer than 2 positions
        are open — callers should check ``df.empty`` rather than ``df is None``.

        Args:
            registry: A ``DataRegistry`` instance for fetching price data.

        Returns:
            Square correlation DataFrame indexed/columned by position name.
            Returns an empty DataFrame if fewer than 2 open positions, or if
            no spread return series can be computed.
        """
        if not self.open_positions:
            # Returns empty DataFrame, not None — callers must use df.empty check.
            return pd.DataFrame()

        series: dict[str, pd.Series] = {}
        for pos in self.open_positions:
            instr = pos.basket.all_instruments
            try:
                prices = registry.get_daily_prices(instr)
                from engine.backtest import prepare_returns
                scaled, _, idx = prepare_returns(prices, instr)
                col = {inst: i for i, inst in enumerate(instr)}
                long_r  = scaled[:, [col[i] for i in pos.basket.long_legs]].mean(axis=1)
                short_r = scaled[:, [col[i] for i in pos.basket.short_legs]].mean(axis=1)
                series[pos.name] = pd.Series(long_r - short_r, index=idx)
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "cross_pair_correlation: could not compute series for %s: %s",
                    pos.name, exc,
                )

        # Invariant: returns empty DataFrame when <2 series (see ARCHITECTURE.md)
        if len(series) < 2:
            return pd.DataFrame()

        result = pd.DataFrame(series).corr()
        assert isinstance(result, pd.DataFrame), (
            "cross_pair_correlation must return a DataFrame, got "
            f"{type(result).__name__}"
        )
        return result

    def capital_at_risk(self, current_prices: dict) -> float:
        """Parametric VaR estimate: 2-SD 10-day loss across open positions.

        Not a full portfolio VaR — does not account for cross-position
        correlation. Deferred full implementation to Sprint 3.

        Args:
            current_prices: Latest ``{instrument: price}`` dict.

        Returns:
            Sum of per-instrument ``|stake| × price × daily_vol × sqrt(10) × 2``
            scaled by pct_open. Zero if no open positions.
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
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "capital_at_risk: could not compute CAR for %s: %s",
                    pos.name, exc,
                )
        return total_car
