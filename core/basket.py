"""
core/basket.py — Basket: the long/short leg definition for a spread.

A Basket is the immutable descriptor of which instruments form the long side
and which form the short side. It carries no price or timing state — those
live in SpreadSignal and Position respectively.

Basket objects are stored inside Position and serialised to positions.json
via ``to_dict`` / ``from_dict``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Basket:
    """Immutable long/short leg descriptor for a spread trade.

    Attributes:
        long_legs: Instrument codes for the long side.
        short_legs: Instrument codes for the short side.
    """

    long_legs:  list[str]
    short_legs: list[str]

    @property
    def n_long(self) -> int:
        """Number of long legs."""
        return len(self.long_legs)

    @property
    def n_short(self) -> int:
        """Number of short legs."""
        return len(self.short_legs)

    @property
    def all_instruments(self) -> list[str]:
        """All instrument codes in leg order: long first, then short.

        Returns:
            Combined list ``long_legs + short_legs``.
        """
        return self.long_legs + self.short_legs

    @property
    def is_cross_asset(self) -> bool:
        """True if legs span more than one asset class.

        Returns:
            True when instruments from at least two different asset classes
            are present across both legs.
        """
        from asset_configs import ASSET_CLASSES

        def _class(code: str) -> str:
            for key, cfg in ASSET_CLASSES.items():
                if code in cfg['instruments']:
                    return key
            return 'unknown'

        classes = {_class(c) for c in self.all_instruments}
        return len(classes) > 1

    @property
    def asset_classes(self) -> dict[str, str]:
        """Return ``{instrument_code: asset_class_key}`` for all legs.

        Returns:
            Dict mapping each instrument code to its asset class key
            (e.g. ``'equity'``, ``'fx'``, ``'commodities'``). Instruments
            not found in any class are omitted.
        """
        from asset_configs import ASSET_CLASSES
        result: dict[str, str] = {}
        for code in self.all_instruments:
            for key, cfg in ASSET_CLASSES.items():
                if code in cfg['instruments']:
                    result[code] = key
                    break
        return result

    def financing_cost_daily(self) -> float:
        """Net daily financing drag across all legs, as a fraction of notional.

        Sums per-leg annual rates (from account.json) then divides by 365.
        Long legs pay their asset-class long rate; short legs pay their
        short rate (negative short_rate = additional cost, not a credit).

        Returns:
            Daily cost as a positive decimal (e.g. 0.00011 = 0.011% per day).
            Annual rates: equity long=4.88%, short=0.88%; FX both sides pay
            1.8% (no rebate).
        """
        from account import get_financing_rates
        from asset_configs import ASSET_CLASSES

        def _asset_class(code: str) -> str:
            for key, cfg in ASSET_CLASSES.items():
                if code in cfg['instruments']:
                    return key
            return 'equity'

        long_cost  = sum(
            get_financing_rates(_asset_class(c))[0] for c in self.long_legs
        )
        short_gain = sum(
            get_financing_rates(_asset_class(c))[1] for c in self.short_legs
        )
        return (long_cost - short_gain) / 365

    def spread_cost(self, registry) -> float:
        """Round-trip bid-ask fraction for the full basket.

        Args:
            registry: A ``DataRegistry`` instance used to fetch latest prices
                for instruments that require price-dependent spread calculation.

        Returns:
            Round-trip spread cost as a fraction (e.g. 0.002 = 0.2%).
            Formula: 4 × mean(spread_pct_i) — factor of 4 covers ×2 round-
            trip and ×2 for both legs.
        """
        from asset_configs import get_spread_cost_lookup, basket_spread_cost
        instruments = self.all_instruments
        latest_px = registry.get_latest_prices(instruments)
        lookup = get_spread_cost_lookup(instruments, latest_px)
        long_idx  = tuple(range(self.n_long))
        short_idx = tuple(range(self.n_long, self.n_long + self.n_short))
        return basket_spread_cost(long_idx, short_idx, instruments, lookup)

    def validate(self) -> None:
        """Validate basket structure.

        Checks:
        - At least one long leg.
        - At least one short leg.
        - No instrument appears on both sides (self-dealing).
        - No instrument code contains whitespace or path-separator characters.
        - All instrument codes are non-empty strings.

        Raises:
            ValueError: If any of the above checks fail. The message identifies
                the failing condition (e.g. which instruments overlap).
        """
        if not self.long_legs:
            raise ValueError('Basket must have at least one long leg')
        if not self.short_legs:
            raise ValueError('Basket must have at least one short leg')

        # Guard: all codes must be non-empty strings (catches None/int slipping in)
        for side_label, legs in (('long', self.long_legs), ('short', self.short_legs)):
            for code in legs:
                if not isinstance(code, str) or not code.strip():
                    raise ValueError(
                        f"Invalid {side_label} instrument code: {code!r} "
                        f"(must be a non-empty string)"
                    )

        overlap = set(self.long_legs) & set(self.short_legs)
        if overlap:
            raise ValueError(f'Instruments in both legs: {overlap}')

        for code in self.all_instruments:
            if any(c in code for c in ' \t\n/\\'):
                raise ValueError(f'Invalid instrument code: {code!r}')

    def to_dict(self) -> dict[str, list[str]]:
        """Serialise to a JSON-safe dict.

        Returns:
            ``{'long_legs': [...], 'short_legs': [...]}``
        """
        return {'long_legs': self.long_legs, 'short_legs': self.short_legs}

    @classmethod
    def from_dict(cls, d: dict) -> Basket:
        """Deserialise from a dict produced by ``to_dict``.

        Args:
            d: Dict with keys ``'long_legs'`` and ``'short_legs'``.

        Returns:
            A new Basket instance.

        Raises:
            KeyError: If ``'long_legs'`` or ``'short_legs'`` are absent.
        """
        return cls(long_legs=d['long_legs'], short_legs=d['short_legs'])

    @classmethod
    def pair(cls, long: str, short: str) -> Basket:
        """Convenience constructor for a 1v1 directional pair.

        Args:
            long: Instrument code for the long leg.
            short: Instrument code for the short leg.

        Returns:
            A Basket with one long leg and one short leg.

        Raises:
            ValueError: If ``long == short`` (self-dealing) or either code
                is empty. Checked via ``validate()`` on construction.
        """
        if not isinstance(long, str) or not long.strip():
            raise ValueError(f"long must be a non-empty string; got {long!r}")
        if not isinstance(short, str) or not short.strip():
            raise ValueError(f"short must be a non-empty string; got {short!r}")
        if long == short:
            raise ValueError(
                f"long and short must be different instruments; both are {long!r}"
            )
        return cls([long], [short])

    @classmethod
    def from_search_result(cls, row) -> Basket:
        """Construct from a search engine result row.

        Args:
            row: A dict or DataFrame row with ``'LongLegs'`` and ``'ShortLegs'``
                as pipe-separated instrument code strings.

        Returns:
            A new Basket instance.
        """
        long_legs  = [s.strip() for s in str(row['LongLegs']).split('|')]
        short_legs = [s.strip() for s in str(row['ShortLegs']).split('|')]
        return cls(long_legs, short_legs)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Basket) and
                self.long_legs == other.long_legs and
                self.short_legs == other.short_legs)

    def __hash__(self) -> int:
        return hash((tuple(self.long_legs), tuple(self.short_legs)))

    def __repr__(self) -> str:
        return f"Basket({'+'.join(self.long_legs)} / {'+'.join(self.short_legs)})"
