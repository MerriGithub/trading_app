from dataclasses import dataclass


@dataclass
class Basket:
    long_legs:  list[str]
    short_legs: list[str]

    @property
    def n_long(self) -> int:
        return len(self.long_legs)

    @property
    def n_short(self) -> int:
        return len(self.short_legs)

    @property
    def all_instruments(self) -> list[str]:
        return self.long_legs + self.short_legs

    @property
    def is_cross_asset(self) -> bool:
        """True if legs span more than one asset class."""
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
        """Return {instrument_code: asset_class_key} for all legs."""
        from asset_configs import ASSET_CLASSES
        result = {}
        for code in self.all_instruments:
            for key, cfg in ASSET_CLASSES.items():
                if code in cfg['instruments']:
                    result[code] = key
                    break
        return result

    def financing_cost_daily(self) -> float:
        """
        Net daily financing drag using per-asset-class rates from account.json.
        Long legs pay their class long_rate.
        Short legs receive their class short_rate (negative = additional cost).
        Returns positive = net cost to holder.
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
        """Round-trip bid-ask fraction for this basket."""
        from asset_configs import get_spread_cost_lookup, basket_spread_cost
        instruments = self.all_instruments
        latest_px = registry.get_latest_prices(instruments)
        lookup = get_spread_cost_lookup(instruments, latest_px)
        long_idx  = tuple(range(self.n_long))
        short_idx = tuple(range(self.n_long, self.n_long + self.n_short))
        return basket_spread_cost(long_idx, short_idx, instruments, lookup)

    def validate(self) -> None:
        if not self.long_legs:
            raise ValueError('Basket must have at least one long leg')
        if not self.short_legs:
            raise ValueError('Basket must have at least one short leg')
        overlap = set(self.long_legs) & set(self.short_legs)
        if overlap:
            raise ValueError(f'Instruments in both legs: {overlap}')
        for code in self.all_instruments:
            if any(c in code for c in ' \t\n/\\'):
                raise ValueError(f'Invalid instrument code: {code!r}')

    def to_dict(self) -> dict:
        return {'long_legs': self.long_legs, 'short_legs': self.short_legs}

    @classmethod
    def from_dict(cls, d: dict) -> 'Basket':
        return cls(long_legs=d['long_legs'], short_legs=d['short_legs'])

    @classmethod
    def pair(cls, long: str, short: str) -> 'Basket':
        return cls([long], [short])

    @classmethod
    def from_search_result(cls, row) -> 'Basket':
        """Construct from search engine result row (pipe-separated strings)."""
        long_legs  = [s.strip() for s in str(row['LongLegs']).split('|')]
        short_legs = [s.strip() for s in str(row['ShortLegs']).split('|')]
        return cls(long_legs, short_legs)

    def __eq__(self, other):
        return (isinstance(other, Basket) and
                self.long_legs == other.long_legs and
                self.short_legs == other.short_legs)

    def __hash__(self):
        return hash((tuple(self.long_legs), tuple(self.short_legs)))

    def __repr__(self):
        return f"Basket({'+'.join(self.long_legs)} / {'+'.join(self.short_legs)})"
