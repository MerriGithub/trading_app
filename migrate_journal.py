"""
migrate_journal.py — One-time migration: legacy journal → new Position format.

Usage:
    python migrate_journal.py [--dry-run]

Reads:  legacy/trade_journal.json
Writes: data/positions.json

Legacy format: list of trades, each with a 'legs' list of
{buy_instrument, sell_instrument, buy_stake, sell_stake, buy_entry_price,
 sell_entry_price, pct_open}.

New format: Position with signed stakes dict and Basket object.
Direction is inferred as 'long_spread' for all legacy trades.
"""
import json, shutil, sys
from pathlib import Path

SRC = Path('legacy/trade_journal.json')
DST = Path('data/positions.json')


def _migrate_trade(trade: dict) -> dict:
    long_legs, short_legs = [], []
    entry_prices, stakes = {}, {}
    pct_opens = []

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

    return {
        'id':              trade['id'],
        'name':            trade['name'],
        'basket':          {'long_legs': long_legs, 'short_legs': short_legs},
        'direction':       'long_spread',
        'entry_date':      trade.get('entry_date', '2000-01-01'),
        'entry_prices':    entry_prices,
        'stakes':          stakes,
        'target_exposure': trade.get('target_exposure', 0.0),
        'comments':        trade.get('comments', ''),
        'status':          trade.get('status', 'open'),
        'exit_date':       trade.get('exit_date'),
        'exit_prices':     trade.get('exit_prices', {}),
        'realised_pnl':    trade.get('realised_pnl', 0.0),
        'pct_open':        min(pct_opens) if pct_opens else 1.0,
    }


def migrate(dry_run: bool = False) -> None:
    if not SRC.exists():
        print(f'No legacy journal at {SRC} — nothing to migrate.')
        return

    if not dry_run:
        shutil.copy(SRC, SRC.with_suffix('.json.bak'))
        print(f'Backed up → {SRC.with_suffix(".json.bak")}')

    legacy = json.loads(SRC.read_text())
    positions = [_migrate_trade(t) for t in legacy]

    for p in positions:
        n_long  = len(p['basket']['long_legs'])
        n_short = len(p['basket']['short_legs'])
        action  = 'Would write' if dry_run else 'Migrated'
        print(f'{action}: {p["name"]} ({n_long}v{n_short}, {p["status"]})')

    if not dry_run:
        DST.write_text(json.dumps(positions, indent=2))
        print(f'\n{len(positions)} positions written to {DST}')
    else:
        print(f'\nDry run — {len(positions)} positions would be written to {DST}')


if __name__ == '__main__':
    migrate('--dry-run' in sys.argv)
