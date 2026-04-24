import json
from datetime import date, datetime
from pathlib import Path

from config import POINT_SIZES, SPREADS

# --- Persistence ---
JOURNAL_PATH = Path(__file__).parent / 'trade_journal.json'


# --- Load ---

def load_trades() -> list[dict]:
    if not JOURNAL_PATH.exists():
        return []
    try:
        trades = json.loads(JOURNAL_PATH.read_text())
    except Exception:
        return []
    # --- Legacy migration ---
    # Older records stored long_flags/short_flags instead of the current legs structure.
    # Patch them on load so the rest of the app always sees the legs format.
    for t in trades:
        if 'legs' not in t:
            lf = t.get('long_flags', {})
            sf = t.get('short_flags', {})
            buy_inst  = next((k for k, v in lf.items() if v), '')
            sell_inst = next((k for k, v in sf.items() if v), '')
            t['legs'] = [{
                'leg_id':           0,
                'buy_instrument':   buy_inst,
                'buy_entry_price':  0.0,
                'buy_stake':        t.get('exposure', 0.0),
                'sell_instrument':  sell_inst,
                'sell_entry_price': 0.0,
                'sell_stake':       t.get('exposure', 0.0),
                'pct_open':         0.0 if t['status'] == 'closed' else 1.0,
                'closes':           [],
            }]
            t['target_exposure'] = t.get('exposure', 0.0)
            t.setdefault('comments', '')
            t.setdefault('realised_pnl', 0.0)
    return trades


# --- Save (atomic write) ---

def _save(trades: list[dict]) -> None:
    # Write to a temp file first, then replace — prevents a corrupt journal if the process is killed mid-write
    tmp = JOURNAL_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(trades, indent=2))
    tmp.replace(JOURNAL_PATH)


# --- Open Trade ---

def open_trade(
    name: str,
    legs: list[dict],
    target_exposure: float,
    entry_date: str,
    comments: str = '',
) -> str:
    # Use timestamp as a unique, sortable trade ID
    trade_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    normalised = []
    for i, leg in enumerate(legs):
        normalised.append({
            'leg_id':           i,
            'buy_instrument':   leg['buy_instrument'],
            'buy_entry_price':  float(leg['buy_entry_price']),
            'buy_stake':        float(leg['buy_stake']),
            'sell_instrument':  leg['sell_instrument'],
            'sell_entry_price': float(leg['sell_entry_price']),
            'sell_stake':       float(leg['sell_stake']),
            'pct_open':         1.0,   # fully open at entry
            'closes':           [],
        })
    trades = load_trades()
    trades.append({
        'id':              trade_id,
        'name':            name,
        'target_exposure': float(target_exposure),
        'entry_date':      entry_date,
        'comments':        comments,
        'status':          'open',
        'legs':            normalised,
        'realised_pnl':    0.0,
        'exit_date':       None,
    })
    _save(trades)
    return trade_id


# --- Partial Close ---

def partial_close_leg(
    trade_id: str,
    leg_id: int,
    pct_to_close: float,
    buy_exit_price: float,
    sell_exit_price: float,
    exit_date: str,
) -> None:
    trades = load_trades()
    for t in trades:
        if t['id'] != trade_id:
            continue
        for leg in t['legs']:
            if leg['leg_id'] != leg_id:
                continue
            open_frac   = leg['pct_open']
            closed_frac = open_frac * pct_to_close   # fraction of the original position being closed now
            bi, si      = leg['buy_instrument'], leg['sell_instrument']
            bpts, spts  = POINT_SIZES.get(bi, 1.0), POINT_SIZES.get(si, 1.0)
            # P&L = gain on long side minus gain on short side for the fraction closed
            leg_pnl = (
                (buy_exit_price  - leg['buy_entry_price'])  * leg['buy_stake']  * closed_frac * bpts
              - (sell_exit_price - leg['sell_entry_price']) * leg['sell_stake'] * closed_frac * spts
            )
            leg['closes'].append({
                'date':            exit_date,
                'pct_closed':      round(closed_frac, 6),
                'buy_exit_price':  float(buy_exit_price),
                'sell_exit_price': float(sell_exit_price),
                'pnl':             round(leg_pnl, 2),
            })
            leg['pct_open']    = round(open_frac - closed_frac, 6)
            t['realised_pnl']  = round(t['realised_pnl'] + leg_pnl, 2)
        # Auto-close the trade when all legs are effectively flat
        if all(leg['pct_open'] < 1e-6 for leg in t['legs']):
            t['status']    = 'closed'
            t['exit_date'] = exit_date
        break
    _save(trades)


# --- Full Close ---

def close_trade(trade_id: str, exit_prices: dict, exit_date: str) -> None:
    """Close all remaining open fractions of every leg at the provided exit prices."""
    trades = load_trades()
    for t in trades:
        if t['id'] != trade_id:
            continue
        for leg in t['legs']:
            if leg['pct_open'] < 1e-6:
                continue   # already closed
            bi, si     = leg['buy_instrument'], leg['sell_instrument']
            buy_price  = exit_prices.get(bi,  leg['buy_entry_price'])
            sell_price = exit_prices.get(si, leg['sell_entry_price'])
            bpts, spts = POINT_SIZES.get(bi, 1.0), POINT_SIZES.get(si, 1.0)
            leg_pnl = (
                (buy_price  - leg['buy_entry_price'])  * leg['buy_stake']  * leg['pct_open'] * bpts
              - (sell_price - leg['sell_entry_price']) * leg['sell_stake'] * leg['pct_open'] * spts
            )
            leg['closes'].append({
                'date':            exit_date,
                'pct_closed':      leg['pct_open'],
                'buy_exit_price':  float(buy_price),
                'sell_exit_price': float(sell_price),
                'pnl':             round(leg_pnl, 2),
            })
            t['realised_pnl'] = round(t['realised_pnl'] + leg_pnl, 2)
            leg['pct_open']   = 0.0
        t['status']    = 'closed'
        t['exit_date'] = exit_date
        break
    _save(trades)


# --- Live P&L ---

def trade_live_pnl(trade: dict, current_prices: dict) -> float:
    """Mark-to-market P&L on all open leg fractions using current prices."""
    total = 0.0
    for leg in trade.get('legs', []):
        if leg['pct_open'] < 1e-6:
            continue
        bi, si     = leg['buy_instrument'], leg['sell_instrument']
        buy_price  = current_prices.get(bi,  leg['buy_entry_price'])
        sell_price = current_prices.get(si, leg['sell_entry_price'])
        bpts, spts = POINT_SIZES.get(bi, 1.0), POINT_SIZES.get(si, 1.0)
        total += (
            (buy_price  - leg['buy_entry_price'])  * leg['buy_stake']  * leg['pct_open'] * bpts
          - (sell_price - leg['sell_entry_price']) * leg['sell_stake'] * leg['pct_open'] * spts
        )
    return total


# --- Delete ---

def delete_trade(trade_id: str) -> None:
    trades = [t for t in load_trades() if t['id'] != trade_id]
    _save(trades)
