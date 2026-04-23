import json
from datetime import date, datetime
from pathlib import Path

JOURNAL_PATH = Path(__file__).parent / 'trade_journal.json'


def load_trades() -> list[dict]:
    if not JOURNAL_PATH.exists():
        return []
    try:
        return json.loads(JOURNAL_PATH.read_text())
    except Exception:
        return []


def _save_trades(trades: list[dict]) -> None:
    JOURNAL_PATH.write_text(json.dumps(trades, indent=2))


def open_trade(
    name: str,
    long_flags: dict,
    short_flags: dict,
    long_display: str,
    short_display: str,
    direction: str,
    exposure: float,
    entry_spread: float,
    entry_date: str,
) -> str:
    trade_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    trades = load_trades()
    trades.append({
        'id':           trade_id,
        'name':         name,
        'long_flags':   long_flags,
        'short_flags':  short_flags,
        'long_display': long_display,
        'short_display': short_display,
        'direction':    direction,       # 'Buy' or 'Sell'
        'exposure':     float(exposure),
        'entry_spread': float(entry_spread),
        'entry_date':   entry_date,
        'status':       'open',
        'exit_spread':  None,
        'exit_date':    None,
        'realised_pnl': None,
    })
    _save_trades(trades)
    return trade_id


def close_trade(trade_id: str, exit_spread: float, exit_date: str) -> None:
    trades = load_trades()
    for t in trades:
        if t['id'] == trade_id:
            t['status']      = 'closed'
            t['exit_spread'] = float(exit_spread)
            t['exit_date']   = exit_date
            sign = 1 if t['direction'] == 'Buy' else -1
            t['realised_pnl'] = round(
                (float(exit_spread) - t['entry_spread']) * t['exposure'] * sign, 2
            )
            break
    _save_trades(trades)


def delete_trade(trade_id: str) -> None:
    trades = [t for t in load_trades() if t['id'] != trade_id]
    _save_trades(trades)
