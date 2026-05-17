import pytest
from datetime import date

from core.basket import Basket
from core.position import Position


def _make_pos(**kwargs) -> Position:
    defaults = dict(
        id='t1',
        name='Test',
        basket=Basket.pair('PLATINUM', 'AUDUSD'),
        direction='long_spread',
        entry_date=date(2026, 1, 1),
        entry_prices={'PLATINUM': 1000.0, 'AUDUSD': 0.64},
        stakes={'PLATINUM': +5.0, 'AUDUSD': -10000.0},
        target_exposure=500.0,
    )
    defaults.update(kwargs)
    return Position(**defaults)


def test_live_pnl_long_spread():
    pos = _make_pos()
    current = {'PLATINUM': 1010.0, 'AUDUSD': 0.64}  # Platinum +10pts, AUDUSD flat
    assert pos.live_pnl(current) == pytest.approx(50.0)  # 5 × 10 pts


def test_live_pnl_flat():
    pos = _make_pos()
    current = {'PLATINUM': 1000.0, 'AUDUSD': 0.64}
    assert pos.live_pnl(current) == pytest.approx(0.0)


def test_financing_cost_positive():
    pos = _make_pos()
    cost = pos.financing_cost_to_date()
    assert cost >= 0  # cost, not gain


def test_days_held():
    pos = _make_pos(entry_date=date(2026, 1, 1))
    # days_held is non-negative regardless of today's date
    assert pos.days_held >= 0


def test_serialisation_roundtrip():
    pos = _make_pos()
    pos2 = Position.from_dict(pos.to_dict())
    assert pos2.id == pos.id
    assert pos2.basket == pos.basket
    assert pos2.stakes == pos.stakes
    assert pos2.entry_date == pos.entry_date


def test_legacy_migration_single_leg():
    legacy = {
        'id': 'abc', 'name': 'Test 1v1', 'status': 'open',
        'entry_date': '2026-01-01', 'target_exposure': 500.0,
        'realised_pnl': 0.0,
        'legs': [{
            'buy_instrument': 'UKX', 'buy_entry_price': 8000.0, 'buy_stake': 1.0,
            'sell_instrument': 'CPH', 'sell_entry_price': 18000.0, 'sell_stake': 0.5,
            'pct_open': 1.0,
        }],
    }
    pos = Position.from_legacy_trade(legacy)
    assert pos.basket.long_legs == ['UKX']
    assert pos.basket.short_legs == ['CPH']
    assert pos.stakes['UKX'] == pytest.approx(+1.0)
    assert pos.stakes['CPH'] == pytest.approx(-0.5)
    assert pos.direction == 'long_spread'


def test_legacy_migration_multi_leg():
    legacy = {
        'id': 'abc', 'name': 'Test 3v3', 'status': 'open',
        'entry_date': '2026-01-01', 'target_exposure': 1500.0,
        'realised_pnl': 0.0,
        'legs': [
            {'buy_instrument': 'UKX', 'buy_entry_price': 8000.0, 'buy_stake': 1.0,
             'sell_instrument': 'CPH', 'sell_entry_price': 27000.0, 'sell_stake': 0.5,
             'pct_open': 1.0},
            {'buy_instrument': 'CBK', 'buy_entry_price': 8200.0, 'buy_stake': 1.2,
             'sell_instrument': 'CTN', 'sell_entry_price': 7100.0, 'sell_stake': 1.4,
             'pct_open': 1.0},
        ],
    }
    pos = Position.from_legacy_trade(legacy)
    assert pos.basket.long_legs == ['UKX', 'CBK']
    assert pos.basket.short_legs == ['CPH', 'CTN']
    assert pos.stakes['UKX'] == pytest.approx(+1.0)
    assert pos.stakes['CPH'] == pytest.approx(-0.5)
    assert pos.direction == 'long_spread'


def test_partial_close():
    pos = _make_pos()
    current = {'PLATINUM': 1010.0, 'AUDUSD': 0.64}
    realised = pos.partial_close(0.5, current, date(2026, 6, 1))
    assert pos.pct_open == pytest.approx(0.5)
    assert realised == pytest.approx(25.0)  # half of 50
    assert pos.status == 'open'


def test_full_close():
    pos = _make_pos()
    current = {'PLATINUM': 1010.0, 'AUDUSD': 0.64}
    realised = pos.close(current, date(2026, 6, 1))
    assert pos.status == 'closed'
    assert pos.pct_open == pytest.approx(0.0)
    assert realised == pytest.approx(50.0)
