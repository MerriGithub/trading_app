import pytest
from core.basket import Basket


def test_pair_construction():
    b = Basket.pair('PLATINUM', 'AUDUSD')
    assert b.n_long == 1 and b.n_short == 1
    assert b.is_cross_asset


def test_nvm_construction():
    b = Basket(['UKX', 'CBK', 'CFR'], ['CPH', 'CTN', 'CTB'])
    assert b.n_long == 3 and b.n_short == 3
    assert not b.is_cross_asset


def test_financing_cost_cross_asset():
    b = Basket.pair('PLATINUM', 'AUDUSD')
    cost = b.financing_cost_daily()
    # Platinum long_rate=4.88%; AUDUSD short_rate=-0.018 (both sides pay swap)
    # Net = (0.0488 - (-0.018)) / 365 = 6.68% p.a.
    expected = (0.0488 + 0.018) / 365
    assert abs(cost - expected) < 1e-8


def test_financing_cost_pure_fx():
    b = Basket.pair('EURUSD', 'GBPUSD')
    cost = b.financing_cost_daily()
    # Both FX: (0.018 - (-0.018)) / 365 = 3.60% p.a.
    expected = (0.018 + 0.018) / 365
    assert abs(cost - expected) < 1e-8


def test_self_dealing_raises():
    with pytest.raises(ValueError):
        Basket(['UKX'], ['UKX']).validate()


def test_empty_long_raises():
    with pytest.raises(ValueError):
        Basket([], ['UKX']).validate()


def test_empty_short_raises():
    with pytest.raises(ValueError):
        Basket(['UKX'], []).validate()


def test_serialisation_roundtrip():
    b = Basket(['PLATINUM', 'GOLD'], ['AUDUSD'])
    assert Basket.from_dict(b.to_dict()) == b


def test_hash_usable_as_dict_key():
    b1 = Basket.pair('UKX', 'CPH')
    b2 = Basket.pair('UKX', 'CPH')
    d = {b1: 'test'}
    assert d[b2] == 'test'


def test_from_search_result():
    class FakeRow:
        def __getitem__(self, key):
            return 'UKX|CBK' if key == 'LongLegs' else 'CPH|CTN'
    b = Basket.from_search_result(FakeRow())
    assert b.long_legs == ['UKX', 'CBK']
    assert b.short_legs == ['CPH', 'CTN']


def test_repr():
    b = Basket.pair('UKX', 'CPH')
    assert 'UKX' in repr(b) and 'CPH' in repr(b)
