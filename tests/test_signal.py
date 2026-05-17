import numpy as np
import pandas as pd
import pytest

from core.basket import Basket
from core.signal import SpreadSignal


@pytest.fixture
def cross_asset_prices():
    """Synthetic prices for PLATINUM and AUDUSD over 500 days."""
    dates = pd.date_range('2024-01-01', periods=500, freq='B')
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'PLATINUM': 1000 * np.cumprod(1 + rng.normal(0, 0.01, 500)),
        'AUDUSD':   0.65 * np.cumprod(1 + rng.normal(0, 0.005, 500)),
    }, index=dates)


def test_signal_computes(cross_asset_prices):
    b = Basket.pair('PLATINUM', 'AUDUSD')
    sig = SpreadSignal(basket=b, prices=cross_asset_prices)
    assert len(sig.distance_sd) > 0
    assert sig.current_sd == pytest.approx(float(sig.distance_sd.iloc[-1]))
    assert sig.signal_state in ('LONG_ENTRY', 'SHORT_ENTRY', 'EXIT', 'NONE')


def test_missing_instrument_raises(cross_asset_prices):
    b = Basket.pair('PLATINUM', 'NATGAS')  # NATGAS not in prices
    with pytest.raises(ValueError, match='NATGAS'):
        SpreadSignal(basket=b, prices=cross_asset_prices)


def test_tvr_is_float(cross_asset_prices):
    b = Basket.pair('PLATINUM', 'AUDUSD')
    sig = SpreadSignal(basket=b, prices=cross_asset_prices)
    assert isinstance(sig.tvr, float)
    assert sig.tvr >= 0.0


def test_spread_ret_length(cross_asset_prices):
    b = Basket.pair('PLATINUM', 'AUDUSD')
    sig = SpreadSignal(basket=b, prices=cross_asset_prices)
    # spread_ret should be shorter than input (vol window drops initial rows)
    assert len(sig.spread_ret) < len(cross_asset_prices)
    assert len(sig.spread_ret) == len(sig.distance_sd)


def test_chart_data_keys(cross_asset_prices):
    b = Basket.pair('PLATINUM', 'AUDUSD')
    sig = SpreadSignal(basket=b, prices=cross_asset_prices)
    chart = sig.chart_data()
    for key in ('dates', 'cum_spread', 'distance_sd', 'current_sd', 'signal_state', 'tvr'):
        assert key in chart


def test_signal_history_length(cross_asset_prices):
    b = Basket.pair('PLATINUM', 'AUDUSD')
    sig = SpreadSignal(basket=b, prices=cross_asset_prices)
    hist = sig.signal_history(n_days=50)
    assert len(hist) <= 50
