"""Tests for data_refresh — staleness helpers and FILE_CONFIG structure."""
from __future__ import annotations

import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from data_refresh import (
    FILE_CONFIG,
    _last_trading_date,
    get_file_last_date,
    is_stale,
    staleness_summary,
)


def test_last_trading_date_weekday_after_18():
    # Wednesday 19:00 UTC — market closed but final prices available
    now = datetime.datetime(2026, 5, 20, 19, 0, tzinfo=datetime.timezone.utc)
    assert _last_trading_date(now) == datetime.date(2026, 5, 20)


def test_last_trading_date_weekday_before_18():
    # Wednesday 09:00 UTC — today's close not yet available, use Tuesday
    now = datetime.datetime(2026, 5, 20, 9, 0, tzinfo=datetime.timezone.utc)
    assert _last_trading_date(now) == datetime.date(2026, 5, 19)


def test_last_trading_date_rolls_back_weekend():
    # Sunday 19:00 UTC — roll back through Saturday → return Friday
    now = datetime.datetime(2026, 5, 17, 19, 0, tzinfo=datetime.timezone.utc)
    assert _last_trading_date(now) == datetime.date(2026, 5, 15)


def test_get_file_last_date_reads_last_row(tmp_path):
    df = pd.DataFrame(
        {'UKX': [100.0, 101.0, 102.0]},
        index=pd.to_datetime(['2026-05-12', '2026-05-13', '2026-05-14']),
    )
    df.index.name = 'Date'
    df.to_csv(tmp_path / 'prices.csv')
    with patch('data_refresh._CACHE_DIR', str(tmp_path)):
        assert get_file_last_date('prices.csv') == datetime.date(2026, 5, 14)


def test_is_stale_when_fresh(tmp_path):
    target = _last_trading_date()
    df = pd.DataFrame({'UKX': [100.0]}, index=pd.to_datetime([target]))
    df.index.name = 'Date'
    df.to_csv(tmp_path / 'prices.csv')
    with patch('data_refresh._CACHE_DIR', str(tmp_path)):
        assert is_stale('prices.csv') is False


def test_is_stale_when_old(tmp_path):
    df = pd.DataFrame({'UKX': [100.0]}, index=pd.to_datetime(['2020-01-01']))
    df.index.name = 'Date'
    df.to_csv(tmp_path / 'prices.csv')
    with patch('data_refresh._CACHE_DIR', str(tmp_path)):
        assert is_stale('prices.csv') is True


def test_staleness_summary_structure(tmp_path):
    with patch('data_refresh._CACHE_DIR', str(tmp_path)):
        summary = staleness_summary()
    assert set(summary.keys()) == set(FILE_CONFIG.keys())
    for v in summary.values():
        assert v is None  # no files in tmp_path


def test_file_config_completeness():
    assert 'prices.csv' in FILE_CONFIG
    assert 'fx_prices.csv' in FILE_CONFIG
    assert 'commodity_prices.csv' in FILE_CONFIG
    for filename, tickers in FILE_CONFIG.items():
        assert len(tickers) > 0, f"{filename} has no tickers"
