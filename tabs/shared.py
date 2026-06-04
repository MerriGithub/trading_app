"""tabs/shared.py — Singletons, cached loaders, and utilities shared across all tabs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.basket import Basket
from core.signal import SpreadSignal
from core.portfolio import Portfolio
from core.data_registry import DataRegistry
from account import load_account, get_margin
from asset_configs import (
    ASSET_CLASSES, ASSET_CLASS_OPTIONS, FI_EXCLUDE,
    get_display_name, get_tradeable_instruments,
    get_spread_cost_lookup,
)

# ── Shared paths ──────────────────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent.parent / 'cache'
_DATA_DIR  = Path(__file__).parent.parent / 'data'
_POSITIONS = _DATA_DIR / 'positions.json'
_ACCOUNT   = _DATA_DIR / 'account.json'


# ── Cached singletons ─────────────────────────────────────────────────────────
@st.cache_resource
def _registry() -> DataRegistry:
    return DataRegistry(_CACHE_DIR)


@st.cache_resource
def _portfolio() -> Portfolio:
    return Portfolio(_POSITIONS, _ACCOUNT)


@st.cache_resource
def _load_account() -> dict:
    return load_account()


registry  = _registry()
portfolio = _portfolio()
account   = _load_account()


# ── Multi-asset instrument lookup ─────────────────────────────────────────────
ALL_INSTRUMENTS: list[str] = []
ALL_DISPLAY: dict[str, str] = {}
for _key, _cfg in ASSET_CLASSES.items():
    for _code in get_tradeable_instruments(_key):
        if _code not in FI_EXCLUDE and _code not in ALL_DISPLAY:
            ALL_INSTRUMENTS.append(_code)
            ALL_DISPLAY[_code] = get_display_name(_key, _code)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _tbl(df: pd.DataFrame, show_index: bool = True, height: int | None = None) -> None:
    """Render a DataFrame as a Plotly Table — zero pyarrow dependency."""
    idx_vals  = [str(v) for v in df.index]
    col_names = ([str(df.index.name or '')] if show_index else []) + [str(c) for c in df.columns]
    col_vals  = ([idx_vals] if show_index else []) + [df[c].astype(str).tolist() for c in df.columns]
    n = len(df)
    row_fill = ['#f0f4f8' if i % 2 == 0 else 'white' for i in range(n)]
    fig = go.Figure(go.Table(
        header=dict(values=col_names, fill_color='#2c6fad',
                    font=dict(color='white', size=13), align='left'),
        cells=dict(values=col_vals, fill_color=[row_fill] * len(col_vals),
                   align='left', font=dict(size=12)),
    ))
    fig.update_layout(height=height or min(80 + n * 28, 550),
                      margin=dict(l=0, r=0, t=4, b=4))
    st.plotly_chart(fig, use_container_width=True)


def _asset_class_of(code: str) -> str:
    for key, cfg in ASSET_CLASSES.items():
        if code in cfg.get('instruments', {}):
            return key
    return 'unknown'


def _signal_state_badge(state: str) -> str:
    return {
        'EXIT':         '🟢 Near target',
        'LONG_ENTRY':   '🟡 At entry level',
        'SHORT_ENTRY':  '🟡 At entry level',
        'NONE':         '⚪ Holding',
    }.get(state, state)


# ── Signal alert helper ───────────────────────────────────────────────────────
def _check_signal_alerts(portfolio: Portfolio, registry: DataRegistry) -> list[dict]:
    alerts = []
    for pos in portfolio.open_positions:
        try:
            prices_hist = registry.get_daily_prices(pos.basket.all_instruments)
            signal = SpreadSignal(basket=pos.basket, prices=prices_hist)
            state  = signal.signal_state
            if state != 'NONE':
                alerts.append({
                    'position':  pos.name,
                    'pair':      ' / '.join(pos.basket.all_instruments),
                    'state':     state,
                    'sd':        signal.current_sd,
                    'days_held': pos.days_held,
                })
        except Exception:
            # Signal computation can fail if price data is stale or unavailable.
            # Skip this position's alert rather than crashing the page load.
            pass
    return alerts


# ── Cached wrappers ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _cached_daily_prices(instruments: tuple[str, ...]) -> pd.DataFrame:
    return registry.get_daily_prices(list(instruments))


@st.cache_data(ttl=300, show_spinner=False)
def _cached_latest_prices(instruments: tuple[str, ...]) -> dict[str, float]:
    return registry.get_latest_prices(list(instruments))


# ── Signal metrics ────────────────────────────────────────────────────────────
def _build_signal_metrics(pos) -> dict:
    ph  = _cached_daily_prices(tuple(pos.basket.all_instruments))
    sig = SpreadSignal(basket=pos.basket, prices=ph)
    return {
        'current_sd':   sig.current_sd,
        'signal_state': sig.signal_state,
        'velocity':     float(sig.velocity.iloc[-1]) if len(sig.velocity) else 0.0,
        'tvr':          sig.tvr,
        'distance_sd':  sig.distance_sd,
        'spread_ret':   sig.spread_ret,
        'cum_spread':   sig.cum_spread,
    }


def _get_signal_metrics(pos) -> dict | None:
    cache = st.session_state.setdefault('signal_metrics', {})
    if pos.id not in cache:
        try:
            cache[pos.id] = _build_signal_metrics(pos)
        except Exception as e:
            cache[pos.id] = {'error': str(e)}
    return cache[pos.id]
