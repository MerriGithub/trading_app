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
        'current_sd':     sig.current_sd,
        'signal_state':   sig.signal_state,
        'velocity':       float(sig.velocity.iloc[-1]) if len(sig.velocity) else 0.0,
        'velocity_series': sig.velocity,
        'tvr':            sig.tvr,
        'distance_sd':    sig.distance_sd,
        'spread_ret':     sig.spread_ret,
        'cum_spread':     sig.cum_spread,
    }


def _get_signal_metrics(pos) -> dict | None:
    cache = st.session_state.setdefault('signal_metrics', {})
    if pos.id not in cache:
        try:
            cache[pos.id] = _build_signal_metrics(pos)
        except Exception as e:
            # Signal computation fails on stale/unavailable prices; return error
            # dict rather than crashing the monitor tab.
            cache[pos.id] = {'error': str(e)}
    return cache[pos.id]


# ── Risk overlay constants ────────────────────────────────────────────────────

# Asset class average hold days — Phase 4b confirmed research optimums.
_AVG_HOLD_DAYS: dict[str, float] = {
    'equity':       8.0,    # EXIT_SD=2.0 scalp regime, avg hold 8d
    'commodities':  104.0,  # EXIT_SD=0.5, avg hold 104d
    'fx':           35.0,
    'fixed_income': 35.0,
    'unknown':      30.0,
}

# Hard stop defaults per asset class (SD units).
# Source: research/hard_stop_analysis.py — Phase 5a, run 2026-06-05.
# Equities: 3.5 SD — research optimal (EV_delta −0.09, stop rate 11.6%).
# Commodities: 4.5 SD — practical choice; algorithm recommends 5.0 SD but it
#   fires only 1.3% of trades (operationally irrelevant). 4.5 SD fires 5.5%
#   and keeps p95 winner MAE (3.82 SD) comfortably inside the buffer.
# FX / fixed income: not yet researched; using conservative 4.0 SD placeholder.
HARD_STOP_SD: dict[str, float] = {
    'equity':       3.5,
    'commodities':  4.5,
    'fx':           4.0,    # not yet researched
    'fixed_income': 4.0,    # not yet researched
    'unknown':      4.0,
}


def _compute_risk_metrics(
    pos,
    signal_metrics: dict,
    stop_sd_override: float | None = None,
) -> dict:
    """Compute risk overlay metrics for a single open position.

    Uses the distance_sd series from signal_metrics (already cached) to derive
    all metrics without additional data fetches.

    Args:
        pos: Open Position object. Uses pos.entry_date, pos.direction,
            pos.basket.all_instruments, pos.days_held, pos.target_exposure.
        signal_metrics: Dict returned by _get_signal_metrics(pos). Must contain
            'distance_sd' (pd.Series), 'current_sd' (float), 'velocity' (float).
            If 'error' key present, returns a degraded metrics dict with data_ok=False.
        stop_sd_override: If provided, overrides the HARD_STOP_SD default for
            this position's asset class.

    Returns:
        Dict with keys: asset_class, avg_hold_days, stop_sd, sd_at_entry,
        sd_change, sd_5d_slope, days_at_extreme, mae_sd, velocity_3d_avg,
        dist_to_stop_sd, dist_to_stop_pct, days_held_ratio, rag, rag_reasons,
        data_ok.

    Raises:
        TypeError: If signal_metrics is not a dict.
        ValueError: If pos.direction is not a valid spread direction.
    """
    if not isinstance(signal_metrics, dict):
        raise TypeError(
            f"signal_metrics must be a dict; got {type(signal_metrics).__name__}"
        )

    _NAN = float('nan')

    # Degraded return when signal data is unavailable
    if 'error' in signal_metrics:
        return {
            'asset_class': 'unknown', 'avg_hold_days': _NAN, 'stop_sd': _NAN,
            'sd_at_entry': _NAN, 'sd_change': _NAN, 'sd_5d_slope': _NAN,
            'days_at_extreme': 0, 'mae_sd': _NAN, 'velocity_3d_avg': _NAN,
            'dist_to_stop_sd': _NAN, 'dist_to_stop_pct': _NAN,
            'days_held_ratio': _NAN, 'rag': 'amber',
            'rag_reasons': ['amber:signal data unavailable'], 'data_ok': False,
        }

    if pos.direction not in ('long_spread', 'short_spread'):
        raise ValueError(
            f"pos.direction must be 'long_spread' or 'short_spread'; got {pos.direction!r}"
        )

    ac         = _asset_class_of(pos.basket.all_instruments[0])
    avg_hold   = _AVG_HOLD_DAYS.get(ac, _AVG_HOLD_DAYS['unknown'])
    stop_sd    = (
        stop_sd_override
        if stop_sd_override is not None
        else HARD_STOP_SD.get(ac, HARD_STOP_SD['unknown'])
    )
    side       = 1 if pos.direction == 'long_spread' else -1
    current_sd = float(signal_metrics.get('current_sd', _NAN))

    dist_sd_series = signal_metrics.get('distance_sd', pd.Series(dtype=float))
    entry_ts       = pd.Timestamp(pos.entry_date)

    # SD at entry: closest available date at or after entry (prices may lag 1d)
    sd_at_entry = _NAN
    if not dist_sd_series.empty:
        mask = dist_sd_series.index >= entry_ts
        if mask.any():
            sd_at_entry = float(dist_sd_series[mask].iloc[0])

    sd_change = (
        current_sd - sd_at_entry
        if not (np.isnan(current_sd) or np.isnan(sd_at_entry))
        else _NAN
    )

    since_entry = (
        dist_sd_series[dist_sd_series.index >= entry_ts]
        if not dist_sd_series.empty
        else pd.Series(dtype=float)
    )

    # MAE: worst adverse excursion since entry (always >= 0)
    if not since_entry.empty:
        mae_sd = abs(float(since_entry.min())) if side == 1 else abs(float(since_entry.max()))
    else:
        mae_sd = abs(current_sd) if not np.isnan(current_sd) else _NAN

    # 5d slope normalised by side: positive = reverting toward zero
    sd_5d_slope = 0.0
    if len(since_entry) >= 2:
        tail = since_entry.iloc[-min(5, len(since_entry)):]
        try:
            slope = float(
                np.polyfit(np.arange(len(tail)), tail.values.astype(float), 1)[0]
            )
            sd_5d_slope = slope * side
        except ValueError:
            logger.warning(
                "np.polyfit failed for position %s; defaulting slope to 0.0", pos.id
            )

    # Days at extreme: consecutive days (from latest) where abs(d) > abs(sd_at_entry)
    days_at_extreme = 0
    if not since_entry.empty and not np.isnan(sd_at_entry):
        for _d in reversed(since_entry.values):
            if abs(_d) > abs(sd_at_entry):
                days_at_extreme += 1
            else:
                break

    # Velocity 3d avg from full series added to signal_metrics by _build_signal_metrics
    vel_series = signal_metrics.get('velocity_series', pd.Series(dtype=float))
    velocity_3d_avg = (
        float(vel_series.iloc[-3:].mean())
        if len(vel_series) >= 3
        else float(signal_metrics.get('velocity', 0.0))
    )

    # Buffer remaining to hard stop
    if not np.isnan(mae_sd):
        dist_to_stop_sd  = max(0.0, stop_sd - mae_sd)
        dist_to_stop_pct = dist_to_stop_sd / stop_sd if stop_sd > 0 else 0.0
    else:
        dist_to_stop_sd  = _NAN
        dist_to_stop_pct = _NAN

    days_held_ratio = pos.days_held / avg_hold if avg_hold > 0 else _NAN

    # RAG classification
    reasons: list[str] = []
    if days_at_extreme > 5:
        reasons.append('red:5+ consecutive days extending')
    if not np.isnan(dist_to_stop_sd) and dist_to_stop_sd < 0.5:
        reasons.append('red:within 0.5 SD of hard stop')
    if not np.isnan(days_held_ratio) and days_held_ratio > 2.0:
        reasons.append('red:held 2× expected avg hold')

    if not any(r.startswith('red:') for r in reasons):
        if days_at_extreme > 3:
            reasons.append('amber:3+ consecutive days extending')
        if not np.isnan(dist_to_stop_sd) and dist_to_stop_sd < 1.0:
            reasons.append('amber:within 1.0 SD of hard stop')
        if not np.isnan(days_held_ratio) and days_held_ratio > 1.5:
            reasons.append('amber:held 1.5× expected avg hold')
        if sd_5d_slope < 0 and days_at_extreme >= 2:
            reasons.append('amber:trend extending for 2+ days')

    rag = (
        'red' if any(r.startswith('red:') for r in reasons)
        else ('amber' if reasons else 'green')
    )

    return {
        'asset_class':      ac,
        'avg_hold_days':    avg_hold,
        'stop_sd':          stop_sd,
        'sd_at_entry':      sd_at_entry,
        'sd_change':        sd_change,
        'sd_5d_slope':      sd_5d_slope,
        'days_at_extreme':  days_at_extreme,
        'mae_sd':           mae_sd,
        'velocity_3d_avg':  velocity_3d_avg,
        'dist_to_stop_sd':  dist_to_stop_sd,
        'dist_to_stop_pct': dist_to_stop_pct,
        'days_held_ratio':  days_held_ratio,
        'rag':              rag,
        'rag_reasons':      reasons,
        'data_ok':          True,
    }
