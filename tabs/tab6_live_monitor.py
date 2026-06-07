"""
Tab 6 — Live Monitor
=====================
Intraday price monitor showing the latest signal z-scores for all active
instruments.  Refreshes on demand or auto-refreshes at a configurable interval.

Note: This is Tab 6 (live_monitor), not the Trade Journal.
The Trade Journal is Tab 7 (tab7_journal.py) — see register item A in CLAUDE.md.
"""
from __future__ import annotations

import logging
import datetime as _dt
import time as _time

logger = logging.getLogger(__name__)

import pandas as pd
import streamlit as st

from tabs.shared import (
    portfolio, registry,
    _cached_latest_prices, _get_signal_metrics, _check_signal_alerts,
    _compute_risk_metrics,
    ALL_DISPLAY, _CACHE_DIR,
)


def render() -> None:
    st.header("Live Monitor")

    col_refresh, col_last = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh now", key="tab6_refresh"):
            open_insts = list({
                inst
                for pos in portfolio.open_positions
                for inst in pos.basket.all_instruments
            })
            if open_insts:
                with st.spinner("Refreshing prices…"):
                    registry.refresh(open_insts)
                st.session_state['signal_alerts'] = _check_signal_alerts(portfolio, registry)
                st.session_state.pop('tab6_intraday', None)
                st.session_state.pop('signal_metrics', None)
                from tabs.shared import _cached_daily_prices, _cached_latest_prices as _clp
                _cached_daily_prices.clear()
                _clp.clear()
                st.success("Prices updated.")

    with col_last:
        _csv_path = _CACHE_DIR / 'prices.csv'
        if _csv_path.exists():
            _mtime = _dt.datetime.fromtimestamp(_csv_path.stat().st_mtime)
            st.caption(f"Cache last modified: {_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    _auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=False,
                                key="tab6_auto_refresh")
    if _auto_refresh:
        if 'tab6_last_refresh' not in st.session_state:
            st.session_state['tab6_last_refresh'] = _time.time()
        _elapsed = _time.time() - st.session_state['tab6_last_refresh']
        st.caption(f"Auto-refresh ON — next in {max(0, 300 - int(_elapsed))}s")
        if _elapsed > 300:
            st.session_state['tab6_last_refresh'] = _time.time()
            _all_open = [
                inst
                for pos in portfolio.open_positions
                for inst in pos.basket.all_instruments
            ]
            registry.refresh(_all_open)
            st.session_state['signal_alerts'] = _check_signal_alerts(portfolio, registry)
            st.rerun()

    st.divider()

    _alerts = st.session_state.get('signal_alerts', [])
    for alert in _alerts:
        if alert['state'] == 'EXIT':
            st.success(
                f"🟢 **EXIT SIGNAL** — {alert['position']} ({alert['pair']}) "
                f"is at {alert['sd']:+.2f} SD after {alert['days_held']} days. "
                f"Consider closing via the Journal tab."
            )
        else:
            st.warning(
                f"🟡 **SIGNAL ACTIVE** — {alert['position']} ({alert['pair']}) "
                f"is at {alert['sd']:+.2f} SD."
            )

    if not portfolio.open_positions:
        st.info("No open positions. Open a trade via the Journal tab.")
        _render_market_overview()
        return

    st.subheader("Open Positions — Live")

    if 'tab6_intraday' not in st.session_state:
        _intraday_cache: dict[tuple, object] = {}
        for _p in portfolio.open_positions:
            _key = tuple(sorted(_p.basket.all_instruments))
            _intraday_cache[_key] = registry.get_intraday(list(_key), interval='5m')
        st.session_state['tab6_intraday'] = _intraday_cache

    for _pos in portfolio.open_positions:
        _insts        = _pos.basket.all_instruments
        _daily_prices = _cached_latest_prices(tuple(_insts))
        _intraday_key = tuple(sorted(_insts))
        _intraday_df  = st.session_state['tab6_intraday'].get(_intraday_key)

        with st.expander(
            f"**{_pos.name}** — {' × '.join(ALL_DISPLAY.get(i, i) for i in _insts)}",
            expanded=True,
        ):
            # Fetch once; reused by col_signal display and risk overlay below.
            _sm = _get_signal_metrics(_pos)
            col_signal, col_prices, col_intraday = st.columns([2, 2, 3])

            with col_signal:
                st.markdown("**Signal**")
                if _sm and 'error' not in _sm:
                    _sd    = _sm['current_sd']
                    _state = _sm['signal_state']
                    st.metric("SD distance", f"{_sd:+.3f}")
                    st.caption({
                        'EXIT':        '🟢 Near exit target',
                        'LONG_ENTRY':  '🟡 At entry level',
                        'SHORT_ENTRY': '🟡 At entry level',
                        'NONE':        '⚪ Holding',
                    }.get(_state, _state))
                    _norm = min(max((_sd + 3) / 6, 0.0), 1.0)
                    st.progress(_norm)
                else:
                    st.caption(f"Signal unavailable: {_sm.get('error') if _sm else 'unknown'}")

            with col_prices:
                st.markdown("**Latest Prices (EOD)**")
                for _inst in _insts:
                    _price = _daily_prices.get(_inst)
                    _entry = _pos.entry_prices.get(_inst)
                    if _price and _entry:
                        _pct  = (_price - _entry) / _entry * 100
                        _side = "Long" if _inst in _pos.basket.long_legs else "Short"
                        st.metric(
                            f"{ALL_DISPLAY.get(_inst, _inst)} ({_side})",
                            f"{_price:,.4f}",
                            delta=f"{_pct:+.2f}% vs entry",
                        )

            with col_intraday:
                st.markdown("**Intraday**")
                if _intraday_df is not None and not _intraday_df.empty:
                    for _inst in _insts:
                        if _inst in _intraday_df.columns:
                            st.caption(ALL_DISPLAY.get(_inst, _inst))
                            _chart = _intraday_df[[_inst]].dropna()
                            if not _chart.empty:
                                st.line_chart(_chart, height=120)
                else:
                    st.caption("Intraday data not available.")
                    st.caption("Showing EOD monitoring only.")

            _pnl     = _pos.live_pnl(_daily_prices)
            _fin     = _pos.financing_cost_to_date()
            _net_pnl = _pos.net_pnl(_daily_prices)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Days held",      _pos.days_held)
            c2.metric("Unrealised P&L", f"£{_pnl:+,.0f}")
            c3.metric("Financing drag", f"-£{_fin:,.0f}")
            c4.metric("Net P&L",        f"£{_net_pnl:+,.0f}",
                      delta="▲" if _net_pnl > 0 else "▼",
                      delta_color="normal")

            # ── Risk overlay ──────────────────────────────────────────────
            if _sm and 'error' not in _sm:
                _risk = _compute_risk_metrics(_pos, _sm)
            else:
                _risk = {
                    'data_ok': False, 'rag': 'amber',
                    'rag_reasons': ['amber:signal data unavailable'],
                }
            _rag_emoji = {'green': '🟢', 'amber': '🟡', 'red': '🔴'}[_risk['rag']]
            _rag_label = {'green': 'Risk OK', 'amber': 'Watch', 'red': 'Alert'}[_risk['rag']]
            with st.expander(f"{_rag_emoji} {_rag_label}", expanded=(_risk['rag'] == 'red')):
                if _risk.get('data_ok', False):
                    _render_risk_overlay(_pos, _risk)
                else:
                    st.caption("⚠ Risk metrics unavailable — signal data could not be computed.")

    st.divider()
    _render_market_overview()


def _render_risk_overlay(pos, risk: dict) -> None:
    """Render the risk overlay panel inside a position expander.

    Three sections (Signal Trend / Stop Context / Hold Context) separated by
    dividers. Called only when risk['data_ok'] is True.

    Args:
        pos: Open Position object. Used for pos.direction and pos.days_held.
        risk: Dict returned by _compute_risk_metrics(). Must have data_ok=True.
    """
    # Fix 2 (review): derive side for direction-aware reverting checks so that
    # both sd_change and velocity captions are correct for longs and shorts.
    _side = 1 if pos.direction == 'long_spread' else -1

    # ── Section 1 — Signal Trend ─────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**SD vs Entry**")
        sd_change = risk['sd_change']
        if not pd.isna(sd_change):
            _reverting = sd_change * _side > 0
            st.metric("Change", f"{sd_change:+.3f} SD")
            st.caption("↗ toward zero ✓" if _reverting else "↘ extending ✗")
        else:
            st.caption("N/A")
    with c2:
        st.markdown("**5d Trend**")
        slope = risk['sd_5d_slope']
        st.metric("Slope", f"{slope:+.4f}")
        st.caption("↗ Reverting" if slope > 0 else "↘ Extending")
    with c3:
        st.markdown("**Days Extending**")
        dae = risk['days_at_extreme']
        _dae_label = "🔴" if dae > 5 else ("🟡" if dae > 3 else "🟢")
        st.metric("Days", dae)
        st.caption(f"{_dae_label} consecutive adverse days")

    st.divider()

    # ── Section 2 — Stop Context ─────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Hard Stop Level**")
        st.metric("Stop", f"{risk['stop_sd']:.1f} SD")
        st.caption("(from research defaults)")
    with c2:
        st.markdown("**Distance to Stop**")
        dts = risk['dist_to_stop_sd']
        dtp = risk['dist_to_stop_pct']
        if not pd.isna(dts):
            st.metric("Buffer", f"{dts:.2f} SD")
            st.caption(f"{dtp * 100:.0f}% of stop range remaining")
        else:
            st.caption("N/A")
    with c3:
        st.markdown("**Worst Point (MAE)**")
        mae = risk['mae_sd']
        if not pd.isna(mae):
            st.metric("MAE", f"{mae:.2f} SD")
            st.caption("since entry")
        else:
            st.caption("N/A")

    st.divider()

    # ── Section 3 — Hold Context ─────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Days Held**")
        st.metric("Days", pos.days_held)
    with c2:
        st.markdown("**vs Expected**")
        ratio = risk['days_held_ratio']
        avg   = risk['avg_hold_days']
        if not pd.isna(ratio):
            _hold_label = "🔴" if ratio > 1.5 else ("🟡" if ratio > 1.0 else "🟢")
            st.metric("Progress", f"{pos.days_held}d / {avg:.0f}d avg")
            st.caption(f"{_hold_label} {ratio * 100:.0f}% of avg hold")
        else:
            st.caption("N/A")
    with c3:
        st.markdown("**Velocity (3d avg)**")
        vel = risk['velocity_3d_avg']
        # Direction-aware: reverting means vel * side > 0
        _vel_reverting = vel * _side > 0
        st.metric("Velocity", f"{vel:.4f}")
        st.caption("↗ reverting" if _vel_reverting else "↘ extending")

    if risk.get('rag_reasons'):
        st.caption("Active alerts: " + " | ".join(
            r.split(':', 1)[1] for r in risk['rag_reasons']
        ))


def _render_market_overview() -> None:
    st.subheader("Market Overview")
    _all_open_insts = list({
        inst
        for pos in portfolio.open_positions
        for inst in pos.basket.all_instruments
    })
    if not _all_open_insts:
        st.caption("No open positions — no instruments to monitor.")
        return

    _latest = _cached_latest_prices(tuple(_all_open_insts))
    _overview = [
        {
            'Instrument':   ALL_DISPLAY.get(inst, inst),
            'Code':         inst,
            'Latest Price': _latest.get(inst, 'N/A'),
        }
        for inst in _all_open_insts
    ]
    import pandas as pd
    st.dataframe(
        pd.DataFrame(_overview).set_index('Code'),
        use_container_width=True,
    )
