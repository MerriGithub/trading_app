from __future__ import annotations

import datetime as _dt
import time as _time

import pandas as pd
import streamlit as st

from tabs.shared import (
    portfolio, registry,
    _cached_latest_prices, _get_signal_metrics, _check_signal_alerts,
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
            col_signal, col_prices, col_intraday = st.columns([2, 2, 3])

            with col_signal:
                st.markdown("**Signal**")
                _sm = _get_signal_metrics(_pos)
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

    st.divider()
    _render_market_overview()


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
