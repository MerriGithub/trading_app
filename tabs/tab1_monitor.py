from __future__ import annotations

import streamlit as st

from tabs.shared import (
    portfolio, _cached_latest_prices, _get_signal_metrics,
    ALL_DISPLAY, _signal_state_badge,
)


def render() -> None:
    st.header("Monitor")
    st.caption("Live signal status and P&L for every open position.")

    if not portfolio.open_positions:
        st.info("No open positions. Open a paper trade in the Journal tab to start monitoring.")
        return

    for pos in portfolio.open_positions:
        with st.container(border=True):
            cp     = _cached_latest_prices(tuple(pos.basket.all_instruments))
            sm     = _get_signal_metrics(pos)
            sig_ok = sm is not None and 'error' not in sm

            hc1, hc2, hc3 = st.columns([3, 2, 2])
            hc1.markdown(f"### {pos.name}")
            hc1.caption(
                f"{' + '.join(pos.basket.long_legs)} vs {' + '.join(pos.basket.short_legs)}  |  "
                f"{pos.direction.replace('_', ' ').title()}"
            )
            hc2.metric("Opened", str(pos.entry_date))
            hc3.metric("Days held", pos.days_held)

            if not sig_ok:
                st.warning(
                    f"Could not build signal for {pos.name}: "
                    f"{sm.get('error') if sm else 'unknown'}"
                )

            if sig_ok:
                sd = sm['current_sd']
                sd_clamped = max(-3.0, min(3.0, sd))
                progress_val = (sd_clamped + 3.0) / 6.0
                sc1, sc2, sc3 = st.columns([2, 2, 2])
                sc1.metric("Signal", f"{sd:+.2f} SD")
                sc1.progress(progress_val)
                sc2.metric("Velocity", f"{sm['velocity']:+.4f}")
                sc2.metric("TVR", f"{sm['tvr']:.3f}")
                sc3.markdown(f"**{_signal_state_badge(sm['signal_state'])}**")
                sc3.caption(f"State: `{sm['signal_state']}`")

            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Unrealised", f"£{pos.live_pnl(cp):+,.0f}")
            pc2.metric("Financing", f"£{-pos.financing_cost_to_date():,.0f}")
            pc3.metric("Net P&L", f"£{pos.net_pnl(cp):+,.0f}")

            _bc1, _bc2 = st.columns(2)
            if _bc1.button("Open in Pair Analysis →", key=f"view_{pos.id}"):
                st.session_state['pa_pair'] = f"{pos.name} ({pos.id})"
                st.session_state['wf_pair'] = {
                    'long':   list(pos.basket.long_legs),
                    'short':  list(pos.basket.short_legs),
                    'source': 'tab1',
                }
                st.session_state['sidebar_nav'] = "📈 Pair Analysis"
                st.rerun()
            if _bc2.button("Validate in Walk-Forward →", key=f"wf_{pos.id}"):
                st.session_state['wf_pair'] = {
                    'long':   list(pos.basket.long_legs),
                    'short':  list(pos.basket.short_legs),
                    'source': 'tab1',
                }
                st.session_state['sidebar_nav'] = "🔀 Walk-Forward"
                st.rerun()
