from __future__ import annotations

import streamlit as st

from core.basket import Basket
from core.signal import SpreadSignal
from data_watchlist import load_monitor_candidates, remove_monitor_candidate
from tabs.shared import (
    portfolio, registry, _cached_latest_prices, _get_signal_metrics,
    ALL_DISPLAY, _signal_state_badge,
)


def _candidate_signal(candidate: dict) -> dict | None:
    try:
        prices = registry.get_daily_prices([candidate['long'], candidate['short']])
        if prices is None or prices.empty:
            return None
        basket = Basket.pair(candidate['long'], candidate['short'])
        sig = SpreadSignal(
            basket, prices,
            vol_window=candidate['vol_window'],
            xing_sd=candidate['entry_sd'],
            exit_sd=candidate['exit_sd'],
        )
        return {'current_sd': sig.current_sd, 'signal_state': sig.signal_state}
    except Exception:
        return None


def _render_candidate_card(candidate: dict) -> None:
    long_display  = ALL_DISPLAY.get(candidate['long'],  candidate['long'])
    short_display = ALL_DISPLAY.get(candidate['short'], candidate['short'])
    with st.container(border=True):
        hc1, hc2 = st.columns([3, 1])
        hc1.markdown(f"### {long_display} / {short_display}")
        hc1.caption(
            f"{candidate.get('asset_class_long', '?')} × {candidate.get('asset_class_short', '?')}"
        )
        _dir_badge = "CT" if "CT" in candidate.get('direction', '') else "WT"
        _sm = candidate.get('scoring_mode')
        hc2.markdown(f"**{_dir_badge}**" + (f" · {_sm}" if _sm else ""))
        hc2.caption(f"E{candidate['entry_sd']} X{candidate['exit_sd']} V{candidate['vol_window']}")

        sm = _candidate_signal(candidate)
        if sm is None:
            st.warning("Could not compute signal — check price data.")
        else:
            sc1, sc2 = st.columns([2, 2])
            sd = sm['current_sd']
            sc1.metric("Current SD", f"{sd:+.2f}")
            sc1.progress((max(-3.0, min(3.0, sd)) + 3.0) / 6.0)
            sc2.markdown(f"**{_signal_state_badge(sm['signal_state'])}**")
            sc2.caption(f"State: `{sm['signal_state']}`")
            if sm['signal_state'] in ('LONG_ENTRY', 'SHORT_ENTRY'):
                st.warning("⚡ Entry signal active — load to Stake Calc to size and book.")

        bc1, bc2 = st.columns(2)
        if bc1.button("🧮 Load to Stake Calc", key=f"mon_to_sc_{candidate['id']}"):
            st.session_state['sc_long_pending']         = [candidate['long']]
            st.session_state['sc_short_pending']        = [candidate['short']]
            st.session_state['tab3_direction']          = candidate['direction']
            st.session_state['tab3_vol_window_pending'] = candidate['vol_window']
            st.session_state['tab3_from_monitor']       = candidate['id']
            st.session_state['sidebar_nav_pending']     = "🧮 Stake Calc"
            st.rerun()
        if bc2.button("✖ Remove", key=f"mon_remove_{candidate['id']}"):
            remove_monitor_candidate(candidate['id'])
            st.rerun()


def render() -> None:
    st.header("Monitor")
    st.caption("Live signal status and P&L for every open position.")

    # ── Section A — Live positions ────────────────────────────────────────────
    st.subheader("Live positions")
    st.caption("Open trades with live P&L and signal state.")
    if not portfolio.open_positions:
        st.info("No open positions.")
    else:
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
                    st.session_state['sidebar_nav_pending'] = "📈 Pair Analysis"
                    st.rerun()
                if _bc2.button("Validate in Walk-Forward →", key=f"wf_{pos.id}"):
                    st.session_state['wf_pair'] = {
                        'long':   list(pos.basket.long_legs),
                        'short':  list(pos.basket.short_legs),
                        'source': 'tab1',
                    }
                    st.session_state['sidebar_nav_pending'] = "🔀 Walk-Forward"
                    st.rerun()

    # ── Section B — Candidates under watch ───────────────────────────────────
    st.divider()
    st.subheader("Candidates under watch")
    st.caption(
        "Watchlist pairs being monitored for entry. "
        "SD hits the threshold → load to Stake Calc."
    )
    candidates = load_monitor_candidates()
    if not candidates:
        st.info("No candidates under watch. Use '👁 Monitor' in Stake Calc to add pairs here.")
    else:
        for candidate in candidates:
            _render_candidate_card(candidate)
