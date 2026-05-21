"""
app.py — Global Spread Trading Platform
Multi-asset, NvM basket spread trading monitor and journal.
"""
from __future__ import annotations

import streamlit as st

from tabs.shared import portfolio, registry, _check_signal_alerts
import tabs.tab1_monitor      as tab1
import tabs.tab2_pair_analysis as tab2
import tabs.tab3_stake_calc    as tab3
import tabs.tab4_portfolio     as tab4
import tabs.tab5_search        as tab5
import tabs.tab6_live_monitor  as tab6
import tabs.tab7_journal       as tab7
import tabs.tab8_backtest      as tab8
import tabs.tab9_walkforward   as tab9
import tabs.tab10_scenario     as tab10
import tabs.tab11_walkforward  as tab11

st.set_page_config(
    page_title="Spread Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if 'pending_close' not in st.session_state:
    st.session_state['pending_close'] = None
if 'leg_count' not in st.session_state:
    st.session_state['leg_count'] = 1

# ── Alert pre-computation (once per page load) ────────────────────────────────
if 'signal_alerts' not in st.session_state:
    st.session_state['signal_alerts'] = _check_signal_alerts(portfolio, registry)

_n_alerts = len(st.session_state['signal_alerts'])

_TABS = [
    "📊 Monitor",
    "📈 Pair Analysis",
    "🧮 Stake Calc",
    "💼 Portfolio",
    "🔍 Search",
    f"{'🔔 Live (' + str(_n_alerts) + ')' if _n_alerts > 0 else '⏱️ Live'}",
    "📓 Journal",
    "📉 Backtest",
    "✅ Trade Validation",
    "🎯 Scenario",
    "🔀 Walk-Forward",
]

with st.sidebar:
    st.markdown("## 📡 Spread Trading")
    st.markdown("---")
    _active_tab = st.radio(
        "Navigation",
        _TABS,
        label_visibility="collapsed",
        key="sidebar_nav",
    )
    st.markdown("---")
    try:
        from account import load_account as _load_account
        _acct = _load_account()
        _open_n = len(portfolio.open_positions)
        st.caption(f"Capital: £{_acct.get('starting_capital', 0):,.0f}")
        st.caption(f"Open positions: {_open_n}")
    except Exception:
        pass

if _active_tab == _TABS[0]:    # Monitor
    tab1.render()
elif _active_tab == _TABS[1]:  # Pair Analysis
    tab2.render()
elif _active_tab == _TABS[2]:  # Stake Calc
    tab3.render()
elif _active_tab == _TABS[3]:  # Portfolio
    tab4.render()
elif _active_tab == _TABS[4]:  # Search
    tab5.render()
elif _active_tab == _TABS[5]:  # Live
    tab6.render()
elif _active_tab == _TABS[6]:  # Journal
    tab7.render()
elif _active_tab == _TABS[7]:  # Backtest
    tab8.render()
elif _active_tab == _TABS[8]:  # Trade Validation
    tab9.render()
elif _active_tab == _TABS[9]:  # Scenario
    tab10.render()
elif _active_tab == _TABS[10]: # Walk-Forward
    tab11.render()
