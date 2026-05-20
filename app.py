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

st.set_page_config(page_title="Spread Trading Platform", page_icon="📈", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────
if 'pending_close' not in st.session_state:
    st.session_state['pending_close'] = None
if 'leg_count' not in st.session_state:
    st.session_state['leg_count'] = 1

# ── Alert pre-computation (once per page load) ────────────────────────────────
if 'signal_alerts' not in st.session_state:
    st.session_state['signal_alerts'] = _check_signal_alerts(portfolio, registry)

_n_alerts   = len(st.session_state['signal_alerts'])
_live_label = f"⏱ Live 🔔 {_n_alerts}" if _n_alerts > 0 else "⏱ Live"

# ── Tabs ──────────────────────────────────────────────────────────────────────
_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _t10, _t11 = st.tabs([
    "📊 Monitor", "📈 Pair Analysis", "🧮 Stake Calc", "🗂 Portfolio",
    "🔍 Search", _live_label, "📓 Journal", "🔬 Backtest", "🔄 Walk-Forward",
    "🔭 Scenario", "📐 Walk-Forward Analysis",
])

with _t1:  tab1.render()
with _t2:  tab2.render()
with _t3:  tab3.render()
with _t4:  tab4.render()
with _t5:  tab5.render()
with _t6:  tab6.render()
with _t7:  tab7.render()
with _t8:  tab8.render()
with _t9:  tab9.render()
with _t10: tab10.render()
with _t11: tab11.render()
