"""
app.py — Global Spread Trading Platform
Multi-asset, NvM basket spread trading monitor and journal.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure trading_app/ is always on sys.path regardless of launch method.
# Streamlit adds the script directory normally, but VS Code and other IDEs
# can launch from the workspace root instead, breaking local package imports.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

from logging_config import configure_logging
configure_logging()  # Sets up stderr logging at INFO level; no-op on reruns.

try:
    import data_refresh as _dr
    _HAS_DR = True
except ImportError:
    _HAS_DR = False

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
import tabs.tab12_watchlist    as tab12
import tabs.tab13_daily_scan   as tab13

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

# ── Data auto-refresh (once per session) ─────────────────────────────────────
if _HAS_DR and 'data_refresh_checked' not in st.session_state:
    st.session_state['data_refresh_checked'] = True
    try:
        if _dr.any_file_stale():
            _refresh_results = _dr.refresh_all()
            _refreshed = [r for r in _refresh_results if r['status'] == 'updated']
            if _refreshed:
                st.cache_data.clear()
                st.session_state['data_refresh_banner'] = _refreshed
    except Exception:
        # Broad catch: data_refresh is a background convenience; any failure
        # (network error, missing file, import error) must not crash app startup.
        pass

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
    "🗂️ Watchlist",
    "📅 Daily Scan",
]

if 'sidebar_nav_pending' in st.session_state:
    st.session_state['sidebar_nav'] = st.session_state.pop('sidebar_nav_pending')

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
        # Sidebar summary is decorative; any load error must not block the UI.
        pass

    if _HAS_DR:
        st.markdown("---")
        st.markdown("**Data**")
        try:
            _staleness = _dr.staleness_summary()
            _labels = {
                'prices.csv':           'Equity',
                'fx_prices.csv':        'FX',
                'commodity_prices.csv': 'Commodity',
            }
            for _fname, _last in _staleness.items():
                _lbl = _labels.get(_fname, _fname)
                if _last is None:
                    st.caption(f"🔴 {_lbl}: no data")
                elif _dr.is_stale(_fname):
                    st.caption(f"🟡 {_lbl}: {_last}")
                else:
                    st.caption(f"🟢 {_lbl}: {_last}")
        except Exception:
            # Staleness display is sidebar decoration; failure must not block the UI.
            pass
        if st.button("🔄 Refresh data", key="sidebar_refresh_btn"):
            with st.spinner("Refreshing…"):
                try:
                    _res = _dr.refresh_all()
                    _refreshed = [r for r in _res if r['status'] == 'updated']
                    if _refreshed:
                        st.cache_data.clear()
                        st.session_state['data_refresh_banner'] = _refreshed
                except Exception as _e:
                    st.error(f"Refresh failed: {_e}")
            st.rerun()
        if 'data_refresh_banner' in st.session_state:
            _banner = st.session_state['data_refresh_banner']
            _names = [_labels.get(r['filename'], r['filename']) for r in _banner]
            st.info(f"Auto-refreshed: {', '.join(_names)}")
            if st.button("✕ Dismiss", key="dismiss_refresh_banner"):
                del st.session_state['data_refresh_banner']
                st.rerun()

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
elif _active_tab == _TABS[11]: # Watchlist
    tab12.render()
elif _active_tab == _TABS[12]: # Daily Scan
    tab13.render()
