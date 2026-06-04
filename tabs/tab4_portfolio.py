"""
Tab 4 — Portfolio Dashboard
=============================
Displays the full position summary (open + closed) with live P&L, financing
costs, and net returns.  Also shows cross-pair correlation and capital-at-risk.
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

from tabs.shared import (
    portfolio, registry, _cached_latest_prices, _asset_class_of,
)


def render() -> None:
    st.header("Portfolio")
    st.caption("Aggregate exposure, correlation and P&L across all open positions.")

    open_positions = portfolio.open_positions
    all_open_instruments = list({
        inst for pos in open_positions for inst in pos.basket.all_instruments
    })
    try:
        current_prices = _cached_latest_prices(tuple(all_open_instruments))
    except Exception:
        # Price fetch failed (network error or empty cache); use empty dict so
        # P&L columns show 0 rather than crashing the whole portfolio tab.
        current_prices = {}

    unrealised = portfolio.total_unrealised_pnl(current_prices)
    realised   = portfolio.total_realised_pnl()
    fin_drag   = sum(p.financing_cost_to_date() for p in open_positions)
    net_total  = unrealised + realised - fin_drag

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unrealised P&L",                       f"£{unrealised:+,.0f}")
    m2.metric("Realised P&L",                         f"£{realised:+,.0f}")
    m3.metric("Financing drag",                       f"£{-fin_drag:,.0f}")
    m4.metric("Net P&L (open + closed − financing)",  f"£{net_total:+,.0f}")

    st.markdown("---")
    st.subheader("Open positions")
    df = portfolio.position_summary(current_prices)
    if df.empty:
        st.caption("No open positions.")
    else:
        df = df.sort_values('days_held', ascending=False).copy()

        def _net_colour(v):
            try:
                return 'color: #1c8a4f' if float(v) >= 0 else 'color: #c0392b'
            except Exception:
                # Non-numeric cell (e.g. NaN or string); return no style.
                return ''

        def _days_colour(v):
            try:
                v = float(v)
                if v > 180:
                    return 'background-color: #fde4e4'
                if v > 90:
                    return 'background-color: #fff5d6'
            except Exception:
                # Non-numeric cell; return no style.
                pass
            return ''

        styled = (df.style
                  .map(_net_colour, subset=['net_pnl', 'live_pnl'])
                  .map(_days_colour, subset=['days_held'])
                  .format({
                      'live_pnl':       '£{:+,.0f}',
                      'financing_cost': '£{:,.0f}',
                      'net_pnl':        '£{:+,.0f}',
                      'pct_open':       '{:.0%}',
                  }))
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Cross-pair correlation")
    try:
        corr = portfolio.cross_pair_correlation(registry)
    except Exception as e:
        corr = pd.DataFrame()
        st.warning(f"Correlation unavailable: {e}")

    if len(corr) > 1:
        st.caption("Pairs with correlation > 0.5 may not be fully independent.")
        st.dataframe(corr.round(2), use_container_width=True)
    else:
        st.caption("Need at least 2 open positions for correlation.")

    st.markdown("---")
    st.subheader("Capital at risk")
    car = portfolio.capital_at_risk(current_prices)
    st.metric("Capital at risk (2 SD, 10-day)", f"£{car:,.0f}",
              help="Parametric VaR: 2 SD × √10 loss estimate per open position.")

    st.markdown("---")
    st.subheader("P&L by asset class")
    if open_positions:
        by_class: dict[str, float] = {}
        for pos in open_positions:
            cls = _asset_class_of(pos.basket.long_legs[0]) if pos.basket.long_legs else 'unknown'
            by_class[cls] = by_class.get(cls, 0.0) + pos.live_pnl(current_prices)
        bc_df = pd.DataFrame(
            {'Asset class': list(by_class.keys()),
             'Unrealised P&L': list(by_class.values())}
        ).set_index('Asset class')
        st.bar_chart(bc_df)
    else:
        st.caption("No open positions to aggregate.")
