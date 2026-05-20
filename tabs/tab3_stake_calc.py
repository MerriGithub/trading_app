from __future__ import annotations

import streamlit as st

from core.basket import Basket
from account import get_margin, get_financing_daily_rate
from tabs.shared import (
    registry, ALL_INSTRUMENTS, ALL_DISPLAY, _asset_class_of,
)


def render() -> None:
    st.header("Stake Calculator")
    st.caption("Vol-targeted stake sizing across any asset class.")

    hdr1, hdr2 = st.columns([3, 1])
    with hdr2:
        broker_profile = st.selectbox(
            "Broker / account type",
            options=["ig_spreadbet", "ig_cfd"],
            format_func=lambda x: {"ig_spreadbet": "IG Spread Bet", "ig_cfd": "IG CFD"}[x],
            key="broker_profile",
        )

    sc1, sc2 = st.columns(2)
    long_picks  = sc1.multiselect(
        "Long instruments", ALL_INSTRUMENTS,
        format_func=lambda c: ALL_DISPLAY.get(c, c),
        key="sc_long",
    )
    short_picks = sc2.multiselect(
        "Short instruments", ALL_INSTRUMENTS,
        format_func=lambda c: ALL_DISPLAY.get(c, c),
        key="sc_short",
    )
    p1, p2, p3 = st.columns(3)
    target_1sd  = p1.number_input("Target 1 SD exposure (£)", value=500.0, step=50.0,
                                  min_value=50.0, key="sc_target")
    vol_window  = p2.slider("Vol window (days)", min_value=130, max_value=524,
                            value=262, step=1, key="tab3_vol_window")
    avg_hold    = p3.number_input("Avg hold (days)", value=30, step=5,
                                  min_value=1, max_value=365, key="sc_avg_hold")

    if not (long_picks and short_picks):
        st.info("Select at least one long and one short instrument.")
        return

    basket = None
    try:
        basket = Basket(long_legs=long_picks, short_legs=short_picks)
        basket.validate()
    except ValueError as e:
        st.error(str(e))
        return

    vols     = registry.get_vols(basket.all_instruments, window=vol_window)
    scalings = registry.get_scalings(basket.all_instruments, target_vol=0.01, window=vol_window)
    latest   = registry.get_latest_prices(basket.all_instruments)

    rows = []
    total_notional = 0.0
    for inst in basket.all_instruments:
        price    = latest.get(inst, 0.0)
        scaling  = scalings.get(inst, 0.0)
        vol      = vols.get(inst, 0.0)
        stake    = (target_1sd * scaling / price) if price > 0 else 0.0
        notional = abs(stake) * price
        total_notional += notional
        side = 'Long' if inst in basket.long_legs else 'Short'
        rows.append({
            'Instrument':  ALL_DISPLAY.get(inst, inst),
            'Side':        side,
            'Asset Class': basket.asset_classes.get(inst, 'unknown'),
            'Vol (ann)':   f"{vol * (252 ** 0.5):.1%}" if vol else 'N/A',
            'Scaling':     f"{scaling:.4f}",
            'Price':       f"{price:,.2f}",
            'Stake':       f"{stake:.3f}",
            'Notional':    f"£{notional:,.0f}",
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    sp_cost = basket.spread_cost(registry)
    margin  = total_notional * get_margin()

    fin_daily_gbp = 0.0
    for inst in basket.long_legs:
        ac   = basket.asset_classes.get(inst, 'equity')
        rate = get_financing_daily_rate(inst, ac, 'long', broker_profile=broker_profile)
        fin_daily_gbp += rate * (target_1sd * scalings.get(inst, 0.0))
    for inst in basket.short_legs:
        ac   = basket.asset_classes.get(inst, 'equity')
        rate = get_financing_daily_rate(inst, ac, 'short', broker_profile=broker_profile)
        fin_daily_gbp += rate * (target_1sd * scalings.get(inst, 0.0))

    total_fin = fin_daily_gbp * avg_hold
    fin_daily_pct = fin_daily_gbp / total_notional if total_notional > 0 else 0.0
    be_days = sp_cost / fin_daily_pct if fin_daily_pct > 0 else float('inf')

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Est. spread cost",               f"{sp_cost:.3%}")
    mc2.metric("Est. daily financing",           f"£{fin_daily_gbp:,.2f}")
    mc3.metric(f"Total financing ({avg_hold}d)", f"£{total_fin:,.2f}")
    mc4.metric("Break-even hold",                f"{be_days:.0f}d")
    mc5.metric("Margin required",                f"£{margin:,.0f}")
