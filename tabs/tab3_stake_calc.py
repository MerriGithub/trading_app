from __future__ import annotations

import pandas as pd
import streamlit as st

from core.basket import Basket
from account import get_margin, get_financing_daily_rate
from asset_configs import get_cfd_contract_size
from tabs.shared import (
    registry, ALL_INSTRUMENTS, ALL_DISPLAY, _asset_class_of,
)

_SB_MIN  = 0.50   # £/pt minimum (IG/CMC)
_CFD_MIN = 1      # contracts minimum

_UNIT_LABELS: dict[str, str] = {
    'equity':       'pts/contract',
    'fx':           'units base ccy',
    'commodities':  'units',
    'fixed_income': 'shares',
}


def render() -> None:
    st.header("Stake Calculator")
    st.caption("Vol-targeted stake sizing across any asset class.")

    # Transfer pending pair from Tab 10 before widgets are instantiated
    for _k in ('sc_long', 'sc_short'):
        _pk = f'{_k}_pending'
        if _pk in st.session_state:
            st.session_state[_k] = st.session_state.pop(_pk)

    hdr1, hdr2 = st.columns([3, 1])
    with hdr2:
        broker_profile = st.selectbox(
            "Broker / account type",
            options=["ig_spreadbet", "ig_cfd"],
            format_func=lambda x: {"ig_spreadbet": "IG Spread Bet", "ig_cfd": "IG CFD"}[x],
            key="tab3_broker_profile",
        )

    # Auto-set mode on first load based on broker profile
    _broker = st.session_state.get('tab3_broker_profile', '')
    if 'sidebar_nav' not in st.session_state:
        if 'cfd' in _broker.lower():
            st.session_state['tab3_calc_mode'] = "CFD (contracts)"
        elif 'spreadbet' in _broker.lower() or 'spread_bet' in _broker.lower():
            st.session_state['tab3_calc_mode'] = "Spreadbet (£/point)"

    _calc_mode = st.radio(
        "Position sizing mode",
        ["Spreadbet (£/point)", "CFD (contracts)"],
        horizontal=True,
        key="tab3_calc_mode",
    )
    _use_cfd = (_calc_mode == "CFD (contracts)")

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
    target_1sd = p1.number_input("Target 1 SD exposure (£)", value=500.0, step=50.0,
                                 min_value=50.0, key="sc_target")
    vol_window = p2.slider("Vol window (days)", min_value=130, max_value=524,
                           value=262, step=1, key="tab3_vol_window")
    avg_hold   = p3.number_input("Avg hold (days)", value=30, step=5,
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

    rows           = []
    total_notional = 0.0

    for inst in basket.all_instruments:
        price       = latest.get(inst, 0.0)
        scaling     = scalings.get(inst, 0.0)
        daily_vol   = vols.get(inst, 0.0)
        side        = 'Long' if inst in basket.long_legs else 'Short'
        asset_class = basket.asset_classes.get(inst, 'equity')
        display_name = ALL_DISPLAY.get(inst, inst)

        # Spreadbet formula — unchanged
        sb_stake = (target_1sd * scaling / price) if price > 0 else 0.0
        notional = abs(sb_stake) * price   # = target_1sd * scaling
        total_notional += notional

        # CFD formula
        cfd_size, cfd_ccy    = get_cfd_contract_size(inst)
        contracts_raw        = notional / (price * cfd_size) if (price > 0 and cfd_size > 0) else 0.0
        contracts_rounded    = max(1, round(contracts_raw))
        unit_label           = _UNIT_LABELS.get(asset_class, 'units')

        with st.container(border=True):
            col_name, col_primary, col_secondary, col_meta = st.columns([2, 2, 2, 2])

            with col_name:
                st.markdown(f"**{display_name}** ({side})")
                st.caption(f"Price: {price:,.2f}")

            with col_primary:
                if _use_cfd:
                    st.metric("CFD Contracts", f"{contracts_rounded}")
                    st.caption(f"Notional: {cfd_size * contracts_rounded:,.0f} "
                               f"{unit_label} · {cfd_ccy}")
                else:
                    st.metric("Spreadbet", f"£{sb_stake:.2f}/pt")

            with col_secondary:
                if _use_cfd:
                    st.caption(f"SB equiv: £{sb_stake:.2f}/pt")
                else:
                    st.caption(f"CFD equiv: {contracts_rounded} contract(s)")

            with col_meta:
                st.caption(f"Vol: {daily_vol*100:.2f}%/day")
                st.caption(f"Scaling: {scaling:.3f}")

        if not _use_cfd and sb_stake < _SB_MIN:
            st.warning(
                f"⚠️ {inst}: stake £{sb_stake:.2f}/pt is below the typical "
                f"minimum of £{_SB_MIN}/pt. Consider more capital or fewer legs."
            )

        if _use_cfd and contracts_raw < 0.5:
            notional_needed = cfd_size * price
            st.warning(
                f"⚠️ {inst}: rounds to 0 contracts. "
                f"1 contract requires ~£{notional_needed * daily_vol / 0.01:,.0f} capital "
                f"at 1% vol target."
            )

        rows.append({
            'Instrument':    display_name,
            'Side':          side,
            'Price':         f"{price:,.2f}",
            'Vol %/d':       f"{daily_vol*100:.2f}%",
            'Scaling':       f"{scaling:.4f}",
            'SB £/pt':       f"£{sb_stake:.3f}",
            'CFD Contracts': str(contracts_rounded),
            'Notional':      f"£{notional:,.0f}",
        })

    st.markdown("---")
    st.subheader("Summary")
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

    total_fin     = fin_daily_gbp * avg_hold
    fin_daily_pct = fin_daily_gbp / total_notional if total_notional > 0 else 0.0
    be_days       = sp_cost / fin_daily_pct if fin_daily_pct > 0 else float('inf')

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Est. spread cost",               f"{sp_cost:.3%}")
    mc2.metric("Est. daily financing",           f"£{fin_daily_gbp:,.2f}")
    mc3.metric(f"Total financing ({avg_hold}d)", f"£{total_fin:,.2f}")
    mc4.metric("Break-even hold",                f"{be_days:.0f}d")
    mc5.metric("Margin required",                f"£{margin:,.0f}")
