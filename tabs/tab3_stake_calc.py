"""
Tab 3 — Stake Calculator
=========================
Computes position sizes, capital requirements, and financing costs for a
proposed spread trade, then optionally books it into the journal.

Session state (widget keys — written by Streamlit widgets in this tab)
----------------------------------------------------------------------
tab3_direction : str
    Spread direction: ``'long_spread'`` or ``'short_spread'``.
_pricing_broker : str
    Broker profile selector key (e.g. ``'ig_spreadbet'``).
tab3_broker_profile : str
    Active broker profile key; synced with ``_pricing_broker``.

Session state written
---------------------
tab7_pending_entry : dict
    Trade metadata dict written when user clicks "Book Trade".
    Consumed by tab7_journal.py to pre-fill the journal confirm form.
    Key is ``tab7_pending_entry``, not ``tab6_pending_entry``
    (register item A — journal is tab7, not tab6).
sidebar_nav_pending : str
    Written after "Book Trade" to navigate user to Journal, or after
    "Analyse Pair" to navigate to Pair Analysis.
    Uses the pending pattern — see register item B in CLAUDE.md.
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

from core.basket import Basket
from account import get_margin, get_margin_rate, get_financing_daily_rate
from asset_configs import get_cfd_contract_size, ASSET_CLASSES
from tabs.shared import (
    registry, ALL_INSTRUMENTS, ALL_DISPLAY, _asset_class_of,
)

_SB_MIN  = 0.04   # £/pt minimum (IG default — per-instrument overrides in asset_configs.py)
_CFD_MIN = 1      # contracts minimum

_UNIT_LABELS: dict[str, str] = {
    'equity':       'pts/contract',
    'fx':           'units base ccy',
    'commodities':  'units',
    'fixed_income': 'shares',
}


def _yahoo_to_ig_price(cfg: dict, csv_price: float) -> float | None:
    """Convert Yahoo CSV price to approximate IG price using ig_price_conversion.
    Returns None if no conversion is defined for the instrument."""
    conv = cfg.get("ig_price_conversion")
    if conv is None:
        return None
    return conv(csv_price) if callable(conv) else csv_price * conv


def _get_gbpusd() -> float:
    """Most recent GBPUSD close from cached FX data; fallback 1.27."""
    try:
        px = registry.get_latest_prices(['GBPUSD'])
        v = px.get('GBPUSD')
        if v:
            return float(v)
    except Exception:
        # Registry unavailable or GBPUSD not in cache; fall back to 1.27.
        pass
    return 1.27


def _cfd_contracts(notional_gbp: float, price: float,
                   cfd_size: float, cfd_ccy: str, gbpusd: float) -> float:
    """
    Convert GBP notional to CFD contracts.
    price: IG pips price for FX instruments; market price for all others.
    For USD-denominated contracts converts via GBPUSD; all others treated as GBP-equivalent.
    """
    if price <= 0 or cfd_size <= 0:
        return 0.0
    if cfd_ccy == 'USD':
        return notional_gbp * gbpusd / (cfd_size * price)
    return notional_gbp / (cfd_size * price)


def render() -> None:
    st.header("Stake Calculator")
    st.caption("Vol-targeted stake sizing across any asset class.")

    # Transfer pending pair from Tab 10/12 before widgets are instantiated
    for _k in ('sc_long', 'sc_short', 'tab3_vol_window'):
        _pk = f'{_k}_pending'
        if _pk in st.session_state:
            st.session_state[_k] = st.session_state.pop(_pk)

    # Restore selections if Streamlit dropped multiselect state during sidebar navigation
    for _k, _bk in (('sc_long', 'sc_long_bak'), ('sc_short', 'sc_short_bak')):
        if not st.session_state.get(_k) and st.session_state.get(_bk):
            st.session_state[_k] = st.session_state[_bk]

    hdr1, hdr2, hdr3 = st.columns([2, 1, 1])
    with hdr2:
        broker_profile = st.selectbox(
            "Broker / account type",
            options=["ig_spreadbet", "ig_cfd"],
            format_func=lambda x: {"ig_spreadbet": "IG Spread Bet", "ig_cfd": "IG CFD"}[x],
            key="tab3_broker_profile",
        )
    with hdr3:
        _pricing_broker = st.radio(
            "Pricing",
            ["IG", "Yahoo"],
            horizontal=True,
            key="tab3_broker",
            help=(
                "IG: uses IG contract price for margin sizing. "
                "Enter live IG price for instruments where Yahoo price differs. "
                "Yahoo: raw CSV price, zero financing — reference baseline for comparison."
            ),
        )

    if st.session_state.get('tab3_from_monitor'):
        st.info(
            f"📋 Loaded from watchlist: `{st.session_state['tab3_from_monitor']}`  "
            f"— pair and vol window pre-filled."
        )

    # Broker profile always determines sizing mode — radio is informational only
    _use_cfd = (broker_profile == "ig_cfd")
    st.session_state["tab3_calc_mode"] = "CFD (contracts)" if _use_cfd else "Spreadbet (£/point)"

    st.radio(
        "Position sizing mode",
        ["Spreadbet (£/point)", "CFD (contracts)"],
        horizontal=True,
        key="tab3_calc_mode",
        disabled=True,
        help="Determined by broker type.",
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
    # Persist selections so they survive sidebar navigation
    if long_picks:
        st.session_state['sc_long_bak'] = long_picks
    if short_picks:
        st.session_state['sc_short_bak'] = short_picks

    p1, p2, p3 = st.columns(3)
    target_1sd = p1.number_input("Target 1 SD exposure (£)", value=500.0, step=50.0,
                                 min_value=50.0, key="sc_target")
    vol_window = p2.slider("Vol window (days)", min_value=130, max_value=524,
                           value=262, step=1, key="tab3_vol_window")
    avg_hold   = p3.number_input("Avg hold (days)", value=30, step=5,
                                 min_value=1, max_value=365, key="sc_avg_hold")

    _dir_opts = ["WT — With-Trend", "CT — Counter-Trend"]
    if "tab3_direction" not in st.session_state:
        st.session_state["tab3_direction"] = "WT — With-Trend"
    _direction = st.radio(
        "Trade direction",
        options=_dir_opts,
        key="tab3_direction",
        horizontal=True,
        help=(
            "WT: Long first instrument, Short second (ride the trend). "
            "CT: Short first instrument, Long second (fade +2 SD back to mean)."
        ),
    )

    if "CT" in _direction:
        _effective_long  = short_picks
        _effective_short = long_picks
    else:
        _effective_long  = long_picks
        _effective_short = short_picks

    if long_picks and short_picks:
        st.caption(
            f"Long: **{', '.join(_effective_long)}**  ·  "
            f"Short: **{', '.join(_effective_short)}**"
        )

    if not (long_picks and short_picks):
        st.info("Select at least one long and one short instrument.")
        return

    basket = None
    try:
        basket = Basket(long_legs=_effective_long, short_legs=_effective_short)
        basket.validate()
    except ValueError as e:
        st.error(str(e))
        return

    vols     = registry.get_vols(basket.all_instruments, window=vol_window)
    scalings = registry.get_scalings(basket.all_instruments, target_vol=0.01, window=vol_window)
    latest   = registry.get_latest_prices(basket.all_instruments)
    _gbpusd  = _get_gbpusd()

    rows           = []
    total_notional = 0.0
    total_margin   = 0.0
    _book_stakes: dict[str, float] = {}
    _book_prices: dict[str, float] = {}

    for inst in basket.all_instruments:
        price       = latest.get(inst, 0.0)
        scaling     = scalings.get(inst, 0.0)
        daily_vol   = vols.get(inst, 0.0)
        side        = 'Long' if inst in basket.long_legs else 'Short'
        asset_class = basket.asset_classes.get(inst, 'equity')
        display_name = ALL_DISPLAY.get(inst, inst)

        # Look up instrument-specific config (min stake, IG price override)
        _inst_cfg = {}
        for _ac_cfg in ASSET_CLASSES.values():
            if inst in _ac_cfg.get('instruments', {}):
                _v = _ac_cfg['instruments'][inst]
                if isinstance(_v, dict):
                    _inst_cfg = _v
                break
        _min_stake = float(_inst_cfg.get('spreadbet_min_stake', _SB_MIN))

        # Effective price — read IG live price from session state before computing stakes.
        # On first render, falls back to ig_default_price; on subsequent renders uses the
        # value the user entered into the number_input widget.
        if _pricing_broker == "IG" and _inst_cfg.get('ig_price_override'):
            _ig_default      = float(_inst_cfg.get('ig_default_price', price))
            _effective_price = float(st.session_state.get(f"tab3_ig_price_{inst}", _ig_default))
        else:
            _effective_price = price

        # Spreadbet formula — vol-targeted: 1SD move delivers target_1sd P&L
        # stake = target × scaling / (price × vol);  notional = target × scaling / vol (price cancels)
        sb_stake = (
            (target_1sd * scaling / (_effective_price * daily_vol))
            if (_effective_price > 0 and daily_vol > 0)
            else 0.0
        )
        # Applied stake — minimum bet floor; this is what will actually be placed
        _sb_stake_applied = max(sb_stake, _min_stake) if not _use_cfd else sb_stake
        _min_applied      = (not _use_cfd) and (sb_stake < _min_stake)
        notional = abs(_sb_stake_applied) * _effective_price
        total_notional += notional

        # CFD formula — minimum and step size are per-instrument (FX = 0.5 lots)
        cfd_size, cfd_ccy    = get_cfd_contract_size(inst)
        _cfd_min             = float(_inst_cfg.get('cfd_min_contracts', _CFD_MIN))
        # FX instruments need IG pips price for contract sizing (cfd_contract_size is $/pip).
        # If Yahoo pricing is active, convert the decimal rate to IG pips equivalent.
        _price_for_cfd = _effective_price
        if _inst_cfg.get('ig_price_override') and _pricing_broker == 'Yahoo':
            _approx_pips = _yahoo_to_ig_price(_inst_cfg, price)
            if _approx_pips:
                _price_for_cfd = _approx_pips
        contracts_raw        = _cfd_contracts(notional, _price_for_cfd, cfd_size, cfd_ccy, _gbpusd)
        contracts_rounded    = max(_cfd_min, round(contracts_raw / _cfd_min) * _cfd_min)
        _cfd_min_applied     = _use_cfd and contracts_raw < _cfd_min
        unit_label           = _UNIT_LABELS.get(asset_class, 'units')

        with st.container(border=True):
            col_name, col_primary, col_secondary, col_meta = st.columns([2, 2, 2, 2])

            with col_name:
                st.markdown(f"**{display_name}** ({side})")
                if _pricing_broker == "IG" and _inst_cfg.get('ig_price_override'):
                    st.caption(
                        f"Yahoo: {price:,.4f} | "
                        f"IG: {_effective_price:,.1f} {_inst_cfg.get('ig_price_unit', '')}"
                    )
                else:
                    st.caption(f"Price: {price:,.4f}")

            with col_primary:
                if _use_cfd:
                    _ctr_label = (f"{contracts_rounded:g}" +
                                  (" (min)" if _cfd_min_applied else ""))
                    st.metric("CFD Contracts", _ctr_label)
                    if cfd_ccy == 'USD':
                        st.caption(f"Notional: £{notional:,.0f} (${notional * _gbpusd:,.0f})")
                    elif cfd_ccy == 'GBP':
                        st.caption(f"Notional: £{notional:,.0f}")
                    else:
                        st.caption(f"Notional: £{notional:,.0f} ({cfd_ccy})")
                    if _cfd_min_applied:
                        st.caption(f"Vol target: {contracts_raw:.3f} contracts")
                else:
                    st.metric("Spreadbet", f"£{_sb_stake_applied:.4f}/pt")
                    if _min_applied:
                        st.caption(f"Vol target: £{sb_stake:.4f}/pt (min applied)")

            with col_secondary:
                if _use_cfd:
                    st.caption(f"SB equiv: £{_sb_stake_applied:.4f}/pt")
                else:
                    st.caption(f"CFD equiv: {contracts_rounded} contract(s)")

            with col_meta:
                st.caption(f"Vol: {daily_vol*100:.2f}%/day")
                st.caption(f"Scaling: {scaling:.3f}")

            # IG live price + Yahoo comparison — IG broker mode only, override instruments only
            if _pricing_broker == "IG" and _inst_cfg.get('ig_price_override'):
                _col_live, _col_conv = st.columns(2)
                with _col_live:
                    _live_price = st.number_input(
                        f"IG live price ({_inst_cfg.get('ig_price_unit', '')})",
                        min_value=0.01,
                        value=_ig_default,
                        step=1.0,
                        key=f"tab3_ig_price_{inst}",
                        help="Enter current IG spreadbet price. Used for stake sizing and margin.",
                    )
                with _col_conv:
                    _approx = _yahoo_to_ig_price(_inst_cfg, price)
                    if _approx is not None:
                        st.metric(
                            label="Yahoo-converted (approx)",
                            value=f"{_approx:,.0f}",
                            delta=f"{_live_price - _approx:+,.0f} vs live",
                            help=(
                                "Approximate IG price derived from Yahoo CSV price. "
                                "Difference from live IG price is normal due to FX and roll basis."
                            ),
                        )
                    else:
                        st.caption("No Yahoo→IG conversion defined for this instrument.")
            else:
                _live_price = price

        total_margin += abs(_sb_stake_applied) * _live_price * get_margin_rate(asset_class)
        _book_stakes[inst] = _sb_stake_applied if side == 'Long' else -_sb_stake_applied
        _book_prices[inst] = _live_price

        if _min_applied:
            st.warning(
                f"⚠️ {display_name}: vol target is £{sb_stake:.4f}/pt but IG minimum is "
                f"£{_min_stake}/pt. Margin calculated at minimum. "
                f"Position carries more risk than the vol target implies."
            )

        if _use_cfd and contracts_raw < _cfd_min / 2:
            _capital_needed = (_cfd_min / contracts_raw) * target_1sd if contracts_raw > 0 else 0.0
            st.warning(
                f"⚠️ {inst}: vol target ({contracts_raw:.3f}) is below half the minimum "
                f"({_cfd_min:g} contract). "
                f"{_cfd_min:g} contract requires ~£{_capital_needed:,.0f} "
                f"target exposure at current vol."
            )

        rows.append({
            'Instrument':    display_name,
            'Side':          side,
            'Price':         f"{_effective_price:,.2f}",
            'Vol %/d':       f"{daily_vol*100:.2f}%",
            'Scaling':       f"{scaling:.4f}",
            'SB £/pt':       f"£{_sb_stake_applied:.4f}" + (" (min)" if _min_applied else ""),
            'CFD Contracts': f"{contracts_rounded:g}" + (" (min)" if _cfd_min_applied else ""),
            'Notional':      (f"£{notional:,.0f} (${notional * _gbpusd:,.0f})"
                             if _use_cfd and cfd_ccy == 'USD' else f"£{notional:,.0f}"),
        })

    st.markdown("---")
    st.subheader("Summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    sp_cost = basket.spread_cost(registry)
    margin  = total_margin

    fin_daily_gbp = 0.0
    if _pricing_broker == "IG":
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

    with st.expander("📊 Margin breakdown", expanded=False):
        _ac_map = basket.asset_classes
        _margin_rows = []
        for inst in basket.all_instruments:
            _ac    = _ac_map.get(inst, 'equity')
            _rate  = get_margin_rate(_ac)
            _notl  = abs(_book_stakes.get(inst, 0.0)) * _book_prices.get(inst, 0.0)
            _margin_rows.append({
                'Instrument':  ALL_DISPLAY.get(inst, inst),
                'Asset class': _ac,
                'Notional':    f"£{_notl:,.0f}",
                'Rate':        f"{_rate:.2%}",
                'Margin':      f"£{_notl * _rate:,.0f}",
            })
        st.dataframe(pd.DataFrame(_margin_rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    nc1, nc2, nc3, nc4 = st.columns([1, 1, 1, 2])
    if nc1.button("→ Pair Analysis", key="tab3_to_pa"):
        st.session_state['pa_long_pending']  = list(_effective_long)
        st.session_state['pa_short_pending'] = list(_effective_short)
        st.session_state['pa_pair_pending']  = '— Custom pair —'
        st.session_state['sidebar_nav_pending'] = "📈 Pair Analysis"  # register item B: use pending, not sidebar_nav
        st.rerun()
    if nc2.button("📒 Book Trade", key="tab3_book_trade", type="primary"):
        _dir_flag = "WT" if "WT" in _direction else "CT"
        st.session_state["tab7_pending_entry"] = {
            "long_legs":       list(_effective_long),
            "short_legs":      list(_effective_short),
            "direction":       "long_spread" if _dir_flag == "WT" else "short_spread",
            "direction_label": _dir_flag,
            "broker":          _pricing_broker,
            "mode":            "CFD" if _use_cfd else "SB",
            "stakes":          dict(_book_stakes),
            "prices":          dict(_book_prices),
            "target_exposure": float(target_1sd),
            "margin":          margin,
            "spread_cost_pct": sp_cost,
            "daily_financing": fin_daily_gbp,
            "breakeven_days":  be_days if be_days != float('inf') else None,
            "entry_date":      pd.Timestamp.today().date().isoformat(),
        }
        st.session_state["sidebar_nav_pending"] = "📓 Journal"
        st.rerun()
    if nc3.button("👁 Monitor", key="tab3_monitor"):
        from data_watchlist import add_monitor_candidate
        if not (long_picks and short_picks):
            st.warning("Select instruments first.")
        elif len(long_picks) > 1 or len(short_picks) > 1:
            st.warning("Monitor is for 1v1 pairs only.")
        else:
            _dir_flag = "WT" if "WT" in _direction else "CT"
            add_monitor_candidate({
                'id':                st.session_state.get('tab3_from_monitor')
                                     or f"{long_picks[0]}_{short_picks[0]}_monitor",
                'long':              long_picks[0],
                'short':             short_picks[0],
                'entry_sd':          st.session_state.get('pa_xing', 2.0),
                'exit_sd':           st.session_state.get('pa_exit', 2.0),
                'vol_window':        vol_window,
                'trend_window':      st.session_state.get('pa_trend_window', 262),
                'trend_mode':        'Both passes',
                'asset_class_long':  _asset_class_of(long_picks[0]),
                'asset_class_short': _asset_class_of(short_picks[0]),
                'scoring_mode':      None,
                'direction':         "CT — Counter-Trend" if _dir_flag == "CT"
                                     else "WT — With-Trend",
                'target_exposure':   float(target_1sd),
                'broker_profile':    broker_profile,
            })
            st.toast("Added to monitor candidates in Tab 1")
