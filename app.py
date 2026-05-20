"""
app.py — Global Spread Trading Platform
Multi-asset, NvM basket spread trading monitor and journal.

Tabs 1-4 + 7 are the new multi-asset UI built on the core/ domain layer.
Tabs 5/8/9 are equity-only and ported from legacy/app_legacy.py.
Tab 6 (Live) is a placeholder — full implementation in Sprint 3.
"""
from __future__ import annotations

from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.basket import Basket
from core.signal import SpreadSignal
from core.position import Position
from core.portfolio import Portfolio
from core.data_registry import DataRegistry
from account import (
    load_account, get_financing_rates,
    get_spread_cost_fallback, get_starting_capital, get_margin,
)
from asset_configs import (
    ASSET_CLASSES, ASSET_CLASS_OPTIONS, FI_EXCLUDE,
    get_display_name, get_tradeable_instruments,
    get_spread_cost_lookup,
)

# ── Shared paths ──────────────────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent / 'cache'
_DATA_DIR  = Path(__file__).parent / 'data'
_POSITIONS = _DATA_DIR / 'positions.json'
_ACCOUNT   = _DATA_DIR / 'account.json'

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spread Trading Platform", page_icon="📈", layout="wide")


# ── Cached singletons ─────────────────────────────────────────────────────────
@st.cache_resource
def _registry() -> DataRegistry:
    return DataRegistry(_CACHE_DIR)


@st.cache_resource
def _portfolio() -> Portfolio:
    return Portfolio(_POSITIONS, _ACCOUNT)


@st.cache_resource
def _load_account() -> dict:
    return load_account()


registry  = _registry()
portfolio = _portfolio()
account   = _load_account()


# ── Multi-asset instrument lookup (used by Tabs 2, 3, 7) ──────────────────────
ALL_INSTRUMENTS: list[str] = []
ALL_DISPLAY: dict[str, str] = {}
for _key, _cfg in ASSET_CLASSES.items():
    for _code in get_tradeable_instruments(_key):
        if _code not in FI_EXCLUDE and _code not in ALL_DISPLAY:
            ALL_INSTRUMENTS.append(_code)
            ALL_DISPLAY[_code] = get_display_name(_key, _code)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _tbl(df: pd.DataFrame, show_index: bool = True, height: int | None = None) -> None:
    """Render a DataFrame as a Plotly Table — zero pyarrow dependency."""
    idx_vals  = [str(v) for v in df.index]
    col_names = ([str(df.index.name or '')] if show_index else []) + [str(c) for c in df.columns]
    col_vals  = ([idx_vals] if show_index else []) + [df[c].astype(str).tolist() for c in df.columns]
    n = len(df)
    row_fill = ['#f0f4f8' if i % 2 == 0 else 'white' for i in range(n)]
    fig = go.Figure(go.Table(
        header=dict(values=col_names, fill_color='#2c6fad',
                    font=dict(color='white', size=13), align='left'),
        cells=dict(values=col_vals, fill_color=[row_fill] * len(col_vals),
                   align='left', font=dict(size=12)),
    ))
    fig.update_layout(height=height or min(80 + n * 28, 550),
                      margin=dict(l=0, r=0, t=4, b=4))
    st.plotly_chart(fig, use_container_width=True)


def _asset_class_of(code: str) -> str:
    for key, cfg in ASSET_CLASSES.items():
        if code in cfg.get('instruments', {}):
            return key
    return 'unknown'


def _signal_state_badge(state: str) -> str:
    return {
        'EXIT':         '🟢 Near target',
        'LONG_ENTRY':   '🟡 At entry level',
        'SHORT_ENTRY':  '🟡 At entry level',
        'NONE':         '⚪ Holding',
    }.get(state, state)


# ── Session state defaults ────────────────────────────────────────────────────
if 'pending_close' not in st.session_state:
    st.session_state['pending_close'] = None
if 'leg_count' not in st.session_state:
    st.session_state['leg_count'] = 1


# ── Signal alert helper ───────────────────────────────────────────────────────
def _check_signal_alerts(portfolio: Portfolio, registry: DataRegistry) -> list[dict]:
    alerts = []
    for pos in portfolio.open_positions:
        try:
            prices_hist = registry.get_daily_prices(pos.basket.all_instruments)
            signal = SpreadSignal(basket=pos.basket, prices=prices_hist)
            state  = signal.signal_state
            if state != 'NONE':
                alerts.append({
                    'position':  pos.name,
                    'pair':      ' / '.join(pos.basket.all_instruments),
                    'state':     state,
                    'sd':        signal.current_sd,
                    'days_held': pos.days_held,
                })
        except Exception:
            pass
    return alerts


# ── Cached wrappers — avoid re-reading CSVs on every Streamlit rerun ─────────
@st.cache_data(ttl=300, show_spinner=False)
def _cached_daily_prices(instruments: tuple[str, ...]) -> pd.DataFrame:
    return registry.get_daily_prices(list(instruments))


@st.cache_data(ttl=300, show_spinner=False)
def _cached_latest_prices(instruments: tuple[str, ...]) -> dict[str, float]:
    return registry.get_latest_prices(list(instruments))


@st.cache_data(ttl=300, show_spinner=False)
def _cached_pair_signal(
    instruments: tuple[str, ...],
    long_legs: tuple[str, ...],
    short_legs: tuple[str, ...],
    vol_window: int,
    xing_sd: float,
    exit_sd: float,
) -> dict:
    prices = registry.get_daily_prices(list(instruments))
    b   = Basket(long_legs=list(long_legs), short_legs=list(short_legs))
    sig = SpreadSignal(basket=b, prices=prices,
                       vol_window=vol_window, xing_sd=xing_sd, exit_sd=exit_sd)
    hist = sig.signal_history(n_days=9999)
    return {
        'current_sd':    sig.current_sd,
        'signal_state':  sig.signal_state,
        'velocity':      float(sig.velocity.iloc[-1]) if len(sig.velocity) else 0.0,
        'tvr':           sig.tvr,
        'hist_index':    hist.index,
        'cum_spread':    hist['cum_spread'],
        'distance_sd':   hist['distance_sd'],
    }


@st.cache_data(ttl=300, show_spinner=False)
def _pair_backtest(
    long_legs: tuple[str, ...],
    short_legs: tuple[str, ...],
    vol_window: int,
    xing_sd: float,
    exit_sd: float,
    window_days: int | None = None,
) -> dict:
    from engine.backtest import prepare_returns, run_backtest
    instr  = list(long_legs) + list(short_legs)
    prices = _cached_daily_prices(tuple(instr))
    scaled, day_ints, index = prepare_returns(prices, instr, vol_window=vol_window,
                                              window_days=window_days)
    n_long, n_short = len(long_legs), len(short_legs)
    long_mask  = np.array([1.0 / n_long  if i < n_long  else 0.0 for i in range(len(instr))])
    short_mask = np.array([1.0 / n_short if i >= n_long else 0.0 for i in range(len(instr))])
    spread_ret = scaled @ (long_mask - short_mask)
    bt = run_backtest(spread_ret, day_ints,
                      vol_window=vol_window, xing_sd=xing_sd, exit_sd=exit_sd,
                      n_legs=n_long + n_short)
    s = bt['summary']
    n = int(bt['n_trades'])
    return {
        'n_trades':    n,
        'gross_wr':    float(s.get('gross_wr', 0.0)),
        'net_wr':      float(s.get('net_wr', 0.0)),
        'avg_gross':   float(s.get('avg_gross', 0.0)),
        'avg_net':     float(s.get('avg_net', 0.0)),
        'avg_holding': float(s.get('avg_holding', 0.0)),
        # Raw data for post-hoc trend alignment analysis
        'trades_raw':  bt['trades_raw'][:n].copy(),
        'cum_spread':  np.cumsum(spread_ret),
        'date_index':  index,
    }


def _build_signal_metrics(pos) -> dict:
    """Compute SpreadSignal metrics for one position. Cached in session_state."""
    ph  = _cached_daily_prices(tuple(pos.basket.all_instruments))
    sig = SpreadSignal(basket=pos.basket, prices=ph)
    return {
        'current_sd':   sig.current_sd,
        'signal_state': sig.signal_state,
        'velocity':     float(sig.velocity.iloc[-1]) if len(sig.velocity) else 0.0,
        'tvr':          sig.tvr,
        'distance_sd':  sig.distance_sd,
        'spread_ret':   sig.spread_ret,
        'cum_spread':   sig.cum_spread,
    }


def _get_signal_metrics(pos) -> dict | None:
    cache = st.session_state.setdefault('signal_metrics', {})
    if pos.id not in cache:
        try:
            cache[pos.id] = _build_signal_metrics(pos)
        except Exception as e:
            cache[pos.id] = {'error': str(e)}
    return cache[pos.id]


# ── Alert pre-computation (once per page load) ────────────────────────────────
if 'signal_alerts' not in st.session_state:
    st.session_state['signal_alerts'] = _check_signal_alerts(portfolio, registry)

_n_alerts = len(st.session_state['signal_alerts'])

# ── Tabs ──────────────────────────────────────────────────────────────────────
_live_label = f"⏱ Live 🔔 {_n_alerts}" if _n_alerts > 0 else "⏱ Live"
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Monitor", "📈 Pair Analysis", "🧮 Stake Calc", "🗂 Portfolio",
    "🔍 Search", _live_label, "📓 Journal", "🔬 Backtest", "🔄 Walk-Forward",
    "🔭 Scenario",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Journal (paper trading)
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Journal")
    st.caption("Paper trade open/close, position history, and live P&L.")

    # ── Open Positions ───────────────────────────────────────────────────────
    st.subheader(f"Open positions ({len(portfolio.open_positions)})")

    if not portfolio.open_positions:
        st.info("No open positions. Use the form below to open one.")
    else:
        for pos in portfolio.open_positions:
            with st.container(border=True):
                try:
                    current_prices = _cached_latest_prices(tuple(pos.basket.all_instruments))
                except Exception as e:
                    st.warning(f"Could not load prices for {pos.name}: {e}")
                    current_prices = pos.entry_prices

                # Header row
                hcol, mcol1, mcol2, mcol3, mcol4 = st.columns([3, 2, 2, 2, 2])
                hcol.markdown(f"**{pos.name}**  \n_{pos.direction.replace('_', ' ').title()}_")
                hcol.caption(
                    f"Long: {' + '.join(pos.basket.long_legs)}  \n"
                    f"Short: {' + '.join(pos.basket.short_legs)}"
                )
                mcol1.metric("Opened", str(pos.entry_date))
                mcol1.caption(f"{pos.days_held}d held")
                live_pnl   = pos.live_pnl(current_prices)
                fin_drag   = pos.financing_cost_to_date()
                net_pnl    = pos.net_pnl(current_prices)
                mcol2.metric("Unrealised", f"£{live_pnl:+,.0f}")
                mcol3.metric("Financing", f"£{-fin_drag:,.0f}")
                mcol4.metric("Net P&L", f"£{net_pnl:+,.0f}",
                             delta_color="normal" if net_pnl >= 0 else "inverse")

                # Action row
                acol1, acol2, acol3 = st.columns([1, 1, 4])
                if acol1.button("Close", key=f"close_{pos.id}"):
                    st.session_state['pending_close'] = pos.id
                    st.rerun()

                with acol2.popover("Partial close"):
                    pct = st.slider("% to close", 10, 100, 50, 10,
                                    key=f"pct_{pos.id}") / 100.0
                    exit_prices = {}
                    for inst in pos.basket.all_instruments:
                        exit_prices[inst] = st.number_input(
                            f"{ALL_DISPLAY.get(inst, inst)} exit price",
                            value=float(current_prices.get(inst, pos.entry_prices.get(inst, 0.0))),
                            key=f"px_{pos.id}_{inst}",
                            format="%.4f",
                        )
                    if st.button("Confirm partial close", key=f"pc_confirm_{pos.id}"):
                        realised = portfolio.partial_close(
                            pos.id, pct, exit_prices, date.today()
                        )
                        st.success(f"Partial closed {pct:.0%} — realised £{realised:+,.0f}")
                        st.rerun()

                # Two-step confirmation
                if st.session_state.get('pending_close') == pos.id:
                    with st.container(border=True):
                        st.warning(f"Confirm full close of **{pos.name}**?")
                        c1, c2 = st.columns(2)
                        if c1.button("Yes, close", key=f"close_yes_{pos.id}", type="primary"):
                            realised = portfolio.close_position(
                                pos.id, current_prices, date.today()
                            )
                            st.session_state['pending_close'] = None
                            st.success(f"Closed {pos.name} — realised £{realised:+,.0f}")
                            st.rerun()
                        if c2.button("Cancel", key=f"close_no_{pos.id}"):
                            st.session_state['pending_close'] = None
                            st.rerun()

                # Leg detail
                with st.expander("Leg detail"):
                    leg_rows = []
                    for inst, stake in pos.stakes.items():
                        entry = pos.entry_prices.get(inst, 0.0)
                        curr  = current_prices.get(inst, entry)
                        leg_pnl = stake * (curr - entry) * pos.pct_open
                        leg_rows.append({
                            'Instrument': ALL_DISPLAY.get(inst, inst),
                            'Side':       'Long' if stake > 0 else 'Short',
                            'Stake':      f"{stake:+.3f}",
                            'Entry':      f"{entry:,.4f}",
                            'Current':    f"{curr:,.4f}",
                            'Leg P&L':    f"£{leg_pnl:+,.2f}",
                        })
                    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Open New Trade ───────────────────────────────────────────────────────
    st.subheader("Open new trade")

    nc1, nc2, nc3 = st.columns(3)
    trade_name      = nc1.text_input("Trade name", value="", key="new_name",
                                     placeholder="e.g. Platinum / AUDUSD")
    direction       = nc2.selectbox("Direction", ['long_spread', 'short_spread'], key="new_dir")
    target_exposure = nc3.number_input("Target exposure (£)", value=500.0, step=50.0,
                                       min_value=50.0, key="new_exp")

    nc4, nc5 = st.columns([1, 2])
    entry_date_sel  = nc4.date_input("Entry date", value=date.today(), key="new_date")
    comments        = nc5.text_input("Comments", value="", key="new_comments")

    st.markdown("**Legs**")
    lc1, lc2, _ = st.columns([1, 1, 4])
    if lc1.button("+ Add leg"):
        st.session_state['leg_count'] += 1
        st.rerun()
    if lc2.button("− Remove leg") and st.session_state['leg_count'] > 1:
        st.session_state['leg_count'] -= 1
        st.rerun()

    legs: list[dict] = []
    for i in range(st.session_state['leg_count']):
        with st.container(border=True):
            st.markdown(f"**Leg {i + 1}**")
            lcols = st.columns([2, 1, 1, 2, 1, 1])
            buy_inst = lcols[0].selectbox(
                "Buy", ALL_INSTRUMENTS,
                format_func=lambda c: f"{ALL_DISPLAY.get(c, c)} ({_asset_class_of(c)})",
                key=f"leg_buy_{i}",
            )
            try:
                buy_default = float(_cached_latest_prices((buy_inst,)).get(buy_inst, 0.0))
            except Exception:
                buy_default = 0.0
            buy_price = lcols[1].number_input(
                "Buy price", value=buy_default, format="%.4f", key=f"leg_bp_{i}",
            )
            buy_stake = lcols[2].number_input(
                "Buy stake", value=1.0, step=0.1, format="%.3f", key=f"leg_bs_{i}",
            )
            sell_inst = lcols[3].selectbox(
                "Sell", ALL_INSTRUMENTS,
                format_func=lambda c: f"{ALL_DISPLAY.get(c, c)} ({_asset_class_of(c)})",
                index=min(1, len(ALL_INSTRUMENTS) - 1),
                key=f"leg_sell_{i}",
            )
            try:
                sell_default = float(_cached_latest_prices((sell_inst,)).get(sell_inst, 0.0))
            except Exception:
                sell_default = 0.0
            sell_price = lcols[4].number_input(
                "Sell price", value=sell_default, format="%.4f", key=f"leg_sp_{i}",
            )
            sell_stake = lcols[5].number_input(
                "Sell stake", value=1.0, step=0.1, format="%.3f", key=f"leg_ss_{i}",
            )
            legs.append({
                'buy': buy_inst, 'buy_price': buy_price, 'buy_stake': buy_stake,
                'sell': sell_inst, 'sell_price': sell_price, 'sell_stake': sell_stake,
            })

    # Preview spread cost / financing
    try:
        prev_basket = Basket(
            long_legs=[l['buy'] for l in legs],
            short_legs=[l['sell'] for l in legs],
        )
        prev_basket.validate()
        sp_cost   = prev_basket.spread_cost(registry)
        daily_fin = prev_basket.financing_cost_daily()
        be_days   = sp_cost / daily_fin if daily_fin > 0 else float('inf')
        st.caption(
            f"Est. spread cost: **{sp_cost:.3%}** round-trip  |  "
            f"Est. daily financing: **£{daily_fin * target_exposure:,.2f}**  |  "
            f"Break-even hold: **{be_days:.0f} days**"
        )
    except Exception as e:
        st.caption(f"_Spread cost preview unavailable: {e}_")

    if st.button("📓 Open position", type="primary", key="new_submit"):
        if not trade_name.strip():
            st.error("Trade name is required.")
        else:
            try:
                basket = Basket(
                    long_legs=[l['buy'] for l in legs],
                    short_legs=[l['sell'] for l in legs],
                )
                basket.validate()
                entry_prices = {}
                stakes = {}
                for l in legs:
                    entry_prices[l['buy']] = l['buy_price']
                    entry_prices[l['sell']] = l['sell_price']
                    stakes[l['buy']]  = stakes.get(l['buy'],  0.0) + l['buy_stake']
                    stakes[l['sell']] = stakes.get(l['sell'], 0.0) - l['sell_stake']

                pos = portfolio.open_position(
                    basket=basket,
                    direction=direction,
                    entry_prices=entry_prices,
                    stakes=stakes,
                    target_exposure=float(target_exposure),
                    name=trade_name.strip(),
                    comments=comments,
                )
                st.success(f"Opened **{pos.name}** (id={pos.id})")
                st.session_state['leg_count'] = 1
                st.rerun()
            except ValueError as e:
                st.error(f"Validation error: {e}")
            except Exception as e:
                st.error(f"Failed to open position: {e}")

    st.markdown("---")

    # ── Trade History ────────────────────────────────────────────────────────
    st.subheader("Trade history")
    closed = portfolio.closed_positions
    if not closed:
        st.caption("No closed positions yet.")
    else:
        total_real = portfolio.total_realised_pnl()
        wins       = sum(1 for p in closed if p.realised_pnl > 0)
        wr         = wins / len(closed) if closed else 0.0
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Realised P&L", f"£{total_real:+,.0f}")
        hm2.metric("Trades closed", f"{len(closed)}")
        hm3.metric("Win rate", f"{wr:.0%}")

        hist_rows = []
        for p in sorted(closed, key=lambda x: x.exit_date or date.min, reverse=True):
            hist_rows.append({
                'Name':       p.name,
                'Pair':       f"{' + '.join(p.basket.long_legs)} vs {' + '.join(p.basket.short_legs)}",
                'Entry':      str(p.entry_date),
                'Close':      str(p.exit_date),
                'Days held':  p.days_held,
                'P&L':        f"£{p.realised_pnl:+,.0f}",
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Monitor
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Monitor")
    st.caption("Live signal status and P&L for every open position.")

    if not portfolio.open_positions:
        st.info("No open positions. Open a paper trade in the Journal tab to start monitoring.")
    else:
        for pos in portfolio.open_positions:
            with st.container(border=True):
                cp    = _cached_latest_prices(tuple(pos.basket.all_instruments))
                sm    = _get_signal_metrics(pos)
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
                    st.warning(f"Could not build signal for {pos.name}: {sm.get('error') if sm else 'unknown'}")

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

                if st.button("Open in Pair Analysis →", key=f"view_{pos.id}"):
                    st.session_state['pa_pair'] = f"{pos.name} ({pos.id})"
                    st.toast(f"Selected {pos.name} — open the Pair Analysis tab", icon="📈")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Pair Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Pair Analysis")
    st.caption("Inspect signal history, charts, and backtest stats for any pair.")

    # Mode selector
    pair_choices = ['— Custom pair —'] + [f"{p.name} ({p.id})" for p in portfolio.open_positions]
    if st.session_state.get('pa_pair') not in pair_choices:
        st.session_state['pa_pair'] = '— Custom pair —'
    pair_choice = st.selectbox("Pair", pair_choices, key="pa_pair")

    basket = None
    open_pos: Position | None = None
    if pair_choice == '— Custom pair —':
        cc1, cc2 = st.columns(2)
        long_picks  = cc1.multiselect(
            "Long legs", ALL_INSTRUMENTS,
            default=[ALL_INSTRUMENTS[0]] if ALL_INSTRUMENTS else [],
            format_func=lambda c: ALL_DISPLAY.get(c, c),
            key="pa_long",
        )
        short_picks = cc2.multiselect(
            "Short legs", ALL_INSTRUMENTS,
            default=[ALL_INSTRUMENTS[1]] if len(ALL_INSTRUMENTS) > 1 else [],
            format_func=lambda c: ALL_DISPLAY.get(c, c),
            key="pa_short",
        )
        if long_picks and short_picks:
            try:
                basket = Basket(long_legs=long_picks, short_legs=short_picks)
                basket.validate()
            except ValueError as e:
                st.error(str(e))
                basket = None
    else:
        for p in portfolio.open_positions:
            if pair_choice.endswith(f"({p.id})"):
                basket = p.basket
                open_pos = p
                break

    cp1, cp2, cp3, cp4, cp5 = st.columns(5)
    vol_window      = cp1.slider("Vol window (days)", 50, 524, 262, 10, key="pa_vol")
    xing_sd         = cp2.slider("Entry SD", 1.0, 3.0, 2.0, 0.1, key="pa_xing")
    exit_sd         = cp3.slider("Exit SD",  0.0, 1.5, 1.0, 0.1, key="pa_exit")
    pa_trend_window = cp4.slider("Trend filter (days)", 130, 756, 262, 1, key="pa_trend_window")
    _pa_window_opts = {'1 year': 262, '2 years': 524, '3 years': 786, '5 years': 1310, 'All': 0}
    pa_window_label = cp5.selectbox("History window", list(_pa_window_opts.keys()),
                                    index=4, key="pa_window")
    pa_hist_days = _pa_window_opts[pa_window_label] or None

    if basket is not None:
        try:
            pa_sig = _cached_pair_signal(
                tuple(basket.all_instruments),
                tuple(basket.long_legs),
                tuple(basket.short_legs),
                int(vol_window), float(xing_sd), float(exit_sd),
            )

            # Metrics row
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Current SD",  f"{pa_sig['current_sd']:+.2f}")
            mc2.metric("TVR",         f"{pa_sig['tvr']:.3f}")
            mc3.metric("Velocity",    f"{pa_sig['velocity']:+.4f}")
            mc4.markdown(f"**{_signal_state_badge(pa_sig['signal_state'])}**")

            # Spread chart — compute rolling stats on full history, then slice for display
            _cs_full   = pa_sig['cum_spread']
            _rm_full   = _cs_full.rolling(int(vol_window), min_periods=10).mean()
            _rstd_full = _cs_full.rolling(int(vol_window), min_periods=10).std()

            # Trend filter computation on full history
            _trend_win  = int(pa_trend_window)
            _trend_full = _cs_full.rolling(_trend_win, min_periods=10).mean()
            _slope_lb   = min(20, _trend_win // 10)
            _trend_valid = _trend_full.dropna()
            if len(_trend_valid) >= _slope_lb + 1:
                _trend_slope = (float(_trend_valid.iloc[-1]) - float(_trend_valid.iloc[-_slope_lb - 1])) / _slope_lb
            else:
                _trend_slope = 0.0

            if pa_hist_days is not None:
                _cutoff      = _cs_full.index[-1] - pd.Timedelta(days=pa_hist_days)
                cum_spread   = _cs_full[_cs_full.index >= _cutoff]
                rolling_mean = _rm_full[_rm_full.index >= _cutoff]
                rolling_std  = _rstd_full[_rstd_full.index >= _cutoff]
                trend_mean   = _trend_full[_trend_full.index >= _cutoff]
            else:
                cum_spread, rolling_mean, rolling_std = _cs_full, _rm_full, _rstd_full
                trend_mean = _trend_full

            fig = go.Figure()
            # Background shading by trend direction
            if abs(_trend_slope) >= 0.0001:
                _shade = 'rgba(0,180,0,0.08)' if _trend_slope > 0 else 'rgba(200,0,0,0.08)'
                fig.add_shape(type='rect',
                              xref='x domain', yref='y domain',
                              x0=0, x1=1, y0=0, y1=1,
                              fillcolor=_shade, line_width=0, layer='below')
            fig.add_trace(go.Scatter(x=cum_spread.index, y=cum_spread,
                                     name='Cumulative spread', line=dict(color='#2c6fad')))
            fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean,
                                     name='Rolling mean', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=rolling_mean.index,
                                     y=rolling_mean + float(xing_sd) * rolling_std,
                                     name=f'+{xing_sd} SD',
                                     line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=rolling_mean.index,
                                     y=rolling_mean - float(xing_sd) * rolling_std,
                                     name=f'-{xing_sd} SD',
                                     line=dict(color='green', dash='dot')))
            fig.add_trace(go.Scatter(x=trend_mean.index, y=trend_mean,
                                     name=f'Trend mean ({_trend_win}d)',
                                     line=dict(color='orange', dash='dash')))
            fig.update_layout(title="Cumulative spread with ±SD bands",
                              height=400, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Trend metrics row
            _trend_dir = ('🟢 Long bias'  if _trend_slope >  0.0001
                          else '🔴 Short bias' if _trend_slope < -0.0001
                          else '⚪ Neutral')
            _cur_sd = pa_sig['current_sd']
            if abs(_cur_sd) < float(xing_sd):
                _align_badge = '— No active signal'
            elif _cur_sd > 0 and _trend_slope < 0:
                _align_badge = '✅ Aligned'
            elif _cur_sd > 0 and _trend_slope > 0:
                _align_badge = '⚠️ Counter-trend'
            elif _cur_sd < 0 and _trend_slope > 0:
                _align_badge = '✅ Aligned'
            else:
                _align_badge = '⚠️ Counter-trend'
            tm1, tm2, tm3 = st.columns(3)
            tm1.metric("Trend direction",  _trend_dir)
            tm2.metric("Signal alignment", _align_badge)
            tm3.metric("Trend slope",      f"{_trend_slope:+.4f}")

            # Open position details
            if open_pos is not None:
                st.subheader("Position detail")
                cp = _cached_latest_prices(tuple(open_pos.basket.all_instruments))
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Opened", str(open_pos.entry_date))
                pc2.metric("Days held", open_pos.days_held)
                pc3.metric("Unrealised", f"£{open_pos.live_pnl(cp):+,.0f}")
                pc4.metric("Net P&L", f"£{open_pos.net_pnl(cp):+,.0f}")


            _bt_pair_key = (tuple(basket.long_legs), tuple(basket.short_legs),
                            pa_hist_days, int(pa_trend_window))
            if st.session_state.get('tab2_bt_pair_key') != _bt_pair_key:
                st.session_state.pop('tab2_backtest_result', None)
            if st.button("Run backtest", key="tab2_run_backtest"):
                st.session_state['tab2_bt_pair_key'] = _bt_pair_key
                try:
                    st.session_state['tab2_backtest_result'] = _pair_backtest(
                        tuple(basket.long_legs), tuple(basket.short_legs),
                        int(vol_window), float(xing_sd), float(exit_sd),
                        pa_hist_days,
                    )
                except Exception as e:
                    st.warning(f"Backtest unavailable: {e}")
            if 'tab2_backtest_result' in st.session_state:
                with st.expander("Backtest summary", expanded=True):
                    bt = st.session_state['tab2_backtest_result']
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Trades", f"{bt['n_trades']:,}")
                    bc1.metric("Avg holding", f"{bt['avg_holding']:.0f}d")
                    bc2.metric("Gross WR", f"{bt['gross_wr']:.1%}")
                    bc2.metric("Avg gross",  f"{bt['avg_gross']:+.4f}")
                    bc3.metric("Net WR",   f"{bt['net_wr']:.1%}")
                    bc3.metric("Avg net",  f"{bt['avg_net']:+.4f}")

                    # Post-hoc trend alignment stat
                    _bt_n = bt.get('n_trades', 0)
                    _bt_trades = bt.get('trades_raw')
                    _bt_cum = bt.get('cum_spread')
                    _bt_idx = bt.get('date_index')
                    if _bt_n > 0 and _bt_trades is not None and _bt_cum is not None:
                        from engine.numba_core import COL_ENTRY_IDX as _CEI, COL_SIDE as _CS
                        _tw = int(pa_trend_window)
                        _slb2 = min(20, _tw // 10)
                        _trend_bt = pd.Series(_bt_cum, index=_bt_idx).rolling(_tw, min_periods=10).mean()
                        _trend_arr = _trend_bt.values
                        _entry_idxs = _bt_trades[:, _CEI].astype(int)
                        _sides      = _bt_trades[:, _CS]
                        _prev_idxs  = np.maximum(0, _entry_idxs - _slb2)
                        _has_trend  = (_entry_idxs >= _slb2) & ~np.isnan(_trend_arr[_entry_idxs])
                        _slopes_bt  = np.where(
                            _has_trend,
                            (_trend_arr[_entry_idxs] - _trend_arr[_prev_idxs]) / _slb2,
                            np.nan,
                        )
                        _al_mask = (((_sides > 0) & (_slopes_bt > 0)) |
                                    ((_sides < 0) & (_slopes_bt < 0)))
                        _valid   = ~np.isnan(_slopes_bt)
                        _al_pct  = float(_al_mask[_valid].mean()) if _valid.any() else float('nan')
                        bc4.metric("Trend-aligned trades",
                                   f"{_al_pct:.0%}" if not np.isnan(_al_pct) else "—")

        except Exception as e:
            st.error(f"Signal computation failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Stake Calculator
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Stake Calculator")
    st.caption("Vol-targeted stake sizing across any asset class.")

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
    target_1sd = st.number_input("Target 1 SD exposure (£)", value=500.0, step=50.0,
                                 min_value=50.0, key="sc_target")
    vol_window = st.slider("Vol window (days)", min_value=130, max_value=524,
                            value=262, step=1, key="tab3_vol_window")

    if long_picks and short_picks:
        try:
            basket = Basket(long_legs=long_picks, short_legs=short_picks)
            basket.validate()
        except ValueError as e:
            st.error(str(e))
            basket = None

        if basket is not None:
            vols     = registry.get_vols(basket.all_instruments, window=vol_window)
            scalings = registry.get_scalings(basket.all_instruments, target_vol=0.01, window=vol_window)
            latest   = registry.get_latest_prices(basket.all_instruments)

            rows = []
            total_notional = 0.0
            for inst in basket.all_instruments:
                price   = latest.get(inst, 0.0)
                scaling = scalings.get(inst, 0.0)
                vol     = vols.get(inst, 0.0)
                stake   = (target_1sd * scaling / price) if price > 0 else 0.0
                notional = abs(stake) * price
                total_notional += notional
                side    = 'Long' if inst in basket.long_legs else 'Short'
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            sp_cost   = basket.spread_cost(registry)
            daily_fin = basket.financing_cost_daily()
            margin    = total_notional * get_margin()
            be_days   = sp_cost / daily_fin if daily_fin > 0 else float('inf')

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Est. spread cost", f"{sp_cost:.3%}")
            mc2.metric("Est. daily financing", f"£{daily_fin * target_1sd:,.2f}")
            mc3.metric("Break-even hold", f"{be_days:.0f}d")
            mc4.metric("Margin required", f"£{margin:,.0f}")
    else:
        st.info("Select at least one long and one short instrument.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Portfolio overview
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Portfolio")
    st.caption("Aggregate exposure, correlation and P&L across all open positions.")

    open_positions = portfolio.open_positions
    all_open_instruments = list({
        inst for pos in open_positions for inst in pos.basket.all_instruments
    })
    try:
        current_prices = _cached_latest_prices(tuple(all_open_instruments))
    except Exception:
        current_prices = {}

    unrealised = portfolio.total_unrealised_pnl(current_prices)
    realised   = portfolio.total_realised_pnl()
    fin_drag   = sum(p.financing_cost_to_date() for p in open_positions)
    net_total  = unrealised + realised - fin_drag

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unrealised P&L", f"£{unrealised:+,.0f}")
    m2.metric("Realised P&L",   f"£{realised:+,.0f}")
    m3.metric("Financing drag", f"£{-fin_drag:,.0f}")
    m4.metric("Net P&L (open + closed − financing)", f"£{net_total:+,.0f}")

    st.markdown("---")

    # Section 2 — Position summary
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
                return ''

        def _days_colour(v):
            try:
                v = float(v)
                if v > 180:
                    return 'background-color: #fde4e4'
                if v > 90:
                    return 'background-color: #fff5d6'
            except Exception:
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

    # Section 3 — Cross-pair correlation
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

    # Section 4 — Capital at risk
    st.subheader("Capital at risk")
    car = portfolio.capital_at_risk(current_prices)
    st.metric("Capital at risk (2 SD, 10-day)", f"£{car:,.0f}",
              help="Parametric VaR: 2 SD × √10 loss estimate per open position. "
                   "Does not account for cross-position correlation.")

    st.markdown("---")

    # Section 5 — P&L by asset class
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


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Search (equity-only, ported from legacy with self-contained loader)
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    from engine.backtest import load_asset_prices
    from engine.calculations import returns as _eq_returns, scaling_vectors as _eq_scalings_fn
    from engine.search import (
        METRICS, METRIC_NAMES, estimate_combinations, run_search,
    )
    from engine.scoring import SCORING_MODES
    from engine.saved import load_saved, save_portfolio, delete_portfolio
    from config import ACTIVE_INSTRUMENTS, DISPLAY_NAMES

    st.header("Portfolio Search")
    st.caption("Equity-only. Enumerates long/short combinations across the 12 equity indices.")

    @st.cache_resource
    def _load_equity_data():
        prices, _ = load_asset_prices(_CACHE_DIR / 'prices.csv')
        rets     = _eq_returns(prices)
        scl      = _eq_scalings_fn(prices, rets)
        return prices, rets, scl

    try:
        _eq_prices, _eq_rets, _eq_scl = _load_equity_data()
    except Exception as e:
        st.error(f"Could not load equity data: {e}")
        st.stop()

    # Saved portfolios
    with st.expander("⭐ Saved portfolios"):
        saved_list = load_saved()
        if not saved_list:
            st.caption("No saved portfolios yet — run a search and save promising results.")
        else:
            for idx, entry in enumerate(saved_list):
                c1, c2, c3 = st.columns([5, 5, 1])
                c1.markdown(f"**{entry['name']}**")
                c1.caption(entry['saved_at'])
                c2.caption(f"L: {entry['long_display']}  |  S: {entry['short_display']}")
                if c3.button("🗑", key=f"sv_del_{idx}"):
                    delete_portfolio(entry['name'])
                    st.rerun()
                st.divider()

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.markdown("**Long legs**")
        min_long = st.number_input("Min", 2, 6, 3, key='s_min_long')
        max_long = st.number_input("Max", 2, 6, 4, key='s_max_long')
        if max_long < min_long:
            st.warning("Max must be ≥ Min")
            max_long = int(min_long)
    with pcol2:
        st.markdown("**Short legs**")
        symmetric = st.checkbox("Same as long", value=True, key='s_symmetric')
        if symmetric:
            min_short, max_short = int(min_long), int(max_long)
        else:
            min_short = st.number_input("Min", 2, 6, 3, key='s_min_short')
            max_short = st.number_input("Max", 2, 6, 4, key='s_max_short')
            if max_short < min_short:
                st.warning("Max must be ≥ Min")
                max_short = int(min_short)
    with pcol3:
        st.markdown("**History window**")
        window_opts = {'6 months': 131, '1 year': 262, '2 years': 524, '3 years': 786}
        window_label = st.selectbox("Window", list(window_opts.keys()), index=1, key='s_window')
        window_days = window_opts[window_label]
    with pcol4:
        st.markdown("**Scale**")
        n_combos = estimate_combinations(int(min_long), int(max_long),
                                          int(min_short), int(max_short))
        st.metric("Combinations", f"{n_combos:,}")
        st.caption(f"Est. run time: ~{max(1, int(n_combos / 80_000))}s")

    with st.expander("Metric filters (leave unchecked to rank without filtering)"):
        filter_cols = st.columns(len(METRICS))
        active_filters: dict = {}
        for col, (name, higher_better, default_limit) in zip(filter_cols, METRICS):
            with col:
                use = st.checkbox(name, key=f's_f_use_{name}')
                limit = st.number_input("Limit", value=float(default_limit),
                                        step=0.1, key=f's_f_lim_{name}',
                                        label_visibility='collapsed')
                if use:
                    active_filters[name] = (1 if higher_better else -1, limit)

    sc1, sc2, _ = st.columns([2, 1, 1])
    with sc1:
        scoring_mode = st.selectbox(
            "Ranking method",
            list(SCORING_MODES.keys()),
            format_func=lambda x: SCORING_MODES[x],
            key='s_scoring_mode',
        )
    with sc2:
        s_exit_sd = st.number_input("Exit SD", 0.0, 2.0, 0.0, 0.5, key='s_exit_sd')

    rcol, tcol = st.columns([2, 1])
    run_btn = rcol.button("▶ Run search", type="primary", use_container_width=True, key='s_run')
    top_n   = tcol.number_input("Show top N", 5, 100, 30, key='s_top_n')

    if run_btn:
        progress_bar = st.progress(0.0)
        status_txt   = st.empty()

        def _progress(pct: float):
            progress_bar.progress(pct)
            status_txt.caption(f"Evaluated {pct * 100:.0f}% of {n_combos:,} combinations…")

        with st.spinner("Searching…"):
            results = run_search(
                _eq_rets, _eq_scl,
                min_long_legs=int(min_long), max_long_legs=int(max_long),
                min_short_legs=int(min_short), max_short_legs=int(max_short),
                window_days=window_days,
                filters=active_filters or None,
                top_n=int(top_n),
                progress_cb=_progress,
                scoring_mode=scoring_mode,
                exit_sd=float(s_exit_sd),
            )
        progress_bar.progress(1.0)
        status_txt.empty()
        st.session_state['search_results'] = results
        st.success(f"Found **{len(results)}** portfolios.")

    results = st.session_state.get('search_results')
    if results is not None and not results.empty:
        st.subheader(f"Results (ranked by: {SCORING_MODES.get(scoring_mode, scoring_mode)})")
        _ordered_cols = [
            'Config', 'Long', 'Short',
            'WinRate', 'Expectancy', 'NetExpectancy', 'EstCost', 'AvgHolding',
            'Trades', 'PayoffRatio',
            'ReturnSD', 'TrendVolRatio', 'ReturnTopology', 'FitDataMinMaxSD', 'LastSD',
        ]
        display_cols = [c for c in _ordered_cols if c in results.columns]
        disp = results[display_cols].copy()
        for c in disp.columns:
            if c in ('Config', 'Long', 'Short'):
                continue
            elif c == 'WinRate':
                disp[c] = disp[c].map('{:.1%}'.format)
            elif c == 'Trades':
                disp[c] = disp[c].map('{:.0f}'.format)
            elif c == 'AvgHolding':
                disp[c] = disp[c].map(lambda v: f'{v:.0f}d')
            else:
                disp[c] = disp[c].map('{:.3f}'.format)
        _tbl(disp, show_index=True)

        st.markdown("---")
        st.subheader("Save / launch")
        rank = st.number_input("Rank # to save", 1, len(results), 1, key='s_save_rank')
        row = results.iloc[rank - 1]
        sc1, sc2 = st.columns(2)
        sc1.markdown(f"**Long:** {row['Long']}")
        sc2.markdown(f"**Short:** {row['Short']}")
        save_name = st.text_input("Save label", value=f"{row['Long']} / {row['Short']}",
                                  key='s_save_name', max_chars=80)
        if st.button("💾 Save", key='s_save_btn'):
            save_portfolio(
                name=save_name,
                long_flags=row['_long_flags'],
                short_flags=row['_short_flags'],
                long_display=row['Long'],
                short_display=row['Short'],
                metrics={m: float(row[m]) for m in METRIC_NAMES},
            )
            st.success(f"Saved: **{save_name}**")
    elif results is not None and results.empty:
        st.warning("No portfolios passed the filters.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Live Monitor
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    import datetime as _dt
    import time as _time

    st.header("Live Monitor")

    # ── Refresh controls ─────────────────────────────────────────────────────
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
                _cached_daily_prices.clear()
                _cached_latest_prices.clear()
                st.success("Prices updated.")

    with col_last:
        _csv_path = _CACHE_DIR / 'prices.csv'
        if _csv_path.exists():
            _mtime = _dt.datetime.fromtimestamp(_csv_path.stat().st_mtime)
            st.caption(f"Cache last modified: {_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh toggle
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

    # ── Signal alerts ────────────────────────────────────────────────────────
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

    # ── Open position live cards ──────────────────────────────────────────────
    if not portfolio.open_positions:
        st.info("No open positions. Open a trade via the Journal tab.")
    else:
        st.subheader("Open Positions — Live")

        if 'tab6_intraday' not in st.session_state:
            _intraday_cache: dict[tuple, object] = {}
            for _p in portfolio.open_positions:
                _key = tuple(sorted(_p.basket.all_instruments))
                _intraday_cache[_key] = registry.get_intraday(list(_key), interval='5m')
            st.session_state['tab6_intraday'] = _intraday_cache

        for _pos in portfolio.open_positions:
            _insts = _pos.basket.all_instruments
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
                            _pct = (_price - _entry) / _entry * 100
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

                # P&L summary row
                _pnl     = _pos.live_pnl(_daily_prices)
                _fin     = _pos.financing_cost_to_date()
                _net_pnl = _pos.net_pnl(_daily_prices)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Days held", _pos.days_held)
                c2.metric("Unrealised P&L", f"£{_pnl:+,.0f}")
                c3.metric("Financing drag", f"-£{_fin:,.0f}")
                c4.metric("Net P&L", f"£{_net_pnl:+,.0f}",
                           delta="▲" if _net_pnl > 0 else "▼",
                           delta_color="normal")

    st.divider()

    # ── Market overview ───────────────────────────────────────────────────────
    st.subheader("Market Overview")
    _all_open_insts = list({
        inst
        for pos in portfolio.open_positions
        for inst in pos.basket.all_instruments
    })
    if _all_open_insts:
        _latest = _cached_latest_prices(tuple(_all_open_insts))
        _overview = [
            {
                'Instrument': ALL_DISPLAY.get(inst, inst),
                'Code':       inst,
                'Latest Price': _latest.get(inst, 'N/A'),
            }
            for inst in _all_open_insts
        ]
        st.dataframe(
            pd.DataFrame(_overview).set_index('Code'),
            use_container_width=True,
        )
    else:
        st.caption("No open positions — no instruments to monitor.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — Backtest (ported from legacy, equity + cross-asset)
# ════════════════════════════════════════════════════════════════════════════
with tab8:
    from engine.backtest import (
        load_asset_prices, prepare_returns, run_backtest, run_exhaustive_search,
        regime_split, sensitivity_grid,
        load_cross_asset_prices, prepare_returns_aligned,
    )
    from engine.scoring import SCORING_MODES

    st.header("Backtest")
    st.caption("Crossing-signal backtest across any asset class.")

    bt_mode = st.radio("Search mode", ["Intra-asset", "Cross-asset"],
                       horizontal=True, key='bt_mode')

    if bt_mode == 'Intra-asset':
        bt_col1, bt_col2 = st.columns([2, 1])
        with bt_col1:
            asset_key = st.selectbox(
                "Data source",
                [k for k, _ in ASSET_CLASS_OPTIONS],
                format_func=lambda k: dict(ASSET_CLASS_OPTIONS)[k],
                key='bt_asset',
            )
        with bt_col2:
            uploaded = st.file_uploader("Or upload custom CSV", type=['csv'], key='bt_upload')

        cfg = ASSET_CLASSES[asset_key]
        csv_path = _CACHE_DIR / cfg['csv_file']

        with st.expander("Signal parameters", expanded=True):
            bc1, bc2, bc3, bc4 = st.columns(4)
            bt_vol_window = bc1.number_input("Vol window", 50, 500, 262, key='bt_vol')
            bt_xing_sd    = bc2.number_input("Entry SD", 0.5, 5.0, 2.0, 0.5, key='bt_xing')
            bt_exit_sd    = bc3.number_input("Exit SD", 0.0, 2.0, 0.0, 0.5, key='bt_exit')
            bt_fin_rate   = bc4.number_input(
                "Financing %pa", 0.0, 10.0,
                cfg['financing']['long_rate'] * 100, 0.1, key='bt_fin',
            )

        st.markdown("**Basket configuration**")
        bk1, bk2, bk3, bk4 = st.columns(4)
        with bk1:
            st.markdown("**Long legs**")
            bt_min_long = st.number_input("Min", 1, 6, 3, key='bt_min_long')
            bt_max_long = st.number_input("Max", 1, 6, 3, key='bt_max_long')
        with bk2:
            st.markdown("**Short legs**")
            bt_symmetric = st.checkbox("Same as long", value=True, key='bt_symmetric')
            if bt_symmetric:
                bt_min_short, bt_max_short = int(bt_min_long), int(bt_max_long)
            else:
                bt_min_short = st.number_input("Min", 1, 6, 3, key='bt_min_short')
                bt_max_short = st.number_input("Max", 1, 6, 3, key='bt_max_short')
        with bk3:
            bt_sample = st.number_input(
                "Sample size (0 = exhaustive)", 0, 50000, 2000, key='bt_sample',
            )
            bt_top_n = st.number_input("Show top N", 5, 100, 30, key='bt_top_n')
        with bk4:
            st.markdown("**History window**")
            bt_window_opts = {'1 year': 262, '2 years': 524, '3 years': 786,
                              '5 years': 1310, 'All': 0}
            bt_window_label = st.selectbox("Window", list(bt_window_opts.keys()),
                                           index=2, key='bt_window')
            bt_window_days = bt_window_opts[bt_window_label] or None

        bt_score_col, _ = st.columns([2, 2])
        with bt_score_col:
            bt_scoring_mode = st.selectbox(
                "Ranking method",
                list(SCORING_MODES.keys()),
                format_func=lambda x: SCORING_MODES[x],
                key='bt_scoring_mode',
            )

        st.markdown("---")
        bt_run = st.button("▶ Run backtest", type="primary",
                           use_container_width=True, key='bt_run')

        if bt_run:
            if not csv_path.exists() and uploaded is None:
                st.error(f"No data file at `{csv_path}`. Upload a CSV or add the file.")
                st.stop()

            with st.spinner("Loading prices…"):
                try:
                    if uploaded is not None:
                        import io
                        _raw = pd.read_csv(io.BytesIO(uploaded.read()),
                                           index_col='Date', parse_dates=True)
                        _raw = _raw.ffill(limit=3).dropna(how='all')
                        _instruments_bt = list(_raw.columns)
                        _prices_bt = _raw
                    else:
                        _prices_bt, _instruments_bt = load_asset_prices(csv_path)
                except Exception as e:
                    st.error(f"Failed to load prices: {e}")
                    st.stop()

            with st.spinner("Preparing returns…"):
                _scaled_bt, _day_ints_bt, _index_bt = prepare_returns(
                    _prices_bt, _instruments_bt,
                    vol_window=int(bt_vol_window),
                    window_days=bt_window_days,
                )

            with st.spinner("Running exhaustive search…"):
                _bt_prog = st.progress(0.0)
                _bt_stat = st.empty()
                def _bt_progress(p):
                    _bt_prog.progress(p)
                    _bt_stat.caption(f"Backtested {p * 100:.0f}% of combinations…")

                _latest_px = dict(zip(_instruments_bt, _prices_bt.iloc[-1].values))
                _spread_lookup = get_spread_cost_lookup(_instruments_bt, _latest_px, asset_key)
                _fin_daily = bt_fin_rate / 100 / 365

                _bt_results = run_exhaustive_search(
                    _scaled_bt, _day_ints_bt, _instruments_bt,
                    display_names=cfg.get('instruments', {}),
                    min_long_legs=int(bt_min_long), max_long_legs=int(bt_max_long),
                    min_short_legs=int(bt_min_short), max_short_legs=int(bt_max_short),
                    vol_window=int(bt_vol_window),
                    xing_sd=float(bt_xing_sd), exit_sd=float(bt_exit_sd),
                    spread_cost_lookup=_spread_lookup,
                    financing_daily_pct=_fin_daily,
                    top_n=int(bt_top_n), sample_n=int(bt_sample),
                    scoring_mode=bt_scoring_mode,
                    progress_cb=_bt_progress,
                )
            _bt_prog.progress(1.0)
            _bt_stat.empty()
            st.session_state['bt_results_cache'] = {
                'df':        _bt_results,
                'asset_key': asset_key,
                'params': {
                    'vol_window': int(bt_vol_window),
                    'xing_sd':    float(bt_xing_sd),
                    'exit_sd':    float(bt_exit_sd),
                    'fin_rate':   float(bt_fin_rate),
                },
            }

        _bt_cache = st.session_state.get('bt_results_cache')
        if _bt_cache is not None:
            _bt_df = _bt_cache['df']
            if _bt_df.empty:
                st.warning("No combinations produced trades.")
            else:
                st.success(f"Found **{len(_bt_df)}** results.")
                _top = _bt_df.iloc[0]
                _m1, _m2, _m3, _m4, _m5 = st.columns(5)
                _m1.metric("Top Trades",   f"{int(_top.get('Trades', 0)):,}")
                _m2.metric("Win Rate",     f"{float(_top.get('WinRate', 0)):.1%}")
                _m3.metric("Expectancy",   f"{float(_top.get('Expectancy', 0)):.3f}")
                _m4.metric("Avg Holding",  f"{float(_top.get('AvgHolding', 0)):.0f}d")
                _m5.metric("Payoff Ratio", f"{float(_top.get('PayoffRatio', 0)):.2f}")

                st.subheader(f"Results (ranked by: {SCORING_MODES.get(bt_scoring_mode, bt_scoring_mode)})")
                _bt_ordered = [
                    'Config', 'Long', 'Short',
                    'WinRate', 'Expectancy', 'NetExpectancy', 'EstCost', 'AvgHolding',
                    'Trades', 'PayoffRatio',
                    'ReturnSD', 'TrendVolRatio', 'ReturnTopology', 'FitDataMinMaxSD', 'LastSD',
                ]
                _bt_display_cols = [c for c in _bt_ordered if c in _bt_df.columns]
                _bt_disp = _bt_df[_bt_display_cols].copy()
                for _c in _bt_disp.columns:
                    if _c in ('Long', 'Short', 'Config'):
                        continue
                    elif _c == 'WinRate':
                        _bt_disp[_c] = _bt_disp[_c].map('{:.1%}'.format)
                    elif _c == 'Trades':
                        _bt_disp[_c] = _bt_disp[_c].map('{:.0f}'.format)
                    elif _c == 'AvgHolding':
                        _bt_disp[_c] = _bt_disp[_c].map(lambda v: f'{v:.0f}d')
                    else:
                        _bt_disp[_c] = _bt_disp[_c].map('{:.3f}'.format)
                _tbl(_bt_disp, show_index=True)

    else:  # Cross-asset mode
        ca_c1, ca_c2 = st.columns(2)
        _long_class  = ca_c1.selectbox(
            "Long-side asset class",
            [k for k, _ in ASSET_CLASS_OPTIONS],
            format_func=lambda k: dict(ASSET_CLASS_OPTIONS)[k],
            key='bt_ca_long_class',
        )
        _short_class = ca_c2.selectbox(
            "Short-side asset class",
            [k for k, _ in ASSET_CLASS_OPTIONS], index=2,
            format_func=lambda k: dict(ASSET_CLASS_OPTIONS)[k],
            key='bt_ca_short_class',
        )

        with st.expander("Signal parameters", expanded=True):
            cc1, cc2, cc3, cc4 = st.columns(4)
            _ca_vol  = cc1.number_input("Vol window", 50, 500, 262, key='bt_ca_vol')
            _ca_xing = cc2.number_input("Entry SD", 0.5, 5.0, 2.0, 0.5, key='bt_ca_xing')
            _ca_exit = cc3.number_input("Exit SD", 0.0, 2.0, 1.0, 0.5, key='bt_ca_exit')
            _long_cfg = ASSET_CLASSES[_long_class]
            _ca_fin   = cc4.number_input(
                "Financing %pa", 0.0, 10.0,
                _long_cfg['financing']['long_rate'] * 100, 0.1, key='bt_ca_fin',
            )

        with st.expander("Basket configuration", expanded=True):
            bc1, bc2, bc3, bc4 = st.columns(4)
            _ca_nl_min = bc1.number_input("Min long legs",  1, 4, 1, key='bt_ca_nl_min')
            _ca_nl_max = bc2.number_input("Max long legs",  1, 4, 1, key='bt_ca_nl_max')
            _ca_ns_min = bc3.number_input("Min short legs", 1, 4, 1, key='bt_ca_ns_min')
            _ca_ns_max = bc4.number_input("Max short legs", 1, 4, 1, key='bt_ca_ns_max')
            _ca_top_n  = st.number_input("Top N results", 10, 200, 50, key='bt_ca_top_n')
            _ca_sample = st.number_input(
                "Sample size (0 = exhaustive)", 0, 20000, 0, key='bt_ca_sample',
            )
            _ca_scoring = st.selectbox(
                "Scoring mode", list(SCORING_MODES.keys()),
                format_func=lambda x: SCORING_MODES[x], key='bt_ca_scoring',
            )

        st.markdown("---")
        _ca_run = st.button("▶ Run cross-asset search", type="primary",
                            use_container_width=True, key='bt_ca_run')

        if _ca_run:
            try:
                with st.spinner(f"Loading {_long_class}/{_short_class} prices..."):
                    _ca_prices, _ca_long_i, _ca_short_i, _ = load_cross_asset_prices(
                        _long_class, _short_class, _CACHE_DIR,
                    )
                st.info(
                    f"Loaded {len(_ca_long_i)} {_long_class} + "
                    f"{len(_ca_short_i)} {_short_class} instruments. "
                    f"Common trading days: {len(_ca_prices):,} "
                    f"({_ca_prices.index[0].date()} – {_ca_prices.index[-1].date()})"
                )

                with st.spinner("Preparing vol-scaled returns..."):
                    _ca_long_sc, _ca_short_sc, _ca_day_ints, _ca_idx = prepare_returns_aligned(
                        _ca_prices, _ca_long_i, _ca_short_i, vol_window=int(_ca_vol),
                    )
                _ca_all_instr  = _ca_long_i + _ca_short_i
                _ca_all_scaled = np.concatenate([_ca_long_sc, _ca_short_sc], axis=1)
                _ca_latest = _ca_prices.iloc[-1].to_dict()
                _ca_lookup = {
                    **get_spread_cost_lookup(_ca_long_i,  _ca_latest, _long_class),
                    **get_spread_cost_lookup(_ca_short_i, _ca_latest, _short_class),
                }
                _ca_fin_daily = (_ca_fin / 100) / 365

                _ca_prog = st.progress(0.0, text="Searching...")
                with st.spinner("Running cross-asset search..."):
                    _ca_df = run_exhaustive_search(
                        _ca_all_scaled, _ca_day_ints, _ca_all_instr,
                        long_instrument_subset=_ca_long_i,
                        short_instrument_subset=_ca_short_i,
                        min_long_legs=int(_ca_nl_min), max_long_legs=int(_ca_nl_max),
                        min_short_legs=int(_ca_ns_min), max_short_legs=int(_ca_ns_max),
                        vol_window=int(_ca_vol),
                        xing_sd=float(_ca_xing), exit_sd=float(_ca_exit),
                        spread_cost_lookup=_ca_lookup,
                        financing_daily_pct=_ca_fin_daily,
                        top_n=int(_ca_top_n), sample_n=int(_ca_sample),
                        scoring_mode=_ca_scoring,
                        progress_cb=lambda p: _ca_prog.progress(p, text=f"Searching... {p:.0%}"),
                    )
                _ca_prog.progress(1.0, text="Done")

                if _ca_df.empty:
                    st.warning("No results found.")
                else:
                    _ca_net_pos = (_ca_df['NetExpectancy'] > 0).sum()
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("Results",   len(_ca_df))
                    cm2.metric("Net+ pairs", f"{_ca_net_pos} ({_ca_net_pos/len(_ca_df):.0%})")
                    cm3.metric("Best net expectancy",
                               f"{_ca_df['NetExpectancy'].max()*100:+.2f}%")
                    cm4.metric("Avg holding (top 10)",
                               f"{_ca_df.head(10)['AvgHolding'].mean():.0f}d")

                    st.subheader(
                        f"Results — {_long_class.title()} vs {_short_class.title()}"
                    )
                    _ca_disp = _ca_df.drop(
                        columns=[c for c in _ca_df.columns if c.startswith('_')],
                        errors='ignore',
                    ).copy()
                    _pct = {'WinRate', 'Expectancy', 'NetExpectancy', 'SpreadCost', 'FinCost'}
                    for _c in _ca_disp.columns:
                        if _c in ('Config', 'Long', 'Short'):
                            continue
                        elif _c in _pct:
                            _ca_disp[_c] = _ca_disp[_c].map('{:.1%}'.format)
                        elif _c == 'AvgHolding':
                            _ca_disp[_c] = _ca_disp[_c].map(lambda v: f'{v:.0f}d')
                        elif _c == 'Trades':
                            _ca_disp[_c] = _ca_disp[_c].map('{:.0f}'.format)
                        else:
                            _ca_disp[_c] = _ca_disp[_c].map('{:.3f}'.format)
                    _tbl(_ca_disp, show_index=True)

            except FileNotFoundError as e:
                st.error(str(e))
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                raise


# ════════════════════════════════════════════════════════════════════════════
# TAB 9 — Walk-Forward Validation
# ════════════════════════════════════════════════════════════════════════════
with tab9:
    from engine.walkforward import run_walk_forward, summarise_walk_forward
    from engine.backtest import load_asset_prices as _wf_load_asset_prices
    from engine.scoring import SCORING_MODES as _WF_SCORING_MODES

    st.header("Walk-Forward Validation")
    st.caption(
        "Tests whether the selected scoring mode predicts out-of-sample performance. "
        "Reproduces the Q11 protocol: score all 1v1 pairs on IS data, evaluate OOS, "
        "compute Spearman ρ(IS rank, OOS gross return)."
    )

    wf_col1, wf_col2, wf_col3 = st.columns(3)
    with wf_col1:
        st.markdown("**Window lengths (trading years)**")
        wf_is_years  = st.number_input("IS window (years)",  3, 10, 5, key='wf_is')
        wf_oos_years = st.number_input("OOS window (years)", 1,  5, 2, key='wf_oos')
        wf_step      = st.number_input("Step size (years)",  1,  3, 1, key='wf_step')
    with wf_col2:
        st.markdown("**Signal parameters**")
        wf_xing_sd = st.number_input("Entry SD", 0.5, 5.0, 2.0, 0.5, key='wf_xing')
        wf_exit_sd = st.number_input("Exit SD",  0.0, 2.0, 0.0, 0.5, key='wf_exit')
        wf_vol_win = st.number_input("Vol window", 100, 500, 262, key='wf_vol')
    with wf_col3:
        st.markdown("**Scoring & data**")
        wf_scoring = st.selectbox(
            "Scoring mode", list(_WF_SCORING_MODES.keys()),
            format_func=lambda x: _WF_SCORING_MODES[x], key='wf_scoring',
        )
        wf_asset = st.selectbox(
            "Asset class", [k for k, _ in ASSET_CLASS_OPTIONS],
            format_func=lambda k: dict(ASSET_CLASS_OPTIONS)[k], key='wf_asset',
        )

    _wf_cfg = ASSET_CLASSES[wf_asset]
    _wf_csv = _CACHE_DIR / _wf_cfg['csv_file']

    st.markdown("---")
    wf_run = st.button("▶ Run walk-forward", type="primary",
                       use_container_width=True, key='wf_run')

    if wf_run:
        if not _wf_csv.exists():
            st.error(f"No data file at `{_wf_csv}`. Run the Backtest tab first to cache this asset class.")
            st.stop()

        with st.spinner("Loading prices…"):
            try:
                _wf_prices, _wf_instruments = _wf_load_asset_prices(_wf_csv)
            except Exception as e:
                st.error(f"Failed to load prices: {e}")
                st.stop()

        if wf_asset == 'fixed_income':
            _wf_instruments = [i for i in _wf_instruments if i not in FI_EXCLUDE]
            _wf_prices = _wf_prices[[c for c in _wf_prices.columns if c not in FI_EXCLUDE]]

        _n_windows_est = max(0, (len(_wf_prices) - wf_is_years * 262 - wf_oos_years * 262)
                             // (wf_step * 262))
        _n_pairs_est = len(_wf_instruments) * (len(_wf_instruments) - 1)
        st.caption(f"~{_n_windows_est} windows × {_n_pairs_est} pairs = "
                   f"~{_n_windows_est * _n_pairs_est:,} observations")

        _wf_prog = st.progress(0.0)
        _wf_stat = st.empty()
        def _wf_progress(pct: float):
            _wf_prog.progress(pct)
            _wf_stat.caption(f"Walk-forward: {pct * 100:.0f}% complete…")

        with st.spinner("Running walk-forward…"):
            _wf_results = run_walk_forward(
                _wf_prices, _wf_instruments,
                is_years=int(wf_is_years), oos_years=int(wf_oos_years),
                step_years=int(wf_step),
                scoring_mode=wf_scoring,
                vol_window=int(wf_vol_win),
                xing_sd=float(wf_xing_sd), exit_sd=float(wf_exit_sd),
                progress_cb=_wf_progress,
            )
        _wf_prog.progress(1.0)
        _wf_stat.empty()
        st.session_state['wf_results_cache'] = {
            'results':      _wf_results,
            'scoring_mode': wf_scoring,
            'asset':        wf_asset,
        }

    _wf_cache_data = st.session_state.get('wf_results_cache')
    if _wf_cache_data is not None:
        _wf_df   = _wf_cache_data['results']
        _wf_mode = _wf_cache_data['scoring_mode']
        if _wf_df.empty:
            st.warning("No results — insufficient data for the selected window lengths.")
        else:
            _wf_sum = summarise_walk_forward(_wf_df)
            rho, pval, n_obs = _wf_sum['rho'], _wf_sum['p_value'], _wf_sum['n_obs']
            st.subheader("Summary")
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Spearman ρ (IS rank vs OOS gross)", f"{rho:+.3f}")
            sm2.metric("p-value", f"{pval:.4f}")
            sm3.metric("Observations", f"{n_obs:,}")

            if abs(rho) < 0.1 and pval >= 0.05:
                st.warning(f"**No predictive power** — ρ = {rho:+.3f}, p = {pval:.4f}.")
            elif rho > 0.1 and pval < 0.05:
                st.success(f"**Positive predictive power** — ρ = {rho:+.3f}, p = {pval:.4f}.")
            elif rho < -0.1 and pval < 0.05:
                st.error(f"**Negative predictor** — ρ = {rho:+.3f}, p = {pval:.4f}.")
            else:
                st.info(f"ρ = {rho:+.3f}, p = {pval:.4f} — weak or marginal result.")

            if not _wf_sum['quintile_df'].empty:
                st.subheader("OOS Performance by IS Quintile")
                _q = _wf_sum['quintile_df'].copy()
                for _c in ['OOS_GrossWR']:
                    _q[_c] = _q[_c].map('{:.1%}'.format)
                for _c in ['OOS_Gross', 'OOS_Net']:
                    _q[_c] = _q[_c].map('{:.4f}'.format)
                for _c in ['OOS_AvgHold']:
                    _q[_c] = _q[_c].map('{:.0f}d'.format)
                _tbl(_q, show_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 10 — Scenario Scanner
# ════════════════════════════════════════════════════════════════════════════
with tab10:
    from engine.backtest import (
        prepare_returns as _sc_prep, run_backtest as _sc_bt,
        aggregate_trades as _sc_agg,
    )
    from engine.numba_core import COL_ENTRY_IDX as _SC_CEI, COL_SIDE as _SC_CS

    st.header("Scenario Scanner")
    st.caption(
        "Grid sweep: crossing signal backtest across asset classes and parameter combinations. "
        "Buckets results by average holding period and ranks by net expectancy."
    )

    # ── Price loading helper (cached, read-only) ──────────────────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _sc_load_prices(asset_key: str) -> tuple[pd.DataFrame, list[str]]:
        """Return (prices_df, valid_instruments) for one asset class."""
        _insts = [c for c in get_tradeable_instruments(asset_key) if c not in FI_EXCLUDE]
        _df    = registry.get_daily_prices(_insts)
        _valid = [c for c in _insts if c in _df.columns]
        return _df, _valid

    # ── Scope selectors ───────────────────────────────────────────────────
    st.subheader("Scope")
    _sc1, _sc2 = st.columns(2)
    with _sc1:
        st.markdown("**Intra-asset**")
        _sc_comm_intra = st.checkbox("Commodities",   key='sc_comm_intra')
        _sc_fx_intra   = st.checkbox("FX",            key='sc_fx_intra')
        _sc_fi_intra   = st.checkbox("Fixed Income",  key='sc_fi_intra')
    with _sc2:
        st.markdown("**Cross-asset**")
        _sc_comm_fx  = st.checkbox("Commodities × FX",           key='sc_comm_fx')
        _sc_comm_eq  = st.checkbox("Commodities × Equity",       key='sc_comm_eq')
        _sc_comm_fi  = st.checkbox("Commodities × Fixed Income", key='sc_comm_fi')
        _sc_fx_eq    = st.checkbox("FX × Equity",                key='sc_fx_eq')

    # ── Parameter grid ────────────────────────────────────────────────────
    st.subheader("Parameter grid")
    _pg1, _pg2, _pg3 = st.columns(3)
    with _pg1:
        _sc_entry_sds = st.multiselect(
            "Entry SD", [2.0, 2.5, 3.0], default=[2.0, 2.5, 3.0], key='sc_entry_sds',
        )
    with _pg2:
        _sc_exit_sds = st.multiselect(
            "Exit SD", [0.5, 1.0, 1.5], default=[0.5, 1.0, 1.5], key='sc_exit_sds',
        )
    with _pg3:
        _sc_vol_wins = st.multiselect(
            "Vol window (days)", [50, 130, 262], default=[50, 130, 262], key='sc_vol_wins',
        )

    _pg4, _pg5, _pg6 = st.columns(3)
    with _pg4:
        _sc_fin_rate = st.number_input(
            "Financing %pa", 0.0, 20.0, 4.88, 0.01, key='sc_fin_rate',
        )
    with _pg5:
        _sc_win_opts  = {'3 years': 786, '5 years': 1310, 'All': 0}
        _sc_win_label = st.selectbox(
            "History window", list(_sc_win_opts.keys()), index=2, key='sc_window',
        )
        _sc_win_days = _sc_win_opts[_sc_win_label] or None
    with _pg6:
        _sc_min_trades = st.number_input(
            "Min trades floor", 1, 100, 10, key='sc_min_trades',
        )

    _pg7, _pg8, _ = st.columns(3)
    with _pg7:
        _sc_trend_win = int(st.number_input(
            "Trend filter window (days)", 130, 756, 262, key='sc_trend_win',
        ))
    with _pg8:
        _sc_trend_mode = st.selectbox(
            "Trend filter mode",
            ["Off", "Aligned only", "Show alignment column"],
            index=1, key='sc_trend_mode',
        )

    _sc_run = st.button(
        "▶ Run scenario scan", type="primary", use_container_width=True, key='sc_run',
    )

    # ── Computation ───────────────────────────────────────────────────────
    if _sc_run:
        _scope_defs: list[tuple[str, str, bool, str]] = []
        if _sc_comm_intra: _scope_defs.append(('commodities',  'commodities',  True,  'Commodity'))
        if _sc_fx_intra:   _scope_defs.append(('fx',           'fx',           True,  'FX'))
        if _sc_fi_intra:   _scope_defs.append(('fixed_income', 'fixed_income', True,  'Fixed Income'))
        if _sc_comm_fx:    _scope_defs.append(('commodities',  'fx',           False, 'Commodity vs FX'))
        if _sc_comm_eq:    _scope_defs.append(('commodities',  'equity',       False, 'Commodity vs Equity'))
        if _sc_comm_fi:    _scope_defs.append(('commodities',  'fixed_income', False, 'Commodity vs Fixed Income'))
        if _sc_fx_eq:      _scope_defs.append(('fx',           'equity',       False, 'FX vs Equity'))

        if not _scope_defs:
            st.warning("Select at least one asset class scope.")
        elif not _sc_entry_sds or not _sc_exit_sds or not _sc_vol_wins:
            st.warning("Select at least one value for each parameter in the grid.")
        else:
            _sc_param_combos = [
                (float(e), float(x), int(v))
                for e in _sc_entry_sds
                for x in _sc_exit_sds
                for v in _sc_vol_wins
            ]
            _sc_fin_daily = _sc_fin_rate / 100.0 / 365.0

            # Build scope → pairs list (instrument lookup only — no price loading yet)
            _sc_scope_work: list[tuple] = []
            for _lk, _sk, _intra, _label in _scope_defs:
                _l_insts = [c for c in get_tradeable_instruments(_lk) if c not in FI_EXCLUDE]
                _s_insts = [c for c in get_tradeable_instruments(_sk) if c not in FI_EXCLUDE]
                if _intra:
                    _pairs = list(combinations(_l_insts, 2))
                else:
                    _pairs = [(a, b) for a in _l_insts for b in _s_insts]
                _sc_scope_work.append((_lk, _sk, _intra, _label, _l_insts, _s_insts, _pairs))

            _sc_total_pairs = sum(len(w[6]) for w in _sc_scope_work)
            if _sc_total_pairs == 0:
                st.warning("No instrument pairs found for the selected scope.")
            else:
                _sc_rows: list[dict] = []
                _sc_prog  = st.progress(0.0)
                _sc_pstat = st.empty()
                _sc_done  = 0

                for _lk, _sk, _intra, _label, _l_insts, _s_insts, _pairs in _sc_scope_work:
                    _l_px, _ = _sc_load_prices(_lk)
                    _s_px, _ = _sc_load_prices(_sk) if not _intra else (_l_px, _l_insts)

                    _sc_slb = min(20, _sc_trend_win // 10)

                    for (_long_i, _short_i) in _pairs:
                        if _long_i not in _l_px.columns or _short_i not in _s_px.columns:
                            _sc_done += 1
                            continue

                        if _intra:
                            _pair_df = _l_px[[_long_i, _short_i]].dropna()
                        else:
                            _pair_df = pd.concat(
                                [_l_px[[_long_i]], _s_px[[_short_i]]],
                                axis=1, join='inner',
                            ).dropna()

                        if len(_pair_df) < 130:
                            _sc_done += 1
                            continue

                        # Round-trip spread cost: 4 × mean(spread_pct_long, spread_pct_short)
                        _lc_cfg = ASSET_CLASSES[_lk]['instruments'].get(_long_i,  {})
                        _sc_cfg = ASSET_CLASSES[_sk]['instruments'].get(_short_i, {})
                        _l_sp = _lc_cfg.get('spread_pct', 0.001) if isinstance(_lc_cfg, dict) else 0.001
                        _s_sp = _sc_cfg.get('spread_pct', 0.001) if isinstance(_sc_cfg, dict) else 0.001
                        _pair_cost = 2.0 * (_l_sp + _s_sp)

                        _l_disp = get_display_name(_lk, _long_i)
                        _s_disp = get_display_name(_sk, _short_i)

                        # Trend series — computed once per pair on raw log-return spread
                        # (independent of vol_window, so valid across all param combos)
                        _raw_lr = (np.log(_pair_df[_long_i] / _pair_df[_short_i])
                                   .diff().fillna(0))
                        _raw_cum_spr = _raw_lr.cumsum()
                        _trend_ser_pair = _raw_cum_spr.rolling(
                            _sc_trend_win, min_periods=10
                        ).mean()
                        _trend_arr_pair = _trend_ser_pair.values

                        for (_e_sd, _x_sd, _v_win) in _sc_param_combos:
                            try:
                                _sc_scaled, _sc_day_ints, _sc_index = _sc_prep(
                                    _pair_df, [_long_i, _short_i],
                                    vol_window=_v_win,
                                    window_days=_sc_win_days,
                                )
                                if _sc_scaled.shape[0] < _v_win:
                                    continue
                                _sc_spread = _sc_scaled[:, 0] - _sc_scaled[:, 1]
                                _sc_bt_res = _sc_bt(
                                    _sc_spread, _sc_day_ints,
                                    vol_window=_v_win,
                                    xing_sd=_e_sd, exit_sd=_x_sd,
                                    spread_cost_pct=_pair_cost,
                                    financing_daily_pct=_sc_fin_daily,
                                    n_legs=2,
                                )
                                _n_tr = _sc_bt_res['n_trades']
                                _sc_s = _sc_bt_res['summary']
                                _use_s = _sc_s
                                _use_n = _n_tr

                                # Post-hoc trend alignment (vectorised)
                                _al_pct = float('nan')
                                if _n_tr > 0:
                                    _raw_t = _sc_bt_res['trades_raw'][:_n_tr]
                                    _eidxs = _raw_t[:, _SC_CEI].astype(int)
                                    _sides = _raw_t[:, _SC_CS]
                                    # Map scaled-array indices → pair_df positions via date
                                    _edates = _sc_index[_eidxs]
                                    _tipos  = _trend_ser_pair.index.get_indexer(
                                        _edates, method='nearest'
                                    )
                                    _prev_p = np.maximum(0, _tipos - _sc_slb)
                                    _has_tr = (_tipos >= _sc_slb) & ~np.isnan(_trend_arr_pair[_tipos])
                                    _slp = np.where(
                                        _has_tr,
                                        (_trend_arr_pair[_tipos] - _trend_arr_pair[_prev_p]) / _sc_slb,
                                        np.nan,
                                    )
                                    _al = (((_sides > 0) & (_slp > 0)) |
                                           ((_sides < 0) & (_slp < 0)))
                                    _vld = ~np.isnan(_slp)
                                    if _vld.any():
                                        _al_pct = float(_al[_vld].mean())

                                    if _sc_trend_mode == "Aligned only":
                                        _al_filter = _al & _vld
                                        _n_al = int(_al_filter.sum())
                                        if _n_al < int(_sc_min_trades):
                                            continue
                                        _use_s = _sc_agg(
                                            _raw_t[_al_filter], _n_al,
                                            spread_cost_pct=_pair_cost,
                                            financing_daily_pct=_sc_fin_daily,
                                            n_legs=2,
                                        )
                                        _use_n = _n_al

                                _sc_rows.append({
                                    '_long':         _long_i,
                                    '_short':        _short_i,
                                    'Long':          _l_disp,
                                    'Short':         _s_disp,
                                    'Asset Classes': _label,
                                    'Entry SD':      _e_sd,
                                    'Exit SD':       _x_sd,
                                    'Vol Window':    _v_win,
                                    'Trend Window':  f'{_sc_trend_win}d',
                                    'Trades':        int(_use_n),
                                    'Gross WR%':     float(_use_s.get('gross_wr',    0.0)),
                                    'Net WR%':       float(_use_s.get('net_wr',      0.0)),
                                    'Avg Gross':     float(_use_s.get('avg_gross',   0.0)),
                                    'Avg Net':       float(_use_s.get('avg_net',     0.0)),
                                    'Avg Hold':      float(_use_s.get('avg_holding', 0.0)),
                                    'Est Cost':      _pair_cost,
                                    'Aligned%':      _al_pct,
                                })
                            except Exception:
                                continue

                        _sc_done += 1
                        _sc_prog.progress(min(1.0, _sc_done / _sc_total_pairs))
                        _sc_pstat.caption(
                            f"Scanning… {_sc_done:,} / {_sc_total_pairs:,} pairs"
                        )

                _sc_prog.progress(1.0)
                _sc_pstat.empty()

                if _sc_rows:
                    _sc_full_df = pd.DataFrame(_sc_rows)
                    st.session_state['scenario_results'] = _sc_full_df
                    st.session_state['scenario_params'] = (
                        f"Entry SD: {_sc_entry_sds}  |  Exit SD: {_sc_exit_sds}  |  "
                        f"Vol window: {_sc_vol_wins}  |  History: {_sc_win_label}  |  "
                        f"Financing: {_sc_fin_rate:.2f}%pa  |  Min trades: {int(_sc_min_trades)}"
                    )
                    st.success(
                        f"Scan complete — {len(_sc_full_df):,} total results across "
                        f"{_sc_total_pairs:,} pairs × {len(_sc_param_combos)} parameter combos."
                    )
                else:
                    st.info("No results — try relaxing parameters or adding more scopes.")

    # ── Results display ───────────────────────────────────────────────────
    if 'scenario_results' in st.session_state:
        _sr_full = st.session_state['scenario_results']
        st.caption(f"**Last run:** {st.session_state.get('scenario_params', '')}")
        st.divider()

        # Apply display filters
        _sr = _sr_full[_sr_full['Trades'] >= int(_sc_min_trades)].copy()
        _sr = _sr[_sr['Avg Net'] > 0].copy()

        if _sr.empty:
            st.info(
                f"No results pass filters (trades ≥ {int(_sc_min_trades)}, avg net > 0). "
                "Use the CSV download below for the full unfiltered dataset."
            )
        else:
            def _sc_bucket(h: float) -> str:
                if h <= 45:  return '🟢 Short'
                if h <= 90:  return '🟡 Medium'
                return '🔴 Long'

            _sr['_bucket'] = _sr['Avg Hold'].apply(_sc_bucket)

            for _bkt, _desc in [
                ('🟢 Short',  '≤ 45d'),
                ('🟡 Medium', '46–90d'),
                ('🔴 Long',   '> 90d'),
            ]:
                _bdf = (
                    _sr[_sr['_bucket'] == _bkt]
                    .sort_values('Avg Net', ascending=False)
                    .reset_index(drop=True)
                )
                with st.expander(
                    f"{_bkt} ({_desc}) — {len(_bdf)} pairs found", expanded=True
                ):
                    if _bdf.empty:
                        st.caption("No results in this bucket.")
                        continue

                    _bdf.insert(0, 'Rank', range(1, len(_bdf) + 1))
                    _disp_cols = [
                        'Rank', 'Long', 'Short', 'Asset Classes',
                        'Entry SD', 'Exit SD', 'Vol Window', 'Trend Window', 'Trades',
                        'Gross WR%', 'Net WR%', 'Avg Gross', 'Avg Net',
                        'Avg Hold', 'Est Cost', 'Aligned%',
                    ]
                    _tbl_df = _bdf[_disp_cols].copy()
                    _tbl_df['Gross WR%'] = _tbl_df['Gross WR%'].map('{:.1%}'.format)
                    _tbl_df['Net WR%']   = _tbl_df['Net WR%'].map('{:.1%}'.format)
                    _tbl_df['Avg Gross'] = _tbl_df['Avg Gross'].map('{:+.4f}'.format)
                    _tbl_df['Avg Net']   = _tbl_df['Avg Net'].map('{:+.4f}'.format)
                    _tbl_df['Avg Hold']  = _tbl_df['Avg Hold'].map('{:.0f}d'.format)
                    _tbl_df['Est Cost']  = _tbl_df['Est Cost'].map('{:.3%}'.format)
                    _tbl_df['Aligned%']  = _tbl_df['Aligned%'].apply(
                        lambda v: f'{v:.0%}' if not (isinstance(v, float) and np.isnan(v)) else '—'
                    )

                    st.dataframe(_tbl_df, use_container_width=True, hide_index=True)

                    # Open in Pair Analysis — selectbox + button
                    _pair_labels = [
                        f"#{r['Rank']} — {r['Long']} / {r['Short']}"
                        for _, r in _bdf.iterrows()
                    ]
                    _sel = st.selectbox(
                        "Open a pair in Pair Analysis →",
                        ['— select a pair —'] + _pair_labels,
                        key=f'sc_sel_{_bkt}',
                    )
                    if _sel != '— select a pair —' and st.button(
                        "Open in Pair Analysis →", key=f'sc_open_{_bkt}',
                    ):
                        _row = _bdf.iloc[_pair_labels.index(_sel)]
                        st.session_state['pa_long']  = [_row['_long']]
                        st.session_state['pa_short'] = [_row['_short']]
                        st.session_state['pa_pair']  = '— Custom pair —'
                        st.toast(
                            f"Loaded {_row['Long']} / {_row['Short']} — "
                            "switch to the Pair Analysis tab",
                            icon="📈",
                        )

        # ── CSV export ────────────────────────────────────────────────────
        st.divider()
        _sc_csv = (
            _sr_full
            .drop(columns=['_long', '_short'], errors='ignore')
            .to_csv(index=False)
            .encode('utf-8')
        )
        st.download_button(
            "⬇ Download full results as CSV",
            _sc_csv,
            file_name="scenario_results.csv",
            mime="text/csv",
            key='sc_download',
        )
