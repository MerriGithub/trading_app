from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.basket import Basket
from core.position import Position
from engine.numba_core import COL_ENTRY_IDX as _CEI, COL_SIDE as _CS
from engine.backtest import aggregate_trades as _agg_bt
from asset_configs import ASSET_CLASSES

from tabs.shared import (
    portfolio, registry, _cached_daily_prices, _cached_latest_prices,
    ALL_INSTRUMENTS, ALL_DISPLAY, _signal_state_badge,
)


def _get_spread_pct(code: str) -> float:
    for ac in ASSET_CLASSES.values():
        cfg = ac.get('instruments', {}).get(code)
        if isinstance(cfg, dict):
            return cfg.get('spread_pct', 0.001)
    return 0.001


def _get_asset_class(code: str) -> str:
    for ac_key, ac in ASSET_CLASSES.items():
        if code in ac.get('instruments', {}):
            return ac_key
    return 'equity'


def _compute_basket_costs(
    long_legs: list[str],
    short_legs: list[str],
    broker_profile: str | None = None,
) -> tuple[float, float]:
    """
    Returns (spread_cost_pct, financing_daily_pct) for use with aggregate_trades.

    spread_cost_pct is the average per-leg one-way spread.
    financing_daily_pct × n_legs gives total daily financing drag as fraction of notional.
    """
    from account import get_financing_daily_rate
    n_legs = len(long_legs) + len(short_legs)
    if n_legs == 0:
        return 0.001, 0.0

    spread_cost_pct = sum(
        _get_spread_pct(leg) for leg in long_legs + short_legs
    ) / n_legs

    fin_daily = 0.0
    for leg in long_legs:
        fin_daily += get_financing_daily_rate(
            leg, _get_asset_class(leg), 'long', broker_profile=broker_profile,
        )
    for leg in short_legs:
        fin_daily += get_financing_daily_rate(
            leg, _get_asset_class(leg), 'short', broker_profile=broker_profile,
        )

    financing_daily_pct = fin_daily / n_legs
    return spread_cost_pct, financing_daily_pct


@st.cache_data(ttl=300, show_spinner=False)
def _cached_pair_signal(
    instruments: tuple[str, ...],
    long_legs: tuple[str, ...],
    short_legs: tuple[str, ...],
    vol_window: int,
    xing_sd: float,
    exit_sd: float,
) -> dict:
    from core.signal import SpreadSignal
    prices = registry.get_daily_prices(list(instruments))
    b   = Basket(long_legs=list(long_legs), short_legs=list(short_legs))
    sig = SpreadSignal(basket=b, prices=prices,
                       vol_window=vol_window, xing_sd=xing_sd, exit_sd=exit_sd)
    hist = sig.signal_history(n_days=9999)
    return {
        'current_sd':   sig.current_sd,
        'signal_state': sig.signal_state,
        'velocity':     float(sig.velocity.iloc[-1]) if len(sig.velocity) else 0.0,
        'tvr':          sig.tvr,
        'hist_index':   hist.index,
        'cum_spread':   hist['cum_spread'],
        'distance_sd':  hist['distance_sd'],
    }


@st.cache_data(ttl=300, show_spinner=False)
def _pair_backtest(
    long_legs: tuple[str, ...],
    short_legs: tuple[str, ...],
    vol_window: int,
    xing_sd: float,
    exit_sd: float,
    window_days: int | None = None,
    spread_cost_pct: float = 0.0,
    financing_daily_pct: float = 0.0,
) -> dict:
    from engine.backtest import prepare_returns, run_backtest
    instr  = list(long_legs) + list(short_legs)
    prices = _cached_daily_prices(tuple(instr))
    scaled, day_ints, index = prepare_returns(prices, instr, vol_window=vol_window,
                                              window_days=window_days)
    n_long, n_short = len(long_legs), len(short_legs)
    n_legs = n_long + n_short
    long_mask  = np.array([1.0 / n_long  if i < n_long  else 0.0 for i in range(len(instr))])
    short_mask = np.array([1.0 / n_short if i >= n_long else 0.0 for i in range(len(instr))])
    spread_ret = scaled @ (long_mask - short_mask)
    bt = run_backtest(spread_ret, day_ints,
                      vol_window=vol_window, xing_sd=xing_sd, exit_sd=exit_sd,
                      spread_cost_pct=spread_cost_pct,
                      financing_daily_pct=financing_daily_pct,
                      n_legs=n_legs)
    s = bt['summary']
    n = int(bt['n_trades'])
    return {
        'n_trades':           n,
        'gross_wr':           float(s.get('gross_wr', 0.0)),
        'net_wr':             float(s.get('net_wr', 0.0)),
        'avg_gross':          float(s.get('avg_gross', 0.0)),
        'avg_net':            float(s.get('avg_net', 0.0)),
        'avg_holding':        float(s.get('avg_holding', 0.0)),
        'trades_raw':         bt['trades_raw'][:n].copy(),
        'cum_spread':         np.cumsum(spread_ret),
        'date_index':         index,
        'spread_cost_pct':    spread_cost_pct,
        'financing_daily_pct': financing_daily_pct,
        'n_legs':             n_legs,
    }


def render() -> None:
    st.header("Pair Analysis")
    st.caption("Inspect signal history, charts, and backtest stats for any pair.")

    # Transfer pending values from Tab 10 before widgets are instantiated
    for _k in ('pa_long', 'pa_short', 'pa_pair'):
        _pk = f'{_k}_pending'
        if _pk in st.session_state:
            st.session_state[_k] = st.session_state.pop(_pk)

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

    if basket is None:
        return

    try:
        pa_sig = _cached_pair_signal(
            tuple(basket.all_instruments),
            tuple(basket.long_legs),
            tuple(basket.short_legs),
            int(vol_window), float(xing_sd), float(exit_sd),
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Current SD",  f"{pa_sig['current_sd']:+.2f}")
        mc2.metric("TVR",         f"{pa_sig['tvr']:.3f}")
        mc3.metric("Velocity",    f"{pa_sig['velocity']:+.4f}")
        mc4.markdown(f"**{_signal_state_badge(pa_sig['signal_state'])}**")

        _cs_full   = pa_sig['cum_spread']
        _rm_full   = _cs_full.rolling(int(vol_window), min_periods=10).mean()
        _rstd_full = _cs_full.rolling(int(vol_window), min_periods=10).std()

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

        _trend_dir = ('🟢 Long bias'  if _trend_slope >  0.0001
                      else '🔴 Short bias' if _trend_slope < -0.0001
                      else '⚪ Neutral')
        _cur_sd = pa_sig['current_sd']
        if abs(_cur_sd) < float(xing_sd):
            _align_badge = '— No active signal'
        elif _cur_sd > 0 and _trend_slope < 0:
            _align_badge = '✅ With-trend'
        elif _cur_sd > 0 and _trend_slope > 0:
            _align_badge = '⚠️ Counter-trend'
        elif _cur_sd < 0 and _trend_slope > 0:
            _align_badge = '✅ With-trend'
        else:
            _align_badge = '⚠️ Counter-trend'
        tm1, tm2, tm3 = st.columns(3)
        tm1.metric("Trend direction",  _trend_dir)
        tm2.metric("Signal alignment", _align_badge)
        tm3.metric("Trend slope",      f"{_trend_slope:+.4f}")

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
                _spread_cost, _fin_daily = _compute_basket_costs(
                    basket.long_legs, basket.short_legs,
                    broker_profile=st.session_state.get('tab3_broker_profile', 'ig_spreadbet'),
                )
                st.session_state['tab2_backtest_result'] = _pair_backtest(
                    tuple(basket.long_legs), tuple(basket.short_legs),
                    int(vol_window), float(xing_sd), float(exit_sd),
                    pa_hist_days,
                    _spread_cost, _fin_daily,
                )
            except Exception as e:
                st.warning(f"Backtest unavailable: {e}")

        if 'tab2_backtest_result' in st.session_state:
            with st.expander("Backtest summary", expanded=True):
                bt = st.session_state['tab2_backtest_result']
                _bt_n      = bt.get('n_trades', 0)
                _bt_trades = bt.get('trades_raw')
                _bt_cum    = bt.get('cum_spread')
                _bt_idx    = bt.get('date_index')

                if _bt_n > 0 and _bt_trades is not None and _bt_cum is not None:
                    _tw    = int(pa_trend_window)
                    _slb2  = min(20, _tw // 10)
                    _trend_bt   = pd.Series(_bt_cum, index=_bt_idx).rolling(_tw, min_periods=10).mean()
                    _trend_arr  = _trend_bt.values
                    _entry_idxs = _bt_trades[:, _CEI].astype(int)
                    _bt_sides   = _bt_trades[:, _CS]
                    _prev_idxs  = np.maximum(0, _entry_idxs - _slb2)
                    _has_trend  = (_entry_idxs >= _slb2) & ~np.isnan(_trend_arr[_entry_idxs])
                    _slopes_bt  = np.where(
                        _has_trend,
                        (_trend_arr[_entry_idxs] - _trend_arr[_prev_idxs]) / _slb2,
                        np.nan,
                    )
                    _valid     = ~np.isnan(_slopes_bt)
                    _wt_mask   = (((_bt_sides > 0) & (_slopes_bt > 0)) |
                                  ((_bt_sides < 0) & (_slopes_bt < 0)))
                    _wt_filter = _wt_mask & _valid
                    _ct_filter = ~_wt_mask & _valid
                    _n_wt      = int(_wt_filter.sum())
                    _n_ct      = int(_ct_filter.sum())

                    st.caption(
                        f"Total: {_bt_n} trades  |  "
                        f"With-trend: {_n_wt}  |  Counter-trend: {_n_ct}"
                    )
                    _col_wt, _col_ct = st.columns(2)

                    _bt_scp = bt.get('spread_cost_pct',    0.0)
                    _bt_fin = bt.get('financing_daily_pct', 0.0)
                    _bt_nl  = bt.get('n_legs', 2)

                    def _show_pass(col, label: str, raw_t, n: int) -> float:
                        col.markdown(f"**{label}** — {n} trades")
                        if n == 0:
                            col.caption("No trades in this subset.")
                            return float('nan')
                        s = _agg_bt(raw_t, n, _bt_scp, _bt_fin, _bt_nl)
                        col.metric("Gross WR",  f"{s.get('gross_wr',    0.0):.1%}")
                        col.metric("Net WR",    f"{s.get('net_wr',      0.0):.1%}")
                        col.metric("Avg gross", f"{s.get('avg_gross',   0.0):+.4f}")
                        col.metric("Avg net",   f"{s.get('avg_net',     0.0):+.4f}")
                        col.metric("Avg hold",  f"{s.get('avg_holding', 0.0):.0f}d")
                        return float(s.get('avg_net', 0.0))

                    _wt_net = _show_pass(_col_wt, "✅ With-trend",    _bt_trades[_wt_filter], _n_wt)
                    _ct_net = _show_pass(_col_ct, "⚠️ Counter-trend", _bt_trades[_ct_filter], _n_ct)

                    _wt_pos = not np.isnan(_wt_net) and _wt_net > 0
                    _ct_pos = not np.isnan(_ct_net) and _ct_net > 0
                    if _wt_pos and _ct_pos:
                        st.success("✅ Both directions viable — consider sizing counter-trend down")
                    elif _wt_pos:
                        st.info("✅ Trade with-trend only")
                    elif _ct_pos:
                        st.warning("⚠️ Counter-trend outperforms — review trend window")
                    else:
                        st.error("❌ No viable direction at current parameters")

                else:
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Trades",      f"{_bt_n:,}")
                    bc1.metric("Avg holding", f"{bt['avg_holding']:.0f}d")
                    bc2.metric("Gross WR",    f"{bt['gross_wr']:.1%}")
                    bc2.metric("Avg gross",   f"{bt['avg_gross']:+.4f}")
                    bc3.metric("Net WR",      f"{bt['net_wr']:.1%}")
                    bc3.metric("Avg net",     f"{bt['avg_net']:+.4f}")

                st.markdown("---")
                if st.button("→ Validate in Walk-Forward", key="tab2_send_wf"):
                    st.session_state['wf_pair'] = {
                        'long':          list(basket.long_legs),
                        'short':         list(basket.short_legs),
                        'vol_window':    int(vol_window),
                        'entry_sd':      float(xing_sd),
                        'exit_sd':       float(exit_sd),
                        'trend_window':  int(pa_trend_window),
                        'trend_mode':    'Both passes',
                        'source':        'tab2',
                    }
                    st.session_state['sidebar_nav_pending'] = "🔀 Walk-Forward"
                    st.rerun()

    except Exception as e:
        st.error(f"Signal computation failed: {e}")
