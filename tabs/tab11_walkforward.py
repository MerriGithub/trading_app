from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.backtest import prepare_returns, run_backtest, aggregate_trades
from engine.numba_core import COL_ENTRY_IDX, COL_SIDE
from asset_configs import ASSET_CLASSES
from tabs.shared import registry, _tbl, ALL_INSTRUMENTS, ALL_DISPLAY


def _get_ac(code: str) -> str:
    for ac_key, ac in ASSET_CLASSES.items():
        if code in ac.get('instruments', {}):
            return ac_key
    return 'equity'


def _basket_costs(long_legs: list[str], short_legs: list[str], broker: str) -> tuple[float, float]:
    from account import get_financing_daily_rate
    n = len(long_legs) + len(short_legs)
    if n == 0:
        return 0.001, 0.0

    def _sp(c: str) -> float:
        for ac in ASSET_CLASSES.values():
            cfg = ac.get('instruments', {}).get(c)
            if isinstance(cfg, dict):
                return cfg.get('spread_pct', 0.001)
        return 0.001

    scp = sum(_sp(l) for l in long_legs + short_legs) / n
    fin = 0.0
    for l in long_legs:
        fin += get_financing_daily_rate(l, _get_ac(l), 'long',  broker_profile=broker)
    for s in short_legs:
        fin += get_financing_daily_rate(s, _get_ac(s), 'short', broker_profile=broker)
    return scp, fin / n


def _trend_split(
    raw_t: np.ndarray,
    n: int,
    spread_ret: np.ndarray,
    idx: pd.DatetimeIndex,
    trend_win: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if n == 0:
        empty = np.empty((0, raw_t.shape[1]))
        return empty, empty, float('nan')
    slb = min(20, trend_win // 10)
    trend_arr = pd.Series(np.cumsum(spread_ret), index=idx).rolling(trend_win, min_periods=10).mean().values
    ei  = raw_t[:n, COL_ENTRY_IDX].astype(int)
    si  = raw_t[:n, COL_SIDE]
    pi  = np.maximum(0, ei - slb)
    htr = (ei >= slb) & ~np.isnan(trend_arr[ei])
    slp = np.where(htr, (trend_arr[ei] - trend_arr[pi]) / slb, np.nan)
    vld = ~np.isnan(slp)
    wt  = (((si > 0) & (slp > 0)) | ((si < 0) & (slp < 0))) & vld
    ct  = ~wt & vld
    al  = float(wt[vld].mean()) if vld.any() else float('nan')
    return raw_t[:n][wt], raw_t[:n][ct], al


def _primary_stats(
    bt: dict,
    spread_ret: np.ndarray,
    idx: pd.DatetimeIndex,
    trend_win: int,
    trend_mode: str,
    scp: float,
    fin: float,
    n_legs: int,
) -> tuple[dict, int]:
    n = bt['n_trades']
    s = bt['summary']
    if n == 0 or trend_mode == 'Off':
        return s, n
    raw_t = bt['trades_raw'][:n]
    wt_t, ct_t, _ = _trend_split(raw_t, n, spread_ret, idx, trend_win)
    if trend_mode == 'With-trend only':
        nw = len(wt_t)
        return aggregate_trades(wt_t, nw, scp, fin, n_legs), nw
    elif trend_mode == 'Counter-trend only':
        nc = len(ct_t)
        return aggregate_trades(ct_t, nc, scp, fin, n_legs), nc
    else:  # Both passes → WT as primary
        nw = len(wt_t)
        return (aggregate_trades(wt_t, nw, scp, fin, n_legs) if nw > 0 else {}), nw


def _run_wf(
    long_legs: list[str],
    short_legs: list[str],
    vol_window: int,
    xing_sd: float,
    exit_sd: float,
    trend_win: int,
    trend_mode: str,
    is_days: int,
    oos_days: int,
    step_days: int,
    broker: str,
) -> tuple[pd.DataFrame | None, str | None]:
    instr  = long_legs + short_legs
    n_legs = len(instr)
    scp, fin = _basket_costs(long_legs, short_legs, broker)

    prices = registry.get_daily_prices(instr)
    if prices.empty:
        return None, "No price data found for the selected instruments."
    prices = prices[instr].dropna()
    n_total = len(prices)

    if n_total < is_days + oos_days:
        return None, f"Insufficient history: {n_total} days, need {is_days + oos_days}."

    starts = []
    pos = 0
    while pos + is_days + oos_days <= n_total:
        starts.append(pos)
        pos += step_days
    if len(starts) < 3:
        return None, f"Only {len(starts)} window(s) possible — need ≥ 3. Reduce IS/OOS sizes or step."

    n_long  = len(long_legs)
    n_short = len(short_legs)
    lm = np.array([1.0/n_long  if i < n_long  else 0.0 for i in range(n_legs)])
    sm = np.array([1.0/n_short if i >= n_long else 0.0 for i in range(n_legs)])

    rows = []
    for w, start in enumerate(starts):
        is_end  = start + is_days
        oos_end = is_end + oos_days

        # IS
        try:
            is_sc, is_di, is_idx = prepare_returns(prices.iloc[start:is_end], instr, vol_window)
        except Exception:
            continue
        if is_sc.shape[0] < max(vol_window // 2, 20):
            continue
        is_spr = is_sc @ (lm - sm)
        is_bt  = run_backtest(is_spr, is_di, vol_window, xing_sd, exit_sd, scp, fin, n_legs)
        is_s, is_n = _primary_stats(is_bt, is_spr, is_idx, trend_win, trend_mode, scp, fin, n_legs)

        # OOS — use IS period as vol warmup, window_days selects last oos_days
        try:
            oos_sc, oos_di, oos_idx = prepare_returns(
                prices.iloc[start:oos_end], instr, vol_window, window_days=oos_days,
            )
        except Exception:
            continue
        if oos_sc.shape[0] < max(vol_window // 4, 10):
            continue
        oos_spr = oos_sc @ (lm - sm)
        oos_bt  = run_backtest(oos_spr, oos_di, vol_window, xing_sd, exit_sd, scp, fin, n_legs)
        oos_s, oos_n = _primary_stats(oos_bt, oos_spr, oos_idx, trend_win, trend_mode, scp, fin, n_legs)

        def _f(d: dict, k: str) -> float:
            v = d.get(k, float('nan'))
            return float(v) if v is not None else float('nan')

        is_nr  = _f(is_s,  'net_wr')
        is_an  = _f(is_s,  'avg_net')
        oos_nr = _f(oos_s, 'net_wr')
        oos_an = _f(oos_s, 'avg_net')
        oos_ah = _f(oos_s, 'avg_holding')

        stable = (
            not np.isnan(oos_nr) and not np.isnan(oos_an)
            and oos_nr > 0.5 and oos_an > 0
        )

        is_str  = f"{prices.index[start].date()} – {prices.index[is_end-1].date()}"
        oos_str = f"{prices.index[is_end].date()} – {prices.index[min(oos_end-1, n_total-1)].date()}"

        rows.append({
            'Window':      w + 1,
            'IS Period':   is_str,
            'OOS Period':  oos_str,
            'IS Net WR':   is_nr,
            'IS Avg Net':  is_an,
            'OOS Net WR':  oos_nr,
            'OOS Avg Net': oos_an,
            'OOS Avg Hold': oos_ah,
            'IS Trades':   is_n,
            'OOS Trades':  oos_n,
            'Stable?':     '✅' if stable else '❌',
            '_is_an':      is_an,
            '_oos_an':     oos_an,
            '_oos_nr':     oos_nr,
        })

    if not rows:
        return None, "No valid windows computed — try reducing vol window or increasing history."
    return pd.DataFrame(rows), None


def render() -> None:
    st.header("Walk-Forward Analysis")
    st.caption("Rolling IS/OOS validation. Tests whether a pair + parameters are robust or curve-fitted.")

    # Transfer pending pair from Tab 10 before widgets are instantiated
    for _k in ('wf11_long', 'wf11_short'):
        _pk = f'{_k}_pending'
        if _pk in st.session_state:
            st.session_state[_k] = st.session_state.pop(_pk)

    wf_pair = st.session_state.get('wf_pair')
    if wf_pair:
        _src = wf_pair.get('source', '').replace('tab', 'Tab ')
        _ll  = ', '.join(wf_pair.get('long',  []))
        _sl  = ', '.join(wf_pair.get('short', []))
        st.info(f"Loaded from {_src}: **{_ll}** (long) / **{_sl}** (short)")

    def _default(key, fallback):
        return wf_pair.get(key, fallback) if wf_pair else fallback

    # ── Pair ──────────────────────────────────────────────────────────────────
    wc1, wc2 = st.columns(2)
    _dl = _default('long',  [ALL_INSTRUMENTS[0]] if ALL_INSTRUMENTS else [])
    _ds = _default('short', [ALL_INSTRUMENTS[1]] if len(ALL_INSTRUMENTS) > 1 else [])
    long_picks  = wc1.multiselect("Long legs",  ALL_INSTRUMENTS, default=_dl,
                                  format_func=lambda c: ALL_DISPLAY.get(c, c), key='wf11_long')
    short_picks = wc2.multiselect("Short legs", ALL_INSTRUMENTS, default=_ds,
                                  format_func=lambda c: ALL_DISPLAY.get(c, c), key='wf11_short')

    # ── Signal params ─────────────────────────────────────────────────────────
    sp1, sp2, sp3, sp4, sp5 = st.columns(5)
    _vol   = int(sp1.number_input("Vol window",    50,  500, _default('vol_window',  262), key='wf11_vol'))
    _entry = float(sp2.number_input("Entry SD",   0.5,  5.0, _default('entry_sd',    2.0), 0.5, key='wf11_entry'))
    _exit  = float(sp3.number_input("Exit SD",    0.0,  2.0, _default('exit_sd',     1.0), 0.5, key='wf11_exit'))
    _tw    = int(sp4.number_input("Trend window", 130,  756, _default('trend_window', 262), key='wf11_tw'))
    _tmode_opts = ["Both passes", "With-trend only", "Counter-trend only", "Off"]
    _tmode_def  = _default('trend_mode', 'Both passes')
    _tmode_idx  = _tmode_opts.index(_tmode_def) if _tmode_def in _tmode_opts else 0
    _tmode = sp5.selectbox("Trend mode", _tmode_opts, index=_tmode_idx, key='wf11_tmode')

    # ── Window params ─────────────────────────────────────────────────────────
    ww1, ww2, ww3, ww4 = st.columns(4)
    _is   = int(ww1.number_input("In-sample (days)",     252, 2520,  756, key='wf11_is'))
    _oos  = int(ww2.number_input("Out-of-sample (days)",  63,  756,  252, key='wf11_oos'))
    _step = int(ww3.number_input("Step size (days)",      63,  756,  252, key='wf11_step'))
    _broker_opts = ["ig_spreadbet", "ig_cfd"]
    _broker_lbl  = {"ig_spreadbet": "IG Spread Bet", "ig_cfd": "IG CFD"}
    _broker_def  = st.session_state.get('tab3_broker_profile', 'ig_spreadbet')
    _broker_idx  = _broker_opts.index(_broker_def) if _broker_def in _broker_opts else 0
    _broker = ww4.selectbox("Broker", _broker_opts, index=_broker_idx,
                            format_func=lambda x: _broker_lbl[x], key='wf11_broker')

    st.markdown("---")
    _run = st.button("▶ Run walk-forward", type="primary", use_container_width=True, key='wf11_run')

    if _run:
        if not long_picks or not short_picks:
            st.error("Select at least one long leg and one short leg.")
        else:
            with st.spinner("Running walk-forward…"):
                _df, _err = _run_wf(
                    list(long_picks), list(short_picks),
                    _vol, _entry, _exit, _tw, _tmode,
                    _is, _oos, _step, _broker,
                )
            if _err:
                st.error(_err)
            else:
                st.session_state['wf11_results'] = {
                    'df':     _df,
                    'long':   list(long_picks),
                    'short':  list(short_picks),
                    'params': dict(vol_window=_vol, entry_sd=_entry, exit_sd=_exit,
                                   trend_window=_tw, trend_mode=_tmode,
                                   is_days=_is, oos_days=_oos, step_days=_step, broker=_broker),
                }

    cache = st.session_state.get('wf11_results')
    if cache is None:
        return

    df     = cache['df']
    _long  = cache['long']
    _short = cache['short']
    _parms = cache['params']

    if df.empty:
        st.warning("No valid windows computed.")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    n_win    = len(df)
    n_stable = int((df['Stable?'] == '✅').sum())

    _common = df[['_is_an', '_oos_an']].replace([np.inf, -np.inf], np.nan).dropna()
    _is_mean = float(_common['_is_an'].mean()) if not _common.empty else float('nan')
    _oos_mean = float(_common['_oos_an'].mean()) if not _common.empty else float('nan')
    _consistency = (_oos_mean / _is_mean) if (not np.isnan(_is_mean) and _is_mean != 0) else float('nan')
    _avg_oos_nr = float(df['_oos_nr'].replace([np.inf, -np.inf], np.nan).dropna().mean())

    st.subheader("Stability Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Stable windows",    f"{n_stable} / {n_win} ({n_stable/n_win:.0%})")
    s2.metric("Avg OOS net exp",   f"{_oos_mean:+.3%}" if not np.isnan(_oos_mean) else "N/A")
    s3.metric("Avg OOS net WR",    f"{_avg_oos_nr:.1%}" if not np.isnan(_avg_oos_nr) else "N/A")
    s4.metric("Consistency score", f"{_consistency:.2f}" if not np.isnan(_consistency) else "N/A")

    _spct = n_stable / n_win
    if not np.isnan(_consistency) and _consistency >= 0.7 and _spct >= 0.7:
        st.success("✅ **Robust** — suitable for live trading (consistency ≥ 0.7, stable ≥ 70%)")
    elif (not np.isnan(_consistency) and _consistency >= 0.4) or _spct >= 0.5:
        st.warning("⚠️ **Moderate** — monitor closely")
    else:
        st.error("❌ **Curve-fitted** — do not trade live")

    # ── Window table ──────────────────────────────────────────────────────────
    st.subheader("Window Results")
    _disp = df[['Window', 'IS Period', 'OOS Period',
                'IS Net WR', 'IS Avg Net', 'OOS Net WR', 'OOS Avg Net',
                'OOS Avg Hold', 'IS Trades', 'OOS Trades', 'Stable?']].copy()

    for _c in ['IS Net WR', 'OOS Net WR']:
        _disp[_c] = _disp[_c].apply(lambda v: f'{v:.1%}' if not (isinstance(v, float) and np.isnan(v)) else '—')
    for _c in ['IS Avg Net', 'OOS Avg Net']:
        _disp[_c] = _disp[_c].apply(lambda v: f'{v:+.3%}' if not (isinstance(v, float) and np.isnan(v)) else '—')
    _disp['OOS Avg Hold'] = _disp['OOS Avg Hold'].apply(
        lambda v: f'{v:.0f}d' if not (isinstance(v, float) and np.isnan(v)) else '—'
    )
    _tbl(_disp, show_index=False)

    # ── IS vs OOS chart ───────────────────────────────────────────────────────
    _chart = df[['Window', '_is_an', '_oos_an']].replace([np.inf, -np.inf], np.nan).dropna()
    if not _chart.empty:
        st.subheader("IS vs OOS Net Expectancy")
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_chart['Window'], y=(_chart['_is_an'] * 100),
            name='IS Avg Net (%)', line=dict(color='#2c6fad'),
        ))
        _fig.add_trace(go.Scatter(
            x=_chart['Window'], y=(_chart['_oos_an'] * 100),
            name='OOS Avg Net (%)', line=dict(color='#e67e22'),
        ))
        _fig.add_hline(y=0, line_dash='dash', line_color='gray')
        _fig.update_layout(
            title=f"{' + '.join(_long)} / {' + '.join(_short)}",
            xaxis_title='Window', yaxis_title='Avg Net Expectancy (%)',
            height=350, margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(_fig, use_container_width=True)
