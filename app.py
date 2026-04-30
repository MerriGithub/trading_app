from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from calculations import (
    compute_contraction_betas, correlation_matrix, crossing_signals,
    intraday_spread, linear_trend, portfolio_returns, portfolio_stats,
    returns, rolling_nd_returns, rolling_volatility, scaling_vectors,
    velocity_acceleration,
)
from config import ACTIVE_INSTRUMENTS, DISPLAY_NAMES, PARAMS, POINT_SIZES, SPREADS
from scipy.stats import norm as _norm
from data import force_intraday_refresh, force_refresh, load_intraday_prices, load_prices
from account import compute_daily_funding, compute_spread_costs, load_account, save_account
from journal import (
    close_trade, delete_trade, load_trades, open_trade,
    partial_close_leg, trade_live_pnl,
)
from saved import delete_portfolio, load_saved, save_portfolio
from search import METRIC_NAMES, METRICS, estimate_combinations, run_search
from scoring import SCORING_MODES
from stake_calc import compute_stakes, pnl_scenario
from asset_configs import ASSET_CLASSES, ASSET_CLASS_OPTIONS
from backtest import (
    load_asset_prices, prepare_returns, run_backtest,
    run_exhaustive_search, regime_split, sensitivity_grid,
    find_breakeven_financing, aggregate_trades,
)

_TDY = PARAMS['trading_days_per_year']

st.set_page_config(page_title="Trading Monitor", page_icon="📈", layout="wide")


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


def _dn(df_or_series):
    """Rename index/columns from internal codes to display names."""
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    return df_or_series.rename(index=DISPLAY_NAMES)


# ── Session state ─────────────────────────────────────────────────────────────
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = None

# If the Search tab loaded a portfolio, pre-set the checkbox keys before render.
# This runs before the sidebar widgets are created so the checkboxes pick up the values.
if st.session_state.get('_load_pending'):
    sel = st.session_state['_load_pending']
    for inst in ACTIVE_INSTRUMENTS:
        st.session_state[f'L_{inst}'] = bool(sel['long'].get(inst, 0))
        st.session_state[f'S_{inst}'] = bool(sel['short'].get(inst, 0))
    st.session_state['_load_pending'] = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Trading Monitor")

    if st.button("🔄 Refresh Data"):
        force_refresh()
        st.rerun()

    target_exposure = st.number_input("Target 1 SD Exposure (£)", value=500, step=50, min_value=50)
    start_year = st.slider("History start year", 1999, date.today().year - 1, 1999)

    st.markdown("---")
    st.markdown("### Signal Selection")
    st.caption("Tick instruments for each leg of the spread trade.")

    c1, c2 = st.columns(2)
    c1.markdown("**Long ▲**")
    c2.markdown("**Short ▼**")

    # Build flag dicts from checkbox state — these drive every calculation below
    long_flags: dict = {}
    short_flags: dict = {}
    for inst in ACTIVE_INSTRUMENTS:
        label = DISPLAY_NAMES.get(inst, inst)
        long_flags[inst]  = int(c1.checkbox(label, key=f"L_{inst}"))
        short_flags[inst] = int(c2.checkbox(label, key=f"S_{inst}"))

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading prices…"):
    prices = load_prices(f"{start_year}-01-01")

if prices is None or prices.empty:
    st.error("No data loaded — check internet connection and try Refresh.")
    st.stop()

# Pre-compute returns, vol, scaling, and portfolio return series once for all tabs
rets     = returns(prices)
vols     = rolling_volatility(rets)
scalings = scaling_vectors(prices, rets)
port_ret = portfolio_returns(rets, scalings, long_flags, short_flags)

# Latest cross-section snapshots used throughout
latest_prices   = prices.iloc[-1]
latest_vols     = vols.iloc[-1]
latest_scalings = scalings.iloc[-1]

# Load trades once per render — reused across Dashboard, Portfolio, and Journal tabs
all_trades = load_trades()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Dashboard", "📈 Analysis", "🧮 Stake Calculator", "🗂 Portfolio",
    "🔍 Search", "⏱ Live", "📓 Journal", "🔬 Backtest", "🔄 Walk-Forward",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dashboard")
    last_date = prices.index[-1].strftime('%d %b %Y')
    st.caption(f"Last data: {last_date}  |  {len(prices)} trading days loaded")

    # ── Price / Signal Table ──────────────────────────────────────────────────
    price_tbl = pd.DataFrame({
        'Price':     latest_prices.map('{:,.1f}'.format),
        '1D Chg':    rets.iloc[-1].map('{:+.2%}'.format),
        'Daily Vol': latest_vols.map(lambda v: f'{v*100:.2f}%' if pd.notna(v) else 'N/A'),
        'Scaling':   latest_scalings.map(lambda v: f'{v*100:.0f}%' if pd.notna(v) else 'N/A'),
        'Long':      pd.Series({k: '✓' if v else '' for k, v in long_flags.items()}),
        'Short':     pd.Series({k: '✓' if v else '' for k, v in short_flags.items()}),
    })
    st.subheader("Prices & Signals")
    _tbl(_dn(price_tbl))

    # ── Account Overview ──────────────────────────────────────────────────────
    st.subheader("Account Overview")
    _acct         = load_account()
    _open_t       = [t for t in all_trades if t['status'] == 'open']
    _closed_t     = [t for t in all_trades if t['status'] == 'closed']
    _cur_px       = {k: float(v) for k, v in latest_prices.items() if pd.notna(v)}
    # Realised P&L comes from closed trades; unrealised (MM) from open trades marked to market
    _realised     = sum(t.get('realised_pnl', 0.0) or 0.0 for t in _closed_t)
    _mm_pnl       = sum(trade_live_pnl(t, _cur_px) for t in _open_t)
    _total_equity = _acct['starting_capital'] + _realised + _mm_pnl
    _free_equity  = _total_equity - _acct['margin']
    _funding      = compute_daily_funding(_open_t, _cur_px, _acct['long_rate'], _acct['short_rate'])
    _spread_costs = compute_spread_costs(_open_t)

    ae1, ae2, ae3 = st.columns(3)
    ae1.metric("Starting Capital", f"£{_acct['starting_capital']:,.0f}")
    ae2.metric("Total Equity",     f"£{_total_equity:,.0f}",
               delta=f"£{(_total_equity - _acct['starting_capital']):+,.0f}")
    ae3.metric("Free Equity",      f"£{_free_equity:,.0f}",
               delta=f"Margin £{_acct['margin']:,.0f}")

    ae4, ae5, ae6 = st.columns(3)
    ae4.metric("Realised P&L",     f"£{_realised:+,.0f}")
    ae5.metric("Unrealised (MM)",  f"£{_mm_pnl:+,.0f}")
    ae6.metric("Open Positions",   str(len(_open_t)))

    ae7, ae8 = st.columns(2)
    ae7.metric("Est. Daily Funding", f"£{_funding:+,.2f}",
               help="Long funding cost minus short rebate (annualised rates / 365)")
    ae8.metric("Open Spread Costs",  f"£{_spread_costs:,.2f}",
               help="Round-trip bid/ask cost if all open positions closed now")

    with st.expander("Account settings"):
        ac1, ac2, ac3, ac4 = st.columns(4)
        _new_cap  = ac1.number_input("Starting capital (£)", value=_acct['starting_capital'],
                                     step=1000.0, key='acct_cap')
        _new_lr   = ac2.number_input("Long rate",  value=_acct['long_rate'],
                                     step=0.001, format='%.4f', key='acct_lr')
        _new_sr   = ac3.number_input("Short rate", value=_acct['short_rate'],
                                     step=0.001, format='%.4f', key='acct_sr')
        _new_marg = ac4.number_input("Margin (£)", value=_acct['margin'],
                                     step=500.0, key='acct_marg')
        if st.button("Save account settings", key='acct_save'):
            save_account({'starting_capital': _new_cap, 'long_rate': _new_lr,
                          'short_rate': _new_sr, 'margin': _new_marg})
            st.success("Saved.")
            st.rerun()

    st.markdown("---")

    # ── Signal Scanner ────────────────────────────────────────────────────────
    # Shows key trend metrics for every instrument, sorted by distance from mean (most extreme first)
    st.subheader("Signal Scanner")
    _tol_sd      = PARAMS['xing_tolerance_sd']
    _n_fit       = PARAMS['linear_fit_points']
    _contractions = compute_contraction_betas(rets)
    _margin_rate  = PARAMS['margin_rate']
    _scan_rows   = []
    for _inst in ACTIVE_INSTRUMENTS:
        if _inst not in rets.columns or _inst not in scalings.columns:
            continue
        _s = (rets[_inst] * scalings[_inst]).dropna()
        if len(_s) < 10:
            continue
        _cum   = (1 + _s).cumprod()
        _va    = velocity_acceleration(_s)
        _cs    = crossing_signals(_s)
        _trend = linear_trend(_cum, _n_fit)
        _vol   = float(_s.std())
        _tvr   = (abs(_trend['slope']) / _vol) if (_vol > 0 and pd.notna(_trend.get('slope'))) else 0.0
        _vel   = float(_va['velocity'].iloc[-1])    if pd.notna(_va['velocity'].iloc[-1])    else 0.0
        _acc   = float(_va['acceleration'].iloc[-1]) if pd.notna(_va['acceleration'].iloc[-1]) else 0.0
        _csd   = float(_cs['distance_sd'].iloc[-1])  if pd.notna(_cs['distance_sd'].iloc[-1]) else 0.0
        _roll  = rolling_nd_returns(_s).iloc[-1]
        _beta  = float(_contractions.get(_inst, float('nan')))

        # Pre-trade recommendation: compute suggested stake and hypothetical P&L at current SDs
        _px   = float(latest_prices.get(_inst, float('nan')))
        _hv   = float(latest_vols.get(_inst, float('nan')))
        _sc2  = float(latest_scalings.get(_inst, 0.0))
        if not (np.isnan(_px) or np.isnan(_hv) or _hv == 0 or _px == 0):
            _rec_stake  = round(float(target_exposure) / (_px * _hv) * _sc2, 2)
            _rec_margin = round(_rec_stake * _px * _margin_rate, 0)
            _rec_hpnl   = round(_rec_stake * _px * _hv * abs(_csd), 0)
        else:
            _rec_stake = _rec_margin = _rec_hpnl = 0.0
        # Direction signal fires only when distance exceeds tolerance threshold
        _direction = 'BUY' if _csd < -_tol_sd else ('SELL' if _csd > _tol_sd else '')

        _scan_rows.append({
            'Instrument': DISPLAY_NAMES.get(_inst, _inst),
            'TVR':        f'{_tvr:.3f}',
            'Velocity':   f'{_vel:+.4f}',
            'Accel.':     f'{_acc:+.4f}',
            'Cur. SDs':   f'{_csd:+.2f}',
            '1D':  f"{float(_roll['1D']):+.2%}" if pd.notna(_roll['1D']) else '',
            '2D':  f"{float(_roll['2D']):+.2%}" if pd.notna(_roll['2D']) else '',
            '3D':  f"{float(_roll['3D']):+.2%}" if pd.notna(_roll['3D']) else '',
            '5D':  f"{float(_roll['5D']):+.2%}" if pd.notna(_roll['5D']) else '',
            'Signal':    '✓' if abs(_csd) > _tol_sd else '',
            'Beta':      f'{_beta:.2f}' if pd.notna(_beta) else '',
            'Direction': _direction,
            'Stake':     f'{_rec_stake:.1f}' if _rec_stake > 0 else '',
            'Margin':    f'£{_rec_margin:,.0f}' if _rec_margin > 0 else '',
            'Hyp P&L':   f'£{_rec_hpnl:,.0f}' if (_rec_hpnl > 0 and _direction) else '',
        })
    if _scan_rows:
        _scan_df = pd.DataFrame(_scan_rows)
        # Sort by absolute SD distance so the most extreme instruments appear first
        _scan_df['_abs'] = _scan_df['Cur. SDs'].apply(lambda x: abs(float(x)) if x else 0.0)
        _scan_df = _scan_df.sort_values('_abs', ascending=False).drop(columns=['_abs'])
        _tbl(_scan_df, show_index=False)

    # ── Multi-Timeframe Range Signals ─────────────────────────────────────────
    # Shows where today's price sits within daily / weekly / monthly price ranges
    st.markdown("---")
    st.subheader("Multi-Timeframe Range Signals")
    st.caption("Percentile of current price within each rolling window. HIGH/LOW = top or bottom 10%.")
    _range_rows = []
    for _inst in ACTIVE_INSTRUMENTS:
        if _inst not in prices.columns:
            continue
        _px_ser = prices[_inst].dropna()
        if len(_px_ser) < 262:
            continue
        _cur_px2 = float(_px_ser.iloc[-1])
        # Compute high/low for each timeframe window
        _d_hi = float(_px_ser.tail(262).max()); _d_lo = float(_px_ser.tail(262).min())
        _w_hi = float(_px_ser.tail(5).max());   _w_lo = float(_px_ser.tail(5).min())
        _m_hi = float(_px_ser.tail(21).max());  _m_lo = float(_px_ser.tail(21).min())
        # Percentile position within each range (0 = at low, 1 = at high)
        _dp = (_cur_px2 - _d_lo) / (_d_hi - _d_lo) if _d_hi > _d_lo else 0.5
        _wp = (_cur_px2 - _w_lo) / (_w_hi - _w_lo) if _w_hi > _w_lo else 0.5
        _mp = (_cur_px2 - _m_lo) / (_m_hi - _m_lo) if _m_hi > _m_lo else 0.5
        _range_rows.append({
            'Instrument': DISPLAY_NAMES.get(_inst, _inst),
            'D %ile':  f'{_dp:.0%}',
            'D Range': 'HIGH' if _dp >= 0.9 else ('LOW' if _dp <= 0.1 else '-'),
            'W %ile':  f'{_wp:.0%}',
            'W Range': 'HIGH' if _wp >= 0.9 else ('LOW' if _wp <= 0.1 else '-'),
            'M %ile':  f'{_mp:.0%}',
            'M Range': 'HIGH' if _mp >= 0.9 else ('LOW' if _mp <= 0.1 else '-'),
        })
    if _range_rows:
        _tbl(pd.DataFrame(_range_rows), show_index=False)

    st.markdown("---")

    # ── Spread Performance ────────────────────────────────────────────────────
    st.subheader("Spread Performance")
    any_selected = any(long_flags.values()) or any(short_flags.values())

    if any_selected and not port_ret.empty:
        roll  = rolling_nd_returns(port_ret)
        stats = portfolio_stats(port_ret)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Today",         f"{port_ret.iloc[-1]:+.2%}")
        c2.metric("5-Day",         f"{roll['5D'].iloc[-1]:+.2%}" if not roll['5D'].isna().all() else "N/A")
        c3.metric("Ann. Sharpe",   f"{stats['sharpe']:.2f}"       if pd.notna(stats['sharpe']) else "N/A")
        c4.metric("Mean Daily",    f"{stats['mean_daily']:+.2%}")

        cum = (1 + port_ret).cumprod()
        fig = px.line(cum, labels={'value': 'Cumulative Return', 'index': 'Date'},
                      title="Cumulative Spread Return")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rolling Returns — last 20 days")
        roll_disp = roll.tail(20).map(lambda v: f'{v:+.2%}' if pd.notna(v) else '')
        _tbl(roll_disp)
    else:
        st.info("Select instruments in the sidebar to see spread performance.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Analysis")

    # ── Normalised Price Chart ────────────────────────────────────────────────
    display_list = [DISPLAY_NAMES.get(i, i) for i in ACTIVE_INSTRUMENTS]
    selected_display = st.multiselect("Instruments to plot", display_list, default=display_list[:4])
    selected = [k for k, v in DISPLAY_NAMES.items() if v in selected_display]

    if selected:
        # Rebase all prices to 1.0 at the start of the history window for easy comparison
        norm = prices[selected] / prices[selected].iloc[0]
        norm = norm.rename(columns=DISPLAY_NAMES)
        st.plotly_chart(
            px.line(norm, title="Normalised Prices (base = 1 at start)",
                    labels={'value': 'Level', 'index': 'Date'}),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(vols[selected].rename(columns=DISPLAY_NAMES) * 100,
                    title="Rolling Daily Volatility (%)",
                    labels={'value': 'Daily Std Dev %', 'index': 'Date'}),
            use_container_width=True,
        )

    # ── Spread Trend Analysis ─────────────────────────────────────────────────
    st.subheader("Spread Trend Analysis")

    any_selected = any(long_flags.values()) or any(short_flags.values())
    if any_selected and not port_ret.empty:
        cum_spread = (1 + port_ret).cumprod()
        n_fit = st.slider("Linear fit points", 5, 50, PARAMS['linear_fit_points'])
        trend = linear_trend(cum_spread, n_fit)

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=cum_spread.index, y=cum_spread.values,
            name='Cumulative Spread', line=dict(color='royalblue'),
        ))
        if trend['fitted'] is not None:
            slope_str = f"{trend['slope']:+.5f}"
            fig_t.add_trace(go.Scatter(
                x=trend['fitted'].index, y=trend['fitted'].values,
                name=f"Linear fit  slope={slope_str}  R²={trend['r2']:.3f}",
                line=dict(color='crimson', dash='dash'),
            ))
        fig_t.update_layout(title="Cumulative Spread with Linear Trend", height=350)
        st.plotly_chart(fig_t, use_container_width=True)

        # ── Velocity & Acceleration ───────────────────────────────────────────
        va = velocity_acceleration(port_ret)
        fig_va = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=('Velocity (ROC)', 'Acceleration'))
        fig_va.add_trace(go.Scatter(x=va.index, y=va['velocity'],    name='Velocity'),    row=1, col=1)
        fig_va.add_trace(go.Scatter(x=va.index, y=va['acceleration'], name='Acceleration'), row=2, col=1)
        fig_va.update_layout(height=380, title="Trend Velocity & Acceleration", showlegend=False)
        st.plotly_chart(fig_va, use_container_width=True)

        # ── Crossing Signals Chart ────────────────────────────────────────────
        cross = crossing_signals(port_ret)
        tol   = PARAMS['xing_tolerance_sd']
        fig_x = go.Figure()
        fig_x.add_trace(go.Scatter(
            x=cross.index, y=cross['distance_sd'],
            name='Distance (SDs)', line=dict(color='purple'),
        ))
        fig_x.add_hline(y= tol, line_dash='dash', line_color='red',   annotation_text=f'+{tol} SD')
        fig_x.add_hline(y=-tol, line_dash='dash', line_color='green',  annotation_text=f'-{tol} SD')
        fig_x.add_hline(y=0,    line_dash='dot',  line_color='grey')
        fig_x.update_layout(title="Spread Distance from Rolling Mean (Standard Deviations)", height=300)
        st.plotly_chart(fig_x, use_container_width=True)

        n_signals = int(cross['signal'].sum())
        last_dist = cross['distance_sd'].iloc[-1]
        st.info(
            f"Total crossing signals in history: **{n_signals}**  |  "
            f"Current distance: **{last_dist:.2f} SDs**"
        )

        # ── Pair Statistics ───────────────────────────────────────────────────
        # Distributional stats for 1/2/3/5-day rolling returns, including current z-score
        st.subheader("Pair Statistics")

        _roll_s = rolling_nd_returns(port_ret)
        _stat_rows = []
        for _n in [1, 2, 3, 5]:
            _col  = f'{_n}D'
            _ser  = _roll_s[_col].dropna()
            if _ser.empty:
                continue
            _avg  = float(_ser.mean())
            _sd   = float(_ser.std())
            _cur  = float(_roll_s[_col].iloc[-1]) if pd.notna(_roll_s[_col].iloc[-1]) else float('nan')
            _csds = (_cur - _avg) / _sd if _sd > 0 and pd.notna(_cur) else float('nan')
            _stat_rows.append({
                'Window':      _col,
                'Avg':         f'{_avg:+.2%}',
                '1 SD':        f'{_sd:.2%}',
                'Max':         f'{float(_ser.max()):+.2%}',
                'Min':         f'{float(_ser.min()):+.2%}',
                'Current':     f'{_cur:+.2%}' if pd.notna(_cur) else '',
                'Current SDs': f'{_csds:+.2f}' if pd.notna(_csds) else '',
            })
        if _stat_rows:
            _tbl(pd.DataFrame(_stat_rows), show_index=False)

        # Volatility scaled to daily / weekly / monthly holding periods
        st.markdown("**Volatility by timeframe**")
        _dv  = float(port_ret.std())
        _wv  = float(port_ret.rolling(5).sum().std())  if len(port_ret) >= 5  else float('nan')
        _mv  = float(port_ret.rolling(21).sum().std()) if len(port_ret) >= 21 else float('nan')
        _vc1, _vc2, _vc3 = st.columns(3)
        _vc1.metric("Daily 1 SD",   f'{_dv:.2%}')
        _vc2.metric("Weekly 1 SD",  f'{_wv:.2%}'  if pd.notna(_wv) else 'N/A')
        _vc3.metric("Monthly 1 SD", f'{_mv:.2%}' if pd.notna(_mv) else 'N/A')

        # Empirical vs Gaussian tail probabilities for daily returns
        st.markdown("**Spread distribution (1-day)**")
        _ds = port_ret.dropna()
        if len(_ds) >= 10:
            _em = float(_ds.mean())
            _es = float(_ds.std())
            _c1 = float(_ds.iloc[-1])
            _cz = (_c1 - _em) / _es if _es > 0 else 0.0
            _dist_rows = []
            for _nsd in [1.0, 1.5, 2.0, 2.5, 3.0]:
                _dist_rows.append({
                    'N SDs': f'{_nsd:.1f}',
                    'Emp. P(>+N)':  f'{float((_ds > _em + _nsd * _es).mean()):.1%}',
                    'Emp. P(<-N)':  f'{float((_ds < _em - _nsd * _es).mean()):.1%}',
                    'Gaussian':     f'{2 * (1 - _norm.cdf(_nsd)):.1%}',
                })
            _tbl(pd.DataFrame(_dist_rows), show_index=False)
            st.caption(
                f"Current 1-day: {_c1:+.2%}  |  Z-score: {_cz:+.2f} SDs  |  "
                f"Mean: {_em:+.2%}  |  1 SD: {_es:.2%}"
            )
    else:
        st.info("Select instruments in the sidebar to see trend analysis.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Stake Calculator
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Stake Calculator")
    st.caption("Prices are pre-filled from the latest data — edit if you want to run scenarios.")

    # ── Price Inputs ──────────────────────────────────────────────────────────
    # Allow the user to override prices for what-if scenarios
    price_cols = st.columns(4)
    price_inputs: dict = {}
    for i, inst in enumerate(ACTIVE_INSTRUMENTS):
        col = price_cols[i % 4]
        default = float(latest_prices.get(inst, 0.0))
        if np.isnan(default):
            default = 0.0
        price_inputs[inst] = col.number_input(
            f"{inst}  (spread {SPREADS.get(inst, 0)} pts)",
            value=round(default, 1),
            step=1.0,
            key=f"stake_price_{inst}",
        )

    vols_dict     = {k: float(v) for k, v in latest_vols.items()     if pd.notna(v)}
    scalings_dict = {k: float(v) for k, v in latest_scalings.items() if pd.notna(v)}

    stakes_df = compute_stakes(
        price_inputs, vols_dict, scalings_dict,
        long_flags, short_flags, float(target_exposure),
    )

    st.subheader("Position Sizes")
    _tbl(stakes_df, show_index=False)

    long_n  = int((stakes_df['Long']  == 1).sum())
    short_n = int((stakes_df['Short'] == 1).sum())
    total_cost = stakes_df['Cost'].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Long instruments",  long_n)
    m2.metric("Short instruments", short_n)
    m3.metric("Total spread cost", f"£{total_cost:,.2f}")

    # ── P&L Scenario ──────────────────────────────────────────────────────────
    # Assumes longs move by +pct_move and shorts move by -pct_move (spread widens)
    st.subheader("P&L Scenario")
    pct_move = st.slider("Long leg % move (short leg moves opposite)", -5.0, 5.0, 1.0, 0.1)
    scenario = {
        inst: pct_move  if long_flags.get(inst)  else
             -pct_move  if short_flags.get(inst) else 0.0
        for inst in ACTIVE_INSTRUMENTS
    }
    pnl = pnl_scenario(stakes_df, scenario)
    st.metric(f"Estimated P&L", f"£{pnl:,.0f}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Portfolio
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Portfolio")

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.subheader(f"Return Correlations — last {PARAMS['vol_calc_days']} days")
    corr = correlation_matrix(rets)
    fig_corr = px.imshow(
        corr,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        text_auto='.0%',
        aspect='auto',
        title="Pairwise Return Correlations",
    )
    fig_corr.update_layout(height=520)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Volatility & Scaling Bar Chart ────────────────────────────────────────
    st.subheader("Current Volatility & Scaling")
    vol_df = pd.DataFrame({
        'Daily Vol %': latest_vols * 100,
        'Scaling %':   latest_scalings * 100,
    }).rename(index=DISPLAY_NAMES).dropna().sort_values('Daily Vol %', ascending=False)

    fig_v = go.Figure()
    fig_v.add_trace(go.Bar(name='Daily Vol %', x=vol_df.index, y=vol_df['Daily Vol %']))
    fig_v.add_trace(go.Bar(name='Scaling %',   x=vol_df.index, y=vol_df['Scaling %']))
    fig_v.update_layout(barmode='group', height=350, title="Daily Volatility and Position Scaling by Instrument")
    st.plotly_chart(fig_v, use_container_width=True)

    # ── Portfolio Performance ─────────────────────────────────────────────────
    st.subheader("Spread Portfolio Performance")
    any_selected = any(long_flags.values()) or any(short_flags.values())
    if any_selected and not port_ret.empty:
        stats = portfolio_stats(port_ret)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Return",  f"{stats['total_return']:.1%}")
        s2.metric("Best period",   f"{stats['max_return']:.1%}")
        s3.metric("Worst period",  f"{stats['min_return']:.1%}")
        s4.metric("Mean Daily",    f"{stats['mean_daily']:+.2%}")
        s5.metric("Ann. Sharpe",   f"{stats['sharpe']:.2f}" if pd.notna(stats['sharpe']) else "N/A")

        roll = rolling_nd_returns(port_ret)
        st.plotly_chart(
            px.line(roll.tail(262), title="Rolling N-Day Spread Returns — last year",
                    labels={'value': 'Return', 'index': 'Date', 'variable': 'Window'}),
            use_container_width=True,
        )
    else:
        st.info("Select instruments in the sidebar to see portfolio performance.")

    # ── P&L Attribution — Today ───────────────────────────────────────────────
    # Breaks down today's P&L by instrument and region using net stakes from open trades
    st.subheader("P&L Attribution — Today")

    _REGIONS = {
        'EU':   ['UKX', 'CBK', 'CEY', 'CFR', 'CMD', 'CEI', 'COI'],
        'US':   ['CPH', 'CTN', 'CTB'],
        'ASIA': ['CRM', 'CIL'],
    }
    _today_ret = rets.iloc[-1]
    _open_t4   = [t for t in all_trades if t['status'] == 'open']

    # Build net stake per instrument across all open trades (long = positive, short = negative)
    _net_stakes: dict = {inst: 0.0 for inst in ACTIVE_INSTRUMENTS}
    for _t4 in _open_t4:
        for _leg in _t4.get('legs', []):
            _pct = _leg.get('pct_open', 0.0)
            _bi  = _leg.get('buy_instrument',  '')
            _si  = _leg.get('sell_instrument', '')
            if _bi in _net_stakes:
                _net_stakes[_bi] += _leg['buy_stake']  * _pct
            if _si in _net_stakes:
                _net_stakes[_si] -= _leg['sell_stake'] * _pct

    _attr_rows = []
    for _region, _insts in _REGIONS.items():
        for _inst in _insts:
            if _inst not in _today_ret.index:
                continue
            _stake   = _net_stakes.get(_inst, 0.0)
            _ret_d   = float(_today_ret[_inst]) if pd.notna(_today_ret.get(_inst)) else 0.0
            _price   = float(latest_prices.get(_inst, 0.0)) if pd.notna(latest_prices.get(_inst)) else 0.0
            _contrib = _stake * _price * _ret_d * POINT_SIZES.get(_inst, 1.0)
            _attr_rows.append({
                'Region':     _region,
                'Instrument': DISPLAY_NAMES.get(_inst, _inst),
                'Net Stake':  round(_stake, 2),
                "Today %":    f'{_ret_d:+.2%}',
                'P&L':        round(_contrib, 0),
            })

    if _attr_rows:
        _attr_df = pd.DataFrame(_attr_rows)
        _tbl(_attr_df, show_index=False)

        _region_totals = _attr_df.groupby('Region')['P&L'].sum()
        _rc = st.columns(len(_region_totals) + 1)
        for _ci, (_reg, _val) in enumerate(sorted(_region_totals.items())):
            _rc[_ci].metric(_reg, f"£{_val:+,.0f}")
        _rc[-1].metric("Total today", f"£{_attr_df['P&L'].sum():+,.0f}")

        _fig_attr = px.bar(
            _attr_df[_attr_df['P&L'] != 0],
            x='Instrument', y='P&L', color='Region',
            title="P&L Attribution by Instrument — Today",
        )
        _fig_attr.update_layout(height=320)
        st.plotly_chart(_fig_attr, use_container_width=True)
    else:
        st.caption("No open positions — open trades in the Journal to see attribution.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — Search
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Portfolio Search")
    st.caption(
        "Enumerates all long/short instrument combinations and scores each on "
        "5 metrics from the original spreadsheet's SearchEngine sheet."
    )

    # ── Saved Portfolios ─────────────────────────────────────────────────────
    with st.expander("⭐ Saved Portfolios"):
        saved_list = load_saved()
        if not saved_list:
            st.caption("No saved portfolios yet — run a search and save promising results.")
        else:
            for idx, entry in enumerate(saved_list):
                c1, c2, c3, c4 = st.columns([4, 4, 1, 1])
                c1.markdown(f"**{entry['name']}**")
                c1.caption(entry['saved_at'])
                c2.caption(f"Long: {entry['long_display']}")
                c2.caption(f"Short: {entry['short_display']}")
                if c3.button("Load", key=f"sv_load_{idx}"):
                    # Stage the flags so the sidebar checkboxes update on next render
                    st.session_state['_load_pending'] = {
                        'long':  entry['long_flags'],
                        'short': entry['short_flags'],
                    }
                    st.rerun()
                if c4.button("🗑", key=f"sv_del_{idx}"):
                    delete_portfolio(entry['name'])
                    st.rerun()
                st.divider()

    # ── Search Parameters ─────────────────────────────────────────────────────
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
        n_combos = estimate_combinations(int(min_long), int(max_long), int(min_short), int(max_short))
        st.metric("Combinations", f"{n_combos:,}")
        est_secs = max(1, int(n_combos / 80_000))
        st.caption(f"Est. run time: ~{est_secs}s")

    # ── Metric Filters ────────────────────────────────────────────────────────
    with st.expander("Metric filters  (leave unchecked to rank without filtering)"):
        st.caption(
            "ReturnSD = annualised Sharpe  |  TrendVolRatio = |trend slope| / vol  |  "
            "ReturnTopology = return skewness  |  FitDataMinMaxSD = price range in SDs  |  "
            "LastSD = distance from rolling mean in SDs"
        )
        filter_cols = st.columns(len(METRICS))
        active_filters: dict = {}
        for col, (name, higher_better, default_limit) in zip(filter_cols, METRICS):
            with col:
                use = st.checkbox(name, key=f'f_use_{name}')
                limit = st.number_input(
                    "Limit", value=float(default_limit),
                    step=0.1, key=f'f_lim_{name}',
                    label_visibility='collapsed',
                )
                if use:
                    # direction=1 means "include if metric >= limit", -1 means "<= limit"
                    active_filters[name] = (1 if higher_better else -1, limit)

    # ── Scoring mode ──────────────────────────────────────────────────────────
    s_score_col, _ = st.columns([2, 2])
    with s_score_col:
        scoring_mode = st.selectbox(
            "Ranking method",
            list(SCORING_MODES.keys()),
            format_func=lambda x: SCORING_MODES[x],
            key='s_scoring_mode',
        )
    with st.expander("Scoring methodology (Q11 research)"):
        st.info(
            "**Q11 walk-forward finding:** The original composite score has no significant "
            "predictive power for out-of-sample performance (Spearman ρ = −0.037, p = 0.15). "
            "FitDataMinMaxSD was removed — it is a *negative* predictor (ρ = −0.129, p < 0.001). "
            "**Cost-Based** ranking is recommended: all pairs have similar gross returns (~83% GWR), "
            "so selecting the cheapest to hold maximises net expectancy. "
            "Use the Walk-Forward tab to validate any scoring approach before trading."
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    run_col, top_col = st.columns([2, 1])
    with run_col:
        run_btn = st.button("▶  Run Search", type="primary", use_container_width=True)
    with top_col:
        top_n = st.number_input("Show top N", 5, 100, 30, key='s_top_n')

    if run_btn:
        if max(max_long, max_short) >= 6:
            st.warning(f"{n_combos:,} combinations — this may take over a minute.")

        progress_bar = st.progress(0.0)
        status_txt   = st.empty()

        def _progress(pct: float):
            progress_bar.progress(pct)
            status_txt.caption(f"Evaluated {pct*100:.0f}% of {n_combos:,} combinations…")

        with st.spinner("Searching…"):
            results = run_search(
                rets, scalings,
                min_long_legs=int(min_long),
                max_long_legs=int(max_long),
                min_short_legs=int(min_short),
                max_short_legs=int(max_short),
                window_days=window_days,
                filters=active_filters or None,
                top_n=int(top_n),
                progress_cb=_progress,
                scoring_mode=scoring_mode,
            )

        progress_bar.progress(1.0)
        status_txt.empty()
        st.session_state['search_results'] = results
        st.success(f"Found **{len(results)}** portfolios passing filters.")

    # ── Results ───────────────────────────────────────────────────────────────
    results = st.session_state.get('search_results')

    if results is not None and not results.empty:
        st.subheader("Results (ranked by composite score)")

        # Display table — hide internal flag columns used only for loading
        _ordered_cols = [
            'Config', 'Long', 'Short',
            'WinRate', 'Expectancy', 'NetExpectancy', 'EstCost', 'AvgHolding',
            'Trades', 'PayoffRatio',
            'ReturnSD', 'TrendVolRatio', 'ReturnTopology', 'FitDataMinMaxSD', 'LastSD',
        ]
        display_cols = [c for c in _ordered_cols if c in results.columns]
        disp = results[display_cols].copy()
        _pct_cols = {'WinRate'}
        _int_cols = {'Trades'}
        for c in disp.columns:
            if c in ('Config', 'Long', 'Short'):
                continue
            elif c in _pct_cols:
                disp[c] = disp[c].map('{:.1%}'.format)
            elif c in _int_cols:
                disp[c] = disp[c].map('{:.0f}'.format)
            else:
                disp[c] = disp[c].map('{:.3f}'.format)

        _tbl(disp, show_index=True)

        # ── Load result into main analysis ────────────────────────────────────
        st.markdown("---")
        st.subheader("Load result into Analysis")

        rank = st.number_input(
            "Rank # to load (from table above)", 1, len(results), 1, key='s_load_rank'
        )
        row = results.iloc[rank - 1]

        lcol, rcol = st.columns(2)
        lcol.markdown(f"**Long:** {row['Long']}")
        rcol.markdown(f"**Short:** {row['Short']}")

        lcol_btn, rcol_btn = st.columns(2)
        if lcol_btn.button("✅ Load to Analysis tabs", use_container_width=True):
            st.session_state['_load_pending'] = {
                'long':  row['_long_flags'],
                'short': row['_short_flags'],
            }
            st.rerun()
        if rcol_btn.button("🔬 Backtest this", key=f's_bt_{rank}', use_container_width=True):
            st.session_state['bt_pending'] = {
                'long_flags':  row['_long_flags'],
                'short_flags': row['_short_flags'],
            }
            st.rerun()

        # ── Save Portfolio ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💾 Save Portfolio")
        default_name = f"{row['Long']} / {row['Short']}"
        save_name = st.text_input("Label", value=default_name, key='s_save_name', max_chars=80)
        if st.button("💾 Save", key='s_save_btn', use_container_width=True):
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
        st.warning("No portfolios passed the current filters — try relaxing the limits.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Live / Intraday Monitor
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Intraday Monitor")
    st.caption("Spread return from yesterday's close vs. ±1 SD historical daily range. Data cached 5 minutes.")

    any_selected = any(long_flags.values()) or any(short_flags.values())
    if not any_selected:
        st.info("Select instruments in the sidebar to monitor the live spread.")
    else:
        lcol, rcol = st.columns([1, 1])
        if lcol.button("🔄 Refresh", key='live_refresh'):
            force_intraday_refresh()
            st.rerun()

        interval_map = {'1 min': '1m', '2 min': '2m', '5 min': '5m', '15 min': '15m'}
        interval_label = rcol.selectbox("Interval", list(interval_map.keys()), index=2, key='live_interval')
        interval = interval_map[interval_label]

        with st.spinner("Fetching intraday prices…"):
            intraday = load_intraday_prices(interval)

        if intraday is None or intraday.empty:
            st.error("Could not load intraday prices — markets may be closed or outside trading hours.")
        else:
            # Strip timezone so Plotly renders timestamps cleanly
            if hasattr(intraday.index, 'tz') and intraday.index.tz is not None:
                intraday = intraday.copy()
                intraday.index = intraday.index.tz_convert(None)

            # Pivot = last close before today's session (yesterday's close)
            today_date = pd.Timestamp.today().normalize()
            pivot = prices.iloc[-2] if prices.index[-1].normalize() >= today_date else prices.iloc[-1]

            spread = intraday_spread(intraday, pivot, latest_scalings, long_flags, short_flags)

            if spread.empty:
                st.warning("No matching instruments found in intraday data.")
            else:
                hist_vol    = port_ret.tail(_TDY).std() if not port_ret.empty else 0.0
                hist_vol_5d = port_ret.rolling(5).sum().tail(_TDY).std() if len(port_ret) >= 5 else 0.0

                current_val = float(spread.iloc[-1])
                current_sd  = (current_val / hist_vol) if hist_vol > 0 else 0.0
                last_ts = spread.index[-1].strftime('%H:%M') if hasattr(spread.index[-1], 'strftime') else str(spread.index[-1])

                m1, m2, m3 = st.columns(3)
                m1.metric("Current spread",  f"{current_val:+.2%}")
                m2.metric("SDs from zero",   f"{current_sd:+.2f}",
                          delta=f"±1 SD = {hist_vol:.2%}")
                m3.metric("Last tick",        last_ts)

                # ── Intraday Chart ────────────────────────────────────────────
                spread_pct = spread * 100
                sd_pct     = hist_vol * 100

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=spread_pct.index, y=spread_pct.values,
                    name='Spread', line=dict(color='royalblue', width=2),
                    fill='tozeroy', fillcolor='rgba(65,105,225,0.08)',
                ))
                fig.add_trace(go.Scatter(
                    x=[spread_pct.index[-1]], y=[float(spread_pct.iloc[-1])],
                    mode='markers', marker=dict(size=10, color='orange', symbol='circle'),
                    name='Current',
                ))
                if hist_vol_5d > 0:
                    sd5_pct = hist_vol_5d * 100
                    fig.add_hline(y= sd5_pct, line_dash='dot', line_color='lightsalmon',
                                  annotation_text='+5D SD', annotation_position='top left')
                    fig.add_hline(y=-sd5_pct, line_dash='dot', line_color='lightseagreen',
                                  annotation_text='-5D SD', annotation_position='bottom left')
                fig.add_hline(y= sd_pct, line_dash='dash', line_color='crimson',
                              annotation_text='+1 SD', annotation_position='top right')
                fig.add_hline(y=-sd_pct, line_dash='dash', line_color='seagreen',
                              annotation_text='-1 SD', annotation_position='bottom right')
                fig.add_hline(y=0, line_dash='dot', line_color='grey')
                fig.update_layout(
                    title="Intraday Spread Return from Yesterday's Close",
                    yaxis_title='Spread Return (%)',
                    xaxis_title='Time',
                    height=420,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Last data point: {spread.index[-1]}  |  Historical daily 1-SD: ±{hist_vol:.2%}  |  5D SD: ±{hist_vol_5d:.2%}")

                # ── Intraday Volatility Tracker ───────────────────────────────
                # Compares today's intrabar vol against historical daily vol for each selected instrument
                st.subheader("Intraday Volatility by Instrument")
                _sel_insts = [
                    i for i in ACTIVE_INSTRUMENTS
                    if (long_flags.get(i) or short_flags.get(i)) and i in intraday.columns
                ]
                if _sel_insts:
                    _ivol_rows = []
                    for _inst in _sel_insts:
                        _ir = intraday[_inst].pct_change().dropna()
                        if _ir.empty:
                            continue
                        _tv = float(_ir.std())
                        _hv2 = float(latest_vols.get(_inst, float('nan')))
                        # Ratio > 1.5 flags unusually high intraday activity
                        _ratio = (_tv / _hv2) if (pd.notna(_hv2) and _hv2 > 0) else float('nan')
                        _ivol_rows.append({
                            'Instrument': DISPLAY_NAMES.get(_inst, _inst),
                            'Side':       'Long' if long_flags.get(_inst) else 'Short',
                            'Today Vol':  f'{_tv*100:.3f}%',
                            'Hist Vol':   f'{_hv2*100:.2f}%' if pd.notna(_hv2) else 'N/A',
                            'Ratio':      f'{_ratio:.2f}x' if pd.notna(_ratio) else 'N/A',
                            'Flag':       'HIGH VOL' if (pd.notna(_ratio) and _ratio > 1.5) else 'normal',
                        })
                    if _ivol_rows:
                        _tbl(pd.DataFrame(_ivol_rows), show_index=False)


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Trade Journal
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Trade Journal")

    _j_cur_px = {k: float(v) for k, v in latest_prices.items() if pd.notna(v)}

    # Initialise session state for close confirmation flow and leg editor
    st.session_state.setdefault('pending_close_id', None)
    st.session_state.setdefault('j_legs', [
        {'buy_instrument':   ACTIVE_INSTRUMENTS[0], 'buy_entry_price': 0.0, 'buy_stake': 1.0,
         'sell_instrument':  ACTIVE_INSTRUMENTS[1], 'sell_entry_price': 0.0, 'sell_stake': 1.0}
    ])

    # ════ OPEN POSITIONS ══════════════════════════════════════════════════════
    st.subheader("Open Positions")
    open_trades = [t for t in all_trades if t['status'] == 'open']

    if not open_trades:
        st.caption("No open positions.")
    else:
        for trade in open_trades:
            live_pnl  = trade_live_pnl(trade, _j_cur_px)
            real_pnl  = trade.get('realised_pnl', 0.0) or 0.0
            total_pnl = live_pnl + real_pnl

            # Detect legs whose instruments are missing from the current price snapshot
            _missing_px = {
                leg[k] for leg in trade.get('legs', [])
                for k in ('buy_instrument', 'sell_instrument')
                if leg.get('pct_open', 0) > 1e-6 and leg[k] not in _j_cur_px
            }

            # Trade header row: name, comments, and P&L metrics
            hcols = st.columns([4, 2, 2, 2, 1])
            hcols[0].markdown(f"**{trade['name']}**")
            if trade.get('comments'):
                hcols[0].caption(trade['comments'])
            hcols[0].caption(
                f"Opened {trade['entry_date']}  ·  "
                f"Target £{trade.get('target_exposure', 0):,.0f}"
            )
            hcols[1].metric("Unrealised",  f"£{live_pnl:+,.0f}")
            hcols[2].metric("Realised",    f"£{real_pnl:+,.0f}")
            hcols[3].metric("Total P&L",   f"£{total_pnl:+,.0f}")
            if _missing_px:
                st.caption(f"⚠️ No current price for {', '.join(_missing_px)} — unrealised P&L is understated.")

            # Two-step confirmation for closing a trade (click once to stage, once to confirm)
            if trade['id'] == st.session_state['pending_close_id']:
                st.warning(
                    f"Close **all legs** of **{trade['name']}** at current prices?  "
                    f"Estimated P&L = **£{total_pnl:+,.0f}**"
                )
                _ca, _cb = st.columns(2)
                if _ca.button("✅ Confirm close all", key=f"confirm_{trade['id']}"):
                    close_trade(trade['id'], _j_cur_px, date.today().isoformat())
                    st.session_state['pending_close_id'] = None
                    st.rerun()
                if _cb.button("Cancel", key=f"cancel_{trade['id']}"):
                    st.session_state['pending_close_id'] = None
                    st.rerun()
            else:
                if hcols[4].button("Close all", key=f"close_{trade['id']}"):
                    st.session_state['pending_close_id'] = trade['id']
                    st.rerun()

            # ── Leg Detail Expander ───────────────────────────────────────────
            with st.expander(f"Legs — {len(trade['legs'])} pair(s)"):
                for leg in trade['legs']:
                    bi, si   = leg['buy_instrument'], leg['sell_instrument']
                    b_cur    = _j_cur_px.get(bi, leg['buy_entry_price'])
                    s_cur    = _j_cur_px.get(si, leg['sell_entry_price'])
                    # Live P&L for this individual leg
                    leg_live = (
                        (b_cur - leg['buy_entry_price'])  * leg['buy_stake']  * leg['pct_open'] * POINT_SIZES.get(bi, 1.0)
                      - (s_cur - leg['sell_entry_price']) * leg['sell_stake'] * leg['pct_open'] * POINT_SIZES.get(si, 1.0)
                    )
                    lc = st.columns([3, 3, 2, 2])
                    lc[0].markdown(
                        f"**Buy** {DISPLAY_NAMES.get(bi, bi)}  "
                        f"@ {leg['buy_entry_price']:,.1f} × {leg['buy_stake']}"
                    )
                    lc[1].markdown(
                        f"**Sell** {DISPLAY_NAMES.get(si, si)}  "
                        f"@ {leg['sell_entry_price']:,.1f} × {leg['sell_stake']}"
                    )
                    lc[2].metric("Live P&L", f"£{leg_live:+,.0f}")
                    lc[3].metric("% Open", f"{leg['pct_open']*100:.0f}%")

                    # Partial close sub-form (only shown if the leg still has open fraction)
                    if leg['pct_open'] > 1e-6:
                        with st.expander(f"Partial close — Leg {leg['leg_id']}"):
                            pc1, pc2, pc3, pc4 = st.columns(4)
                            pct_slider = pc1.slider(
                                "% of open to close", 10, 100, 100, 10,
                                key=f"pc_pct_{trade['id']}_{leg['leg_id']}"
                            )
                            buy_ep  = pc2.number_input(
                                f"Exit {DISPLAY_NAMES.get(bi, bi)}",
                                value=round(b_cur, 1), step=1.0,
                                key=f"pc_buy_{trade['id']}_{leg['leg_id']}"
                            )
                            sell_ep = pc3.number_input(
                                f"Exit {DISPLAY_NAMES.get(si, si)}",
                                value=round(s_cur, 1), step=1.0,
                                key=f"pc_sell_{trade['id']}_{leg['leg_id']}"
                            )
                            if pc4.button(
                                "Confirm", key=f"pc_btn_{trade['id']}_{leg['leg_id']}"
                            ):
                                partial_close_leg(
                                    trade['id'], leg['leg_id'],
                                    pct_slider / 100.0,
                                    float(buy_ep), float(sell_ep),
                                    date.today().isoformat(),
                                )
                                st.rerun()

                    # Close history table for this leg
                    if leg['closes']:
                        _cr = pd.DataFrame(leg['closes'])
                        _cr['pnl']        = _cr['pnl'].map('£{:+,.0f}'.format)
                        _cr['pct_closed'] = (_cr['pct_closed'] * 100).map('{:.0f}%'.format)
                        st.dataframe(
                            _cr.rename(columns={
                                'date': 'Date', 'pct_closed': '% Closed',
                                'buy_exit_price': 'Buy exit',
                                'sell_exit_price': 'Sell exit', 'pnl': 'P&L',
                            }),
                            use_container_width=True, hide_index=True,
                        )

            st.divider()

    # ════ OPEN NEW TRADE ══════════════════════════════════════════════════════
    st.subheader("Open New Trade")

    _nt1, _nt2 = st.columns([3, 1])
    _trade_name      = _nt1.text_input("Trade name", key='j_trade_name')
    _target_exp      = _nt2.number_input("Target 1 SD exposure (£)", value=500, step=50,
                                          min_value=0, key='j_t_exp')
    _trade_comments  = st.text_input("Comments (optional)", key='j_comments')
    _entry_date_new  = st.date_input("Entry date", value=date.today(), key='j_entry_date_new')

    st.markdown("**Legs** — each row is one buy / sell pair")

    # Legs are stored in session state so adding/removing rows survives reruns
    _legs_state = st.session_state['j_legs']
    _updated_legs = []

    for _idx, _leg in enumerate(_legs_state):
        _lc = st.columns([1, 2, 1, 1, 2, 1, 0.4])
        _lc[0].markdown(f"**Leg {_idx + 1}**")

        _bi_def = _leg.get('buy_instrument', ACTIVE_INSTRUMENTS[0])
        _bi_idx = ACTIVE_INSTRUMENTS.index(_bi_def) if _bi_def in ACTIVE_INSTRUMENTS else 0
        _buy_inst = _lc[1].selectbox(
            "Buy", ACTIVE_INSTRUMENTS, index=_bi_idx,
            key=f'j_buy_inst_{_idx}',
            format_func=lambda x: DISPLAY_NAMES.get(x, x),
        )
        _bp_def = float(latest_prices.get(_buy_inst, 0.0))
        if np.isnan(_bp_def):
            _bp_def = 0.0
        _buy_price = _lc[2].number_input(
            "Price", value=round(_bp_def, 1), step=1.0, key=f'j_buy_price_{_idx}'
        )
        _buy_stake = _lc[3].number_input(
            "Stake", value=float(_leg.get('buy_stake', 1.0)),
            step=0.5, min_value=0.0, key=f'j_buy_stake_{_idx}'
        )

        _si_def = _leg.get('sell_instrument', ACTIVE_INSTRUMENTS[1] if len(ACTIVE_INSTRUMENTS) > 1 else ACTIVE_INSTRUMENTS[0])
        _si_idx = ACTIVE_INSTRUMENTS.index(_si_def) if _si_def in ACTIVE_INSTRUMENTS else min(1, len(ACTIVE_INSTRUMENTS)-1)
        _sell_inst = _lc[4].selectbox(
            "Sell", ACTIVE_INSTRUMENTS, index=_si_idx,
            key=f'j_sell_inst_{_idx}',
            format_func=lambda x: DISPLAY_NAMES.get(x, x),
        )
        _sp_def = float(latest_prices.get(_sell_inst, 0.0))
        if np.isnan(_sp_def):
            _sp_def = 0.0
        _sell_price = _lc[5].number_input(
            "Price", value=round(_sp_def, 1), step=1.0, key=f'j_sell_price_{_idx}'
        )
        _sell_stake = _lc[6].number_input(
            "Stake", value=float(_leg.get('sell_stake', 1.0)),
            step=0.5, min_value=0.0, key=f'j_sell_stake_{_idx}'
        )

        _updated_legs.append({
            'buy_instrument':   _buy_inst,  'buy_entry_price':  _buy_price,  'buy_stake':  _buy_stake,
            'sell_instrument':  _sell_inst, 'sell_entry_price': _sell_price, 'sell_stake': _sell_stake,
        })

    # Single explicit write back to session state after all widgets have been read
    st.session_state['j_legs'] = _updated_legs

    # Remove / Add leg controls
    _al, _rl, _ol = st.columns([1, 1, 2])
    if _al.button("➕ Add leg", key='j_add_leg'):
        _legs_state.append({
            'buy_instrument':   ACTIVE_INSTRUMENTS[0], 'buy_entry_price': 0.0, 'buy_stake': 1.0,
            'sell_instrument':  ACTIVE_INSTRUMENTS[1] if len(ACTIVE_INSTRUMENTS) > 1 else ACTIVE_INSTRUMENTS[0],
            'sell_entry_price': 0.0, 'sell_stake': 1.0,
        })
        st.rerun()
    if _rl.button("✕ Remove last leg", key='j_remove_last') and len(_legs_state) > 1:
        _legs_state.pop()
        st.rerun()

    if _ol.button("📝 Open Trade", type="primary", key='j_open_btn'):
        _errors = []
        if not _trade_name.strip():
            _errors.append("Enter a trade name.")
        for _vi, _vleg in enumerate(_legs_state):
            if _vleg['buy_instrument'] == _vleg['sell_instrument']:
                _errors.append(f"Leg {_vi + 1}: buy and sell instrument cannot be the same.")
            if _vleg['buy_stake'] <= 0 or _vleg['sell_stake'] <= 0:
                _errors.append(f"Leg {_vi + 1}: stakes must be greater than zero.")
            if _vleg['buy_entry_price'] <= 0 or _vleg['sell_entry_price'] <= 0:
                _errors.append(f"Leg {_vi + 1}: entry prices must be greater than zero.")
        for _e in _errors:
            st.error(_e)
        if not _errors:
            open_trade(
                name=_trade_name.strip(),
                legs=list(_legs_state),
                target_exposure=float(_target_exp),
                entry_date=_entry_date_new.isoformat(),
                comments=_trade_comments.strip(),
            )
            # Reset the leg editor to a single blank leg after opening
            st.session_state['j_legs'] = [{
                'buy_instrument':   ACTIVE_INSTRUMENTS[0], 'buy_entry_price': 0.0, 'buy_stake': 1.0,
                'sell_instrument':  ACTIVE_INSTRUMENTS[1] if len(ACTIVE_INSTRUMENTS) > 1 else ACTIVE_INSTRUMENTS[0],
                'sell_entry_price': 0.0, 'sell_stake': 1.0,
            }]
            st.success(f"Trade opened: **{_trade_name.strip()}**")
            st.rerun()

    # ════ TRADE HISTORY ═══════════════════════════════════════════════════════
    st.subheader("Trade History")
    closed_trades = [t for t in all_trades if t['status'] == 'closed']

    if not closed_trades:
        st.caption("No closed trades yet.")
    else:
        _total_pnl = sum(t.get('realised_pnl', 0.0) or 0.0 for t in closed_trades)
        _wins      = sum(1 for t in closed_trades if (t.get('realised_pnl') or 0.0) >= 0)
        st.metric("Total realised P&L", f"£{_total_pnl:+,.0f}",
                  delta=f"{_wins}/{len(closed_trades)} winning trades")

        # Show most recent trades first
        _hist_rows = []
        for _t in reversed(closed_trades):
            _hist_rows.append({
                'Trade':      _t['name'],
                'Legs':       len(_t.get('legs', [])),
                'Entry date': _t['entry_date'],
                'Close date': _t.get('exit_date', ''),
                'P&L':        f"£{(_t.get('realised_pnl') or 0.0):+,.0f}",
                '_id':        _t['id'],
            })
        _hist_df = pd.DataFrame(_hist_rows)
        _tbl(_hist_df.drop(columns=['_id']), show_index=False)

        with st.expander("Delete a closed trade"):
            _del_opts = {
                f"{_t['name']}  [{_t.get('exit_date','')}]  (id:{_t['id'][-6:]})": _t['id']
                for _t in reversed(closed_trades)
            }
            _del_key = st.selectbox("Select trade", list(_del_opts.keys()), key='j_del_select')
            if st.button("🗑 Delete", key='j_del_btn'):
                delete_trade(_del_opts[_del_key])
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — Backtest
# ════════════════════════════════════════════════════════════════════════════
with tab8:
    st.header("Backtest")
    st.caption(
        "Run the crossing signal backtest on any asset class. "
        "Data is loaded from the cache directory CSV files."
    )

    from pathlib import Path as _Path
    _CACHE_DIR = _Path(__file__).parent / 'cache'

    # ── Data source ───────────────────────────────────────────────────────────
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

    # ── Signal parameters ─────────────────────────────────────────────────────
    with st.expander("Signal Parameters", expanded=True):
        bc1, bc2, bc3, bc4 = st.columns(4)
        bt_vol_window = bc1.number_input("Vol window", 50, 500, 262, key='bt_vol')
        bt_xing_sd    = bc2.number_input("Entry SD", 0.5, 5.0, 2.0, 0.5, key='bt_xing')
        bt_exit_sd    = bc3.number_input(
            "Exit SD", 0.0, 2.0, 0.0, 0.5, key='bt_exit',
            help="0.0 = full reversion. 0.5 = partial (shorter holds, less financing).",
        )
        bt_fin_rate   = bc4.number_input(
            "Financing %pa", 0.0, 10.0,
            cfg['financing']['long_rate'] * 100, 0.1, key='bt_fin',
        )

    # ── Basket configuration ──────────────────────────────────────────────────
    st.markdown("**Basket configuration**")
    bk1, bk2, bk3, bk4 = st.columns(4)

    with bk1:
        st.markdown("**Long legs**")
        bt_min_long = st.number_input("Min", 2, 6, 3, key='bt_min_long')
        bt_max_long = st.number_input("Max", 2, 6, 3, key='bt_max_long')

    with bk2:
        st.markdown("**Short legs**")
        bt_symmetric = st.checkbox("Same as long", value=True, key='bt_symmetric')
        if bt_symmetric:
            bt_min_short, bt_max_short = int(bt_min_long), int(bt_max_long)
        else:
            bt_min_short = st.number_input("Min", 2, 6, 3, key='bt_min_short')
            bt_max_short = st.number_input("Max", 2, 6, 3, key='bt_max_short')

    with bk3:
        bt_sample = st.number_input(
            "Sample size (0 = exhaustive)", 0, 50000, 2000, key='bt_sample',
            help="Randomly sample this many combinations. 0 runs all combinations.",
        )
        bt_top_n = st.number_input("Show top N", 5, 100, 30, key='bt_top_n')

    with bk4:
        st.markdown("**History window**")
        bt_window_opts = {'1 year': 262, '2 years': 524, '3 years': 786, '5 years': 1310, 'All': 0}
        bt_window_label = st.selectbox("Window", list(bt_window_opts.keys()), index=2, key='bt_window')
        bt_window_days = bt_window_opts[bt_window_label] or None

    # Pre-populate basket from Search tab "Backtest this" button
    _bt_pending = st.session_state.pop('bt_pending', None)

    # ── Scoring mode ──────────────────────────────────────────────────────────
    bt_score_col, _ = st.columns([2, 2])
    with bt_score_col:
        bt_scoring_mode = st.selectbox(
            "Ranking method",
            list(SCORING_MODES.keys()),
            format_func=lambda x: SCORING_MODES[x],
            key='bt_scoring_mode',
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    bt_run = st.button("▶  Run Backtest", type="primary", use_container_width=True, key='bt_run')

    if bt_run:
        if not csv_path.exists() and uploaded is None:
            st.error(
                f"No data file found at `{csv_path}`. "
                "Upload a CSV above or add the file to the cache directory."
            )
            st.stop()

        with st.spinner("Loading prices…"):
            try:
                if uploaded is not None:
                    import io
                    _raw = pd.read_csv(io.BytesIO(uploaded.read()), index_col='Date', parse_dates=True)
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
            _bt_progress_bar = st.progress(0.0)
            _bt_status = st.empty()

            def _bt_progress(pct: float):
                _bt_progress_bar.progress(pct)
                _bt_status.caption(f"Backtested {pct*100:.0f}% of combinations…")

            _display_names_bt = cfg.get('instruments', {})
            _fin_daily = bt_fin_rate / 100 / 365

            _bt_results = run_exhaustive_search(
                _scaled_bt, _day_ints_bt, _instruments_bt,
                display_names=_display_names_bt,
                min_long_legs=int(bt_min_long),
                max_long_legs=int(bt_max_long),
                min_short_legs=int(bt_min_short),
                max_short_legs=int(bt_max_short),
                vol_window=int(bt_vol_window),
                xing_sd=float(bt_xing_sd),
                exit_sd=float(bt_exit_sd),
                financing_daily_pct=_fin_daily,
                top_n=int(bt_top_n),
                sample_n=int(bt_sample),
                scoring_mode=bt_scoring_mode,
                progress_cb=_bt_progress,
            )

        _bt_progress_bar.progress(1.0)
        _bt_status.empty()
        st.session_state['bt_results_cache'] = {
            'df': _bt_results,
            'asset_key': asset_key,
            'params': {
                'vol_window': int(bt_vol_window),
                'xing_sd': float(bt_xing_sd),
                'exit_sd': float(bt_exit_sd),
                'fin_rate': float(bt_fin_rate),
            },
        }

    # ── Results display ───────────────────────────────────────────────────────
    _bt_cache = st.session_state.get('bt_results_cache')

    if _bt_cache is not None:
        _bt_df = _bt_cache['df']

        if _bt_df.empty:
            st.warning("No combinations produced trades with the current parameters.")
        else:
            st.success(f"Found **{len(_bt_df)}** results.")

            # Summary metrics from top result
            _top = _bt_df.iloc[0]
            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            _m1.metric("Top Trades",      f"{int(_top.get('Trades', 0)):,}")
            _m2.metric("Win Rate",        f"{float(_top.get('WinRate', 0)):.1%}")
            _m3.metric("Expectancy",      f"{float(_top.get('Expectancy', 0)):.3f}")
            _m4.metric("Avg Holding",     f"{float(_top.get('AvgHolding', 0)):.0f}d")
            _m5.metric("Payoff Ratio",    f"{float(_top.get('PayoffRatio', 0)):.2f}")

            # Results table
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
                else:
                    _bt_disp[_c] = _bt_disp[_c].map('{:.3f}'.format)
            _tbl(_bt_disp, show_index=True)

            # ── Charts ────────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Top Result — Trade Analysis")

            # Re-run single backtest on top result to get trade-level data
            _top_long  = list(_top.get('_long_flags', {}).keys()) if '_long_flags' in _bt_df.columns else []
            _top_short = list(_top.get('_short_flags', {}).keys()) if '_short_flags' in _bt_df.columns else []

            if '_long_flags' in _bt_df.columns and not _bt_df.iloc[0]['_long_flags'] == {}:
                try:
                    _top_long_instr  = list(_bt_df.iloc[0]['_long_flags'].keys())
                    _top_short_instr = list(_bt_df.iloc[0]['_short_flags'].keys())
                    _all_instr = list(dict.fromkeys(_top_long_instr + _top_short_instr))

                    # Re-run single backtest to get full trade array
                    if hasattr(_scaled_bt, '__len__') and 'bt_results_cache' in st.session_state:
                        pass  # scaled available from last run in this session

                    with st.spinner("Computing trade details…"):
                        _reload_prices, _reload_instr = load_asset_prices(csv_path) if csv_path.exists() else (_prices_bt, _instruments_bt)
                        _reload_scaled, _reload_day_ints, _reload_index = prepare_returns(
                            _reload_prices, _reload_instr,
                            vol_window=int(bt_vol_window),
                            window_days=bt_window_days,
                        )
                        _long_mask  = np.array([1 if i in _top_long_instr  else 0 for i in _reload_instr], dtype=np.float64)
                        _short_mask = np.array([1 if i in _top_short_instr else 0 for i in _reload_instr], dtype=np.float64)
                        _n_long  = max(1, _long_mask.sum())
                        _n_short = max(1, _short_mask.sum())
                        _spread_ret = (
                            (_reload_scaled @ (_long_mask / _n_long)) -
                            (_reload_scaled @ (_short_mask / _n_short))
                        )
                        _single_bt = run_backtest(
                            _spread_ret, _reload_day_ints,
                            vol_window=int(bt_vol_window),
                            xing_sd=float(bt_xing_sd),
                            exit_sd=float(bt_exit_sd),
                            financing_daily_pct=_fin_daily,
                            n_legs=int(_n_long + _n_short),
                        )

                    _n_bt = _single_bt['n_trades']
                    if _n_bt > 0:
                        _trades_arr = _single_bt['trades_raw'][:_n_bt]
                        from numba_core import COL_GROSS_RETURN, COL_HOLDING_DAYS, COL_SIDE

                        _gross_rets  = _trades_arr[:, COL_GROSS_RETURN]
                        _hold_days   = _trades_arr[:, COL_HOLDING_DAYS]

                        # Trade return distribution
                        _ch1, _ch2 = st.columns(2)
                        with _ch1:
                            _fig_dist = px.histogram(
                                x=_gross_rets, nbins=30,
                                title="Gross Return Distribution",
                                labels={'x': 'Gross Return', 'count': 'Trades'},
                                color_discrete_sequence=['#2c6fad'],
                            )
                            _fig_dist.add_vline(x=0, line_dash='dash', line_color='red')
                            _fig_dist.update_layout(showlegend=False)
                            st.plotly_chart(_fig_dist, use_container_width=True)

                        with _ch2:
                            _fig_hold = px.histogram(
                                x=_hold_days, nbins=30,
                                title="Holding Period Distribution (days)",
                                labels={'x': 'Days', 'count': 'Trades'},
                                color_discrete_sequence=['#2c6fad'],
                            )
                            _p50 = np.median(_hold_days)
                            _p90 = np.percentile(_hold_days, 90)
                            _fig_hold.add_vline(x=_p50, line_dash='dash', line_color='orange',
                                                annotation_text=f"p50={_p50:.0f}d")
                            _fig_hold.add_vline(x=_p90, line_dash='dot',  line_color='red',
                                                annotation_text=f"p90={_p90:.0f}d")
                            st.plotly_chart(_fig_hold, use_container_width=True)

                        # Regime analysis
                        _regime_data = regime_split(
                            _single_bt['trades_raw'], _n_bt, _reload_index,
                            financing_daily_pct=_fin_daily,
                            n_legs=int(_n_long + _n_short),
                        )
                        if _regime_data:
                            st.subheader("Regime Analysis")
                            _reg_df = pd.DataFrame(_regime_data)
                            _fig_reg = px.bar(
                                _reg_df, x='regime', y=['gross_wr', 'avg_gross'],
                                barmode='group',
                                title="Win Rate & Expectancy by Regime",
                                labels={'value': 'Value', 'variable': 'Metric'},
                                color_discrete_sequence=['#2c6fad', '#f0a500'],
                            )
                            st.plotly_chart(_fig_reg, use_container_width=True)

                        # Sensitivity grid
                        with st.expander("Sensitivity Analysis"):
                            st.caption("Parameter sweep across entry SD, exit target, and financing rate.")
                            with st.spinner("Running sensitivity grid…"):
                                _sens_df = sensitivity_grid(
                                    _spread_ret, _reload_day_ints,
                                    vol_window=int(bt_vol_window),
                                    n_legs=int(_n_long + _n_short),
                                )
                            _sens_disp = _sens_df[[
                                'SD_Threshold', 'Exit_Target', 'Financing_Ann_Pct',
                                'n_trades', 'net_wr', 'avg_net', 'avg_holding',
                            ]].copy()
                            _sens_disp.columns = [
                                'Entry SD', 'Exit SD', 'Fin %pa',
                                'Trades', 'Net WR', 'Net Exp', 'Avg Hold',
                            ]
                            for _c in ['Net WR']:
                                _sens_disp[_c] = _sens_disp[_c].map('{:.1%}'.format)
                            for _c in ['Net Exp']:
                                _sens_disp[_c] = _sens_disp[_c].map('{:.4f}'.format)
                            for _c in ['Avg Hold']:
                                _sens_disp[_c] = _sens_disp[_c].map('{:.0f}d'.format)
                            _tbl(_sens_disp, show_index=False)

                except Exception as _e:
                    st.warning(f"Could not compute trade details: {_e}")

            # ── Export ────────────────────────────────────────────────────────
            st.markdown("---")
            if st.button("💾 Export results", key='bt_export'):
                import json as _json
                from datetime import datetime as _dt
                _ts = _dt.now().strftime('%Y%m%d_%H%M%S')
                _export_dir = _Path(__file__).parent / 'backtest_exports'
                _export_dir.mkdir(exist_ok=True)

                # Summary JSON
                _summary = {
                    'timestamp': _ts,
                    'asset_class': asset_key,
                    'params': _bt_cache.get('params', {}),
                    'n_results': len(_bt_df),
                    'top_result': {k: v for k, v in _top.items()
                                   if not k.startswith('_')},
                }
                (_export_dir / f'summary_{_ts}.json').write_text(
                    _json.dumps(_summary, indent=2, default=str)
                )

                # Trade list CSV
                _export_cols = [c for c in _bt_df.columns if not c.startswith('_')]
                _bt_df[_export_cols].to_csv(_export_dir / f'results_{_ts}.csv')

                st.success(
                    f"Exported to `backtest_exports/summary_{_ts}.json` "
                    f"and `results_{_ts}.csv`"
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 9 — Walk-Forward Validation
# ════════════════════════════════════════════════════════════════════════════
with tab9:
    from walkforward import run_walk_forward, summarise_walk_forward

    st.header("Walk-Forward Validation")
    st.caption(
        "Tests whether the selected scoring mode predicts out-of-sample performance. "
        "Reproduces the Q11 research protocol: score all 1v1 pairs on IS data, "
        "evaluate on OOS data, compute Spearman ρ(IS rank, OOS gross return)."
    )

    # ── Parameters ────────────────────────────────────────────────────────────
    wf_col1, wf_col2, wf_col3 = st.columns(3)
    with wf_col1:
        st.markdown("**Window lengths (trading years)**")
        wf_is_years  = st.number_input("IS window (years)",  3, 10, 5, key='wf_is')
        wf_oos_years = st.number_input("OOS window (years)", 1,  5, 2, key='wf_oos')
        wf_step      = st.number_input("Step size (years)",  1,  3, 1, key='wf_step')

    with wf_col2:
        st.markdown("**Signal parameters**")
        wf_xing_sd = st.number_input("Entry SD",  0.5, 5.0, 2.0, 0.5, key='wf_xing')
        wf_exit_sd = st.number_input("Exit SD",   0.0, 2.0, 0.0, 0.5, key='wf_exit')
        wf_vol_win = st.number_input("Vol window", 100, 500, 262, key='wf_vol')

    with wf_col3:
        st.markdown("**Scoring & data**")
        wf_scoring = st.selectbox(
            "Scoring mode",
            list(SCORING_MODES.keys()),
            format_func=lambda x: SCORING_MODES[x],
            key='wf_scoring',
        )
        wf_asset = st.selectbox(
            "Asset class",
            [k for k, _ in ASSET_CLASS_OPTIONS],
            format_func=lambda k: dict(ASSET_CLASS_OPTIONS)[k],
            key='wf_asset',
        )

    # Estimate window count for UI feedback
    from pathlib import Path as _WfPath
    _wf_cache = _WfPath(__file__).parent / 'cache'
    _wf_cfg   = ASSET_CLASSES[wf_asset]
    _wf_csv   = _wf_cache / _wf_cfg['csv_file']

    # ── Run ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    wf_run = st.button("▶  Run Walk-Forward", type="primary", use_container_width=True, key='wf_run')

    if wf_run:
        if not _wf_csv.exists():
            st.error(f"No data file found at `{_wf_csv}`. Run the Backtest tab first to cache this asset class.")
            st.stop()

        with st.spinner("Loading prices…"):
            try:
                _wf_prices, _wf_instruments = load_asset_prices(_wf_csv)
            except Exception as _e:
                st.error(f"Failed to load prices: {_e}")
                st.stop()

        _n_windows_est = max(0, (len(_wf_prices) - wf_is_years * 262 - wf_oos_years * 262) // (wf_step * 262))
        _n_pairs_est   = len(_wf_instruments) * (len(_wf_instruments) - 1)
        st.caption(f"~{_n_windows_est} windows × {_n_pairs_est} pairs = ~{_n_windows_est * _n_pairs_est:,} observations")

        _wf_prog = st.progress(0.0)
        _wf_stat = st.empty()

        def _wf_progress(pct: float):
            _wf_prog.progress(pct)
            _wf_stat.caption(f"Walk-forward: {pct*100:.0f}% complete…")

        with st.spinner("Running walk-forward…"):
            _wf_results = run_walk_forward(
                _wf_prices, _wf_instruments,
                is_years=int(wf_is_years),
                oos_years=int(wf_oos_years),
                step_years=int(wf_step),
                scoring_mode=wf_scoring,
                vol_window=int(wf_vol_win),
                xing_sd=float(wf_xing_sd),
                exit_sd=float(wf_exit_sd),
                progress_cb=_wf_progress,
            )

        _wf_prog.progress(1.0)
        _wf_stat.empty()
        st.session_state['wf_results_cache'] = {
            'results':      _wf_results,
            'scoring_mode': wf_scoring,
            'asset':        wf_asset,
        }

    # ── Results display ───────────────────────────────────────────────────────
    _wf_cache_data = st.session_state.get('wf_results_cache')

    if _wf_cache_data is not None:
        _wf_df   = _wf_cache_data['results']
        _wf_mode = _wf_cache_data['scoring_mode']

        if _wf_df.empty:
            st.warning("No results — insufficient data for the selected window lengths.")
        else:
            _wf_sum = summarise_walk_forward(_wf_df)
            rho     = _wf_sum['rho']
            pval    = _wf_sum['p_value']
            n_obs   = _wf_sum['n_obs']

            # ── Summary metrics ───────────────────────────────────────────────
            st.subheader("Summary")
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Spearman ρ (IS rank vs OOS gross)",
                       f"{rho:+.3f}",
                       help="Positive means IS rank predicts OOS outperformers. "
                            "Near zero means IS scoring has no predictive power.")
            sm2.metric("p-value", f"{pval:.4f}",
                       delta="significant" if pval < 0.05 else "not significant",
                       delta_color="normal" if pval < 0.05 else "off")
            sm3.metric("Observations", f"{n_obs:,}")

            sm4, sm5, sm6 = st.columns(3)
            sm4.metric("Q1 − Q5 OOS Gross",
                       f"{_wf_sum['q1_mean'] - _wf_sum['q5_mean']:+.4f}",
                       help="Positive means IS top quintile outperforms IS bottom quintile OOS.")
            sm5.metric("Q1 vs Q5 t-statistic",
                       f"{_wf_sum['t_stat']:+.3f}",
                       help="Welch t-test: IS best quintile vs IS worst quintile OOS gross return.")
            sm6.metric("Q1 vs Q5 p-value",
                       f"{_wf_sum['t_p']:.4f}",
                       delta="significant" if _wf_sum['t_p'] < 0.05 else "not significant",
                       delta_color="normal" if _wf_sum['t_p'] < 0.05 else "off")

            # ── Interpretation ────────────────────────────────────────────────
            if abs(rho) < 0.1 and pval >= 0.05:
                st.warning(
                    f"**No predictive power** — the '{SCORING_MODES.get(_wf_mode, _wf_mode)}' scoring mode "
                    f"has no significant relationship with OOS performance "
                    f"(ρ = {rho:+.3f}, p = {pval:.4f}). "
                    "Consider Cost-Based ranking as a structural alternative."
                )
            elif rho > 0.1 and pval < 0.05:
                st.success(
                    f"**Positive predictive power** — IS rank positively correlates with OOS gross return "
                    f"(ρ = {rho:+.3f}, p = {pval:.4f}). "
                    "This scoring mode identifies better pairs out-of-sample."
                )
            elif rho < -0.1 and pval < 0.05:
                st.error(
                    f"**Negative predictor** — IS rank is negatively correlated with OOS gross return "
                    f"(ρ = {rho:+.3f}, p = {pval:.4f}). "
                    "IS top-ranked pairs underperform OOS. Consider the Contrarian mode."
                )
            else:
                st.info(f"ρ = {rho:+.3f}, p = {pval:.4f} — weak or marginal result.")

            st.markdown("---")

            # ── Quintile table ────────────────────────────────────────────────
            st.subheader("OOS Performance by IS Quintile")
            st.caption("Q1 = IS top-ranked; Q5 = IS bottom-ranked. A flat table means scoring has no predictive power.")
            if not _wf_sum['quintile_df'].empty:
                _q_disp = _wf_sum['quintile_df'].copy()
                for _c in ['OOS_GrossWR']:
                    _q_disp[_c] = _q_disp[_c].map('{:.1%}'.format)
                for _c in ['OOS_Gross', 'OOS_Net']:
                    _q_disp[_c] = _q_disp[_c].map('{:.4f}'.format)
                for _c in ['OOS_AvgHold']:
                    _q_disp[_c] = _q_disp[_c].map('{:.0f}d'.format)
                _tbl(_q_disp, show_index=True)

            # ── Bar chart: OOS Gross by quintile ─────────────────────────────
            if not _wf_sum['quintile_df'].empty:
                _q_plot = _wf_sum['quintile_df'].reset_index()
                _fig_q = px.bar(
                    _q_plot, x='Q', y='OOS_Gross',
                    title="OOS Gross Return by IS Quintile (flat = no predictive power)",
                    labels={'Q': 'IS Quintile', 'OOS_Gross': 'Mean OOS Gross Return'},
                    color_discrete_sequence=['#2c6fad'],
                )
                _fig_q.add_hline(
                    y=float(_wf_sum['quintile_df']['OOS_Gross'].mean()),
                    line_dash='dash', line_color='orange',
                    annotation_text='Mean',
                )
                _fig_q.update_layout(showlegend=False)
                st.plotly_chart(_fig_q, use_container_width=True)

            st.markdown("---")

            # ── Scatter plot: IS rank vs OOS gross ────────────────────────────
            st.subheader("IS Rank vs OOS Gross Return")
            st.caption("No pattern = no predictive power. Each point is one pair in one walk-forward window.")
            _valid = _wf_sum.get('valid', pd.DataFrame())
            if not _valid.empty:
                _fig_sc = px.scatter(
                    _valid, x='IS_Rank', y='OOS_Gross',
                    color='window',
                    opacity=0.5,
                    trendline='ols',
                    title=f"IS Rank vs OOS Gross Return (ρ = {rho:+.3f}, p = {pval:.4f})",
                    labels={'IS_Rank': 'IS Rank (1 = best)', 'OOS_Gross': 'OOS Gross Return'},
                )
                _fig_sc.add_hline(y=0, line_dash='dash', line_color='red', opacity=0.5)
                _fig_sc.update_layout(coloraxis_showscale=False)
                st.plotly_chart(_fig_sc, use_container_width=True)

            # ── Per-window ρ table ────────────────────────────────────────────
            if not _wf_sum['window_df'].empty:
                with st.expander("Per-window Spearman ρ"):
                    _tbl(_wf_sum['window_df'], show_index=False)

            # ── Raw data download ─────────────────────────────────────────────
            st.markdown("---")
            if st.button("💾 Export walk-forward results", key='wf_export'):
                _wf_export = _wf_df[[c for c in _wf_df.columns if not c.startswith('_')]].copy()
                import json as _wfjson
                from datetime import datetime as _wfdt
                _wf_ts  = _wfdt.now().strftime('%Y%m%d_%H%M%S')
                _wf_dir = _WfPath(__file__).parent / 'backtest_exports'
                _wf_dir.mkdir(exist_ok=True)
                _wf_export.to_csv(_wf_dir / f'walkforward_{_wf_ts}.csv', index=False)
                st.success(f"Exported to `backtest_exports/walkforward_{_wf_ts}.csv`")
