from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from calculations import (
    correlation_matrix, crossing_signals, intraday_spread, linear_trend,
    portfolio_returns, portfolio_stats, returns,
    rolling_nd_returns, rolling_volatility, scaling_vectors,
    velocity_acceleration,
)
from config import ACTIVE_INSTRUMENTS, DISPLAY_NAMES, PARAMS, SPREADS
from data import force_intraday_refresh, force_refresh, load_intraday_prices, load_prices
from journal import close_trade, delete_trade, load_trades, open_trade
from saved import delete_portfolio, load_saved, save_portfolio
from search import METRIC_NAMES, METRICS, estimate_combinations, run_search
from stake_calc import compute_stakes, pnl_scenario

_TDY = PARAMS['trading_days_per_year']

st.set_page_config(page_title="Trading Monitor", page_icon="📈", layout="wide")


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

# If the Search tab loaded a portfolio, pre-set the checkbox keys before render
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
    start_year = st.slider("History start year", 1999, 2020, 1999)

    st.markdown("---")
    st.markdown("### Signal Selection")
    st.caption("Tick instruments for each leg of the spread trade.")

    c1, c2 = st.columns(2)
    c1.markdown("**Long ▲**")
    c2.markdown("**Short ▼**")

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

rets     = returns(prices)
vols     = rolling_volatility(rets)
scalings = scaling_vectors(prices, rets)
port_ret = portfolio_returns(rets, scalings, long_flags, short_flags)

latest_prices   = prices.iloc[-1]
latest_vols     = vols.iloc[-1]
latest_scalings = scalings.iloc[-1]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Dashboard", "📈 Analysis", "🧮 Stake Calculator", "🗂 Portfolio", "🔍 Search", "⏱ Live", "📓 Journal"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dashboard")
    last_date = prices.index[-1].strftime('%d %b %Y')
    st.caption(f"Last data: {last_date}  |  {len(prices)} trading days loaded")

    # Price / signal table
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

    # Spread summary metrics
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

    display_list = [DISPLAY_NAMES.get(i, i) for i in ACTIVE_INSTRUMENTS]
    selected_display = st.multiselect("Instruments to plot", display_list, default=display_list[:4])
    selected = [k for k, v in DISPLAY_NAMES.items() if v in selected_display]

    if selected:
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

        # Velocity & acceleration
        va = velocity_acceleration(port_ret)
        fig_va = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=('Velocity (ROC)', 'Acceleration'))
        fig_va.add_trace(go.Scatter(x=va.index, y=va['velocity'],    name='Velocity'),    row=1, col=1)
        fig_va.add_trace(go.Scatter(x=va.index, y=va['acceleration'], name='Acceleration'), row=2, col=1)
        fig_va.update_layout(height=380, title="Trend Velocity & Acceleration", showlegend=False)
        st.plotly_chart(fig_va, use_container_width=True)

        # Crossing signals
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
    else:
        st.info("Select instruments in the sidebar to see trend analysis.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Stake Calculator
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Stake Calculator")
    st.caption("Prices are pre-filled from the latest data — edit if you want to run scenarios.")

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

    # Correlation heatmap
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

    # Volatility & scaling bar chart
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

    # Portfolio performance (if signals selected)
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
                    st.session_state['_load_pending'] = {
                        'long':  entry['long_flags'],
                        'short': entry['short_flags'],
                    }
                    st.rerun()
                if c4.button("🗑", key=f"sv_del_{idx}"):
                    delete_portfolio(entry['name'])
                    st.rerun()
                st.divider()

    # ── Parameters ──────────────────────────────────────────────────────────
    pcol1, pcol2, pcol3 = st.columns(3)

    with pcol1:
        st.markdown("**Legs per side**")
        min_legs = st.number_input("Min", 2, 6, 3, key='s_min_legs')
        max_legs = st.number_input("Max", 2, 6, 4, key='s_max_legs')
        if max_legs < min_legs:
            st.warning("Max must be ≥ Min")
            max_legs = min_legs

    with pcol2:
        st.markdown("**History window**")
        window_opts = {'6 months': 131, '1 year': 262, '2 years': 524, '3 years': 786}
        window_label = st.selectbox("Window", list(window_opts.keys()), index=1)
        window_days = window_opts[window_label]

    with pcol3:
        st.markdown("**Scale**")
        n_combos = estimate_combinations(min_legs, max_legs)
        st.metric("Combinations", f"{n_combos:,}")
        est_secs = max(1, int(n_combos / 80_000))
        st.caption(f"Est. run time: ~{est_secs}s")

    # ── Metric filters (optional) ────────────────────────────────────────────
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
                    active_filters[name] = (1 if higher_better else -1, limit)

    # ── Run ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    run_col, top_col = st.columns([2, 1])
    with run_col:
        run_btn = st.button("▶  Run Search", type="primary", use_container_width=True)
    with top_col:
        top_n = st.number_input("Show top N", 5, 100, 30, key='s_top_n')

    if run_btn:
        if max_legs >= 6:
            st.warning(f"{n_combos:,} combinations — this may take over a minute.")

        progress_bar = st.progress(0.0)
        status_txt   = st.empty()

        def _progress(pct: float):
            progress_bar.progress(pct)
            status_txt.caption(f"Evaluated {pct*100:.0f}% of {n_combos:,} combinations…")

        with st.spinner("Searching…"):
            results = run_search(
                rets, scalings,
                min_legs=int(min_legs),
                max_legs=int(max_legs),
                window_days=window_days,
                filters=active_filters or None,
                top_n=int(top_n),
                progress_cb=_progress,
            )

        progress_bar.progress(1.0)
        status_txt.empty()
        st.session_state['search_results'] = results
        st.success(f"Found **{len(results)}** portfolios passing filters.")

    # ── Results ───────────────────────────────────────────────────────────────
    results = st.session_state.get('search_results')

    if results is not None and not results.empty:
        st.subheader("Results (ranked by composite score)")

        # Display table — hide internal flag columns
        display_cols = ['Long', 'Short'] + METRIC_NAMES
        disp = results[display_cols].copy()
        for m in METRIC_NAMES:
            disp[m] = disp[m].map('{:.2f}'.format)

        _tbl(disp, show_index=True)

        # ── Load a result into the main analysis ─────────────────────────────
        st.markdown("---")
        st.subheader("Load result into Analysis")

        rank = st.number_input(
            "Rank # to load (from table above)", 1, len(results), 1, key='s_load_rank'
        )
        row = results.iloc[rank - 1]

        lcol, rcol = st.columns(2)
        lcol.markdown(f"**Long:** {row['Long']}")
        rcol.markdown(f"**Short:** {row['Short']}")

        if st.button("✅ Load to Analysis tabs", use_container_width=True):
            st.session_state['_load_pending'] = {
                'long':  row['_long_flags'],
                'short': row['_short_flags'],
            }
            st.rerun()

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
            # Strip timezone info for clean display
            if hasattr(intraday.index, 'tz') and intraday.index.tz is not None:
                intraday = intraday.copy()
                intraday.index = intraday.index.tz_convert(None)

            # Pivot = last close before today's session
            today_date = pd.Timestamp.today().normalize()
            pivot = prices.iloc[-2] if prices.index[-1].normalize() >= today_date else prices.iloc[-1]

            spread = intraday_spread(intraday, pivot, latest_scalings, long_flags, short_flags)

            if spread.empty:
                st.warning("No matching instruments found in intraday data.")
            else:
                hist_vol = port_ret.tail(_TDY).std() if not port_ret.empty else 0.0

                current_val = float(spread.iloc[-1])
                current_sd  = (current_val / hist_vol) if hist_vol > 0 else 0.0
                last_ts = spread.index[-1].strftime('%H:%M') if hasattr(spread.index[-1], 'strftime') else str(spread.index[-1])

                m1, m2, m3 = st.columns(3)
                m1.metric("Current spread",  f"{current_val:+.2%}")
                m2.metric("SDs from zero",   f"{current_sd:+.2f}",
                          delta=f"±1 SD = {hist_vol:.2%}")
                m3.metric("Last tick",        last_ts)

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
                st.caption(f"Last data point: {spread.index[-1]}  |  Historical daily 1-SD: ±{hist_vol:.2%}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Trade Journal
# ════════════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Trade Journal")

    # ── helper: compute cumulative spread level for a set of flags ───────────
    def _cum_spread(lf: dict, sf: dict) -> float:
        pr = portfolio_returns(rets, scalings, lf, sf)
        if pr.empty or pr.isna().all():
            return 1.0
        return float((1 + pr).cumprod().iloc[-1])

    # ── session state for close confirmation ─────────────────────────────────
    if 'pending_close_id' not in st.session_state:
        st.session_state['pending_close_id'] = None

    # ════ OPEN POSITIONS ══════════════════════════════════════════════════════
    st.subheader("Open Positions")
    open_trades = [t for t in load_trades() if t['status'] == 'open']

    if not open_trades:
        st.caption("No open positions.")
    else:
        for trade in open_trades:
            current = _cum_spread(trade['long_flags'], trade['short_flags'])
            sign    = 1 if trade['direction'] == 'Buy' else -1
            unreal  = (current - trade['entry_spread']) * trade['exposure'] * sign

            cols = st.columns([3, 2, 2, 2, 2, 1])
            cols[0].markdown(f"**{trade['name']}**")
            cols[0].caption(f"{trade['direction']} · opened {trade['entry_date']}")
            cols[1].metric("Entry level", f"{trade['entry_spread']:.4f}")
            cols[2].metric("Current level", f"{current:.4f}",
                           delta=f"£{unreal:+,.0f}")
            cols[3].metric("Exposure", f"£{trade['exposure']:,.0f}")
            cols[4].metric("Unrealised P&L", f"£{unreal:+,.0f}")

            # Close / confirm flow
            if trade['id'] == st.session_state['pending_close_id']:
                cols[5].write("")  # spacer
                st.warning(
                    f"Close **{trade['name']}** at {current:.4f}?  "
                    f"Realised P&L = **£{unreal:+,.0f}**"
                )
                ca, cb = st.columns(2)
                if ca.button("✅ Confirm", key=f"confirm_{trade['id']}"):
                    close_trade(trade['id'], current, date.today().isoformat())
                    st.session_state['pending_close_id'] = None
                    st.rerun()
                if cb.button("Cancel", key=f"cancel_{trade['id']}"):
                    st.session_state['pending_close_id'] = None
                    st.rerun()
            else:
                if cols[5].button("Close", key=f"close_{trade['id']}"):
                    st.session_state['pending_close_id'] = trade['id']
                    st.rerun()

            st.divider()

    # ════ OPEN NEW TRADE ══════════════════════════════════════════════════════
    st.subheader("Open New Trade")

    # Build portfolio options: sidebar selection + saved portfolios
    portfolio_options = {}
    sidebar_has_selection = any(long_flags.values()) or any(short_flags.values())
    if sidebar_has_selection:
        long_disp  = ' | '.join(DISPLAY_NAMES.get(k, k) for k, v in long_flags.items()  if v)
        short_disp = ' | '.join(DISPLAY_NAMES.get(k, k) for k, v in short_flags.items() if v)
        portfolio_options['— Current sidebar selection —'] = {
            'long_flags':   {k: v for k, v in long_flags.items()  if v},
            'short_flags':  {k: v for k, v in short_flags.items() if v},
            'long_display': long_disp,
            'short_display': short_disp,
            'name': f"{long_disp} / {short_disp}",
        }
    for s in load_saved():
        portfolio_options[s['name']] = {
            'long_flags':   s['long_flags'],
            'short_flags':  s['short_flags'],
            'long_display': s['long_display'],
            'short_display': s['short_display'],
            'name': s['name'],
        }

    if not portfolio_options:
        st.info("Select instruments in the sidebar or save a portfolio from the Search tab first.")
    else:
        jc1, jc2, jc3 = st.columns([3, 2, 2])

        with jc1:
            chosen_key = st.selectbox("Portfolio", list(portfolio_options.keys()), key='j_portfolio')
            chosen = portfolio_options[chosen_key]

        with jc2:
            direction = st.radio("Direction", ['Buy', 'Sell'], key='j_direction',
                                 help="Buy = long the spread (expect long basket to outperform). Sell = reverse.")
            exposure = st.number_input("Exposure (£)", value=500, step=50, min_value=0, key='j_exposure')

        with jc3:
            auto_level = _cum_spread(chosen['long_flags'], chosen['short_flags'])
            entry_level = st.number_input(
                "Entry spread level", value=round(auto_level, 4),
                format='%.4f', step=0.0001, key='j_entry_level',
                help="Auto-filled from current cumulative spread. Edit if opening at a different level.",
            )
            entry_date = st.date_input("Entry date", value=date.today(), key='j_entry_date')

        st.markdown(f"Long: `{chosen['long_display']}`  ·  Short: `{chosen['short_display']}`")
        if st.button("📝 Open Trade", type="primary", key='j_open_btn'):
            open_trade(
                name         = chosen['name'],
                long_flags   = chosen['long_flags'],
                short_flags  = chosen['short_flags'],
                long_display = chosen['long_display'],
                short_display= chosen['short_display'],
                direction    = direction,
                exposure     = float(exposure),
                entry_spread = float(entry_level),
                entry_date   = entry_date.isoformat(),
            )
            st.success(f"Trade opened: **{chosen['name']}** ({direction})")
            st.rerun()

    # ════ TRADE HISTORY ═══════════════════════════════════════════════════════
    st.subheader("Trade History")
    closed_trades = [t for t in load_trades() if t['status'] == 'closed']

    if not closed_trades:
        st.caption("No closed trades yet.")
    else:
        hist_df = pd.DataFrame([{
            'Portfolio':   t['name'],
            'Direction':   t['direction'],
            'Exposure':    f"£{t['exposure']:,.0f}",
            'Entry level': f"{t['entry_spread']:.4f}",
            'Exit level':  f"{t['exit_spread']:.4f}",
            'Entry date':  t['entry_date'],
            'Close date':  t['exit_date'],
            'P&L':         f"£{t['realised_pnl']:+,.0f}",
            '_id':         t['id'],
        } for t in reversed(closed_trades)])

        total_pnl = sum(t['realised_pnl'] for t in closed_trades)
        wins      = sum(1 for t in closed_trades if t['realised_pnl'] >= 0)
        st.metric("Total realised P&L", f"£{total_pnl:+,.0f}",
                  delta=f"{wins}/{len(closed_trades)} winning trades")

        _tbl(hist_df.drop(columns=['_id']), show_index=False)

        with st.expander("Delete a closed trade"):
            del_name = st.selectbox("Select trade to delete",
                                    [t['name'] + ' · ' + t['exit_date'] for t in reversed(closed_trades)],
                                    key='j_del_select')
            if st.button("🗑 Delete", key='j_del_btn'):
                # Match by exit_date suffix
                target_date = del_name.rsplit(' · ', 1)[-1]
                for t in closed_trades:
                    if t['exit_date'] == target_date and t['name'] in del_name:
                        delete_trade(t['id'])
                        break
                st.rerun()
