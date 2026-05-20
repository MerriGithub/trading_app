from __future__ import annotations

import pandas as pd
import streamlit as st

from engine.backtest import load_asset_prices
from engine.calculations import returns as _eq_returns, scaling_vectors as _eq_scalings_fn
from engine.search import METRICS, METRIC_NAMES, estimate_combinations, run_search
from engine.scoring import SCORING_MODES
from engine.saved import load_saved, save_portfolio, delete_portfolio
from tabs.shared import _CACHE_DIR, _tbl


@st.cache_resource
def _load_equity_data():
    prices, _ = load_asset_prices(_CACHE_DIR / 'prices.csv')
    rets = _eq_returns(prices)
    scl  = _eq_scalings_fn(prices, rets)
    return prices, rets, scl


def render() -> None:
    st.header("Portfolio Search")
    st.caption("Equity-only. Enumerates long/short combinations across the 12 equity indices.")

    try:
        _eq_prices, _eq_rets, _eq_scl = _load_equity_data()
    except Exception as e:
        st.error(f"Could not load equity data: {e}")
        return

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
        window_opts  = {'6 months': 131, '1 year': 262, '2 years': 524, '3 years': 786}
        window_label = st.selectbox("Window", list(window_opts.keys()), index=1, key='s_window')
        window_days  = window_opts[window_label]
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
    if results is None:
        return

    if results.empty:
        st.warning("No portfolios passed the filters.")
        return

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
    row  = results.iloc[rank - 1]
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
