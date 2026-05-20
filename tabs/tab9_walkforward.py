from __future__ import annotations

import streamlit as st

from engine.walkforward import run_walk_forward, summarise_walk_forward
from engine.backtest import load_asset_prices as _wf_load_asset_prices
from engine.scoring import SCORING_MODES as _WF_SCORING_MODES
from asset_configs import ASSET_CLASSES, ASSET_CLASS_OPTIONS, FI_EXCLUDE
from tabs.shared import _CACHE_DIR, _tbl


def render() -> None:
    st.header("Trade Validation")
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
            st.error(
                f"No data file at `{_wf_csv}`. "
                "Run the Backtest tab first to cache this asset class."
            )
            return

        with st.spinner("Loading prices…"):
            try:
                _wf_prices, _wf_instruments = _wf_load_asset_prices(_wf_csv)
            except Exception as e:
                st.error(f"Failed to load prices: {e}")
                return

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
    if _wf_cache_data is None:
        return

    _wf_df   = _wf_cache_data['results']
    _wf_mode = _wf_cache_data['scoring_mode']
    if _wf_df.empty:
        st.warning("No results — insufficient data for the selected window lengths.")
        return

    _wf_sum = summarise_walk_forward(_wf_df)
    rho, pval, n_obs = _wf_sum['rho'], _wf_sum['p_value'], _wf_sum['n_obs']
    st.subheader("Summary")
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Spearman ρ (IS rank vs OOS gross)", f"{rho:+.3f}")
    sm2.metric("p-value",      f"{pval:.4f}")
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
