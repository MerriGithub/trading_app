from __future__ import annotations

import streamlit as st

from engine.walkforward import run_walk_forward, run_cross_asset_walkforward, summarise_walk_forward
from engine.backtest import load_asset_prices as _wf_load_asset_prices
from engine.scoring import SCORING_MODES as _WF_SCORING_MODES
from asset_configs import (
    ASSET_CLASSES, ASSET_CLASS_OPTIONS, FI_EXCLUDE, COMMODITY_EXCLUDE,
    CROSS_ASSET_COMBINATIONS, _DEFAULT_SCORING_MODE, get_cross_asset_scoring_default,
)
from tabs.shared import _CACHE_DIR, _tbl


def _interpret_rho(rho: float, p: float) -> tuple[str, str, str]:
    """
    Returns (label_text, detail_text, colour_key).
    Separates statistical significance (p-value) from effect magnitude (|ρ|).
    Significance threshold: p < 0.05.
    Magnitude thresholds: |ρ| < 0.05 = negligible, 0.05–0.15 = weak, > 0.15 = moderate+.
    """
    significant = p < 0.05
    mag = abs(rho)

    if not significant:
        return (
            "No predictive power",
            f"ρ = {rho:+.3f}, p = {p:.4f} — not statistically significant.",
            "grey"
        )

    if mag < 0.05:
        magnitude_label = "statistically significant but negligible effect"
    elif mag < 0.15:
        magnitude_label = "statistically significant, weak effect"
    else:
        magnitude_label = "statistically significant, moderate effect"

    if rho > 0:
        return (
            "Positive predictive power",
            f"ρ = {rho:+.3f}, p = {p:.4f} — {magnitude_label}. "
            f"IS rank predicts OOS performance.",
            "green"
        )
    else:
        return (
            "Negative predictor",
            f"ρ = {rho:+.3f}, p = {p:.4f} — {magnitude_label}. "
            f"IS rank is an anti-predictor; consider contrarian mode.",
            "red"
        )


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
        # Auto-default scoring mode when asset class or CA selectors change
        _cur_wf_asset  = st.session_state.get('wf_asset', 'equity')
        _prev_wf_asset = st.session_state.get('wf_asset_prev', _cur_wf_asset)
        if _cur_wf_asset != _prev_wf_asset and _cur_wf_asset in _DEFAULT_SCORING_MODE:
            st.session_state['wf_scoring'] = _DEFAULT_SCORING_MODE[_cur_wf_asset]

        if _cur_wf_asset == 'cross_asset':
            _ss_ca_long  = st.session_state.get('wf_ca_long', 'commodities')
            _ss_ca_short = st.session_state.get('wf_ca_short', 'fx')
            _ca_default  = get_cross_asset_scoring_default(_ss_ca_long, _ss_ca_short)
            if st.session_state.get('_prev_ca_long') != _ss_ca_long \
               or st.session_state.get('_prev_ca_short') != _ss_ca_short:
                st.session_state['wf_scoring'] = _ca_default
                st.session_state['_prev_ca_long'] = _ss_ca_long
                st.session_state['_prev_ca_short'] = _ss_ca_short

        wf_scoring = st.selectbox(
            "Scoring mode", list(_WF_SCORING_MODES.keys()),
            format_func=lambda x: _WF_SCORING_MODES[x], key='wf_scoring',
        )
        wf_asset = st.selectbox(
            "Asset class", [k for k, _ in ASSET_CLASS_OPTIONS],
            format_func=lambda k: dict(ASSET_CLASS_OPTIONS)[k], key='wf_asset',
        )
        st.session_state['wf_asset_prev'] = wf_asset

    # ── Scoring mode recommendation warning ───────────────────────────────────
    if wf_asset == 'cross_asset':
        _ac_labels = dict(ASSET_CLASS_OPTIONS)
        _ca_rec = get_cross_asset_scoring_default(
            st.session_state.get('wf_ca_long', 'commodities'),
            st.session_state.get('wf_ca_short', 'fx'),
        )
        if wf_scoring != _ca_rec:
            st.warning(
                f"⚠️ Walk-forward suggests **{_ca_rec.title()}** mode for "
                f"{_ac_labels.get(st.session_state.get('wf_ca_long', 'commodities'), '')} × "
                f"{_ac_labels.get(st.session_state.get('wf_ca_short', 'fx'), '')} pairs."
            )
    else:
        _wf_rec = _DEFAULT_SCORING_MODE.get(wf_asset)
        if _wf_rec and wf_scoring != _wf_rec:
            st.warning(
                f"⚠️ Walk-forward suggests **{_WF_SCORING_MODES[_wf_rec]}** "
                f"for {dict(ASSET_CLASS_OPTIONS)[wf_asset]} pairs."
            )

    # ── Cross-asset secondary selectors ───────────────────────────────────────
    _ca_long_ac  = 'commodities'
    _ca_short_ac = 'fx'

    if wf_asset == 'cross_asset':
        _valid_long  = sorted({c[0] for c in CROSS_ASSET_COMBINATIONS})
        _valid_short = sorted({c[1] for c in CROSS_ASSET_COMBINATIONS})
        _ac_labels   = dict(ASSET_CLASS_OPTIONS)

        ca_col1, ca_col2 = st.columns(2)
        _ca_long_ac = ca_col1.selectbox(
            "Long leg asset class",
            _valid_long,
            format_func=lambda k: _ac_labels.get(k, k),
            key='wf_ca_long',
        )
        _ca_short_ac = ca_col2.selectbox(
            "Short leg asset class",
            _valid_short,
            format_func=lambda k: _ac_labels.get(k, k),
            key='wf_ca_short',
        )
        if _ca_long_ac == _ca_short_ac:
            st.error("Long and short leg asset classes must differ.")
            return

    st.markdown("---")
    wf_run = st.button("▶ Run walk-forward", type="primary",
                       use_container_width=True, key='wf_run')

    if wf_run:
        if wf_asset == 'cross_asset':
            _long_cfg  = ASSET_CLASSES[_ca_long_ac]
            _short_cfg = ASSET_CLASSES[_ca_short_ac]
            _long_csv  = _CACHE_DIR / _long_cfg['csv_file']
            _short_csv = _CACHE_DIR / _short_cfg['csv_file']

            for _csv, _label in [(_long_csv, _long_cfg['label']),
                                  (_short_csv, _short_cfg['label'])]:
                if not _csv.exists():
                    st.error(
                        f"No data file at `{_csv}`. "
                        f"Run the Backtest tab first to cache {_label}."
                    )
                    return

            with st.spinner("Loading prices…"):
                try:
                    _prices_long,  _instr_long  = _wf_load_asset_prices(_long_csv)
                    _prices_short, _instr_short = _wf_load_asset_prices(_short_csv)
                except Exception as e:
                    st.error(f"Failed to load prices: {e}")
                    return

            # Apply exclusions
            if _ca_long_ac == 'commodities':
                _instr_long  = [i for i in _instr_long  if i not in COMMODITY_EXCLUDE]
                _prices_long = _prices_long[[c for c in _prices_long.columns  if c not in COMMODITY_EXCLUDE]]
            if _ca_long_ac == 'fixed_income':
                _instr_long  = [i for i in _instr_long  if i not in FI_EXCLUDE]
                _prices_long = _prices_long[[c for c in _prices_long.columns  if c not in FI_EXCLUDE]]
            if _ca_short_ac == 'commodities':
                _instr_short  = [i for i in _instr_short if i not in COMMODITY_EXCLUDE]
                _prices_short = _prices_short[[c for c in _prices_short.columns if c not in COMMODITY_EXCLUDE]]
            if _ca_short_ac == 'fixed_income':
                _instr_short  = [i for i in _instr_short if i not in FI_EXCLUDE]
                _prices_short = _prices_short[[c for c in _prices_short.columns if c not in FI_EXCLUDE]]

            _n_pairs_est   = len(_instr_long) * len(_instr_short) * 2
            _common_len    = len(_prices_long.index.intersection(_prices_short.index))
            _n_windows_est = max(
                0,
                (_common_len - int(wf_is_years) * 262 - int(wf_oos_years) * 262)
                // (int(wf_step) * 262),
            )
            st.caption(
                f"~{_n_windows_est} windows × {_n_pairs_est} pairs = "
                f"~{_n_windows_est * _n_pairs_est:,} observations  "
                f"({len(_instr_long)} {_ac_labels.get(_ca_long_ac, _ca_long_ac)} × "
                f"{len(_instr_short)} {_ac_labels.get(_ca_short_ac, _ca_short_ac)} × 2 directions)"
            )

            if _n_windows_est < 1:
                st.error(
                    f"Insufficient overlapping history ({_common_len} days) for "
                    f"{int(wf_is_years)}y IS + {int(wf_oos_years)}y OOS. "
                    "Reduce window lengths or use a longer data range."
                )
                return

            _wf_prog = st.progress(0.0)
            _wf_stat = st.empty()

            def _wf_progress(pct: float):
                _wf_prog.progress(pct)
                _wf_stat.caption(f"Walk-forward: {pct * 100:.0f}% complete…")

            with st.spinner("Running cross-asset walk-forward…"):
                _wf_results = run_cross_asset_walkforward(
                    prices_long=_prices_long,
                    prices_short=_prices_short,
                    instruments_long=_instr_long,
                    instruments_short=_instr_short,
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
                'results':       _wf_results,
                'scoring_mode':  wf_scoring,
                'asset':         wf_asset,
                'ca_long_ac':    _ca_long_ac,
                'ca_short_ac':   _ca_short_ac,
            }

        else:
            # ── Single-asset path (unchanged) ─────────────────────────────────
            _wf_cfg = ASSET_CLASSES[wf_asset]
            _wf_csv = _CACHE_DIR / _wf_cfg['csv_file']

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
            elif wf_asset == 'commodities':
                _wf_instruments = [i for i in _wf_instruments if i not in COMMODITY_EXCLUDE]
                _wf_prices = _wf_prices[[c for c in _wf_prices.columns if c not in COMMODITY_EXCLUDE]]

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

    _rho_label, _rho_detail, _rho_colour = _interpret_rho(rho, pval)
    _rho_msg = f"**{_rho_label}** — {_rho_detail}"
    if _rho_colour == 'green':
        st.success(_rho_msg)
    elif _rho_colour == 'red':
        st.error(_rho_msg)
    else:
        st.info(_rho_msg)

    if not _wf_sum['quintile_df'].empty:
        st.subheader("OOS Performance by IS Quintile")
        _q = _wf_sum['quintile_df'].copy()
        for _c in ['OOS_GrossWR']:
            _q[_c] = _q[_c].map('{:.1%}'.format)
        for _c in ['OOS_Gross', 'OOS_Net']:
            _q[_c] = _q[_c].map('{:+.2%}'.format)
        for _c in ['OOS_AvgHold']:
            _q[_c] = _q[_c].map('{:.0f}d'.format)
        _tbl(_q, show_index=True)

    # ── Top / Bottom pairs table ───────────────────────────────────────────────
    _valid = _wf_sum.get('valid')
    if _valid is not None and not _valid.empty and 'pair' in _valid.columns:
        _pair_agg = (
            _valid.groupby('pair', observed=True)
            .agg(
                IS_Score  =('IS_Score',  'mean'),
                OOS_Gross =('OOS_Gross', 'mean'),
                OOS_Net   =('OOS_Net',   'mean'),
                N_obs     =('OOS_Gross', 'count'),
            )
            .reset_index()
            .sort_values('IS_Score', ascending=False)
        )

        with st.expander("Top / Bottom pairs by IS score", expanded=False):
            _top10 = _pair_agg.head(10).copy()
            _bot10 = _pair_agg.tail(10).copy()

            for _df in (_top10, _bot10):
                _df['IS_Score'] = _df['IS_Score'].map('{:+.4f}'.format)
                _df['OOS_Gross'] = _df['OOS_Gross'].map('{:+.3%}'.format)
                _df['OOS_Net']   = _df['OOS_Net'].map('{:+.3%}'.format)

            tb1, tb2 = st.columns(2)
            with tb1:
                st.caption("Top 10 by IS score")
                _tbl(_top10[['pair', 'IS_Score', 'OOS_Gross', 'OOS_Net', 'N_obs']],
                     show_index=False)
            with tb2:
                st.caption("Bottom 10 by IS score")
                _tbl(_bot10[['pair', 'IS_Score', 'OOS_Gross', 'OOS_Net', 'N_obs']],
                     show_index=False)
