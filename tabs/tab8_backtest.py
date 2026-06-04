"""
Tab 8 — Backtest
=================
Exhaustive crossing-signal backtest across all instrument pairs for a
selected asset class (intra-asset) or cross-asset combination.

Scoring mode
------------
Tab 8 auto-defaults to the WF-validated scoring mode for the selected
asset class and warns on deviation.  Scoring mode is a Tab 8 concern —
Tab 10 sorts by AvgNet_WT directly; Tab 11 has no scoring mode selector
(register item F in CLAUDE.md).

Session state (widget keys)
---------------------------
bt_scoring_mode : str
    Selected scoring mode for intra-asset backtest.
bt_ca_scoring : str
    Selected scoring mode for cross-asset backtest.
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

from engine.backtest import (
    load_asset_prices, prepare_returns, run_backtest, aggregate_trades,
    run_exhaustive_search,
    load_cross_asset_prices, prepare_returns_aligned,
)
from engine.numba_core import COL_ENTRY_IDX, COL_SIDE
from engine.scoring import SCORING_MODES
from asset_configs import ASSET_CLASSES, ASSET_CLASS_OPTIONS, FI_EXCLUDE, COMMODITY_EXCLUDE, _DEFAULT_SCORING_MODE, get_spread_cost_lookup
from tabs.shared import _CACHE_DIR, _tbl


def render() -> None:
    st.header("Backtest")
    st.caption("Crossing-signal backtest across any asset class.")

    bt_mode = st.radio("Search mode", ["Intra-asset", "Cross-asset"],
                       horizontal=True, key='bt_mode')

    if bt_mode == 'Intra-asset':
        _render_intra_asset()
    else:
        _render_cross_asset()


# ── Trend filter ─────────────────────────────────────────────────────────────

def _bt8_apply_trend_filter(
    results_df: pd.DataFrame,
    scaled: np.ndarray,
    day_ints: np.ndarray,
    index: pd.DatetimeIndex,
    instruments: list[str],
    vol_window: int,
    xing_sd: float,
    exit_sd: float,
    fin_daily: float,
    trend_win: int,
    trend_mode: str,
) -> pd.DataFrame:
    """Apply trend filter post-hoc to top results. Re-runs backtest per row to get raw trades."""
    if trend_mode == 'Off' or results_df.empty:
        return results_df

    slb       = min(20, trend_win // 10)
    instr_idx = {code: i for i, code in enumerate(instruments)}

    trend_rows: list[dict | None] = []

    for _, row in results_df.iterrows():
        lf = row.get('_long_flags')
        sf = row.get('_short_flags')
        if not isinstance(lf, dict) or not isinstance(sf, dict):
            trend_rows.append(None)
            continue

        lc = [c for c in lf if c in instr_idx]
        sc = [c for c in sf if c in instr_idx]
        if not lc or not sc:
            trend_rows.append(None)
            continue

        n_legs    = len(lc) + len(sc)
        li        = [instr_idx[c] for c in lc]
        si        = [instr_idx[c] for c in sc]
        spr       = scaled[:, li].mean(axis=1) - scaled[:, si].mean(axis=1)

        bt = run_backtest(spr, day_ints, vol_window, xing_sd, exit_sd, 0.0, 0.0, n_legs)
        n_tr = bt['n_trades']
        if n_tr == 0:
            trend_rows.append(None)
            continue

        raw_t    = bt['trades_raw'][:n_tr]
        trend_v  = pd.Series(np.cumsum(spr), index=index).rolling(trend_win, min_periods=10).mean().values
        ei       = raw_t[:, COL_ENTRY_IDX].astype(int)
        sides    = raw_t[:, COL_SIDE]
        pi       = np.maximum(0, ei - slb)
        htr      = (ei >= slb) & ~np.isnan(trend_v[ei])
        slopes   = np.where(htr, (trend_v[ei] - trend_v[pi]) / slb, np.nan)
        valid    = ~np.isnan(slopes)
        wt_f     = (((sides > 0) & (slopes > 0)) | ((sides < 0) & (slopes < 0))) & valid
        ct_f     = ~wt_f & valid
        n_wt     = int(wt_f.sum())
        n_ct     = int(ct_f.sum())

        sp_cost = float(row.get('SpreadCost', 0.001))
        scp     = sp_cost / (n_legs * 2) if n_legs > 0 else 0.001

        wt_s = aggregate_trades(raw_t[wt_f], n_wt, scp, fin_daily, n_legs) if n_wt > 0 else {}
        ct_s = aggregate_trades(raw_t[ct_f], n_ct, scp, fin_daily, n_legs) if n_ct > 0 else {}

        al_pct   = float(wt_f[valid].mean()) if valid.any() else float('nan')
        wt_net   = float(wt_s.get('avg_net', float('nan')))
        ct_net   = float(ct_s.get('avg_net', float('nan')))
        wt_pos   = not np.isnan(wt_net) and wt_net > 0
        ct_pos   = not np.isnan(ct_net) and ct_net > 0
        best_dir = ('Both' if wt_pos and ct_pos else
                    'WT'   if wt_pos else
                    'CT'   if ct_pos else 'Neither')

        trend_rows.append({
            'Trades_WT':  n_wt,
            'NetWR_WT':   float(wt_s.get('net_wr',      float('nan'))),
            'AvgNet_WT':  float(wt_s.get('avg_net',     float('nan'))),
            'AvgHold_WT': float(wt_s.get('avg_holding', float('nan'))),
            'Trades_CT':  n_ct,
            'NetWR_CT':   float(ct_s.get('net_wr',      float('nan'))),
            'AvgNet_CT':  float(ct_s.get('avg_net',     float('nan'))),
            'AvgHold_CT': float(ct_s.get('avg_holding', float('nan'))),
            'Aligned%':   al_pct,
            'Best Dir':   best_dir,
        })

    if not any(r is not None for r in trend_rows):
        return results_df

    df = results_df.copy()
    trend_cols = ['Trades_WT', 'NetWR_WT', 'AvgNet_WT', 'AvgHold_WT',
                  'Trades_CT', 'NetWR_CT', 'AvgNet_CT', 'AvgHold_CT',
                  'Aligned%', 'Best Dir']
    for c in trend_cols:
        df[c] = np.nan if c != 'Best Dir' else ''

    for i, td in enumerate(trend_rows):
        if td is None:
            continue
        for c, v in td.items():
            df.iloc[i, df.columns.get_loc(c)] = v

    return df


# ── Navigation buttons ────────────────────────────────────────────────────────

def _bt8_nav_buttons(df: pd.DataFrame, params: dict, ac_long: str, ac_short: str,
                     key_sfx: str = '') -> None:
    _labels = [
        f"#{i+1} — {row.get('Long', '')} / {row.get('Short', '')}"
        for i, (_, row) in enumerate(df.iterrows())
    ]
    _sel = st.selectbox(
        "Select a result for navigation",
        ['— select —'] + _labels,
        key=f'bt8_nav{key_sfx}',
    )
    if _sel == '— select —':
        return

    _idx = _labels.index(_sel)
    _row = df.iloc[_idx]
    _lf  = _row.get('_long_flags', {})
    _sf  = _row.get('_short_flags', {})
    _lc  = list(_lf.keys()) if isinstance(_lf, dict) else []
    _sc  = list(_sf.keys()) if isinstance(_sf, dict) else []

    nc1, nc2, _ = st.columns([2, 2, 2])
    if nc1.button("Open in Pair Analysis →", key=f'bt8_pa{key_sfx}'):
        if _lc and _sc:
            st.session_state['pa_long_pending']  = _lc
            st.session_state['pa_short_pending'] = _sc
            st.session_state['pa_pair_pending']  = '— Custom pair —'
            st.toast(
                f"Loaded {_row.get('Long','')} / {_row.get('Short','')} — "
                "switch to Pair Analysis tab", icon="📈",
            )

    if nc2.button("Send to Walk-Forward →", key=f'bt8_wf{key_sfx}'):
        if _lc and _sc:
            st.session_state['wf_pair'] = {
                'long':             _lc,
                'short':            _sc,
                'vol_window':       params.get('vol_window', 262),
                'entry_sd':         params.get('xing_sd',    2.0),
                'exit_sd':          params.get('exit_sd',    1.0),
                'trend_window':     params.get('trend_window', 262),
                'trend_mode':       params.get('trend_mode',   'Both passes'),
                'asset_class_long': ac_long,
                'asset_class_short':ac_short,
                'source':           'tab8',
            }
            st.toast(
                f"Loaded {_row.get('Long','')} / {_row.get('Short','')} — "
                "switch to Walk-Forward tab", icon="🔬",
            )


# ── Results display ───────────────────────────────────────────────────────────

def _bt8_render_results(cache: dict) -> None:
    df          = cache['df']
    params      = cache['params']
    trend_mode  = params.get('trend_mode', 'Off')
    ac_long     = cache.get('asset_class_long',  cache.get('asset_key', 'commodities'))
    ac_short    = cache.get('asset_class_short', cache.get('asset_key', 'commodities'))
    scoring_mode = cache.get('scoring_mode', 'composite')

    if df.empty:
        st.warning("No combinations produced trades.")
        return

    st.success(f"Found **{len(df)}** results.")
    _top = df.iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Top Trades",   f"{int(_top.get('Trades', 0)):,}")
    m2.metric("Win Rate",     f"{float(_top.get('WinRate', 0)):.1%}")
    m3.metric("Expectancy",   f"{float(_top.get('Expectancy', 0))*100:+.2f}%")
    m4.metric("Avg Holding",  f"{float(_top.get('AvgHolding', 0)):.0f}d")
    m5.metric("Payoff Ratio", f"{float(_top.get('PayoffRatio', 0)):.2f}")

    st.subheader(f"Results (ranked by: {SCORING_MODES.get(scoring_mode, scoring_mode)})")

    _is_both = trend_mode == 'Both passes' and 'Trades_WT' in df.columns

    if _is_both:
        _bt8_render_both_passes(df, params, ac_long, ac_short)
    else:
        _bt8_render_standard(df, params, ac_long, ac_short, trend_mode)


def _bt8_render_both_passes(df: pd.DataFrame, params: dict, ac_long: str, ac_short: str) -> None:
    def _bucket(h: float) -> str:
        if h <= 45:   return '🟢 Short'
        if h <= 90:   return '🟡 Medium'
        return '🔴 Long'

    _df = df.copy()
    _df['_bucket'] = _df['AvgHold_WT'].apply(
        lambda v: _bucket(v) if not (isinstance(v, float) and np.isnan(v)) else '🔴 Long'
    )

    for _bkt, _desc in [('🟢 Short', '≤ 45d'), ('🟡 Medium', '46–90d'), ('🔴 Long', '> 90d')]:
        _bdf = (
            _df[_df['_bucket'] == _bkt]
            .sort_values('AvgNet_WT', ascending=False)
            .reset_index(drop=True)
        )
        with st.expander(f"{_bkt} ({_desc}) — {len(_bdf)} pairs", expanded=True):
            if _bdf.empty:
                st.caption("No results in this bucket.")
                continue

            _bdf.insert(0, 'Rank', range(1, len(_bdf) + 1))
            _cols = ['Rank', 'Long', 'Short', 'Entry SD', 'Exit SD', 'Vol Window',
                     'Trades_WT', 'NetWR_WT', 'AvgNet_WT', 'AvgHold_WT',
                     'Trades_CT', 'NetWR_CT', 'AvgNet_CT', 'AvgHold_CT',
                     'Aligned%', 'Best Dir']
            _tbl_df = _bdf[[c for c in _cols if c in _bdf.columns]].copy()

            for _c in ['NetWR_WT', 'NetWR_CT']:
                if _c in _tbl_df.columns:
                    _tbl_df[_c] = _tbl_df[_c].apply(
                        lambda v: f'{v:.1%}' if not (isinstance(v, float) and np.isnan(v)) else '—'
                    )
            for _c in ['AvgNet_WT', 'AvgNet_CT']:
                if _c in _tbl_df.columns:
                    _tbl_df[_c] = _tbl_df[_c].apply(
                        lambda v: f'{v*100:+.2f}%' if not (isinstance(v, float) and np.isnan(v)) else '—'
                    )
            for _c in ['AvgHold_WT', 'AvgHold_CT']:
                if _c in _tbl_df.columns:
                    _tbl_df[_c] = _tbl_df[_c].apply(
                        lambda v: f'{v:.0f}d' if not (isinstance(v, float) and np.isnan(v)) else '—'
                    )
            if 'Aligned%' in _tbl_df.columns:
                _tbl_df['Aligned%'] = _tbl_df['Aligned%'].apply(
                    lambda v: f'{v:.0%}' if not (isinstance(v, float) and np.isnan(v)) else '—'
                )
            st.dataframe(_tbl_df, use_container_width=True, hide_index=True)
            _bt8_nav_buttons(_bdf, params, ac_long, ac_short, key_sfx=f'_{_bkt}')


def _bt8_render_standard(df: pd.DataFrame, params: dict, ac_long: str, ac_short: str,
                         trend_mode: str) -> None:
    _ordered = ['Config', 'Long', 'Short',
                'WinRate', 'Expectancy', 'NetExpectancy', 'EstCost', 'AvgHolding',
                'Trades', 'PayoffRatio',
                'ReturnSD', 'TrendVolRatio', 'ReturnTopology', 'FitDataMinMaxSD', 'LastSD',
                'Aligned%']
    _dcols = [c for c in _ordered if c in df.columns]
    _disp  = df[_dcols].copy()
    for _c in _disp.columns:
        if _c in ('Long', 'Short', 'Config'):
            continue
        elif _c == 'WinRate':
            _disp[_c] = _disp[_c].map('{:.1%}'.format)
        elif _c == 'Trades':
            _disp[_c] = _disp[_c].map('{:.0f}'.format)
        elif _c == 'AvgHolding':
            _disp[_c] = _disp[_c].map(lambda v: f'{v:.0f}d')
        elif _c == 'Aligned%':
            _disp[_c] = _disp[_c].apply(
                lambda v: f'{v:.0%}' if not (isinstance(v, float) and np.isnan(v)) else '—'
            )
        elif _c in ('Expectancy', 'NetExpectancy'):
            _disp[_c] = _disp[_c].map(lambda v: f'{v*100:+.2f}%')
        elif _c == 'EstCost':
            _disp[_c] = _disp[_c].map('{:.3%}'.format)
        else:
            _disp[_c] = _disp[_c].map('{:.3f}'.format)
    _tbl(_disp, show_index=True)
    st.markdown("")
    _bt8_nav_buttons(df, params, ac_long, ac_short, key_sfx='_std')


# ── Intra-asset ───────────────────────────────────────────────────────────────

def _render_intra_asset() -> None:
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

    cfg      = ASSET_CLASSES[asset_key]
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

    with st.expander("Trend filter", expanded=True):
        tc1, tc2 = st.columns(2)
        bt_trend_win  = int(tc1.number_input("Trend filter window (days)", 130, 756, 262, key='bt_trend_win'))
        bt_trend_mode = tc2.selectbox(
            "Trend filter mode",
            ["Both passes", "With-trend only", "Counter-trend only", "Off"],
            index=0, key='bt_trend_mode',
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
        bt_window_opts  = {'1 year': 262, '2 years': 524, '3 years': 786,
                           '5 years': 1310, 'All': 0}
        bt_window_label = st.selectbox("Window", list(bt_window_opts.keys()),
                                       index=2, key='bt_window')
        bt_window_days  = bt_window_opts[bt_window_label] or None

    # Auto-default scoring mode when asset class changes
    _prev_bt_asset = st.session_state.get('bt_asset_prev')
    if _prev_bt_asset != asset_key and asset_key in _DEFAULT_SCORING_MODE:
        st.session_state['bt_scoring_mode'] = _DEFAULT_SCORING_MODE[asset_key]
    st.session_state['bt_asset_prev'] = asset_key

    bt_score_col, _ = st.columns([2, 2])
    with bt_score_col:
        bt_scoring_mode = st.selectbox(
            "Ranking method",
            list(SCORING_MODES.keys()),
            format_func=lambda x: SCORING_MODES[x],
            key='bt_scoring_mode',
        )
    _bt_rec = _DEFAULT_SCORING_MODE.get(asset_key)
    if _bt_rec and bt_scoring_mode != _bt_rec:
        st.warning(
            f"⚠️ Walk-forward suggests **{SCORING_MODES[_bt_rec]}** "
            f"for {dict(ASSET_CLASS_OPTIONS)[asset_key]} pairs."
        )

    st.markdown("---")
    bt_run = st.button("▶ Run backtest", type="primary",
                       use_container_width=True, key='bt_run')

    if bt_run:
        if not csv_path.exists() and uploaded is None:
            st.error(f"No data file at `{csv_path}`. Upload a CSV or add the file.")
            return

        with st.spinner("Loading prices…"):
            try:
                if uploaded is not None:
                    _raw = pd.read_csv(io.BytesIO(uploaded.read()),
                                       index_col='Date', parse_dates=True)
                    _raw = _raw.ffill(limit=3).dropna(how='all')
                    _instruments_bt = list(_raw.columns)
                    _prices_bt = _raw
                else:
                    _prices_bt, _instruments_bt = load_asset_prices(csv_path)
            except Exception as e:
                st.error(f"Failed to load prices: {e}")
                return

        if asset_key == 'commodities':
            _instruments_bt = [i for i in _instruments_bt if i not in COMMODITY_EXCLUDE]
            _prices_bt = _prices_bt[[c for c in _prices_bt.columns if c not in COMMODITY_EXCLUDE]]

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

            _latest_px     = dict(zip(_instruments_bt, _prices_bt.iloc[-1].values))
            _spread_lookup = get_spread_cost_lookup(_instruments_bt, _latest_px, asset_key)
            _fin_daily     = bt_fin_rate / 100 / 365

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

        # Apply trend filter to top results
        if bt_trend_mode != 'Off' and not _bt_results.empty:
            with st.spinner("Applying trend filter…"):
                _bt_results = _bt8_apply_trend_filter(
                    _bt_results, _scaled_bt, _day_ints_bt, _index_bt, _instruments_bt,
                    int(bt_vol_window), float(bt_xing_sd), float(bt_exit_sd),
                    _fin_daily, bt_trend_win, bt_trend_mode,
                )

        st.session_state['bt_results_cache'] = {
            'df':               _bt_results,
            'asset_key':        asset_key,
            'asset_class_long': asset_key,
            'asset_class_short':asset_key,
            'scoring_mode':     bt_scoring_mode,
            'params': {
                'vol_window':   int(bt_vol_window),
                'xing_sd':      float(bt_xing_sd),
                'exit_sd':      float(bt_exit_sd),
                'fin_rate':     float(bt_fin_rate),
                'trend_window': bt_trend_win,
                'trend_mode':   bt_trend_mode,
            },
        }

    _bt_cache = st.session_state.get('bt_results_cache')
    if _bt_cache is None:
        return

    _bt8_render_results(_bt_cache)


# ── Cross-asset ───────────────────────────────────────────────────────────────

def _render_cross_asset() -> None:
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

    with st.expander("Trend filter", expanded=True):
        tc1, tc2 = st.columns(2)
        _ca_trend_win  = int(tc1.number_input("Trend filter window (days)", 130, 756, 262, key='bt_ca_trend_win'))
        _ca_trend_mode = tc2.selectbox(
            "Trend filter mode",
            ["Both passes", "With-trend only", "Counter-trend only", "Off"],
            index=0, key='bt_ca_trend_mode',
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
        # Auto-default scoring mode when either asset class changes
        _prev_ca_long  = st.session_state.get('bt_ca_long_prev')
        _prev_ca_short = st.session_state.get('bt_ca_short_prev')
        if _prev_ca_long != _long_class or _prev_ca_short != _short_class:
            if 'commodities' in (_long_class, _short_class):
                st.session_state['bt_ca_scoring'] = _DEFAULT_SCORING_MODE.get('commodities', 'contrarian')
            else:
                st.session_state['bt_ca_scoring'] = 'composite'
        st.session_state['bt_ca_long_prev']  = _long_class
        st.session_state['bt_ca_short_prev'] = _short_class
        _ca_scoring = st.selectbox(
            "Scoring mode", list(SCORING_MODES.keys()),
            format_func=lambda x: SCORING_MODES[x], key='bt_ca_scoring',
        )
        if 'commodities' in (_long_class, _short_class):
            _ca_rec = _DEFAULT_SCORING_MODE.get('commodities')
            if _ca_rec and _ca_scoring != _ca_rec:
                st.warning(
                    f"⚠️ Walk-forward suggests **{SCORING_MODES[_ca_rec]}** "
                    "when either side includes Commodities."
                )

    st.markdown("---")
    _ca_run = st.button("▶ Run cross-asset search", type="primary",
                        use_container_width=True, key='bt_ca_run')

    if not _ca_run:
        _ca_cache = st.session_state.get('bt_ca_results_cache')
        if _ca_cache:
            _bt8_render_results(_ca_cache)
        return

    try:
        with st.spinner(f"Loading {_long_class}/{_short_class} prices..."):
            _ca_prices, _ca_long_i, _ca_short_i, _ = load_cross_asset_prices(
                _long_class, _short_class, _CACHE_DIR,
            )
        if _long_class == 'commodities':
            _ca_long_i = [i for i in _ca_long_i if i not in COMMODITY_EXCLUDE]
        if _short_class == 'commodities':
            _ca_short_i = [i for i in _ca_short_i if i not in COMMODITY_EXCLUDE]
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
            return

        # Apply trend filter
        if _ca_trend_mode != 'Off':
            with st.spinner("Applying trend filter…"):
                _ca_df = _bt8_apply_trend_filter(
                    _ca_df, _ca_all_scaled, _ca_day_ints, _ca_idx, _ca_all_instr,
                    int(_ca_vol), float(_ca_xing), float(_ca_exit),
                    _ca_fin_daily, _ca_trend_win, _ca_trend_mode,
                )

        _ca_net_pos = (_ca_df['NetExpectancy'] > 0).sum()
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("Results",    len(_ca_df))
        cm2.metric("Net+ pairs", f"{_ca_net_pos} ({_ca_net_pos/len(_ca_df):.0%})")
        cm3.metric("Best net expectancy",
                   f"{_ca_df['NetExpectancy'].max()*100:+.2f}%")
        cm4.metric("Avg holding (top 10)",
                   f"{_ca_df.head(10)['AvgHolding'].mean():.0f}d")

        _ca_cache = {
            'df':               _ca_df,
            'asset_key':        _long_class,
            'asset_class_long': _long_class,
            'asset_class_short':_short_class,
            'scoring_mode':     _ca_scoring,
            'params': {
                'vol_window':   int(_ca_vol),
                'xing_sd':      float(_ca_xing),
                'exit_sd':      float(_ca_exit),
                'fin_rate':     float(_ca_fin),
                'trend_window': _ca_trend_win,
                'trend_mode':   _ca_trend_mode,
            },
        }
        st.session_state['bt_ca_results_cache'] = _ca_cache
        _bt8_render_results(_ca_cache)

    except FileNotFoundError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        raise
