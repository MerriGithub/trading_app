from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pandas as pd
import streamlit as st

from engine.backtest import (
    prepare_returns as _sc_prep,
    run_backtest as _sc_bt,
    aggregate_trades as _sc_agg,
)
from engine.numba_core import COL_ENTRY_IDX as _SC_CEI, COL_SIDE as _SC_CS
from asset_configs import ASSET_CLASSES, FI_EXCLUDE, COMMODITY_EXCLUDE, _DEFAULT_SCORING_MODE, get_tradeable_instruments, get_display_name
from data_watchlist import add_to_watchlist
from tabs.shared import registry, _tbl


def _get_wt_trades(row) -> int:
    """Safely get WT trade count from a result row (handles both scan modes)."""
    for col in ('Trades_WT', 'trades_wt', 'Trades', 'trades'):
        if col in row.index and pd.notna(row[col]):
            return int(row[col])
    return 0


def _resolve_trend_mode(row) -> str:
    """
    Derive the correct WF trend mode for a scan result row.

    When the scan ran in "Both passes" mode, Best Dir identifies which
    direction was actually profitable — the WF should test that direction,
    not re-run both passes. For single-direction scans the scan mode is
    already correct.
    """
    scan_mode = str(row.get('_trend_mode', 'Both passes'))
    if scan_mode != 'Both passes':
        return scan_mode
    return {
        'WT':   'With-trend only',
        'CT':   'Counter-trend only',
        'Both': 'Both passes',
    }.get(str(row.get('Best Dir', '')), 'Both passes')


@st.cache_data(ttl=300, show_spinner=False)
def _sc_load_prices(asset_key: str) -> tuple[pd.DataFrame, list[str]]:
    _excl  = FI_EXCLUDE if asset_key == 'fixed_income' else (COMMODITY_EXCLUDE if asset_key == 'commodities' else frozenset())
    _insts = [c for c in get_tradeable_instruments(asset_key) if c not in _excl]
    _df    = registry.get_daily_prices(_insts)
    _valid = [c for c in _insts if c in _df.columns]
    return _df, _valid


def render() -> None:
    st.header("Scenario Scanner")
    st.caption(
        "Grid sweep: crossing signal backtest across asset classes and parameter combinations. "
        "Buckets results by average holding period and ranks by net expectancy."
    )

    st.subheader("Scope")
    _sc1, _sc2 = st.columns(2)
    with _sc1:
        st.markdown("**Intra-asset**")
        _sc_comm_intra = st.checkbox("Commodities",  key='sc_comm_intra')
        _sc_fx_intra   = st.checkbox("FX",           key='sc_fx_intra')
        _sc_fi_intra   = st.checkbox("Fixed Income", key='sc_fi_intra')
        _sc_eq_intra   = st.checkbox("Equity",       key='sc_eq_intra')
    with _sc2:
        st.markdown("**Cross-asset**")
        _sc_comm_fx  = st.checkbox("Commodities × FX",           key='sc_comm_fx')
        _sc_comm_eq  = st.checkbox("Commodities × Equity",       key='sc_comm_eq')
        _sc_comm_fi  = st.checkbox("Commodities × Fixed Income", key='sc_comm_fi')
        _sc_fx_eq    = st.checkbox("FX × Equity",                key='sc_fx_eq')

    st.subheader("Parameter grid")
    _pg1, _pg2, _pg3 = st.columns(3)
    with _pg1:
        _sc_entry_sds = st.multiselect(
            "Entry SD", [2.0, 2.5, 3.0], default=[2.0, 2.5, 3.0], key='sc_entry_sds',
        )
    with _pg2:
        _sc_exit_sds = st.multiselect(
            "Exit SD", [0.5, 1.0, 1.5, 2.0], default=[0.5, 1.0, 1.5], key='sc_exit_sds',
        )
    with _pg3:
        _sc_vol_wins = st.multiselect(
            "Vol window (days)", [50, 130, 262], default=[50, 130, 262], key='sc_vol_wins',
        )

    _pg4, _pg5, _pg6 = st.columns(3)
    with _pg4:
        _sc_broker = st.selectbox(
            "Broker / account type",
            options=["ig_spreadbet", "ig_cfd"],
            format_func=lambda x: {"ig_spreadbet": "IG Spread Bet", "ig_cfd": "IG CFD"}[x],
            key="tab10_broker_profile",
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
            ["Both passes", "With-trend only", "Counter-trend only", "Off"],
            index=0, key='sc_trend_mode',
        )

    _sc_run = st.button(
        "▶ Run scenario scan", type="primary", use_container_width=True, key='sc_run',
    )

    if _sc_run:
        _run_scan(
            _sc_comm_intra, _sc_fx_intra, _sc_fi_intra, _sc_eq_intra,
            _sc_comm_fx, _sc_comm_eq, _sc_comm_fi, _sc_fx_eq,
            _sc_entry_sds, _sc_exit_sds, _sc_vol_wins,
            _sc_broker, _sc_win_days, _sc_win_label, _sc_min_trades,
            _sc_trend_win, _sc_trend_mode,
        )

    if 'scenario_results' in st.session_state:
        _render_results(int(_sc_min_trades))


def _run_scan(
    sc_comm_intra, sc_fx_intra, sc_fi_intra, sc_eq_intra,
    sc_comm_fx, sc_comm_eq, sc_comm_fi, sc_fx_eq,
    sc_entry_sds, sc_exit_sds, sc_vol_wins,
    sc_broker, sc_win_days, sc_win_label, sc_min_trades,
    sc_trend_win, sc_trend_mode,
) -> None:
    from account import get_financing_daily_rate, BROKER_PROFILE_LABELS
    _scope_defs: list[tuple[str, str, bool, str]] = []
    if sc_comm_intra: _scope_defs.append(('commodities',  'commodities',  True,  'Commodity'))
    if sc_fx_intra:   _scope_defs.append(('fx',           'fx',           True,  'FX'))
    if sc_fi_intra:   _scope_defs.append(('fixed_income', 'fixed_income', True,  'Fixed Income'))
    if sc_eq_intra:   _scope_defs.append(('equity',       'equity',       True,  'Equity'))
    if sc_comm_fx:    _scope_defs.append(('commodities',  'fx',           False, 'Commodity vs FX'))
    if sc_comm_eq:    _scope_defs.append(('commodities',  'equity',       False, 'Commodity vs Equity'))
    if sc_comm_fi:    _scope_defs.append(('commodities',  'fixed_income', False, 'Commodity vs Fixed Income'))
    if sc_fx_eq:      _scope_defs.append(('fx',           'equity',       False, 'FX vs Equity'))

    if not _scope_defs:
        st.warning("Select at least one asset class scope.")
        return
    if not sc_entry_sds or not sc_exit_sds or not sc_vol_wins:
        st.warning("Select at least one value for each parameter in the grid.")
        return

    _sc_param_combos = [
        (float(e), float(x), int(v))
        for e in sc_entry_sds
        for x in sc_exit_sds
        for v in sc_vol_wins
    ]
    _sc_slb = min(20, sc_trend_win // 10)

    _sc_scope_work: list[tuple] = []
    def _excl(ac: str) -> frozenset:
        if ac == 'fixed_income': return FI_EXCLUDE
        if ac == 'commodities':  return COMMODITY_EXCLUDE
        return frozenset()

    for _lk, _sk, _intra, _label in _scope_defs:
        _l_insts = [c for c in get_tradeable_instruments(_lk) if c not in _excl(_lk)]
        _s_insts = [c for c in get_tradeable_instruments(_sk) if c not in _excl(_sk)]
        if _intra:
            # Equity uses permutations (both directions scored independently).
            # All other intra-asset classes use combinations (direction-agnostic).
            _pairs = (list(permutations(_l_insts, 2))
                      if _lk == 'equity'
                      else list(combinations(_l_insts, 2)))
        else:
            _pairs = [(a, b) for a in _l_insts for b in _s_insts]
        _sc_scope_work.append((_lk, _sk, _intra, _label, _l_insts, _s_insts, _pairs))

    _sc_total_pairs = sum(len(w[6]) for w in _sc_scope_work)
    if _sc_total_pairs == 0:
        st.warning("No instrument pairs found for the selected scope.")
        return

    _sc_rows: list[dict] = []
    _sc_prog  = st.progress(0.0)
    _sc_pstat = st.empty()
    _sc_done  = 0

    for _lk, _sk, _intra, _label, _l_insts, _s_insts, _pairs in _sc_scope_work:
        _l_px, _ = _sc_load_prices(_lk)
        _s_px, _ = _sc_load_prices(_sk) if not _intra else (_l_px, _l_insts)

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

            _lc_cfg = ASSET_CLASSES[_lk]['instruments'].get(_long_i,  {})
            _sc_cfg = ASSET_CLASSES[_sk]['instruments'].get(_short_i, {})
            _l_sp = _lc_cfg.get('spread_pct', 0.001) if isinstance(_lc_cfg, dict) else 0.001
            _s_sp = _sc_cfg.get('spread_pct', 0.001) if isinstance(_sc_cfg, dict) else 0.001
            _pair_cost = 2.0 * (_l_sp + _s_sp)

            _long_fin  = get_financing_daily_rate(_long_i,  _lk, 'long',  broker_profile=sc_broker)
            _short_fin = get_financing_daily_rate(_short_i, _sk, 'short', broker_profile=sc_broker)
            _pair_fin_daily = (_long_fin + _short_fin) / 2  # per-leg average; n_legs=2

            _l_disp = get_display_name(_lk, _long_i)
            _s_disp = get_display_name(_sk, _short_i)

            _raw_lr = (np.log(_pair_df[_long_i] / _pair_df[_short_i]).diff().fillna(0))
            _raw_cum_spr = _raw_lr.cumsum()
            _trend_ser_pair = _raw_cum_spr.rolling(sc_trend_win, min_periods=10).mean()
            _trend_arr_pair = _trend_ser_pair.values

            for (_e_sd, _x_sd, _v_win) in _sc_param_combos:
                try:
                    _sc_scaled, _sc_day_ints, _sc_index = _sc_prep(
                        _pair_df, [_long_i, _short_i],
                        vol_window=_v_win,
                        window_days=sc_win_days,
                    )
                    if _sc_scaled.shape[0] < _v_win:
                        continue
                    _sc_spread = _sc_scaled[:, 0] - _sc_scaled[:, 1]
                    _sc_bt_res = _sc_bt(
                        _sc_spread, _sc_day_ints,
                        vol_window=_v_win,
                        xing_sd=_e_sd, exit_sd=_x_sd,
                        spread_cost_pct=_pair_cost,
                        financing_daily_pct=_pair_fin_daily,
                        n_legs=2,
                    )
                    _n_tr = _sc_bt_res['n_trades']
                    _sc_s = _sc_bt_res['summary']
                    _al_pct = float('nan')

                    if _n_tr > 0 and sc_trend_mode != "Off":
                        _raw_t  = _sc_bt_res['trades_raw'][:_n_tr]
                        _eidxs  = _raw_t[:, _SC_CEI].astype(int)
                        _sides  = _raw_t[:, _SC_CS]
                        _edates = _sc_index[_eidxs]
                        _tipos  = _trend_ser_pair.index.get_indexer(_edates, method='nearest')
                        _prev_p = np.maximum(0, _tipos - _sc_slb)
                        _has_tr = (_tipos >= _sc_slb) & ~np.isnan(_trend_arr_pair[_tipos])
                        _slp = np.where(
                            _has_tr,
                            (_trend_arr_pair[_tipos] - _trend_arr_pair[_prev_p]) / _sc_slb,
                            np.nan,
                        )
                        _al  = (((_sides > 0) & (_slp > 0)) |
                                ((_sides < 0) & (_slp < 0)))
                        _vld = ~np.isnan(_slp)
                        if _vld.any():
                            _al_pct = float(_al[_vld].mean())
                        _wt_f = _al & _vld
                        _ct_f = ~_al & _vld
                        _n_wt = int(_wt_f.sum())
                        _n_ct = int(_ct_f.sum())

                        if sc_trend_mode == "Both passes":
                            _wt_s = (_sc_agg(_raw_t[_wt_f], _n_wt,
                                             spread_cost_pct=_pair_cost,
                                             financing_daily_pct=_pair_fin_daily,
                                             n_legs=2)
                                     if _n_wt > 0 else {})
                            _ct_s = (_sc_agg(_raw_t[_ct_f], _n_ct,
                                             spread_cost_pct=_pair_cost,
                                             financing_daily_pct=_pair_fin_daily,
                                             n_legs=2)
                                     if _n_ct > 0 else {})
                            _wt_net = float(_wt_s.get('avg_net', float('nan')))
                            _ct_net = float(_ct_s.get('avg_net', float('nan')))
                            _wt_pos = not np.isnan(_wt_net) and _wt_net > 0
                            _ct_pos = not np.isnan(_ct_net) and _ct_net > 0
                            _best_dir = ('Both'    if _wt_pos and _ct_pos else
                                         'WT'      if _wt_pos else
                                         'CT'      if _ct_pos else 'Neither')
                            _sc_rows.append({
                                '_long':              _long_i,
                                '_short':             _short_i,
                                '_asset_class_long':  _lk,
                                '_asset_class_short': _sk,
                                '_trend_mode':        sc_trend_mode,
                                'Long':          _l_disp,
                                'Short':         _s_disp,
                                'Asset Classes': _label,
                                'Entry SD':      _e_sd,
                                'Exit SD':       _x_sd,
                                'Vol Window':    _v_win,
                                'Trend Window':  f'{sc_trend_win}d',
                                'Trades_WT':     _n_wt,
                                'NetWR_WT':      float(_wt_s.get('net_wr',      float('nan'))),
                                'AvgNet_WT':     float(_wt_s.get('avg_net',     float('nan'))),
                                'AvgHold_WT':    float(_wt_s.get('avg_holding', float('nan'))),
                                'Trades_CT':     _n_ct,
                                'NetWR_CT':      float(_ct_s.get('net_wr',      float('nan'))),
                                'AvgNet_CT':     float(_ct_s.get('avg_net',     float('nan'))),
                                'AvgHold_CT':    float(_ct_s.get('avg_holding', float('nan'))),
                                'Aligned%':      _al_pct,
                                'Best Dir':      _best_dir,
                            })
                            continue

                        _use_f = _wt_f if sc_trend_mode == "With-trend only" else _ct_f
                        _use_n = _n_wt if sc_trend_mode == "With-trend only" else _n_ct
                        if _use_n < int(sc_min_trades):
                            continue
                        _use_s = _sc_agg(_raw_t[_use_f], _use_n,
                                         spread_cost_pct=_pair_cost,
                                         financing_daily_pct=_pair_fin_daily,
                                         n_legs=2)
                    else:
                        _use_s = _sc_s
                        _use_n = _n_tr

                    _sc_rows.append({
                        '_long':              _long_i,
                        '_short':             _short_i,
                        '_asset_class_long':  _lk,
                        '_asset_class_short': _sk,
                        '_trend_mode':        sc_trend_mode,
                        'Long':          _l_disp,
                        'Short':         _s_disp,
                        'Asset Classes': _label,
                        'Entry SD':      _e_sd,
                        'Exit SD':       _x_sd,
                        'Vol Window':    _v_win,
                        'Trend Window':  f'{sc_trend_win}d',
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
            _sc_pstat.caption(f"Scanning… {_sc_done:,} / {_sc_total_pairs:,} pairs")

    _sc_prog.progress(1.0)
    _sc_pstat.empty()

    if _sc_rows:
        _sc_full_df = pd.DataFrame(_sc_rows)
        st.session_state['scenario_results'] = _sc_full_df
        st.session_state['scenario_params'] = (
            f"Entry SD: {sc_entry_sds}  |  Exit SD: {sc_exit_sds}  |  "
            f"Vol window: {sc_vol_wins}  |  History: {sc_win_label}  |  "
            f"Broker: {BROKER_PROFILE_LABELS.get(sc_broker, sc_broker)}  |  Min trades: {int(sc_min_trades)}"
        )
        st.success(
            f"Scan complete — {len(_sc_full_df):,} total results across "
            f"{_sc_total_pairs:,} pairs × {len(_sc_param_combos)} parameter combos."
        )
    else:
        st.info("No results — try relaxing parameters or adding more scopes.")


def _render_results(sc_min_trades: int) -> None:
    _sr_full = st.session_state['scenario_results']
    st.caption(f"**Last run:** {st.session_state.get('scenario_params', '')}")
    st.divider()

    _is_both_passes = 'Trades_WT' in _sr_full.columns

    if _is_both_passes:
        _sr = _sr_full[
            (_sr_full['Trades_WT'] >= sc_min_trades) |
            (_sr_full['Trades_CT'] >= sc_min_trades)
        ].copy()
        _sr = _sr[(_sr['AvgNet_WT'] > 0) | (_sr['AvgNet_CT'] > 0)].copy()
    else:
        _sr = _sr_full[_sr_full['Trades'] >= sc_min_trades].copy()
        _sr = _sr[_sr['Avg Net'] > 0].copy()

    if _sr.empty:
        st.info(
            f"No results pass filters (trades ≥ {sc_min_trades}, avg net > 0). "
            "Use the CSV download below for the full unfiltered dataset."
        )
    else:
        def _sc_bucket(h: float) -> str:
            if h <= 45:  return '🟢 Short'
            if h <= 90:  return '🟡 Medium'
            return '🔴 Long'

        _bucket_col = 'AvgHold_WT' if _is_both_passes else 'Avg Hold'
        _sort_col   = 'AvgNet_WT'  if _is_both_passes else 'Avg Net'
        _sr['_bucket'] = _sr[_bucket_col].apply(_sc_bucket)

        for _bkt, _desc in [
            ('🟢 Short',  '≤ 45d'),
            ('🟡 Medium', '46–90d'),
            ('🔴 Long',   '> 90d'),
        ]:
            _bdf = (
                _sr[_sr['_bucket'] == _bkt]
                .sort_values(_sort_col, ascending=False)
                .reset_index(drop=True)
            )
            with st.expander(f"{_bkt} ({_desc}) — {len(_bdf)} pairs found", expanded=True):
                if _bdf.empty:
                    st.caption("No results in this bucket.")
                    continue

                _bdf.insert(0, 'Rank', range(1, len(_bdf) + 1))

                if _is_both_passes:
                    _disp_cols = [
                        'Rank', 'Long', 'Short', 'Asset Classes',
                        'Entry SD', 'Exit SD', 'Vol Window', 'Trend Window',
                        'Trades_WT', 'NetWR_WT', 'AvgNet_WT', 'AvgHold_WT',
                        'Trades_CT', 'NetWR_CT', 'AvgNet_CT', 'AvgHold_CT',
                        'Aligned%', 'Best Dir',
                    ]
                    _tbl_df = _bdf[[c for c in _disp_cols if c in _bdf.columns]].copy()
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
                    _tbl_df['Aligned%'] = _tbl_df['Aligned%'].apply(
                        lambda v: f'{v:.0%}' if not (isinstance(v, float) and np.isnan(v)) else '—'
                    )
                else:
                    _disp_cols = [
                        'Rank', 'Long', 'Short', 'Asset Classes',
                        'Entry SD', 'Exit SD', 'Vol Window', 'Trend Window', 'Trades',
                        'Gross WR%', 'Net WR%', 'Avg Gross', 'Avg Net',
                        'Avg Hold', 'Est Cost', 'Aligned%',
                    ]
                    _tbl_df = _bdf[[c for c in _disp_cols if c in _bdf.columns]].copy()
                    _tbl_df['Gross WR%'] = _tbl_df['Gross WR%'].map('{:.1%}'.format)
                    _tbl_df['Net WR%']   = _tbl_df['Net WR%'].map('{:.1%}'.format)
                    _tbl_df['Avg Gross'] = _tbl_df['Avg Gross'].map(lambda v: f'{v*100:+.2f}%')
                    _tbl_df['Avg Net']   = _tbl_df['Avg Net'].map(lambda v: f'{v*100:+.2f}%')
                    _tbl_df['Avg Hold']  = _tbl_df['Avg Hold'].map('{:.0f}d'.format)
                    _tbl_df['Est Cost']  = _tbl_df['Est Cost'].map('{:.3%}'.format)
                    _tbl_df['Aligned%']  = _tbl_df['Aligned%'].apply(
                        lambda v: f'{v:.0%}' if not (isinstance(v, float) and np.isnan(v)) else '—'
                    )

                st.dataframe(_tbl_df, use_container_width=True, hide_index=True)

                # ── Send to Batch Walk-Forward (🟢 Short bucket only) ────────
                if _bkt == '🟢 Short':
                    _min_wt = int(st.number_input(
                        "Min WT trades to include in batch",
                        min_value=1, max_value=50, value=8, step=1,
                        key="tab10_min_wt_trades",
                        help=(
                            "Only pairs with at least this many With-Trend trades are sent to "
                            "Walk-Forward. Filters out rows where the WT signal has too little "
                            "history for a meaningful walk-forward result."
                        ),
                    ))
                    _green_wf_rows = _bdf[_bdf.apply(_get_wt_trades, axis=1) >= _min_wt]
                    _excluded = len(_bdf) - len(_green_wf_rows)

                    if len(_green_wf_rows) > 0:
                        _cb1, _cb2 = st.columns([2, 5])
                        with _cb1:
                            if st.button(
                                f"📐 Send {len(_green_wf_rows)} pairs to Walk-Forward →",
                                key='sc_send_wf_green',
                                type='primary',
                            ):
                                _tw_raw = lambda r: int(str(r.get('Trend Window', '262d')).rstrip('d'))
                                _wf_batch = []
                                for _, _r in _green_wf_rows.iterrows():
                                    _wf_batch.append({
                                        'long':              str(_r['_long']),
                                        'short':             str(_r['_short']),
                                        'asset_class_long':  str(_r.get('_asset_class_long', '')),
                                        'asset_class_short': str(_r.get('_asset_class_short', '')),
                                        'entry_sd':          float(_r['Entry SD']),
                                        'exit_sd':           float(_r['Exit SD']),
                                        'vol_window':        int(_r['Vol Window']),
                                        'trend_window':      _tw_raw(_r),
                                        'trend_mode':        _resolve_trend_mode(_r),
                                        'scan_metrics': {
                                            'trades_wt':   _get_wt_trades(_r),
                                            'net_wr_wt':   float(_r.get('NetWR_WT', _r.get('Net WR%', 0)) or 0),
                                            'avg_net_wt':  float(_r.get('AvgNet_WT', _r.get('Avg Net', 0)) or 0),
                                            'avg_hold_wt': int(_r.get('AvgHold_WT', _r.get('Avg Hold', 0)) or 0),
                                            'trades_ct':   int(_r.get('Trades_CT', 0)),
                                            'net_wr_ct':   float(_r.get('NetWR_CT', 0) or 0),
                                            'avg_net_ct':  float(_r.get('AvgNet_CT', 0) or 0),
                                            'avg_hold_ct': int(_r.get('AvgHold_CT', 0) or 0),
                                            'best_dir':    str(_r.get('Best Dir', '')),
                                        },
                                    })
                                st.session_state['wf_batch'] = _wf_batch
                                st.session_state['wf_batch_source'] = 'tab10_green'
                                st.session_state.pop('wf_batch_results', None)
                                st.session_state['sidebar_nav_pending'] = "🔀 Walk-Forward"
                                st.rerun()
                        with _cb2:
                            if _excluded > 0:
                                st.caption(
                                    f"{len(_green_wf_rows)} pairs · avg hold ≤ 45d · "
                                    f"{_excluded} excluded (< {_min_wt} WT trades)"
                                )
                            else:
                                st.caption(
                                    f"{len(_green_wf_rows)} pairs · avg hold ≤ 45d · "
                                    "all pass WT trades filter"
                                )
                    else:
                        st.info(
                            f"No 🟢 pairs pass the Min WT trades filter (≥ {_min_wt}). "
                            "Lower the filter or adjust scan parameters."
                        )

                # ── Per-row action area ──────────────────────────────────────
                _pair_labels = [
                    f"#{r['Rank']} — {r['Long']} / {r['Short']}"
                    for _, r in _bdf.iterrows()
                ]
                _sel = st.selectbox(
                    "Select a pair",
                    ['— select a pair —'] + _pair_labels,
                    key=f'sc_sel_{_bkt}',
                )
                if _sel != '— select a pair —':
                    _row = _bdf.iloc[_pair_labels.index(_sel)]
                    _tw_val = int(str(_row.get('Trend Window', '262d')).rstrip('d'))
                    _act1, _act2 = st.columns(2)

                    if _act1.button("Open in Pair Analysis →", key=f'sc_open_{_bkt}'):
                        st.session_state['pa_long_pending']   = [_row['_long']]
                        st.session_state['pa_short_pending']  = [_row['_short']]
                        st.session_state['pa_pair_pending']   = '— Custom pair —'
                        st.session_state['pa_vol']            = int(_row['Vol Window'])
                        st.session_state['pa_xing']           = float(_row['Entry SD'])
                        st.session_state['pa_exit']           = float(_row['Exit SD'])
                        st.session_state['pa_trend_window']   = _tw_val
                        st.session_state['sc_long_pending']   = [_row['_long']]
                        st.session_state['sc_short_pending']  = [_row['_short']]
                        st.session_state['wf11_long_pending'] = [_row['_long']]
                        st.session_state['wf11_short_pending']= [_row['_short']]
                        st.session_state['wf_pair'] = {
                            'long':              [_row['_long']],
                            'short':             [_row['_short']],
                            'vol_window':        int(_row['Vol Window']),
                            'entry_sd':          float(_row['Entry SD']),
                            'exit_sd':           float(_row['Exit SD']),
                            'trend_window':      _tw_val,
                            'trend_mode':        _resolve_trend_mode(_row),
                            'asset_class_long':  str(_row.get('_asset_class_long', '')),
                            'asset_class_short': str(_row.get('_asset_class_short', '')),
                            'source':            'tab10',
                        }
                        st.session_state['sidebar_nav_pending'] = "📈 Pair Analysis"
                        st.rerun()

                    if _act2.button("★ Add to Watchlist", key=f'sc_wl_{_bkt}'):
                        _wl_ac_l = str(_row.get('_asset_class_long', ''))
                        _wl_ac_s = str(_row.get('_asset_class_short', ''))
                        _wl_mode = (
                            _DEFAULT_SCORING_MODE.get('commodities', 'contrarian')
                            if 'commodities' in (_wl_ac_l, _wl_ac_s)
                            else _DEFAULT_SCORING_MODE.get(_wl_ac_l, 'composite')
                        )
                        _eid = add_to_watchlist({
                            'long':              str(_row['_long']),
                            'short':             str(_row['_short']),
                            'asset_class_long':  _wl_ac_l,
                            'asset_class_short': _wl_ac_s,
                            'entry_sd':          float(_row['Entry SD']),
                            'exit_sd':           float(_row['Exit SD']),
                            'vol_window':        int(_row['Vol Window']),
                            'trend_window':      _tw_val,
                            'trend_mode':        _resolve_trend_mode(_row),
                            'scoring_mode':      _wl_mode,
                            'source':            'tab10',
                            'scan_metrics': {
                                'trades_wt':  int(_row.get('Trades_WT', _row.get('Trades', 0))),
                                'net_wr_wt':  float(_row.get('NetWR_WT', _row.get('Net WR%', 0)) or 0),
                                'avg_net_wt': float(_row.get('AvgNet_WT', _row.get('Avg Net', 0)) or 0),
                                'avg_hold_wt':int(_row.get('AvgHold_WT', _row.get('Avg Hold', 0)) or 0),
                                'trades_ct':  int(_row.get('Trades_CT', 0)),
                                'net_wr_ct':  float(_row.get('NetWR_CT', 0) or 0),
                                'avg_net_ct': float(_row.get('AvgNet_CT', 0) or 0),
                                'avg_hold_ct':int(_row.get('AvgHold_CT', 0) or 0),
                                'best_dir':   str(_row.get('Best Dir', '')),
                            },
                        })
                        st.toast(f"★ {_row['Long']}/{_row['Short']} added to watchlist")

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
