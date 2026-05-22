from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data_watchlist import (
    load_watchlist, load_wf_cache,
    add_to_watchlist, remove_from_watchlist,
    update_notes, save_wf_result,
)
from tabs.tab11_walkforward import _run_wf


def _get_wf_status(entry: dict, cache: dict) -> dict | None:
    if entry.get('wf_metrics'):
        return entry['wf_metrics']
    return cache.get(entry['id'])


def _wf_badge(eid: str, wf_cache: dict, wfs: dict | None = None) -> str:
    if wfs is None:
        wfs = wf_cache.get(eid)
    if wfs is None:
        return "— Not run"
    rec = wfs.get('recommendation', '')
    return {"Robust": "✅ Robust", "Moderate": "⚠️ Moderate"}.get(rec, "❌ Curve-fitted")


def _wf_consistency(eid: str, wf_cache: dict, wfs: dict | None = None) -> str:
    if wfs is None:
        wfs = wf_cache.get(eid)
    v = wfs.get('consistency_score') if wfs else None
    return f"{v:.2f}" if v is not None else "—"


def _compute_wf_summary(df: pd.DataFrame) -> dict:
    n_windows  = len(df)
    stable_pct = float((df['Stable?'] == '✅').sum() / n_windows * 100) if n_windows > 0 else 0.0

    an_vals = df['_oos_an'].dropna()
    nr_vals = df['_oos_nr'].dropna()
    avg_oos_net = float(an_vals.mean()) if len(an_vals) > 0 else None
    avg_oos_wr  = float(nr_vals.mean()) if len(nr_vals) > 0 else None

    consistency_score = None
    valid_both = df[['_is_an', '_oos_an']].dropna()
    if len(valid_both) >= 3:
        try:
            from scipy.stats import spearmanr
            r, _ = spearmanr(valid_both['_is_an'].values, valid_both['_oos_an'].values)
            consistency_score = float(r)
        except Exception:
            pass

    if stable_pct >= 75:
        recommendation = "Robust"
    elif stable_pct >= 50:
        recommendation = "Moderate"
    else:
        recommendation = "Curve-fitted"

    windows_data = []
    for _, row in df.iterrows():
        def _safe(v):
            try:
                return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
            except Exception:
                return v
        windows_data.append({
            'Window':      int(row['Window']),
            'IS Period':   str(row['IS Period']),
            'OOS Period':  str(row['OOS Period']),
            'IS Net WR':   _safe(row.get('IS Net WR')),
            'IS Avg Net':  _safe(row.get('IS Avg Net')),
            'OOS Net WR':  _safe(row.get('OOS Net WR')),
            'OOS Avg Net': _safe(row.get('OOS Avg Net')),
            'OOS Avg Hold':_safe(row.get('OOS Avg Hold')),
            'IS Trades':   int(row.get('IS Trades', 0) or 0),
            'OOS Trades':  int(row.get('OOS Trades', 0) or 0),
            'Stable':      str(row.get('Stable?', '')),
        })

    return {
        'stable_pct':        stable_pct,
        'avg_oos_net':       avg_oos_net,
        'avg_oos_wr':        avg_oos_wr,
        'consistency_score': consistency_score,
        'recommendation':    recommendation,
        'n_windows':         n_windows,
        'windows':           windows_data,
    }


_SOURCE_BADGES = {
    'batch_wf': '📐 Batch WF',
    'tab10':    '🔍 Manual',
    'tab12':    '🔍 Manual',
    'tab1':     '📊 Monitor',
}


def _build_table_df(entries: list, wf_cache: dict) -> pd.DataFrame:
    rows = []
    for e in entries:
        sm   = e.get('scan_metrics', {})
        wfs  = _get_wf_status(e, wf_cache)
        _cs  = wfs.get('consistency_score') if wfs else None
        rows.append({
            'id':           e['id'],
            '_consist':     _cs if _cs is not None else -999.0,
            'Pair':         f"{e['long']} / {e['short']}",
            'Asset Classes':f"{e.get('asset_class_long', '?')} × {e.get('asset_class_short', '?')}",
            'Params':       f"E{e['entry_sd']} X{e['exit_sd']} V{e['vol_window']} T{e['trend_window']}",
            'Best Dir':     sm.get('best_dir', '—'),
            'WT Trades':    sm.get('trades_wt', '—'),
            'WT Net WR':    f"{sm['net_wr_wt']:.1%}" if sm.get('net_wr_wt') is not None else '—',
            'WT Avg Net':   f"{sm['avg_net_wt']:+.4f}" if sm.get('avg_net_wt') is not None else '—',
            'WT Hold':      f"{sm.get('avg_hold_wt', '—')}d" if sm.get('avg_hold_wt') else '—',
            'WF Status':    _wf_badge(e['id'], wf_cache, wfs),
            'WF Consist.':  _wf_consistency(e['id'], wf_cache, wfs),
            'Source':       _SOURCE_BADGES.get(e.get('source', ''), e.get('source', '—') or '—'),
            'Added':        e.get('added', ''),
            'Notes':        (e.get('notes', '') or '')[:40],
        })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['id', '_consist', 'Pair', 'Asset Classes', 'Params', 'Best Dir',
                 'WT Trades', 'WT Net WR', 'WT Avg Net', 'WT Hold',
                 'WF Status', 'WF Consist.', 'Source', 'Added', 'Notes']
    )
    return df.sort_values('_consist', ascending=False).reset_index(drop=True) if not df.empty else df


def render() -> None:
    st.header("Watchlist")
    st.caption("Pairs saved from the Scenario Scanner. Send to Pair Analysis, Stake Calc, or Walk-Forward.")

    entries  = load_watchlist()
    wf_cache = load_wf_cache()

    hdr1, hdr2 = st.columns([6, 1])
    hdr1.markdown(f"**{len(entries)} pair(s) saved**")
    if hdr2.button("↻ Refresh", key="wl_refresh"):
        st.rerun()

    if not entries:
        st.info("No pairs saved yet. Run a Scenario scan and click ★ Add to Watchlist.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns(4)

    _all_ac = sorted({e.get('asset_class_long', '') for e in entries}
                     | {e.get('asset_class_short', '') for e in entries}
                     if entries else set())
    _ac_filter = fc1.selectbox("Asset class", ["All"] + _all_ac, key="wl_f_ac")

    _dir_opts  = ["All", "WT", "CT", "Both", "Neither", "— Not run"]
    _dir_filter = fc2.selectbox("Best direction", _dir_opts, key="wl_f_dir")

    _search = fc3.text_input("Instrument search", key="wl_f_search").strip().upper()

    _wf_only = fc4.checkbox("Only WF-verified", key="wl_f_wf")

    # Apply filters
    def _passes(e: dict) -> bool:
        sm = e.get('scan_metrics', {})
        bd = sm.get('best_dir', '')
        if _ac_filter != "All":
            if _ac_filter not in (e.get('asset_class_long', ''), e.get('asset_class_short', '')):
                return False
        if _dir_filter != "All" and _dir_filter != "— Not run":
            if bd != _dir_filter:
                return False
        if _dir_filter == "— Not run":
            if bd:
                return False
        if _search:
            if _search not in e.get('long', '').upper() and _search not in e.get('short', '').upper():
                return False
        if _wf_only and _get_wf_status(e, wf_cache) is None:
            return False
        return True

    filtered = [e for e in entries if _passes(e)]

    # ── Table ─────────────────────────────────────────────────────────────────
    tbl_df = _build_table_df(filtered, wf_cache)
    st.dataframe(
        tbl_df.drop(columns=['id', '_consist']),
        use_container_width=True,
        hide_index=True,
    )

    # CSV export
    _export_cols = [c for c in tbl_df.columns if c not in ('id', '_consist')]
    _csv = tbl_df[_export_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇ Export CSV", _csv,
        file_name="watchlist_export.csv",
        mime="text/csv",
        key="wl_export",
    )

    if not filtered:
        return

    st.divider()

    # ── Entry selector ────────────────────────────────────────────────────────
    _pair_opts = [f"{e['long']}/{e['short']} — E{e['entry_sd']} X{e['exit_sd']} V{e['vol_window']}"
                  for e in filtered]
    _sel_idx = st.selectbox(
        "Select pair for actions",
        range(len(_pair_opts)),
        format_func=lambda i: _pair_opts[i],
        key="wl_sel",
    )
    entry = filtered[_sel_idx]

    # ── Detail panel ─────────────────────────────────────────────────────────
    with st.container(border=True):
        sm   = entry.get('scan_metrics', {})
        _wfs = _get_wf_status(entry, wf_cache)
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Pair",       f"{entry['long']} / {entry['short']}")
        d2.metric("Best Dir",   sm.get('best_dir', '—'))
        d3.metric("WT Trades",  sm.get('trades_wt', '—'))
        d4.metric("WT Avg Net", f"{sm['avg_net_wt']:+.4f}" if sm.get('avg_net_wt') is not None else '—')
        d5.metric("WF Status",  _wf_badge(entry['id'], wf_cache, _wfs))

        st.caption(
            f"**Params:** Entry {entry['entry_sd']} | Exit {entry['exit_sd']} | "
            f"Vol {entry['vol_window']}d | Trend {entry['trend_window']}d | "
            f"Mode: {entry.get('trend_mode', '—')} | Added: {entry.get('added', '—')}"
        )

        # ── Action buttons ────────────────────────────────────────────────────
        ab1, ab2, ab3 = st.columns(3)

        if ab1.button("→ Pair Analysis", key=f"wl_to_pa_{entry['id']}"):
            st.session_state['pa_long_pending']  = [entry['long']]
            st.session_state['pa_short_pending'] = [entry['short']]
            st.session_state['pa_pair_pending']  = '— Custom pair —'
            st.session_state['pa_vol']           = entry['vol_window']
            st.session_state['pa_xing']          = entry['entry_sd']
            st.session_state['pa_exit']          = entry['exit_sd']
            st.session_state['pa_trend_window']  = entry['trend_window']
            st.session_state['sidebar_nav']      = "📈 Pair Analysis"
            st.rerun()

        if ab2.button("→ Stake Calc", key=f"wl_to_sc_{entry['id']}"):
            st.session_state['sc_long_pending']  = [entry['long']]
            st.session_state['sc_short_pending'] = [entry['short']]
            st.session_state['sidebar_nav']      = "🧮 Stake Calc"
            st.rerun()

        if ab3.button("→ Walk-Forward", key=f"wl_to_wf_{entry['id']}"):
            st.session_state['wf_pair'] = {
                'long':              entry['long'],
                'short':             entry['short'],
                'vol_window':        entry['vol_window'],
                'entry_sd':          entry['entry_sd'],
                'exit_sd':           entry['exit_sd'],
                'trend_window':      entry['trend_window'],
                'trend_mode':        entry.get('trend_mode', 'Both passes'),
                'asset_class_long':  entry.get('asset_class_long', ''),
                'asset_class_short': entry.get('asset_class_short', ''),
                'source':            'tab12',
            }
            st.session_state['wf11_long_pending']  = [entry['long']]
            st.session_state['wf11_short_pending'] = [entry['short']]
            st.session_state['sidebar_nav']        = "🔀 Walk-Forward"
            st.rerun()

        # ── Inline Walk-Forward ───────────────────────────────────────────────
        with st.expander("▶ Run Walk-Forward inline"):
            wc1, wc2, wc3, wc4 = st.columns(4)
            _is_days   = int(wc1.number_input("In-sample (days)",     252, 2520,  756, key=f"wl_is_{entry['id']}"))
            _oos_days  = int(wc2.number_input("Out-of-sample (days)",  63,  756,  252, key=f"wl_oos_{entry['id']}"))
            _step_days = int(wc3.number_input("Step size (days)",       63,  756,  252, key=f"wl_step_{entry['id']}"))
            _broker    = wc4.selectbox(
                "Broker",
                ["ig_spreadbet", "ig_cfd"],
                format_func=lambda x: {"ig_spreadbet": "IG Spread Bet", "ig_cfd": "IG CFD"}[x],
                key=f"wl_broker_{entry['id']}",
            )

            if st.button("▶ Run", key=f"wl_run_wf_{entry['id']}"):
                with st.spinner("Running walk-forward…"):
                    try:
                        _wf_df, _wf_err = _run_wf(
                            long_legs  = [entry['long']],
                            short_legs = [entry['short']],
                            vol_window = entry['vol_window'],
                            xing_sd    = entry['entry_sd'],
                            exit_sd    = entry['exit_sd'],
                            trend_win  = entry['trend_window'],
                            trend_mode = entry.get('trend_mode', 'Both passes'),
                            is_days    = _is_days,
                            oos_days   = _oos_days,
                            step_days  = _step_days,
                            broker     = _broker,
                        )
                        if _wf_err:
                            st.error(_wf_err)
                        else:
                            _summary = _compute_wf_summary(_wf_df)
                            save_wf_result(entry['id'], _summary)
                            st.success(
                                f"{_summary['recommendation']} — "
                                f"{_summary['stable_pct']:.0f}% stable windows "
                                f"({_summary['n_windows']} total)"
                            )
                            st.rerun()
                    except Exception as _e:
                        st.error(f"Walk-forward error: {_e}")

            # Show cached result if available
            if entry['id'] in wf_cache:
                _cr = wf_cache[entry['id']]
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Result",     _cr.get('recommendation', '—'))
                rc2.metric("Stable %",   f"{_cr.get('stable_pct', 0):.0f}%")
                rc3.metric("Avg OOS Net",f"{_cr['avg_oos_net']:+.4f}" if _cr.get('avg_oos_net') is not None else '—')
                rc4.metric("Consistency",f"{_cr['consistency_score']:.2f}" if _cr.get('consistency_score') is not None else '—')
                st.caption(f"Last run: {_cr.get('run_at', '—')}")

        # ── Notes ─────────────────────────────────────────────────────────────
        st.markdown("**Notes**")
        _new_notes = st.text_input(
            "Notes", value=entry.get('notes', ''),
            label_visibility="collapsed",
            key=f"wl_notes_{entry['id']}",
        )
        if st.button("💾 Save notes", key=f"wl_save_{entry['id']}"):
            update_notes(entry['id'], _new_notes)
            st.toast("Notes saved")
            st.rerun()

        # ── Remove (two-step) ─────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🗑️ Remove from watchlist", key=f"wl_remove_{entry['id']}"):
            st.session_state['wl_pending_remove'] = entry['id']

        if st.session_state.get('wl_pending_remove') == entry['id']:
            st.warning(f"Remove **{entry['long']}/{entry['short']}** from the watchlist?")
            rc1, rc2 = st.columns(2)
            if rc1.button("✔ Confirm", key=f"wl_confirm_{entry['id']}"):
                remove_from_watchlist(entry['id'])
                st.session_state.pop('wl_pending_remove', None)
                st.rerun()
            if rc2.button("✖ Cancel", key=f"wl_cancel_{entry['id']}"):
                st.session_state.pop('wl_pending_remove', None)
                st.rerun()
