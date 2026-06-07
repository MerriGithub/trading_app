"""
Tab 13 — Daily Scan
====================
Displays results from the overnight pair scan (research/daily_scan.py).
Shows AI morning briefing, open positions, per-search candidate tables,
scan history, and a search configuration panel.

Session state (widget keys)
---------------------------
tab13_dedup : bool
    When True, deduplicate mirror pairs (keep higher AvgNet_WT direction).

Session state written
---------------------
sidebar_nav_pending : str
    Written by "Open in Pair Analysis" button (register item B).
pa_vol, pa_xing, pa_exit, pa_trend_window, pa_wl_id : various
    Pair Analysis algo parameter pre-fill.
pa_long_pending, pa_short_pending, pa_pair_pending : list / str
    Pair Analysis instrument selection.
sc_long_pending, sc_short_pending : list
    Stake Calc pre-fill.
wf11_long_pending, wf11_short_pending, wf_pair : list / dict
    Tab 11 single-pair WF pre-fill.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from asset_configs import _DEFAULT_SCORING_MODE
from data_watchlist import add_to_watchlist

logger = logging.getLogger(__name__)

_ROOT         = Path(__file__).parent.parent
_RESULTS_PATH = _ROOT / 'data' / 'daily_scan_results.json'
_CONFIG_PATH  = _ROOT / 'data' / 'scan_config.json'
_HISTORY_DIR  = _ROOT / 'data' / 'scan_history'


# ─── helpers ──────────────────────────────────────────────────────────────────

def _isnan(v: object) -> bool:
    """Return True for None, NaN, or any other missing-value sentinel."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _fmt_ts(ts_str: str) -> tuple[str, bool]:
    """Parse an ISO-format timestamp string.

    Args:
        ts_str: ISO format timestamp (e.g. '2026-06-06T15:34:08.377415').

    Returns:
        Tuple of (formatted display string, True if the timestamp is >24 h old).
    """
    try:
        dt = datetime.fromisoformat(ts_str)
        age_h = (datetime.now() - dt).total_seconds() / 3600
        return dt.strftime('%d %b %Y %H:%M'), age_h > 24
    except (ValueError, TypeError):
        return ts_str or '—', False


def _dedup_candidates(candidates: list[dict]) -> list[dict]:
    """Remove mirror pairs, keeping the higher AvgNet_WT direction for each pair.

    Only deduplicates intra-asset pairs where frozenset({long, short}) matches.
    OPEN_POSITION rows are never removed.
    Preserves original ordering; replaces in-place when a better direction found.

    Args:
        candidates: Candidate dicts (any status) from a single search result.

    Returns:
        Filtered list with at most one direction per instrument pair.
    """
    seen: dict[frozenset, int] = {}
    result: list[dict] = []

    for c in candidates:
        if c.get('status') == 'OPEN_POSITION':
            result.append(c)
            continue
        key = frozenset({c['long'], c['short']})
        if key not in seen:
            seen[key] = len(result)
            result.append(c)
        else:
            i = seen[key]
            if (c.get('AvgNet_WT') or 0.0) > (result[i].get('AvgNet_WT') or 0.0):
                result[i] = c
    return result


# ─── candidate row ────────────────────────────────────────────────────────────

def _render_candidate_row(c: dict, search_cfg: dict, row_key: str) -> None:
    """Render one candidate: stats line + "Analyse" and "Watch" buttons.

    pair_cost and fin_daily are absent from non-OPEN_POSITION rows (CR1).
    Always accessed via .get() to avoid KeyError.

    Args:
        c: Candidate dict from daily_scan_results.json.
        search_cfg: The search's config entry from scan_config.json.
        row_key: Unique prefix for all Streamlit widget keys in this row.
    """
    algo = search_cfg.get('algo_params', {})

    col_info, col_btn1, col_btn2 = st.columns([5, 1, 1])
    with col_info:
        sd_str   = f"SD={c['current_sd']:+.2f}" if not _isnan(c.get('current_sd')) else "SD=n/a"
        rho_str  = (
            f"ρ={c['wf_rho']:+.3f} p={c['wf_p']:.3f}"
            if not _isnan(c.get('wf_rho')) else "ρ=n/a"
        )
        net_str  = f"Net={c['AvgNet_WT'] * 100:+.2f}%" if not _isnan(c.get('AvgNet_WT')) else "Net=—"
        hold_str = f"Hold={c['AvgHold_WT']:.0f}d"      if not _isnan(c.get('AvgHold_WT')) else ""
        open_tag = "  *(held)*" if c.get('already_open') else ""
        st.markdown(
            f"**{c['pair']}**  |  {sd_str}  |  {rho_str}  |  "
            f"T={c.get('Trades_WT', 0)} {net_str} {hold_str}  |  "
            f"Dir={c.get('Best_Dir', '—')}{open_tag}"
        )

    with col_btn1:
        if st.button("→ Analyse", key=f"{row_key}_pa", use_container_width=True):
            # Full 13-key pre-fill block (PF2). pa_wl_id=None clears any stale
            # watchlist selection in Pair Analysis before navigating there.
            st.session_state['pa_vol']             = algo.get('vol_window', 262)
            st.session_state['pa_xing']            = algo.get('xing_sd', 2.0)
            st.session_state['pa_exit']            = algo.get('exit_sd', 0.5)
            st.session_state['pa_trend_window']    = algo.get('trend_window', 262)
            st.session_state['pa_wl_id']           = None
            st.session_state['pa_long_pending']    = [c['long']]
            st.session_state['pa_short_pending']   = [c['short']]
            st.session_state['pa_pair_pending']    = '— Custom pair —'
            st.session_state['sc_long_pending']    = [c['long']]
            st.session_state['sc_short_pending']   = [c['short']]
            st.session_state['wf11_long_pending']  = [c['long']]
            st.session_state['wf11_short_pending'] = [c['short']]
            st.session_state['wf_pair'] = {
                'long':              [c['long']],
                'short':             [c['short']],
                'vol_window':        algo.get('vol_window', 262),
                'entry_sd':          algo.get('xing_sd', 2.0),
                'exit_sd':           algo.get('exit_sd', 0.5),
                'trend_window':      algo.get('trend_window', 262),
                'trend_mode':        algo.get('trend_mode', 'Both passes'),
                'asset_class_long':  c['long_ac'],
                'asset_class_short': c['short_ac'],
                'source':            'tab13',
            }
            # PF1: emoji required — must match exact sidebar entry string
            st.session_state['sidebar_nav_pending'] = '📈 Pair Analysis'
            st.rerun()

    with col_btn2:
        if st.button("★ Watch", key=f"{row_key}_wl", use_container_width=True):
            wl_ac_l = c['long_ac']
            wl_ac_s = c['short_ac']
            wl_mode = (
                _DEFAULT_SCORING_MODE.get('commodities', 'contrarian')
                if 'commodities' in (wl_ac_l, wl_ac_s)
                else _DEFAULT_SCORING_MODE.get(wl_ac_l, 'composite')
            )
            add_to_watchlist({
                'long':              c['long'],
                'short':             c['short'],
                'asset_class_long':  wl_ac_l,
                'asset_class_short': wl_ac_s,
                'entry_sd':          float(algo.get('xing_sd', 2.0)),
                'exit_sd':           float(algo.get('exit_sd', 0.5)),
                'vol_window':        int(algo.get('vol_window', 262)),
                'trend_window':      int(algo.get('trend_window', 262)),
                'trend_mode':        algo.get('trend_mode', 'Both passes'),
                'scoring_mode':      wl_mode,
                'source':            'tab13',
                'scan_metrics': {
                    'trades_wt':   int(c.get('Trades_WT', 0) or 0),
                    'avg_net_wt':  float(c.get('AvgNet_WT', 0.0) or 0.0),
                    'avg_hold_wt': int(c.get('AvgHold_WT', 0) or 0),
                    'best_dir':    c.get('Best_Dir', ''),
                    'wf_rho':      float(c.get('wf_rho', 0.0) or 0.0),
                    'wf_p':        float(c.get('wf_p', 1.0) or 1.0),
                },
            })
            st.toast(f"★ {c['long']}/{c['short']} added to watchlist")


# ─── section renderers ────────────────────────────────────────────────────────

def _render_open_positions(open_rows: list[dict]) -> None:
    """Render the cross-search open positions section.

    Args:
        open_rows: All candidates with status=='OPEN_POSITION' from all searches.
    """
    st.info(f"{len(open_rows)} position(s) currently held — showing live SD below")
    rows = []
    for c in open_rows:
        pair_cost = c.get('pair_cost', float('nan'))
        fin_daily = c.get('fin_daily', float('nan'))
        rows.append({
            'Pair':        c['pair'],
            'Current SD':  f"{c['current_sd']:+.2f}" if not _isnan(c.get('current_sd')) else '—',
            'Asset Class': f"{c.get('long_ac', '—')} / {c.get('short_ac', '—')}",
            'Pair Cost':   f"{pair_cost * 100:.3f}%" if not _isnan(pair_cost) else '—',
            'Fin Daily':   f"{fin_daily * 100:.4f}%" if not _isnan(fin_daily) else '—',
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Position status as of last scan run.")
    st.divider()


def _render_search_expander(
    result: dict,
    search_cfg: dict,
    dedup: bool,
) -> None:
    """Render one per-search collapsible section.

    Args:
        result: One search result dict from daily_scan_results.json.
        search_cfg: Matching entry from scan_config.json (may be empty dict if
            the search ID was not found — treat as unknown/disabled).
        dedup: Whether to apply mirror-pair deduplication.
    """
    name   = result.get('search_name', result.get('search_id', '?'))
    status = result.get('status', 'unknown')

    # Exclude OPEN_POSITION rows — rendered separately above
    candidates = [c for c in result.get('candidates', []) if c.get('status') != 'OPEN_POSITION']
    if dedup:
        candidates = _dedup_candidates(candidates)

    n_strong = sum(1 for c in candidates if c['status'] == 'STRONG')
    expanded = (status == 'completed') and (n_strong > 0)

    with st.expander(name, expanded=expanded):
        if status == 'failed':
            st.error(f"Search failed: {result.get('error', 'unknown error')}")
            return

        if status == 'skipped':
            # blocked_reason is in scan_config.json, not in results (CR3, CR4)
            blocked = search_cfg.get('blocked_reason', 'Disabled — not yet researched.')
            st.caption(blocked)
            return

        # completed
        n_watch   = sum(1 for c in candidates if c['status'] == 'WATCH')
        n_monitor = sum(1 for c in candidates if c['status'] == 'MONITOR')
        n_weak    = sum(1 for c in candidates if c['status'] == 'WEAK')

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pairs scanned",  result.get('pairs_scanned', 0))
        m2.metric("WF validated",   result.get('pairs_wf_validated', 0))
        m3.metric("STRONG",         n_strong)
        m4.metric("WATCH",          n_watch)

        strong_list = [c for c in candidates if c['status'] == 'STRONG']
        watch_list  = [c for c in candidates if c['status'] == 'WATCH']
        sid         = result.get('search_id', 'unknown')

        if strong_list:
            st.markdown("#### 🟢 STRONG")
            for idx, c in enumerate(strong_list):
                _render_candidate_row(c, search_cfg, f"t13_{sid}_s{idx}")
                st.divider()

        if watch_list:
            st.markdown("#### 🔵 WATCH")
            for idx, c in enumerate(watch_list):
                _render_candidate_row(c, search_cfg, f"t13_{sid}_w{idx}")
                st.divider()

        if n_monitor:
            st.caption(f"🟡 MONITOR: {n_monitor} pair(s) — not yet near signal threshold.")
        if n_weak:
            st.caption(f"⚪ WEAK: {n_weak} pair(s) — failed WF filter.")


def _render_config_panel(scan_config: dict | None) -> None:
    """Render the search configuration expander with enable/disable toggles.

    Iterates scan_config['searches'] (not results) so disabled searches appear.
    blocked_reason searches render as disabled toggles with an explanatory caption.

    Args:
        scan_config: Parsed scan_config.json, or None if the file could not be read.
    """
    with st.expander("Search configuration", expanded=False):
        if scan_config is None:
            st.caption("scan_config.json not found.")
            return

        searches = scan_config.get('searches', [])
        for s in searches:
            sid     = s['id']
            blocked = s.get('blocked_reason')  # CR3: may be absent
            if blocked:
                st.toggle(s['name'], value=False, disabled=True, key=f'tab13_cfg_{sid}')
                st.caption(blocked)
            else:
                st.toggle(
                    s['name'],
                    value=bool(s.get('enabled', False)),
                    key=f'tab13_cfg_{sid}',
                )

        if st.button("💾 Save config", key='tab13_cfg_save'):
            updated = dict(scan_config)
            updated_searches = []
            for s in searches:
                entry = dict(s)
                if not s.get('blocked_reason'):
                    entry['enabled'] = bool(
                        st.session_state.get(f'tab13_cfg_{s["id"]}', s.get('enabled', False))
                    )
                updated_searches.append(entry)
            updated['searches'] = updated_searches
            _CONFIG_PATH.write_text(json.dumps(updated, indent=2), encoding='utf-8')
            st.success("Configuration saved.")


def _render_history_panel() -> None:
    """Render the scan history expander showing the last 7 runs."""
    with st.expander("Scan history", expanded=False):
        if not _HISTORY_DIR.exists():
            st.caption("No scan history yet.")
            return

        history_files = sorted(_HISTORY_DIR.glob('*.json'), reverse=True)[:7]
        if not history_files:
            st.caption("No scan history yet.")
            return

        for f in history_files:
            try:
                dt = datetime.strptime(f.stem, '%Y-%m-%d_%H%M')  # CR6
                display = dt.strftime('%d %b %Y %H:%M')
            except ValueError:
                display = f.stem

            try:
                h = json.loads(f.read_text(encoding='utf-8'))
                n_strong = sum(
                    sum(1 for c in r.get('candidates', []) if c.get('status') == 'STRONG')
                    for r in h.get('searches', [])
                )
                duration = h.get('run_duration_seconds', '—')
                dur_str  = f"{duration}s" if isinstance(duration, (int, float)) else '—'
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                logger.warning("Could not parse history file %s: %s", f.name, exc)
                continue

            hc1, hc2, hc3, hc4 = st.columns([3, 1, 1, 1])
            hc1.write(display)
            hc2.write(dur_str)
            hc3.write(f"🟢 {n_strong}")
            hc4.download_button(
                "⬇",
                f.read_bytes(),
                file_name=f.name,
                mime='application/json',
                key=f"t13_hist_{f.stem}",
            )


# ─── entry point ──────────────────────────────────────────────────────────────

def render() -> None:
    """Render the Daily Scan tab.

    Loads data/daily_scan_results.json on each call — no caching, always shows
    the latest run. Loads scan_config.json separately to retrieve blocked_reason
    for skipped searches, which are absent from the results JSON (CR4).
    """
    # Poll any running background scan; auto-rerun every 2 s until it finishes
    if 'tab13_scan_proc' in st.session_state:
        proc    = st.session_state['tab13_scan_proc']
        retcode = proc.poll()
        if retcode is None:
            st.info("⏳ Scan running… page refreshes automatically.")
            time.sleep(2)
            st.rerun()
        else:
            del st.session_state['tab13_scan_proc']
            if retcode != 0:
                st.error(f"Scan failed (exit code {retcode}).")
            st.rerun()

    st.header("📅 Daily Scan")

    hcol1, hcol2 = st.columns([4, 1])

    if not _RESULTS_PATH.exists():
        with hcol1:
            st.info("No scan results yet. Click 'Run scan now' to generate the first scan.")
        with hcol2:
            if st.button("▶ Run scan now", key='tab13_run_btn'):
                proc = subprocess.Popen(
                    [sys.executable, str(_ROOT / 'research' / 'daily_scan.py')],
                    cwd=str(_ROOT),
                )
                st.session_state['tab13_scan_proc'] = proc
                st.rerun()
        try:
            scan_config: dict | None = json.loads(_CONFIG_PATH.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            scan_config = None
        _render_config_panel(scan_config)
        return

    results = json.loads(_RESULTS_PATH.read_text(encoding='utf-8'))

    try:
        scan_config = json.loads(_CONFIG_PATH.read_text(encoding='utf-8'))
        config_by_id: dict[str, dict] = {
            s['id']: s for s in scan_config.get('searches', [])
        }
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load scan_config.json: %s", exc)
        scan_config  = None
        config_by_id = {}

    ts_display, is_stale = _fmt_ts(results.get('run_timestamp', ''))
    with hcol1:
        if is_stale:
            st.markdown(
                f"<span style='color:grey'>Last run: {ts_display} (>24 h ago)</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"Last run: **{ts_display}**")

    with hcol2:
        if st.button(
            "▶ Run scan now",
            key='tab13_run_btn',
            disabled='tab13_scan_proc' in st.session_state,
        ):
            proc = subprocess.Popen(
                [sys.executable, str(_ROOT / 'research' / 'daily_scan.py')],
                cwd=str(_ROOT),
            )
            st.session_state['tab13_scan_proc'] = proc
            st.rerun()

    # ── AI Morning Briefing ───────────────────────────────────────────────────
    ai_review = results.get('ai_review', '')
    with st.expander("AI Morning Briefing", expanded=bool(ai_review)):
        if ai_review:
            st.markdown(ai_review)
        else:
            st.caption("No AI review available for this run.")

    # ── Dedup toggle ──────────────────────────────────────────────────────────
    dedup = st.checkbox("Deduplicate mirror pairs", value=True, key='tab13_dedup')

    # ── Open Positions (cross-search, rendered before per-search expanders) ───
    all_open: list[dict] = [
        c
        for r in results.get('searches', [])
        for c in r.get('candidates', [])
        if c.get('status') == 'OPEN_POSITION'
    ]
    if all_open:
        _render_open_positions(all_open)

    # ── Per-search expanders ──────────────────────────────────────────────────
    for result in results.get('searches', []):
        sid        = result.get('search_id', '')
        search_cfg = config_by_id.get(sid, {})
        _render_search_expander(result, search_cfg, dedup)

    st.divider()

    # ── Config and history panels ─────────────────────────────────────────────
    _render_config_panel(scan_config)
    _render_history_panel()
