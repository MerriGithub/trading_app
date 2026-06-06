"""
daily_scan.py — Overnight pair scan batch script
=================================================

Runs all enabled named searches from data/scan_config.json, performs
a two-stage gate (scenario scan → WF validation on shortlist), calls
the Claude API for an AI review, and writes results to:
  - data/daily_scan_results.json  (overwritten each run)
  - data/scan_history/YYYY-MM-DD_HHMM.json  (archived per run)
  - Obsidian note (if obsidian_output_enabled)

Run from trading_app/:
    python research/daily_scan.py

No Streamlit dependency — uses DataRegistry and engine modules directly.
"""
from __future__ import annotations

import json
import keyring
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow running from trading_app/ without installing as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_registry import DataRegistry
from engine.backtest import (
    aggregate_trades,
    load_asset_prices,
    prepare_returns,
    run_backtest,
)
from engine.numba_core import COL_ENTRY_IDX, COL_SIDE, rolling_mean_std
from engine.walkforward import (
    run_cross_asset_walkforward,
    run_walk_forward,
    summarise_walk_forward,
)
from asset_configs import (
    ASSET_CLASSES,
    COMMODITY_EXCLUDE,
    FI_EXCLUDE,
    get_display_name,
    get_tradeable_instruments,
)
from account import get_financing_daily_rate

logger = logging.getLogger(__name__)

_TRADING_APP_ROOT = Path(__file__).parent.parent
_CACHE_DIR        = _TRADING_APP_ROOT / 'cache'
_DATA_DIR         = _TRADING_APP_ROOT / 'data'
_CONFIG_PATH      = _DATA_DIR / 'scan_config.json'
_RESULTS_PATH     = _DATA_DIR / 'daily_scan_results.json'
_HISTORY_DIR      = _DATA_DIR / 'scan_history'


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def load_scan_config() -> dict:
    """Load and validate scan_config.json.

    Returns:
        Parsed config dict with 'searches' list and 'global_settings'.

    Raises:
        FileNotFoundError: If scan_config.json does not exist.
        ValueError: If required keys are absent or JSON is malformed.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"scan_config.json not found at {_CONFIG_PATH}")
    try:
        config = json.loads(_CONFIG_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f"scan_config.json is malformed: {exc}") from exc
    for key in ('searches', 'global_settings'):
        if key not in config:
            raise ValueError(f"scan_config.json missing required key: '{key}'")
    return config


# ═══════════════════════════════════════════════════════════════════════════
# INSTRUMENT / COST HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_exclusions(search: dict) -> frozenset[str]:
    """Return combined exclusion set for a search, including standard class exclusions.

    Args:
        search: Search config dict from scan_config.json.

    Returns:
        frozenset of instrument codes to exclude from pair generation.
    """
    ac: str = search.get('asset_class', '')
    excludes: set[str] = set(search.get('exclude', []))
    if ac == 'commodities':
        excludes |= COMMODITY_EXCLUDE
    elif ac == 'fixed_income':
        excludes |= FI_EXCLUDE
    return frozenset(excludes)


def _get_instruments(asset_class: str, exclude: frozenset[str]) -> list[str]:
    """Return tradeable instruments for an asset class minus exclusions.

    Args:
        asset_class: Asset class key (e.g. 'equity', 'commodities').
        exclude: Instrument codes to remove from the tradeable list.

    Returns:
        List of instrument codes available for pair generation.

    Raises:
        ValueError: If fewer than 2 instruments remain after exclusion.
    """
    instruments = [i for i in get_tradeable_instruments(asset_class) if i not in exclude]
    if len(instruments) < 2:
        raise ValueError(
            f"Fewer than 2 instruments remain for {asset_class!r} after exclusions: {instruments}"
        )
    return instruments


def _infer_asset_class(instrument: str) -> str:
    """Return the asset_class key for an instrument code, or '' if not found.

    Iterates over ASSET_CLASSES and returns the first key whose 'instruments'
    dict contains the given code. Used by the open-positions overlay to route
    each position to the correct search.

    Args:
        instrument: Instrument code (e.g. 'CIL', 'GOLD', 'EURUSD').

    Returns:
        Asset class key string (e.g. 'equity', 'commodities', 'fx') or ''
        if the instrument is not found in any asset class.
    """
    for ac_key, ac_cfg in ASSET_CLASSES.items():
        if instrument in ac_cfg.get('instruments', {}):
            return ac_key
    return ''


def _get_pair_cost(
    long_inst: str,
    long_ac: str,
    short_inst: str,
    short_ac: str,
) -> float:
    """Compute round-trip spread cost for a pair, replicating Tab 10 lines 250-253.

    Args:
        long_inst: Long instrument code.
        long_ac: Long instrument's asset class key.
        short_inst: Short instrument code.
        short_ac: Short instrument's asset class key.

    Returns:
        Round-trip cost as fraction: 2 × (long_spread + short_spread).
    """
    _lc_cfg = ASSET_CLASSES[long_ac]['instruments'].get(long_inst, {})
    _sc_cfg = ASSET_CLASSES[short_ac]['instruments'].get(short_inst, {})
    _l_sp = _lc_cfg.get('spread_pct', 0.001) if isinstance(_lc_cfg, dict) else 0.001
    _s_sp = _sc_cfg.get('spread_pct', 0.001) if isinstance(_sc_cfg, dict) else 0.001
    return 2.0 * (_l_sp + _s_sp)


def _get_fin_daily(
    long_inst: str,
    long_ac: str,
    short_inst: str,
    short_ac: str,
    broker_profile: str,
) -> float:
    """Compute average daily financing cost for a pair, replicating Tab 10 lines 255-257.

    Args:
        long_inst: Long instrument code.
        long_ac: Long instrument's asset class key.
        short_inst: Short instrument code.
        short_ac: Short instrument's asset class key.
        broker_profile: Broker profile key (e.g. 'ig_spreadbet').

    Returns:
        Average of long and short daily financing rates as fraction of notional.
    """
    long_fin  = get_financing_daily_rate(long_inst,  long_ac,  'long',  broker_profile=broker_profile)
    short_fin = get_financing_daily_rate(short_inst, short_ac, 'short', broker_profile=broker_profile)
    return (long_fin + short_fin) / 2


# ═══════════════════════════════════════════════════════════════════════════
# CORE PER-PAIR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _build_open_position_row(
    long_i: str,
    long_ac: str,
    short_i: str,
    short_ac: str,
    prices_df: pd.DataFrame,
    vol_window: int,
    broker_profile: str,
) -> dict:
    """Build a candidate dict for an open position that failed the scan filter.

    Used by the open-positions overlay pass in run_intra_asset_search() and
    run_cross_asset_search(). Computes current_sd so Tab 13 and the AI review
    can show whether the position is reverting or extending.

    All backtest fields (Trades_WT, AvgNet_WT, etc.) are NaN — this row was not
    ranked by the scanner. The 'status' field is set to 'OPEN_POSITION' so Tab 13
    can render it in a dedicated section rather than mixing it with candidates.

    Args:
        long_i: Long instrument code.
        long_ac: Long instrument asset class key.
        short_i: Short instrument code.
        short_ac: Short instrument asset class key.
        prices_df: Price DataFrame containing both instruments (DatetimeIndex).
            For intra-asset: the full asset class prices_df.
            For cross-asset: an inner-joined pair slice built by the caller.
        vol_window: Vol window to use for current_sd calculation.
        broker_profile: Broker profile for financing rate lookup.

    Returns:
        Candidate dict with status='OPEN_POSITION'. All scan/WF fields are NaN
        or None except current_sd, pair_cost, fin_daily, already_open=True.
    """
    nan = float('nan')

    current_sd = nan
    if long_i in prices_df.columns and short_i in prices_df.columns:
        current_sd = _compute_current_sd(prices_df, long_i, short_i, vol_window)

    pair_cost = _get_pair_cost(long_i, long_ac, short_i, short_ac)
    fin_daily = _get_fin_daily(long_i, long_ac, short_i, short_ac, broker_profile)

    long_disp  = get_display_name(long_ac,  long_i)
    short_disp = get_display_name(short_ac, short_i)

    return {
        'pair':          f'{long_disp} / {short_disp}',
        'long':          long_i,
        'short':         short_i,
        'long_display':  long_disp,
        'short_display': short_disp,
        'long_ac':       long_ac,
        'short_ac':      short_ac,
        'current_sd':    current_sd,
        'wf_rho':        nan,
        'wf_p':          nan,
        'oos_trades':    0,
        'oos_win_rate':  nan,
        'oos_avg_gross': nan,
        'Trades_WT':     0,
        'AvgNet_WT':     nan,
        'AvgHold_WT':    nan,
        'Best_Dir':      'n/a',
        'status':        'OPEN_POSITION',
        'already_open':  True,
        'pair_cost':     pair_cost,
        'fin_daily':     fin_daily,
    }


def _run_pair_both_passes(
    pair_df: pd.DataFrame,
    long_i: str,
    short_i: str,
    long_ac: str,
    short_ac: str,
    algo_params: dict,
    scan_params: dict,
    broker_profile: str,
) -> Optional[dict]:
    """Run Tab 10 'Both passes' analysis on a single pair.

    Replicates Tab 10 lines 262-354 exactly: log-return trend detection,
    vol-scaled backtest, and trend-aligned / counter-trend split.

    Args:
        pair_df: DataFrame with columns [long_i, short_i], DatetimeIndex.
        long_i: Long instrument code.
        short_i: Short instrument code.
        long_ac: Long instrument's asset class key.
        short_ac: Short instrument's asset class key.
        algo_params: Algo params dict (xing_sd, exit_sd, vol_window, trend_window,
            history_days).
        scan_params: Scan params dict (min_wt_trades, min_avg_net_wt).
        broker_profile: Broker profile key.

    Returns:
        Dict with keys long, short, long_display, short_display, long_ac, short_ac,
        Trades_WT, NetWR_WT, AvgNet_WT, AvgHold_WT, Trades_CT, NetWR_CT, AvgNet_CT,
        AvgHold_CT, Best_Dir, Aligned_pct, pair_cost, fin_daily.
        None if the pair fails filters (insufficient data, < min_wt_trades, or
        AvgNet_WT <= min_avg_net_wt).
    """
    # Step 1: Log-return spread for trend direction detection
    raw_lr = np.log(pair_df[long_i] / pair_df[short_i]).diff().fillna(0)
    raw_cum_spr = raw_lr.cumsum()
    trend_win = algo_params['trend_window']
    sc_slb = min(20, trend_win // 10)
    trend_ser = raw_cum_spr.rolling(trend_win, min_periods=10).mean()
    trend_arr = trend_ser.values

    # Step 2: Vol-scaled spread via prepare_returns
    scaled, day_ints, index = prepare_returns(
        pair_df, [long_i, short_i],
        vol_window=algo_params['vol_window'],
        window_days=algo_params.get('history_days'),
    )
    if scaled.shape[0] < algo_params['vol_window']:
        return None
    spread = scaled[:, 0] - scaled[:, 1]

    # Step 3: Full backtest
    pair_cost = _get_pair_cost(long_i, long_ac, short_i, short_ac)
    fin_daily = _get_fin_daily(long_i, long_ac, short_i, short_ac, broker_profile)
    bt_res = run_backtest(
        spread, day_ints,
        vol_window=algo_params['vol_window'],
        xing_sd=algo_params['xing_sd'],
        exit_sd=algo_params['exit_sd'],
        spread_cost_pct=pair_cost,
        financing_daily_pct=fin_daily,
        n_legs=2,
    )
    n_tr = bt_res['n_trades']
    if n_tr == 0:
        return None

    # Step 4: Both passes trend split (replicates Tab 10 lines 289-354)
    raw_t  = bt_res['trades_raw'][:n_tr]
    eidxs  = raw_t[:, COL_ENTRY_IDX].astype(int)
    sides  = raw_t[:, COL_SIDE]
    edates = index[eidxs]

    tipos  = trend_ser.index.get_indexer(edates, method='nearest')
    prev_p = np.maximum(0, tipos - sc_slb)
    has_tr = (tipos >= sc_slb) & ~np.isnan(trend_arr[tipos])
    slp    = np.where(
        has_tr,
        (trend_arr[tipos] - trend_arr[prev_p]) / sc_slb,
        np.nan,
    )
    al  = (((sides > 0) & (slp > 0)) | ((sides < 0) & (slp < 0)))
    vld = ~np.isnan(slp)
    al_pct = float(al[vld].mean()) if vld.any() else float('nan')

    wt_f = al & vld
    ct_f = ~al & vld
    n_wt = int(wt_f.sum())
    n_ct = int(ct_f.sum())

    wt_s = aggregate_trades(raw_t[wt_f], n_wt, pair_cost, fin_daily, 2) if n_wt > 0 else {}
    ct_s = aggregate_trades(raw_t[ct_f], n_ct, pair_cost, fin_daily, 2) if n_ct > 0 else {}

    wt_net = float(wt_s.get('avg_net', float('nan')))
    ct_net = float(ct_s.get('avg_net', float('nan')))
    wt_pos = not np.isnan(wt_net) and wt_net > 0
    ct_pos = not np.isnan(ct_net) and ct_net > 0
    best_dir = (
        'Both'    if wt_pos and ct_pos else
        'WT'      if wt_pos else
        'CT'      if ct_pos else 'Neither'
    )

    # Step 5: Apply scan filters
    min_wt = scan_params['min_wt_trades']
    if n_wt < min_wt:
        return None
    if np.isnan(wt_net) or wt_net <= scan_params.get('min_avg_net_wt', 0.0):
        return None

    long_disp  = get_display_name(long_ac,  long_i)
    short_disp = get_display_name(short_ac, short_i)

    return {
        'long':          long_i,
        'short':         short_i,
        'long_display':  long_disp,
        'short_display': short_disp,
        'long_ac':       long_ac,
        'short_ac':      short_ac,
        'Trades_WT':     n_wt,
        'NetWR_WT':      float(wt_s.get('net_wr', float('nan'))),
        'AvgNet_WT':     wt_net,
        'AvgHold_WT':    float(wt_s.get('avg_holding', float('nan'))),
        'Trades_CT':     n_ct,
        'NetWR_CT':      float(ct_s.get('net_wr', float('nan'))),
        'AvgNet_CT':     ct_net,
        'AvgHold_CT':    float(ct_s.get('avg_holding', float('nan'))),
        'Best_Dir':      best_dir,
        'Aligned_pct':   al_pct,
        'pair_cost':     pair_cost,
        'fin_daily':     fin_daily,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_current_sd(
    prices_df: pd.DataFrame,
    long_i: str,
    short_i: str,
    vol_window: int,
) -> float:
    """Compute the current spread SD position for near-signal detection.

    Uses the same vol-scaled spread approach as run_backtest to compute
    where the spread currently sits relative to its rolling mean, in
    rolling-std units.

    Args:
        prices_df: Price DataFrame containing at least long_i and short_i columns.
        long_i: Long instrument code.
        short_i: Short instrument code.
        vol_window: Rolling window in trading days.

    Returns:
        Current dist_sd value. NaN if data is insufficient or std is zero.
    """
    scaled, _, _ = prepare_returns(
        prices_df[[long_i, short_i]], [long_i, short_i],
        vol_window=vol_window,
    )
    spread = scaled[:, 0] - scaled[:, 1]
    roll_mean, roll_std = rolling_mean_std(spread, vol_window)
    if len(roll_std) == 0 or np.isnan(roll_std[-1]) or roll_std[-1] == 0:
        return float('nan')
    return float((spread[-1] - roll_mean[-1]) / roll_std[-1])


def _classify_candidate(
    row: dict,
    wf_summary: dict,
    current_sd: float,
    xing_sd: float,
    near_signal_buffer: float,
) -> str:
    """Classify a WF-validated candidate into STRONG / WATCH / MONITOR / WEAK.

    Args:
        row: Scan result dict from _run_pair_both_passes.
        wf_summary: Output of summarise_walk_forward.
        current_sd: Current spread SD position from _compute_current_sd.
        xing_sd: Entry threshold (e.g. 2.0).
        near_signal_buffer: Distance from xing_sd that counts as near-signal.

    Returns:
        One of 'STRONG', 'WATCH', 'MONITOR', 'WEAK'.
        Note: 'OPEN_POSITION' is a fifth status produced by _build_open_position_row()
        for positions that failed scan filters but are currently held. Tab 13 renders
        these in a dedicated section separate from ranked candidates.
    """
    _rho = wf_summary.get('rho')
    _p   = wf_summary.get('p_value')
    rho  = float(_rho) if _rho is not None else 0.0
    p    = float(_p)   if _p   is not None else 1.0
    oos  = int(wf_summary.get('n_obs') or 0)
    near_signal = (
        not np.isnan(current_sd)
        and abs(current_sd) >= (xing_sd - near_signal_buffer)
    )

    if rho > 0.05 and p < 0.10 and near_signal and row['AvgNet_WT'] > 0:
        return 'STRONG'
    if rho > 0.05 and p < 0.10 and row['AvgNet_WT'] > 0:
        return 'WATCH'
    if rho > 0 and oos >= 5:
        return 'MONITOR'
    return 'WEAK'


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO / POSITION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _check_already_open(long_inst: str, short_inst: str) -> bool:
    """Check if an open position already holds both instruments.

    Args:
        long_inst: Long instrument code.
        short_inst: Short instrument code.

    Returns:
        True if any non-closed position contains both instruments. False on
        file-not-found or parse errors (treated as no open positions).
    """
    try:
        positions = json.loads((_DATA_DIR / 'positions.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    for pos in positions:
        if pos.get('status') == 'closed':
            continue
        basket = pos.get('basket', {})
        all_legs = basket.get('long_legs', []) + basket.get('short_legs', [])
        if long_inst in all_legs and short_inst in all_legs:
            return True
    return False


def _load_open_positions() -> list[dict]:
    """Load all open positions from data/positions.json.

    Returns:
        List of open position dicts. Each dict has the positions.json structure:
        instruments are nested as pos['basket']['long_legs'][0] and
        pos['basket']['short_legs'][0]. Returns empty list if the file cannot
        be read or contains no open positions.

    Notes:
        Silently returns [] on any file/parse error — the overlay is informational
        and must not abort the scan on data-access failure.
    """
    positions_path = _DATA_DIR / 'positions.json'
    try:
        raw = json.loads(positions_path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

    positions: list[dict] = raw if isinstance(raw, list) else raw.get('positions', [])
    return [p for p in positions if str(p.get('status', '')).lower() == 'open']


def _load_open_positions_summary() -> str:
    """Load open positions from positions.json and return a summary string.

    Returns:
        Human-readable summary for the AI review prompt. Returns a safe
        fallback string if the file is missing or malformed.
    """
    open_pos = _load_open_positions()
    if not open_pos:
        return 'No open positions.'

    parts: list[str] = []
    for p in open_pos:
        long_i  = p.get('basket', {}).get('long_legs',  ['?'])[0]
        short_i = p.get('basket', {}).get('short_legs', ['?'])[0]
        side    = p.get('direction', p.get('side', '?'))
        days_str = ''
        entry_date_raw = p.get('entry_date')
        if entry_date_raw:
            try:
                entry_dt = datetime.fromisoformat(str(entry_date_raw)[:10])
                days_held = (datetime.now() - entry_dt).days
                days_str = f', {days_held}d'
            except ValueError:
                pass
        parts.append(f'{long_i}/{short_i} ({side}{days_str})')

    return f'{len(open_pos)} open: {", ".join(parts)}'


# ═══════════════════════════════════════════════════════════════════════════
# OOS STATS HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _oos_stats_from_wf(wf_df: pd.DataFrame, long_i: str) -> tuple[int, float, float]:
    """Extract per-pair OOS statistics from walk-forward results DataFrame.

    Args:
        wf_df: DataFrame returned by run_walk_forward or run_cross_asset_walkforward.
        long_i: Long instrument code used to filter to the correct direction.

    Returns:
        Tuple (oos_trades, oos_win_rate, oos_avg_gross). All floats are NaN
        if the DataFrame is empty or missing the expected columns.
    """
    if wf_df.empty or 'long' not in wf_df.columns:
        return 0, float('nan'), float('nan')
    pair_rows = wf_df[wf_df['long'] == long_i]
    oos_trades    = int(pair_rows['OOS_Trades'].sum()) if 'OOS_Trades' in pair_rows.columns else 0
    oos_win_rate  = float(pair_rows['OOS_WinRate'].mean()) if 'OOS_WinRate' in pair_rows.columns else float('nan')
    oos_avg_gross = float(pair_rows['OOS_Gross'].mean()) if 'OOS_Gross' in pair_rows.columns else float('nan')
    return oos_trades, oos_win_rate, oos_avg_gross


def _build_candidate(
    row: dict,
    long_ac: str,
    short_ac: str,
    wf_summary: dict,
    wf_df: pd.DataFrame,
    current_sd: float,
    xing_sd: float,
    near_buf: float,
) -> dict:
    """Assemble the final candidate dict for a WF-validated pair.

    Args:
        row: Scan result dict from _run_pair_both_passes.
        long_ac: Long asset class key.
        short_ac: Short asset class key.
        wf_summary: Output of summarise_walk_forward.
        wf_df: Raw WF results DataFrame (for per-direction OOS stats).
        current_sd: Current dist_sd value.
        xing_sd: Entry threshold.
        near_buf: Near-signal buffer SD.

    Returns:
        Candidate dict matching the results JSON spec.
    """
    long_i  = row['long']
    short_i = row['short']
    status  = _classify_candidate(row, wf_summary, current_sd, xing_sd, near_buf)
    oos_trades, oos_win_rate, oos_avg_gross = _oos_stats_from_wf(wf_df, long_i)

    _rho = wf_summary.get('rho')
    _p   = wf_summary.get('p_value')

    return {
        'pair':          f"{row['long_display']} / {row['short_display']}",
        'long':          long_i,
        'short':         short_i,
        'long_display':  row['long_display'],
        'short_display': row['short_display'],
        'long_ac':       long_ac,
        'short_ac':      short_ac,
        'current_sd':    current_sd,
        'wf_rho':        float(_rho) if _rho is not None else float('nan'),
        'wf_p':          float(_p)   if _p   is not None else float('nan'),
        'oos_trades':    oos_trades,
        'oos_win_rate':  oos_win_rate,
        'oos_avg_gross': oos_avg_gross,
        'Trades_WT':     row['Trades_WT'],
        'AvgNet_WT':     row['AvgNet_WT'],
        'AvgHold_WT':    row['AvgHold_WT'],
        'Best_Dir':      row['Best_Dir'],
        'status':        status,
        'already_open':  _check_already_open(long_i, short_i),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SEARCH ORCHESTRATORS
# ═══════════════════════════════════════════════════════════════════════════

def run_intra_asset_search(search: dict, global_settings: dict) -> dict:
    """Orchestrate a full intra-asset pair scan with WF validation.

    Two-stage gate:
        1. Scan all pairs via _run_pair_both_passes, apply WT filters.
        2. Take top top_n_for_wf by AvgNet_WT, run WF on each.

    Args:
        search: Search config dict from scan_config.json.
        global_settings: Global settings dict from scan_config.json.

    Returns:
        Results dict with status, pairs_scanned, pairs_wf_validated, candidates.

    Raises:
        ValueError: If fewer than 2 instruments survive exclusion or price loading.
    """
    t0             = time.time()
    ac             = search['asset_class']
    algo_params    = search['algo_params']
    scan_params    = search['scan_params']
    wf_params      = search['wf_params']
    broker_profile = scan_params.get('broker_profile', 'ig_spreadbet')
    top_n_for_wf   = scan_params.get('top_n_for_wf', 10)
    near_buf       = global_settings.get('near_signal_buffer_sd', 0.3)

    # Load prices
    csv_file = ASSET_CLASSES[ac]['csv_file']
    prices_df, _ = load_asset_prices(str(_CACHE_DIR / csv_file))

    # Instruments: apply exclusions and filter to what's in the price data
    exclude     = _get_exclusions(search)
    instruments = _get_instruments(ac, exclude)
    instruments = [i for i in instruments if i in prices_df.columns]
    if len(instruments) < 2:
        raise ValueError(
            f"Fewer than 2 instruments available in price data for {ac!r}: {instruments}"
        )
    prices_df = prices_df[instruments]

    # Generate pairs
    if ac == 'equity':
        pairs: list[tuple[str, str]] = list(permutations(instruments, 2))
    else:
        pairs = list(combinations(instruments, 2))

    # Stage 1: scan all pairs
    rows: list[dict] = []
    for long_i, short_i in pairs:
        pair_df = prices_df[[long_i, short_i]].dropna()
        result  = _run_pair_both_passes(
            pair_df, long_i, short_i, ac, ac,
            algo_params, scan_params, broker_profile,
        )
        if result is not None:
            rows.append(result)

    logger.info(
        "%s: scanned %d pairs, %d passed filters",
        search['name'], len(pairs), len(rows),
    )

    # Stage 2: WF validation on shortlist
    rows.sort(key=lambda r: r['AvgNet_WT'], reverse=True)
    shortlist = rows[:top_n_for_wf]
    candidates: list[dict] = []

    for row in shortlist:
        long_i  = row['long']
        short_i = row['short']
        pair_prices = prices_df[[long_i, short_i]].dropna()

        try:
            wf_df      = run_walk_forward(
                prices=pair_prices,
                instruments=[long_i, short_i],
                is_years=wf_params['is_years'],
                oos_years=wf_params['oos_years'],
                step_years=wf_params['step_years'],
                scoring_mode=wf_params['scoring_mode'],
                vol_window=algo_params['vol_window'],
                xing_sd=algo_params['xing_sd'],
                exit_sd=algo_params['exit_sd'],
                spread_cost_pct=row['pair_cost'],
            )
            wf_summary = summarise_walk_forward(wf_df)
        except Exception as exc:  # broad: WF can fail for short history or numerical edge cases
            logger.warning("WF failed for %s/%s: %s", long_i, short_i, exc)
            wf_df      = pd.DataFrame()
            wf_summary = {}

        try:
            current_sd = _compute_current_sd(prices_df, long_i, short_i, algo_params['vol_window'])
        except Exception as exc:  # broad: numerical failures in vol-scaling
            logger.warning("current_sd failed for %s/%s: %s", long_i, short_i, exc)
            current_sd = float('nan')

        candidates.append(_build_candidate(
            row, ac, ac, wf_summary, wf_df,
            current_sd, algo_params['xing_sd'], near_buf,
        ))

    # Open positions overlay — append any open position not already in candidates.
    # Ensures that positions currently held always appear in scan output regardless
    # of whether they passed the min_wt_trades / min_avg_net_wt filters.
    candidate_pairs = {(c['long'], c['short']) for c in candidates}
    for pos in _load_open_positions():
        long_i  = pos.get('basket', {}).get('long_legs',  [''])[0]
        short_i = pos.get('basket', {}).get('short_legs', [''])[0]
        if not long_i or not short_i:
            continue
        if _infer_asset_class(long_i) != ac:
            continue
        if (long_i, short_i) in candidate_pairs:
            continue  # already present — already_open flag set by _check_already_open()
        try:
            row = _build_open_position_row(
                long_i, ac, short_i, ac,
                prices_df, algo_params['vol_window'],
                scan_params.get('broker_profile', 'ig_spreadbet'),
            )
            candidates.append(row)
            logger.info("Open position overlay: %s/%s added to %s output", long_i, short_i, search['id'])
        except Exception as exc:  # overlay failure must not abort result write
            logger.warning("Open position overlay failed for %s/%s: %s", long_i, short_i, exc)

    return {
        'search_id':          search['id'],
        'search_name':        search['name'],
        'status':             'completed',
        'error':              None,
        'pairs_scanned':      len(rows),
        'pairs_wf_validated': len(shortlist),
        'run_seconds':        round(time.time() - t0, 1),
        'candidates':         candidates,
    }


def run_cross_asset_search(search: dict, global_settings: dict) -> dict:
    """Orchestrate a full cross-asset pair scan with WF validation.

    Same two-stage gate as run_intra_asset_search but loads two separate
    price files and uses run_cross_asset_walkforward for WF.

    Args:
        search: Search config dict. Must contain 'long_asset_class' and
            'short_asset_class' keys.
        global_settings: Global settings dict from scan_config.json.

    Returns:
        Results dict with status, pairs_scanned, pairs_wf_validated, candidates.

    Raises:
        ValueError: If insufficient instruments survive exclusion.
    """
    t0             = time.time()
    long_ac        = search['long_asset_class']
    short_ac       = search['short_asset_class']
    algo_params    = search['algo_params']
    scan_params    = search['scan_params']
    wf_params      = search['wf_params']
    broker_profile = scan_params.get('broker_profile', 'ig_spreadbet')
    top_n_for_wf   = scan_params.get('top_n_for_wf', 10)
    near_buf       = global_settings.get('near_signal_buffer_sd', 0.3)

    # Load prices for both asset classes
    long_csv  = ASSET_CLASSES[long_ac]['csv_file']
    short_csv = ASSET_CLASSES[short_ac]['csv_file']
    long_prices_df,  _ = load_asset_prices(str(_CACHE_DIR / long_csv))
    short_prices_df, _ = load_asset_prices(str(_CACHE_DIR / short_csv))

    # Instruments with exclusions, filtered to what's in price data
    excl_set     = frozenset(search.get('exclude', []))
    long_exclude  = _get_exclusions({'asset_class': long_ac,  'exclude': list(excl_set)})
    short_exclude = _get_exclusions({'asset_class': short_ac, 'exclude': list(excl_set)})
    long_insts  = [i for i in _get_instruments(long_ac,  long_exclude)  if i in long_prices_df.columns]
    short_insts = [i for i in _get_instruments(short_ac, short_exclude) if i in short_prices_df.columns]

    if len(long_insts) < 1 or len(short_insts) < 1:
        raise ValueError(
            f"Insufficient instruments for cross-asset search: "
            f"long={long_insts}, short={short_insts}"
        )

    # Stage 1: scan all ordered (long, short) pairs
    pairs: list[tuple[str, str]] = [(a, b) for a in long_insts for b in short_insts]
    rows: list[dict] = []

    for long_i, short_i in pairs:
        # Merge on inner date intersection
        pair_df = (
            long_prices_df[[long_i]]
            .join(short_prices_df[[short_i]], how='inner')
            .dropna()
        )
        if pair_df.empty:
            continue
        result = _run_pair_both_passes(
            pair_df, long_i, short_i, long_ac, short_ac,
            algo_params, scan_params, broker_profile,
        )
        if result is not None:
            rows.append(result)

    logger.info(
        "%s: scanned %d pairs, %d passed filters",
        search['name'], len(pairs), len(rows),
    )

    # Stage 2: WF validation on shortlist
    rows.sort(key=lambda r: r['AvgNet_WT'], reverse=True)
    shortlist = rows[:top_n_for_wf]
    candidates: list[dict] = []

    for row in shortlist:
        long_i  = row['long']
        short_i = row['short']

        try:
            wf_df      = run_cross_asset_walkforward(
                prices_long=long_prices_df[[long_i]],
                prices_short=short_prices_df[[short_i]],
                instruments_long=[long_i],
                instruments_short=[short_i],
                is_years=wf_params['is_years'],
                oos_years=wf_params['oos_years'],
                step_years=wf_params['step_years'],
                scoring_mode=wf_params['scoring_mode'],
                vol_window=algo_params['vol_window'],
                xing_sd=algo_params['xing_sd'],
                exit_sd=algo_params['exit_sd'],
                spread_cost_pct=row['pair_cost'],
            )
            wf_summary = summarise_walk_forward(wf_df)
        except Exception as exc:  # broad: WF can fail for short history or numerical edge cases
            logger.warning("WF failed for %s/%s: %s", long_i, short_i, exc)
            wf_df      = pd.DataFrame()
            wf_summary = {}

        try:
            pair_prices = (
                long_prices_df[[long_i]]
                .join(short_prices_df[[short_i]], how='inner')
                .dropna()
            )
            current_sd = _compute_current_sd(pair_prices, long_i, short_i, algo_params['vol_window'])
        except Exception as exc:  # broad: numerical failures in vol-scaling
            logger.warning("current_sd failed for %s/%s: %s", long_i, short_i, exc)
            current_sd = float('nan')

        candidates.append(_build_candidate(
            row, long_ac, short_ac, wf_summary, wf_df,
            current_sd, algo_params['xing_sd'], near_buf,
        ))

    # Open positions overlay — cross-asset variant.
    # A cross-asset position belongs to this search if one leg is in long_insts
    # and the other is in short_insts.
    candidate_pairs = {(c['long'], c['short']) for c in candidates}
    for pos in _load_open_positions():
        long_i  = pos.get('basket', {}).get('long_legs',  [''])[0]
        short_i = pos.get('basket', {}).get('short_legs', [''])[0]
        if not long_i or not short_i:
            continue
        if long_i not in long_insts or short_i not in short_insts:
            continue
        if (long_i, short_i) in candidate_pairs:
            continue
        try:
            # prices_df does not exist in run_cross_asset_search() — build a
            # per-pair joined DataFrame from the two separate price DataFrames.
            pair_prices_df = (
                long_prices_df[[long_i]]
                .join(short_prices_df[[short_i]], how='inner')
                .dropna()
            )
            row = _build_open_position_row(
                long_i, long_ac, short_i, short_ac,
                pair_prices_df, algo_params['vol_window'],
                scan_params.get('broker_profile', 'ig_spreadbet'),
            )
            candidates.append(row)
            logger.info(
                "Open position overlay (cross-asset): %s/%s added to %s output",
                long_i, short_i, search['id'],
            )
        except Exception as exc:  # overlay failure must not abort result write
            logger.warning("Open position overlay failed for %s/%s: %s", long_i, short_i, exc)

    return {
        'search_id':          search['id'],
        'search_name':        search['name'],
        'status':             'completed',
        'error':              None,
        'pairs_scanned':      len(rows),
        'pairs_wf_validated': len(shortlist),
        'run_seconds':        round(time.time() - t0, 1),
        'candidates':         candidates,
    }


# ═══════════════════════════════════════════════════════════════════════════
# AI REVIEW
# ═══════════════════════════════════════════════════════════════════════════

def _get_api_key() -> str:
    """Retrieve the Anthropic API key from Windows Credential Manager.

    Returns:
        The API key string, or empty string if not found.

    Notes:
        Store once via:
            python -c "import keyring; keyring.set_password('trading_app', 'anthropic_api_key', 'sk-ant-...')"
        or via Windows Credential Manager UI:
            Internet or network address: trading_app
            User name:                   anthropic_api_key
            Password:                    sk-ant-...
    """
    key = keyring.get_password('trading_app', 'anthropic_api_key')
    if not key:
        logger.warning(
            "Anthropic API key not found in credential store "
            "(service='trading_app', username='anthropic_api_key'). "
            "AI review will be skipped."
        )
        return ''
    return key


_AI_SYSTEM_PROMPT = (
    "You are a systematic spread trading analyst reviewing overnight scan results.\n"
    "The strategy is statistical mean-reversion on spread pairs "
    "(long one index, short another).\n\n"
    "Review the scan results and produce a concise morning briefing.\n\n"
    "Format exactly as:\n"
    "HEADLINE: [one sentence — how many strong candidates, any immediate action needed]\n\n"
    "STRONG CANDIDATES:\n"
    "[bullet per STRONG candidate: pair name, current SD, WF rho/p, WT trades, "
    "AvgNet_WT, recommendation]\n\n"
    "WATCH LIST:\n"
    "[bullet per WATCH candidate: pair, why watching]\n\n"
    "FLAGS:\n"
    "[any concerns: borderline stats, low trade counts, already-open pairs, unusual patterns]\n\n"
    "OPEN POSITIONS:\n"
    "[brief note on any already-open positions appearing in scan results]\n\n"
    "Be direct. No filler. The trader reads this at 7am."
)


def _call_ai_review(
    all_results: list[dict],
    open_positions_summary: str,
    global_settings: dict,
) -> str:
    """Call the Claude API to generate a morning briefing on scan results.

    Args:
        all_results: List of search result dicts from the scan.
        open_positions_summary: Formatted string of current open positions.
        global_settings: Global settings dict (provides model, max_tokens).

    Returns:
        AI-generated review string, or empty string if the API key is absent
        or the request fails.
    """
    api_key = _get_api_key()
    if not api_key:
        return ''

    model      = global_settings.get('ai_review_model', 'claude-sonnet-4-20250514')
    max_tokens = global_settings.get('ai_max_tokens', 1000)

    candidate_lines: list[str] = []
    for result in all_results:
        if result.get('status') != 'completed':
            continue
        search_name = result.get('search_name', '')
        for c in result.get('candidates', []):
            sd_str = f"{c['current_sd']:.2f}" if not np.isnan(c['current_sd']) else 'n/a'
            rho_str = f"{c['wf_rho']:.3f}" if not np.isnan(c['wf_rho']) else 'n/a'
            p_str   = f"{c['wf_p']:.3f}"   if not np.isnan(c['wf_p'])   else 'n/a'
            open_tag = " [OPEN]" if c.get('already_open') else ""
            candidate_lines.append(
                f"  [{c['status']}] {c['pair']} ({search_name}): "
                f"SD={sd_str}, rho={rho_str}, p={p_str}, "
                f"WT_trades={c['Trades_WT']}, AvgNet_WT={c['AvgNet_WT']:.3f}%, "
                f"AvgHold={c['AvgHold_WT']:.1f}d, best_dir={c['Best_Dir']}"
                + open_tag
            )

    user_message = (
        f"Daily scan results — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"CANDIDATES:\n"
        + ("\n".join(candidate_lines) if candidate_lines else "  None")
        + f"\n\nOPEN POSITIONS:\n{open_positions_summary}"
    )

    payload = json.dumps({
        "model":      model,
        "max_tokens": max_tokens,
        "system":     _AI_SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": user_message}],
    }).encode('utf-8')

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key":         api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        return body['content'][0]['text']
    except urllib.error.HTTPError as exc:
        logger.warning(
            "AI review HTTP error %s: %s",
            exc.code,
            exc.read().decode('utf-8', errors='replace'),
        )
        return ''
    except urllib.error.URLError as exc:
        logger.warning("AI review network error: %s", exc.reason)
        return ''


# ═══════════════════════════════════════════════════════════════════════════
# OBSIDIAN OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def _write_obsidian_note(output: dict, config: dict) -> None:
    """Write daily scan results as an Obsidian note, overwriting same-day file.

    Args:
        output: Full scan output dict (run_timestamp, searches, ai_review, etc.).
        config: Global settings dict (provides obsidian_notes_path).
    """
    notes_path_str = config.get('obsidian_notes_path', '')
    if not notes_path_str:
        logger.warning("obsidian_notes_path not configured; skipping Obsidian write")
        return
    notes_path = Path(notes_path_str)
    if not notes_path.exists():
        logger.warning("Obsidian notes path not found: %s", notes_path)
        return

    now          = datetime.now()
    date_str     = now.strftime('%Y-%m-%d')
    datetime_str = now.strftime('%Y-%m-%d %H:%M')
    out_path     = notes_path / f"{date_str}-Daily-Scan-Results.md"

    all_candidates = [
        c
        for r in output.get('searches', [])
        for c in r.get('candidates', [])
    ]
    strong = [c for c in all_candidates if c['status'] == 'STRONG']
    watch  = [c for c in all_candidates if c['status'] == 'WATCH']

    lines: list[str] = [
        "---",
        f"created: {date_str}",
        f"updated: {datetime_str}",
        "tags:",
        "  - daily-scan",
        "  - trading",
        "type: note",
        "---",
        "",
        f"# Daily Scan Results — {date_str}",
        "",
    ]

    ai_review = output.get('ai_review', '')
    if ai_review:
        lines += ["## AI Review", "", ai_review, ""]

    lines += [
        "## Search Summary",
        "",
        "| Search | Pairs Scanned | WF Validated | STRONG | WATCH |",
        "|--------|--------------|--------------|--------|-------|",
    ]
    for r in output.get('searches', []):
        if r.get('status') == 'skipped':
            continue
        n_s = sum(1 for c in r.get('candidates', []) if c['status'] == 'STRONG')
        n_w = sum(1 for c in r.get('candidates', []) if c['status'] == 'WATCH')
        lines.append(
            f"| {r['search_name']} | {r.get('pairs_scanned', 0)} | "
            f"{r.get('pairs_wf_validated', 0)} | {n_s} | {n_w} |"
        )
    lines.append("")

    if strong:
        lines += [
            "## Strong Candidates",
            "",
            "| Pair | SD | WF rho | WF p | WT Trades | AvgNet WT | Avg Hold | Best Dir |",
            "|------|-----|--------|------|-----------|-----------|----------|----------|",
        ]
        for c in strong:
            sd_s = f"{c['current_sd']:.2f}" if not np.isnan(c['current_sd']) else "n/a"
            rho_s = f"{c['wf_rho']:.3f}" if not np.isnan(c['wf_rho']) else "n/a"
            p_s   = f"{c['wf_p']:.3f}"   if not np.isnan(c['wf_p'])   else "n/a"
            lines.append(
                f"| {c['pair']} | {sd_s} | {rho_s} | {p_s} | "
                f"{c['Trades_WT']} | {c['AvgNet_WT']:.3f}% | "
                f"{c['AvgHold_WT']:.1f}d | {c['Best_Dir']} |"
            )
        lines.append("")

    if watch:
        lines += [
            "## Watch List",
            "",
            "| Pair | SD | WF rho | WF p | WT Trades | AvgNet WT |",
            "|------|-----|--------|------|-----------|-----------|",
        ]
        for c in watch:
            sd_s  = f"{c['current_sd']:.2f}" if not np.isnan(c['current_sd']) else "n/a"
            rho_s = f"{c['wf_rho']:.3f}" if not np.isnan(c['wf_rho']) else "n/a"
            p_s   = f"{c['wf_p']:.3f}"   if not np.isnan(c['wf_p'])   else "n/a"
            lines.append(
                f"| {c['pair']} | {sd_s} | {rho_s} | {p_s} | "
                f"{c['Trades_WT']} | {c['AvgNet_WT']:.3f}% |"
            )
        lines.append("")

    lines += [
        "## Run Metadata",
        "",
        f"- Timestamp: {output.get('run_timestamp', '')}",
        f"- Duration: {output.get('run_duration_seconds', 0)}s",
        f"- Data refreshed: {output.get('data_refreshed', False)}",
        f"- Searches run: {len([r for r in output.get('searches', []) if r.get('status') != 'skipped'])}",
    ]

    out_path.write_text("\n".join(lines), encoding='utf-8')
    logger.info("Obsidian note written: %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def _setup_logging() -> None:
    """Configure root logger for standalone script use."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def main() -> None:
    """Entry point for overnight batch execution."""
    _setup_logging()
    t_start   = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')

    config  = load_scan_config()
    searches = config['searches']
    gs       = config['global_settings']

    # Step 1: Refresh price cache
    logger.info("Refreshing price cache...")
    registry = DataRegistry(_CACHE_DIR)
    registry.refresh(instruments=None)

    # Step 2: Run enabled searches
    all_results: list[dict] = []
    for search in searches:
        if not search.get('enabled', False):
            blocked = search.get('blocked_reason', 'disabled')
            logger.info("Skipping %s: %s", search['id'], blocked[:80])
            all_results.append({
                'search_id':          search['id'],
                'search_name':        search['name'],
                'status':             'skipped',
                'error':              None,
                'pairs_scanned':      0,
                'pairs_wf_validated': 0,
                'run_seconds':        0,
                'candidates':         [],
            })
            continue

        logger.info("Running search: %s", search['name'])
        try:
            ac = search.get('asset_class', '')
            if ac == 'cross_asset':
                result = run_cross_asset_search(search, gs)
            else:
                result = run_intra_asset_search(search, gs)
        except Exception as exc:  # broad: individual search failures must not abort the batch
            logger.error("Search %s failed: %s", search['id'], exc)
            result = {
                'search_id':          search['id'],
                'search_name':        search['name'],
                'status':             'failed',
                'error':              str(exc),
                'pairs_scanned':      0,
                'pairs_wf_validated': 0,
                'run_seconds':        0,
                'candidates':         [],
            }
        all_results.append(result)

    # Step 3: AI review
    ai_review = ''
    if gs.get('ai_review_enabled', False):
        try:
            open_positions = _load_open_positions_summary()
            ai_review      = _call_ai_review(all_results, open_positions, gs)
        except Exception as exc:  # broad: AI review failures must not abort the batch
            logger.warning("AI review failed: %s", exc)

    # Step 4: Write results
    run_duration = time.time() - t_start
    output = {
        'run_timestamp':         datetime.now().isoformat(),
        'run_duration_seconds':  round(run_duration, 1),
        'data_refreshed':        True,
        'searches':              all_results,
        'ai_review':             ai_review,
        'ai_review_timestamp':   datetime.now().isoformat() if ai_review else None,
    }

    _RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding='utf-8')

    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = _HISTORY_DIR / f'{timestamp}.json'
    history_path.write_text(json.dumps(output, indent=2), encoding='utf-8')

    if gs.get('obsidian_output_enabled', False):
        try:
            _write_obsidian_note(output, gs)
        except Exception as exc:  # broad: Obsidian write failures must not abort the batch
            logger.warning("Obsidian write failed: %s", exc)

    # Step 5: Print summary to stdout (only in main — not inside functions)
    n_strong = sum(
        sum(1 for c in r['candidates'] if c['status'] == 'STRONG')
        for r in all_results
    )
    n_watch = sum(
        sum(1 for c in r['candidates'] if c['status'] == 'WATCH')
        for r in all_results
    )
    print(f"\nDaily scan complete — {round(run_duration)}s")
    print(f"STRONG: {n_strong}  WATCH: {n_watch}")
    if ai_review:
        print(f"\n{ai_review}")


if __name__ == '__main__':
    main()
