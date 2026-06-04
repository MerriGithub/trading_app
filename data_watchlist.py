"""
data_watchlist.py — Watchlist persistence helpers.
Reads/writes data/watchlist.json and data/walkforward_cache.json.
No Streamlit imports — pure Python only.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

_WATCHLIST_PATH = os.path.join(os.path.dirname(__file__), "data", "watchlist.json")
_WF_CACHE_PATH  = os.path.join(os.path.dirname(__file__), "data", "walkforward_cache.json")
_MONITOR_PATH   = os.path.join(os.path.dirname(__file__), "data", "monitor_candidates.json")


def _make_id(
    long: str, short: str,
    entry_sd: float, exit_sd: float,
    vol_window: int, trend_window: int, trend_mode: str,
) -> str:
    mode_slug = (
        trend_mode.lower()
        .replace(" ", "_").replace("-", "_")
        .replace("(", "").replace(")", "")
    )
    return f"{long}_{short}_{entry_sd}_{exit_sd}_{vol_window}_{trend_window}_{mode_slug}"


def _clean_for_json(obj: object) -> object:
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    return obj


def load_watchlist() -> list:
    if not os.path.exists(_WATCHLIST_PATH):
        return []
    try:
        with open(_WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # File corrupt or unreadable; return empty list so the UI starts clean.
        return []


def save_watchlist(entries: list) -> None:
    os.makedirs(os.path.dirname(_WATCHLIST_PATH), exist_ok=True)
    with open(_WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def add_to_watchlist(entry: dict) -> str:
    """
    Add or update a watchlist entry. Returns the entry id.
    entry must contain: long, short, entry_sd, exit_sd, vol_window,
    trend_window, trend_mode, asset_class_long, asset_class_short, source.
    Optionally: scan_metrics, notes.
    If the same pair+params already exists, overwrites scan_metrics/added/source.
    """
    eid = _make_id(
        entry["long"], entry["short"],
        entry["entry_sd"], entry["exit_sd"],
        entry["vol_window"], entry["trend_window"], entry["trend_mode"],
    )
    entry = dict(entry)
    entry["id"] = eid
    if "added" not in entry:
        entry["added"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if "notes" not in entry:
        entry["notes"] = ""
    if "scan_metrics" not in entry:
        entry["scan_metrics"] = {}
    if "scoring_mode" not in entry:
        entry["scoring_mode"] = None

    entries = load_watchlist()
    idx = next((i for i, e in enumerate(entries) if e.get("id") == eid), None)
    if idx is not None:
        entries[idx] = _clean_for_json(entry)
    else:
        entries.append(_clean_for_json(entry))
    save_watchlist(entries)
    return eid


def remove_from_watchlist(entry_id: str) -> None:
    entries = [e for e in load_watchlist() if e.get("id") != entry_id]
    save_watchlist(entries)


def clear_watchlist() -> None:
    save_watchlist([])


def update_notes(entry_id: str, notes: str) -> None:
    entries = load_watchlist()
    for e in entries:
        if e.get("id") == entry_id:
            e["notes"] = notes
    save_watchlist(entries)


def load_monitor_candidates() -> list[dict]:
    if not os.path.exists(_MONITOR_PATH):
        return []
    try:
        with open(_MONITOR_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # File corrupt or unreadable; return empty list so monitor starts clean.
        return []


def save_monitor_candidates(candidates: list[dict]) -> None:
    os.makedirs(os.path.dirname(_MONITOR_PATH), exist_ok=True)
    with open(_MONITOR_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)


def add_monitor_candidate(candidate: dict) -> str:
    candidate = dict(candidate)
    if "added_to_monitor" not in candidate:
        candidate["added_to_monitor"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    candidates = load_monitor_candidates()
    idx = next((i for i, c in enumerate(candidates) if c.get("id") == candidate["id"]), None)
    if idx is not None:
        candidates[idx] = _clean_for_json(candidate)
    else:
        candidates.append(_clean_for_json(candidate))
    save_monitor_candidates(candidates)
    return candidate["id"]


def remove_monitor_candidate(candidate_id: str) -> None:
    candidates = [c for c in load_monitor_candidates() if c.get("id") != candidate_id]
    save_monitor_candidates(candidates)


def load_wf_cache() -> dict:
    if not os.path.exists(_WF_CACHE_PATH):
        return {}
    try:
        with open(_WF_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # File corrupt or unreadable; return empty dict so WF tab starts clean.
        return {}


def save_wf_result(entry_id: str, result: dict) -> None:
    cache = load_wf_cache()
    result = dict(result)
    result["run_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cache[entry_id] = _clean_for_json(result)
    os.makedirs(os.path.dirname(_WF_CACHE_PATH), exist_ok=True)
    with open(_WF_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
