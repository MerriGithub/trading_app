"""
crossing_signal_mcp/server.py  —  v2.0 (wired)
"""

import asyncio
import functools
import sys
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from asset_configs import ASSET_CLASSES, COMMODITY_EXCLUDE, FI_EXCLUDE, _DEFAULT_SCORING_MODE
from engine.backtest import (
    run_backtest, prepare_returns, run_exhaustive_search,
    regime_split, aggregate_trades,
)
from engine.numba_core import (
    backtest_spread,
    COL_ENTRY_IDX, COL_EXIT_IDX, COL_SIDE, COL_GROSS_RETURN, COL_HOLDING_DAYS,
)
from engine.unified_loader import load_aligned_prices_unified


# ── Instrument registry ───────────────────────────────────────────────────────

INSTRUMENT_REGISTRY = {
    "commodities":  [t for t in ASSET_CLASSES["commodities"]["instruments"]
                     if t not in COMMODITY_EXCLUDE],
    "fx":           list(ASSET_CLASSES["fx"]["instruments"]),
    "fixed_income": [t for t in ASSET_CLASSES["fixed_income"]["instruments"]
                     if t not in FI_EXCLUDE],
}

SHORTHAND = {
    "ALL_COMMODITIES": INSTRUMENT_REGISTRY["commodities"],
    "ALL_FX":          INSTRUMENT_REGISTRY["fx"],
    "ALL_FI":          INSTRUMENT_REGISTRY["fixed_income"],
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")


# ── App + job store ───────────────────────────────────────────────────────────

app = FastAPI(title="Crossing Signal MCP Server", version="2.0.0")

JOB_STORE: dict = {}
_JOB_STORE_LOCK = asyncio.Lock()
JOB_TTL_SECONDS = 3600


# ── Models ────────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    instruments: list[str]
    sd_threshold: float = 2.0
    exit_sd: float = 0.0
    vol_window: int = 262
    max_hold_days: int = 300
    financing_rate_long: float = 4.88
    financing_rate_short: float = 0.88
    bid_ask_override: Optional[float] = None
    date_from: str = "1999-01-04"
    date_to: str = "2026-04-24"
    pair_mode: str = "exhaustive"
    explicit_pairs: Optional[list[dict]] = None
    top_n_pairs: int = 20


class SensitivityRequest(BaseModel):
    base_instruments: list[str]
    parameter: str
    values: list[float]
    fixed_params: Optional[dict] = None


class CorrelationRequest(BaseModel):
    instruments: list[str]
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class PairDeepDiveRequest(BaseModel):
    long_instrument: str
    short_instrument: str
    sd_threshold: float = 2.0
    exit_sd: float = 0.0
    financing_rate_long: float = 4.88
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    include_trade_log: bool = False


class CompareAssetClassesRequest(BaseModel):
    sd_threshold: float = 2.0
    exit_sd: float = 0.0
    financing_rate_long: float = 4.88
    include_cross_asset: bool = True
    date_from: Optional[str] = None
    date_to: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_instruments(instrument_list: list[str]) -> list[str]:
    resolved = []
    for ticker in instrument_list:
        resolved.extend(SHORTHAND.get(ticker, [ticker]))
    return list(dict.fromkeys(resolved))


def get_asset_class(ticker: str) -> str:
    for cls, cfg in ASSET_CLASSES.items():
        if ticker in cfg.get("instruments", {}):
            return cls
    return "unknown"


def is_cross_asset_pair(long_ticker: str, short_ticker: str) -> bool:
    return get_asset_class(long_ticker) != get_asset_class(short_ticker)


def resolve_scoring_mode(instruments: list[str]) -> str:
    classes = {get_asset_class(t) for t in instruments}
    cls = "cross_asset" if len(classes) > 1 else classes.pop()
    return _DEFAULT_SCORING_MODE.get(cls, "composite")


def make_job_id() -> str:
    return f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}"


def net_financing_daily(rate_long_pct: float, rate_short_pct: float) -> float:
    return (rate_long_pct - rate_short_pct) / 100 / 365


def add_significance_stats(trade_returns: np.ndarray) -> dict:
    if len(trade_returns) < 3:
        return {"t_stat_net": None, "p_value_net": None, "bootstrap_ci_net": [None, None]}
    t_stat, p_value = scipy_stats.ttest_1samp(trade_returns, 0)
    rng = np.random.default_rng(42)
    bootstrap_means = [
        float(np.mean(rng.choice(trade_returns, len(trade_returns), replace=True)))
        for _ in range(10_000)
    ]
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "t_stat_net":      round(float(t_stat), 3),
        "p_value_net":     round(float(p_value), 6),
        "bootstrap_ci_net": [round(float(ci_low), 6), round(float(ci_high), 6)],
    }


async def evict_old_jobs() -> None:
    async with _JOB_STORE_LOCK:
        now = datetime.now(timezone.utc)
        to_delete = [
            jid for jid, job in JOB_STORE.items()
            if job["status"] in ("complete", "failed")
            and (now - datetime.fromisoformat(job["created_at"]).replace(tzinfo=timezone.utc)
                 ).total_seconds() > JOB_TTL_SECONDS
        ]
        for jid in to_delete:
            del JOB_STORE[jid]


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to Python native types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


# ── Numba JIT warmup ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def warmup_numba():
    """
    Trigger Numba JIT compilation before the first real job arrives.
    First call after server restart otherwise takes 30-60s to compile.
    """
    loop = asyncio.get_running_loop()
    dummy_returns = np.random.randn(100) * 0.01
    dummy_days    = np.arange(18000, 18100, dtype=np.int64)
    try:
        await loop.run_in_executor(None, functools.partial(
            run_backtest,
            dummy_returns, dummy_days,
            vol_window=20, xing_sd=2.0, exit_sd=0.0,
            spread_cost_pct=0.001, financing_daily_pct=0.00011,
        ))
    except Exception:
        pass  # warmup failure is non-fatal


# ── Background worker ─────────────────────────────────────────────────────────

async def run_backtest_worker(job_id: str, params: dict) -> None:
    async with _JOB_STORE_LOCK:
        JOB_STORE[job_id]["status"] = "running"

    try:
        instruments = params["instruments"]
        fin_daily   = net_financing_daily(
            params["financing_rate_long"], params["financing_rate_short"]
        )
        bid_ask    = params.get("bid_ask_override") or 0.001
        vol_window = params["vol_window"]
        max_hold   = params["max_hold_days"]

        # ── Step 1: Load aligned prices ───────────────────────────────────────
        prices = load_aligned_prices_unified(
            tickers=instruments,
            cache_dir=CACHE_DIR,
            date_from=params["date_from"],
            date_to=params["date_to"],
        )
        prices = prices[instruments]  # enforce column order

        # ── Step 2: Vol-scale ─────────────────────────────────────────────────
        scaled, day_ints, index = prepare_returns(
            prices, instruments, vol_window=vol_window
        )

        loop = asyncio.get_running_loop()

        # ── Step 3 + 4a: Exhaustive mode ──────────────────────────────────────
        if params["pair_mode"] == "exhaustive":
            raw_df = await loop.run_in_executor(None, functools.partial(
                run_exhaustive_search,
                scaled, day_ints, instruments,
                min_long_legs=1, max_long_legs=1,
                min_short_legs=1, max_short_legs=1,
                xing_sd=params["sd_threshold"],
                exit_sd=params["exit_sd"],
                spread_cost_pct=bid_ask,
                financing_daily_pct=fin_daily,
                max_hold_days=max_hold,
                scoring_mode=resolve_scoring_mode(instruments),
            ))

            safe_df   = raw_df.drop(columns=["_long_flags", "_short_flags"], errors="ignore")
            top_n     = params.get("top_n_pairs", 20)
            aggregate = {
                "total_trades":       int(raw_df["Trades"].sum()),
                "gross_win_rate":     float(raw_df["WinRate"].mean()),
                "net_win_rate":       None,
                "avg_gross_return":   float(raw_df["Expectancy"].mean()),
                "avg_net_return":     float(raw_df["NetExpectancy"].mean()),
                "avg_hold_days":      float(raw_df["AvgHolding"].mean()),
                "avg_financing_cost": float(raw_df["FinCost"].mean()),
                "t_stat_net":         None,
                "p_value_net":        None,
                "bootstrap_ci_net":   [None, None],
            }
            top_pairs    = _to_native(safe_df.nlargest(top_n, "NetExpectancy").to_dict("records"))
            bottom_pairs = _to_native(safe_df.nsmallest(top_n, "NetExpectancy").to_dict("records"))
            regime_data  = []

        # ── Step 3 + 4b: Specified-pair mode ──────────────────────────────────
        else:
            pair_results = []
            for pair in params["explicit_pairs"]:
                long_idx  = instruments.index(pair["long"])
                short_idx = instruments.index(pair["short"])
                spread_returns = scaled[:, long_idx] - scaled[:, short_idx]

                result = await loop.run_in_executor(None, functools.partial(
                    run_backtest,
                    spread_returns, day_ints,
                    vol_window=vol_window,
                    xing_sd=params["sd_threshold"],
                    exit_sd=params["exit_sd"],
                    spread_cost_pct=bid_ask,
                    financing_daily_pct=fin_daily,
                    max_hold_days=max_hold,
                ))
                pair_results.append((pair, result))

            # Aggregate across all pairs
            all_summaries = [r["summary"] for _, r in pair_results]
            aggregate = {
                "total_trades":
                    sum(s["n_trades"] for s in all_summaries),
                "gross_win_rate":
                    float(np.mean([s["gross_wr"] for s in all_summaries])),
                "net_win_rate":
                    float(np.mean([s["net_wr"] for s in all_summaries])),
                "avg_gross_return":
                    float(np.mean([s["avg_gross"] for s in all_summaries])),
                "avg_net_return":
                    float(np.mean([s["avg_net"] for s in all_summaries])),
                "avg_hold_days":
                    float(np.mean([s["avg_holding"] for s in all_summaries])),
                "avg_financing_cost":
                    float(np.mean([s["avg_fin_cost"] for s in all_summaries])),
            }

            # Significance stats on first pair using compounding net (spec §M)
            _, first = pair_results[0]
            t_arr  = first["trades_raw"][:first["n_trades"]]
            if len(t_arr) > 0:
                gross    = t_arr[:, COL_GROSS_RETURN]
                fin_cost = (1 + fin_daily) ** (2 * t_arr[:, COL_HOLDING_DAYS]) - 1
                net      = gross - bid_ask - fin_cost
                sig      = add_significance_stats(net)
            else:
                sig = {"t_stat_net": None, "p_value_net": None, "bootstrap_ci_net": [None, None]}
            aggregate.update(sig)

            # Per-pair results for response
            top_pairs = _to_native([
                {
                    "Long": p["long"], "Short": p["short"],
                    "Trades":        r["summary"]["n_trades"],
                    "WinRate":       r["summary"]["gross_wr"],
                    "NetExpectancy": r["summary"]["avg_net"],
                    "AvgHolding":    r["summary"]["avg_holding"],
                }
                for p, r in pair_results
            ])
            bottom_pairs = []

            # Regime split (first pair only)
            regime_data = _to_native(regime_split(
                trades=first["trades_raw"],
                n_trades=first["n_trades"],
                index=index,
                spread_cost_pct=bid_ask,
                financing_daily_pct=fin_daily,
            ))

            # Optional trade log (deep dive)
            if params.get("include_trade_log") and len(t_arr) > 0:
                trade_log = _to_native([{
                    "entry_idx":    int(t_arr[k, COL_ENTRY_IDX]),
                    "exit_idx":     int(t_arr[k, COL_EXIT_IDX]),
                    "side":         int(t_arr[k, COL_SIDE]),
                    "gross_return": float(t_arr[k, COL_GROSS_RETURN]),
                    "holding_days": float(t_arr[k, COL_HOLDING_DAYS]),
                } for k in range(len(t_arr))])
            else:
                trade_log = None

        results = {
            "aggregate": aggregate,
            "pairs":     {"top": top_pairs, "bottom": bottom_pairs},
            "regime":    {"periods": regime_data},
        }
        if params["pair_mode"] != "exhaustive" and trade_log is not None:
            results["trade_log"] = trade_log

        async with _JOB_STORE_LOCK:
            JOB_STORE[job_id].update({
                "status":       "complete",
                "results":      results,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })

    except Exception as e:
        async with _JOB_STORE_LOCK:
            JOB_STORE[job_id].update({"status": "failed", "error": str(e)})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":      "ok",
        "active_jobs": sum(1 for j in JOB_STORE.values() if j["status"] == "running"),
        "total_jobs":  len(JOB_STORE),
    }


@app.get("/tools/list_instruments")
async def list_instruments(asset_class: str = "all"):
    classes = list(INSTRUMENT_REGISTRY.keys()) if asset_class == "all" else [asset_class]
    result = [
        {"ticker": t, "asset_class": c}
        for c in classes
        for t in INSTRUMENT_REGISTRY.get(c, [])
    ]
    return {"instruments": result, "total": len(result), "cross_asset_enabled": True}


@app.post("/tools/run_backtest")
async def run_backtest_endpoint(req: BacktestRequest, background_tasks: BackgroundTasks):
    await evict_old_jobs()
    instruments = resolve_instruments(req.instruments)
    if len(instruments) < 2:
        raise HTTPException(400, "Need at least 2 instruments")
    if req.pair_mode == "specified" and not req.explicit_pairs:
        raise HTTPException(400, "explicit_pairs required for pair_mode=specified")

    job_id = make_job_id()
    params = req.model_dump()
    params["instruments"] = instruments
    n_pairs = len(instruments) * (len(instruments) - 1) // 2

    async with _JOB_STORE_LOCK:
        JOB_STORE[job_id] = {
            "status":     "queued",
            "params":     params,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pair_count": n_pairs,
        }
    background_tasks.add_task(run_backtest_worker, job_id, params)
    return {
        "job_id":           job_id,
        "status":           "queued",
        "estimated_seconds": max(5, n_pairs * 2),
        "pair_count":       n_pairs,
        "instruments":      instruments,
    }


@app.get("/tools/get_backtest_status/{job_id}")
async def get_backtest_status(job_id: str):
    async with _JOB_STORE_LOCK:
        job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return {
        "job_id":       job_id,
        "status":       job["status"],
        "error":        job.get("error"),
        "completed_at": job.get("completed_at"),
    }


@app.get("/tools/get_backtest_results/{job_id}")
async def get_backtest_results(
    job_id: str,
    breakdown: str = "aggregate",
    top_n_pairs: int = 20,
    sort_by: str = "avg_net_return",
):
    async with _JOB_STORE_LOCK:
        job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "complete":
        raise HTTPException(409, f"Job status: {job['status']}")

    requested = set(breakdown.split(","))
    response  = {"job_id": job_id, "aggregate": job["results"]["aggregate"]}
    if "pairs"  in requested:
        response["pairs"]  = job["results"].get("pairs", {})
    if "regime" in requested:
        response["regime"] = job["results"].get("regime", {})
    if "trade_log" in requested and "trade_log" in job["results"]:
        response["trade_log"] = job["results"]["trade_log"]
    return response


@app.post("/tools/run_sensitivity")
async def run_sensitivity(req: SensitivityRequest, background_tasks: BackgroundTasks):
    """
    Sweep one parameter by submitting one backtest job per value.
    Supported parameters: sd_threshold, exit_sd, vol_window, max_hold_days,
    financing_rate_long, financing_rate_short.
    """
    await evict_old_jobs()
    jobs = []
    for val in req.values:
        base = BacktestRequest(instruments=req.base_instruments)
        if req.fixed_params:
            for k, v in req.fixed_params.items():
                if hasattr(base, k):
                    setattr(base, k, v)
        if not hasattr(base, req.parameter):
            raise HTTPException(400, f"Unknown parameter: {req.parameter}")
        setattr(base, req.parameter, val)

        job_id = make_job_id()
        p = base.model_dump()
        p["instruments"] = resolve_instruments(p["instruments"])
        async with _JOB_STORE_LOCK:
            JOB_STORE[job_id] = {
                "status":            "queued",
                "params":            p,
                "created_at":        datetime.now(timezone.utc).isoformat(),
                "sensitivity_value": val,
            }
        background_tasks.add_task(run_backtest_worker, job_id, p)
        jobs.append({"value": val, "job_id": job_id})
    return {"parameter": req.parameter, "values": req.values, "jobs": jobs}


@app.post("/tools/get_correlation_matrix")
async def get_correlation_matrix(req: CorrelationRequest):
    instruments = resolve_instruments(req.instruments)
    prices = load_aligned_prices_unified(
        instruments, CACHE_DIR,
        date_from=req.date_from,
        date_to=req.date_to,
    )
    corr = prices.pct_change().dropna().corr()
    return {
        "instruments": instruments,
        "matrix":      _to_native(corr.to_dict()),
        "date_from":   str(prices.index[0].date()),
        "date_to":     str(prices.index[-1].date()),
        "n_days":      len(prices),
    }


@app.post("/tools/get_pair_deep_dive")
async def get_pair_deep_dive(req: PairDeepDiveRequest, background_tasks: BackgroundTasks):
    await evict_old_jobs()
    job_id = make_job_id()
    params = {
        "instruments":         [req.long_instrument, req.short_instrument],
        "pair_mode":           "specified",
        "explicit_pairs":      [{"long": req.long_instrument, "short": req.short_instrument}],
        "sd_threshold":        req.sd_threshold,
        "exit_sd":             req.exit_sd,
        "financing_rate_long": req.financing_rate_long,
        "financing_rate_short": 0.88,
        "bid_ask_override":    None,
        "vol_window":          262,
        "max_hold_days":       300,
        "date_from":           req.date_from or "1999-01-04",
        "date_to":             req.date_to or "2026-04-24",
        "include_trade_log":   req.include_trade_log,
        "top_n_pairs":         20,
    }
    async with _JOB_STORE_LOCK:
        JOB_STORE[job_id] = {
            "status":      "queued",
            "params":      params,
            "created_at":  datetime.now(timezone.utc).isoformat(),
            "is_deep_dive": True,
        }
    background_tasks.add_task(run_backtest_worker, job_id, params)
    return {
        "job_id":      job_id,
        "status":      "queued",
        "long":        req.long_instrument,
        "short":       req.short_instrument,
        "cross_asset": is_cross_asset_pair(req.long_instrument, req.short_instrument),
    }


@app.post("/tools/compare_asset_classes")
async def compare_asset_classes(
    req: CompareAssetClassesRequest,
    background_tasks: BackgroundTasks,
):
    await evict_old_jobs()
    groups: dict[str, list[str]] = {
        "commodities":  ["ALL_COMMODITIES"],
        "fx":           ["ALL_FX"],
        "fixed_income": ["ALL_FI"],
    }
    if req.include_cross_asset:
        groups["cross_asset"] = ["ALL_COMMODITIES", "ALL_FX", "ALL_FI"]

    jobs: dict[str, str] = {}
    for label, shorthand_list in groups.items():
        job_id = make_job_id()
        p = {
            "instruments":          resolve_instruments(shorthand_list),
            "sd_threshold":         req.sd_threshold,
            "exit_sd":              req.exit_sd,
            "financing_rate_long":  req.financing_rate_long,
            "financing_rate_short": 0.88,
            "bid_ask_override":     None,
            "vol_window":           262,
            "max_hold_days":        300,
            "pair_mode":            "exhaustive",
            "explicit_pairs":       None,
            "date_from":            req.date_from or "1999-01-04",
            "date_to":              req.date_to or "2026-04-24",
            "top_n_pairs":          20,
        }
        async with _JOB_STORE_LOCK:
            JOB_STORE[job_id] = {
                "status":             "queued",
                "params":             p,
                "created_at":         datetime.now(timezone.utc).isoformat(),
                "asset_class_label":  label,
            }
        background_tasks.add_task(run_backtest_worker, job_id, p)
        jobs[label] = job_id

    return {
        "jobs": jobs,
        "note": "Poll each job_id, then retrieve results to assemble league table",
    }
