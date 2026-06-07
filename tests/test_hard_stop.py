"""
test_hard_stop.py — Behavioural tests for backtest_spread_with_stop()
======================================================================

Tests the hard stop, MAE tracking, and stop flag logic introduced in
engine/numba_core.py alongside the existing backtest_spread() function.

Run with:  pytest tests/test_hard_stop.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.numba_core import (
    _ref_rolling_mean_std,
    backtest_spread_with_stop,
    COL_GROSS_RETURN,
    COL_MAE_SD,
    COL_STOPPED,
)

# ── Synthetic series helpers ───────────────────────────────────────────────

_WUP = 200      # warm-up bars; must be > vol_window
_VOL = 50       # small vol_window keeps tests fast


def _make_wup(seed: int = 0) -> np.ndarray:
    """Return a warm-up return series with stable rolling variance."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.003, _WUP).astype(np.float64)


def _append_at_dist_sd(returns: np.ndarray, target_d: float, vol_window: int = _VOL) -> np.ndarray:
    """Append one return that achieves target_d on the resulting bar.

    Computes the cumulative series, reads the rolling mean/std at the last
    position, and back-solves for the return that places the next cum value
    at mean + target_d * std.

    Args:
        returns: Existing return series (modified in place conceptually;
            actually returns a new array).
        target_d: Desired dist_sd for the appended bar.
        vol_window: Rolling window used in the backtest.

    Returns:
        New array with one bar appended at the desired dist_sd.
    """
    cum = np.cumprod(1.0 + returns)
    rm, rs = _ref_rolling_mean_std(cum, vol_window)
    target_cum = rm[-1] + target_d * rs[-1]
    r = float(target_cum / cum[-1] - 1)
    return np.append(returns, r)


def _day_ints(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Stop does not fire at entry
# ═══════════════════════════════════════════════════════════════════════════

def test_stop_does_not_fire_at_entry() -> None:
    """Stop must not fire on the entry bar or while d is inside stop_sd.

    Uses probe values from register item H: d=+2.2, side=-1, stop_sd=3.0.
    For a SHORT trade (side=-1), stop fires when d > stop_sd=3.0.
    At entry d=+2.2 < 3.0, so stop must not fire.
    """
    # Verify stop condition logic with probe values from register item H
    d_probe, side_probe, stop_sd_probe = 2.2, -1, 3.0
    stop_hit_probe = (side_probe == 1 and d_probe < -stop_sd_probe) or (
        side_probe == -1 and d_probe > stop_sd_probe
    )
    assert not stop_hit_probe, (
        f"Stop logic error: stop_hit should be False at d={d_probe}, "
        f"side={side_probe}, stop_sd={stop_sd_probe}"
    )

    # End-to-end: build a series that enters SHORT at ~2.2 SD, stays inside
    # stop_sd=3.0, then exits naturally.  All trades must have COL_STOPPED=0.0.
    xing_sd, exit_sd, stop_sd = 2.0, 0.5, 3.0
    rets = _make_wup()
    # Inject entry (SHORT, d > xing_sd=2.0, d < stop_sd=3.0)
    rets = _append_at_dist_sd(rets, 2.2)
    # Hover just inside stop threshold for two bars
    rets = _append_at_dist_sd(rets, 2.5)
    rets = _append_at_dist_sd(rets, 2.1)
    # Natural exit (d <= exit_sd=0.5)
    rets = _append_at_dist_sd(rets, 0.3)
    day_ints = _day_ints(len(rets))

    trades, n = backtest_spread_with_stop(
        rets, _VOL, xing_sd, exit_sd, stop_sd, day_ints
    )
    assert n >= 1, "Expected at least one trade"
    # No trade from the constructed tail should be a hard stop
    last = trades[n - 1]
    assert last[COL_STOPPED] == 0.0, (
        f"Trade flagged as stop (COL_STOPPED={last[COL_STOPPED]}) "
        f"when spread stayed inside stop_sd={stop_sd}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Stop fires on adverse extension
# ═══════════════════════════════════════════════════════════════════════════

def test_stop_fires_on_adverse_extension() -> None:
    """Hard stop fires when spread extends past stop_sd in the adverse direction.

    Constructs a SHORT trade that extends well past stop_sd.  At least one
    trade must have COL_STOPPED=1.0 and a positive gross return loss.
    """
    xing_sd, exit_sd, stop_sd = 2.0, 0.5, 2.5
    rets = _make_wup()
    # Inject SHORT entry just above xing_sd
    rets = _append_at_dist_sd(rets, 2.2)
    # Adverse extension well past stop_sd (d > 2.5 for SHORT)
    rets = _append_at_dist_sd(rets, 3.2)
    day_ints = _day_ints(len(rets))

    trades, n = backtest_spread_with_stop(
        rets, _VOL, xing_sd, exit_sd, stop_sd, day_ints
    )
    assert n >= 1, "Expected at least one trade after the crossing"

    # The last trade should have been stopped out
    last = trades[n - 1]
    assert last[COL_STOPPED] == 1.0, (
        f"Expected COL_STOPPED=1.0 for a trade that extended to d>stop_sd={stop_sd}; "
        f"got {last[COL_STOPPED]}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: MAE tracks worst point
# ═══════════════════════════════════════════════════════════════════════════

def test_mae_tracks_worst_point() -> None:
    """COL_MAE_SD equals the maximum adverse dist_sd reached during the trade.

    Constructs a LONG trade (d < -xing_sd) that extends adversely then
    reverts naturally.  The MAE is verified against the worst adverse dist_sd
    independently computed from the series.
    """
    xing_sd, exit_sd, stop_sd = 2.0, 0.5, 5.0  # large stop so it never fires
    rets = _make_wup()
    # LONG entry (d < -2.0)
    rets = _append_at_dist_sd(rets, -2.2)
    # Adverse extension: d goes more negative (worse for LONG)
    rets = _append_at_dist_sd(rets, -2.8)   # adverse_d = 2.8 for LONG (side=+1)
    rets = _append_at_dist_sd(rets, -3.1)   # new worst point; adverse_d = 3.1
    rets = _append_at_dist_sd(rets, -2.5)   # partially better
    # Natural exit (d >= -exit_sd = -0.5 for LONG)
    rets = _append_at_dist_sd(rets, -0.3)
    day_ints = _day_ints(len(rets))

    trades, n = backtest_spread_with_stop(
        rets, _VOL, xing_sd, exit_sd, stop_sd, day_ints
    )
    assert n >= 1, "Expected at least one trade"

    last = trades[n - 1]
    assert last[COL_STOPPED] == 0.0, "Trade should exit normally, not via hard stop"

    # Independently compute the worst adverse dist_sd over the trade bars
    cum = np.cumprod(1.0 + rets)
    rm, rs = _ref_rolling_mean_std(cum, _VOL)
    dist = np.full(len(cum), np.nan)
    for i in range(len(cum)):
        if not np.isnan(rs[i]) and rs[i] > 0:
            dist[i] = (cum[i] - rm[i]) / rs[i]

    entry_idx = int(last[0])   # COL_ENTRY_IDX
    exit_idx = int(last[1])    # COL_EXIT_IDX
    trade_dist = dist[entry_idx + 1 : exit_idx + 1]  # bars after entry up to exit
    expected_mae = float(np.nanmax(-trade_dist))  # LONG adverse = -d

    assert abs(last[COL_MAE_SD] - expected_mae) < 1e-10, (
        f"COL_MAE_SD={last[COL_MAE_SD]:.6f} does not match "
        f"independently computed worst adverse dist_sd={expected_mae:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Normal exit is not flagged as stop
# ═══════════════════════════════════════════════════════════════════════════

def test_normal_exit_not_flagged_as_stop() -> None:
    """A trade that reverts naturally through exit_sd has COL_STOPPED=0.0."""
    xing_sd, exit_sd, stop_sd = 2.0, 0.5, 4.0
    rets = _make_wup()
    # SHORT entry
    rets = _append_at_dist_sd(rets, 2.3)
    # Mild adverse move — stays well inside stop_sd
    rets = _append_at_dist_sd(rets, 2.6)
    # Reversion toward mean
    rets = _append_at_dist_sd(rets, 1.8)
    rets = _append_at_dist_sd(rets, 0.8)
    # Exit (d <= exit_sd=0.5 for SHORT)
    rets = _append_at_dist_sd(rets, 0.2)
    day_ints = _day_ints(len(rets))

    trades, n = backtest_spread_with_stop(
        rets, _VOL, xing_sd, exit_sd, stop_sd, day_ints
    )
    assert n >= 1, "Expected at least one trade"
    last = trades[n - 1]
    assert last[COL_STOPPED] == 0.0, (
        f"Natural reversion exit incorrectly flagged as hard stop "
        f"(COL_STOPPED={last[COL_STOPPED]})"
    )
    assert last[COL_GROSS_RETURN] > 0.0, (
        "Expected a positive gross return for a trade that reverted through exit_sd"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: stop_sd validation
# ═══════════════════════════════════════════════════════════════════════════

def test_stop_sd_validation() -> None:
    """ValueError is raised when stop_sd <= xing_sd or stop_sd <= exit_sd."""
    rets = _make_wup()
    day_ints = _day_ints(len(rets))

    # stop_sd equal to xing_sd — must raise
    with pytest.raises(ValueError, match="stop_sd"):
        backtest_spread_with_stop(rets, _VOL, xing_sd=2.0, exit_sd=0.5,
                                  stop_sd=2.0, day_ints=day_ints)

    # stop_sd less than xing_sd — must raise
    with pytest.raises(ValueError, match="stop_sd"):
        backtest_spread_with_stop(rets, _VOL, xing_sd=2.0, exit_sd=0.5,
                                  stop_sd=1.5, day_ints=day_ints)

    # stop_sd equal to exit_sd (but > xing_sd) — must raise
    with pytest.raises(ValueError, match="stop_sd"):
        backtest_spread_with_stop(rets, _VOL, xing_sd=2.0, exit_sd=2.5,
                                  stop_sd=2.5, day_ints=day_ints)

    # Valid parameters — must not raise
    trades, n = backtest_spread_with_stop(
        rets, _VOL, xing_sd=2.0, exit_sd=0.5, stop_sd=3.0, day_ints=day_ints
    )
    assert isinstance(trades, np.ndarray)
    assert trades.shape[1] == 8
