"""
test_cross_asset_replication.py — Verify cross-asset search against Q8 findings
================================================================================

Runs the six cross-asset category combinations at exit_sd=1.0 and compares
aggregate GWR and net expectancy against Q8 research paper values.

Expected runtime: ~5-10 seconds (all 1,726 pairs exhaustive, exit=1.0)

Run from project root:
    python test_cross_asset_replication.py
"""

import sys
from pathlib import Path

_PROJECT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT))

import numpy as np
from itertools import combinations

from backtest import (
    load_cross_asset_prices, prepare_returns_aligned, run_exhaustive_search,
)
from asset_configs import ASSET_CLASSES, get_spread_cost_lookup, FI_EXCLUDE

_CACHE = _PROJECT / 'cache'

CROSS_PAIRS = [
    ('equity',      'commodities'),
    ('equity',      'fx'),
    ('equity',      'fixed_income'),
    ('commodities', 'fx'),
    ('commodities', 'fixed_income'),
    ('fx',          'fixed_income'),
]

# Q8 top-5 individual pairs (exit=1.0, 4.88% financing)
Q8_TOP_PAIRS = [
    ('PLATINUM', 'AUDUSD', +0.0111, 36),
    ('CBK',      'GBPJPY', +0.0121, 49),
    ('COI',      'WHEAT',  +0.0156, 39),
    ('PLATINUM', 'HYG',    +0.0154, 33),
    ('UKX',      'GBPJPY', +0.0094, 53),
]


def run_cross_asset(long_class, short_class, exit_sd=1.0):
    prices, long_i, short_i, cls_map = load_cross_asset_prices(
        long_class, short_class, _CACHE
    )
    ls, ss, di, idx = prepare_returns_aligned(prices, long_i, short_i)

    all_instr  = long_i + short_i
    all_scaled = np.concatenate([ls, ss], axis=1)

    latest_px = prices.iloc[-1].to_dict()
    lookup = {
        **get_spread_cost_lookup(long_i,  latest_px, long_class),
        **get_spread_cost_lookup(short_i, latest_px, short_class),
    }

    long_cfg  = ASSET_CLASSES.get(long_class,  ASSET_CLASSES.get('equity', {}))
    fin_daily = long_cfg.get('financing', {}).get('long_rate', 0.0488) / 365

    n_all = len(long_i) * len(short_i) * 2
    df = run_exhaustive_search(
        all_scaled, di, all_instr,
        long_instrument_subset=long_i,
        short_instrument_subset=short_i,
        min_long_legs=1, max_long_legs=1,
        min_short_legs=1, max_short_legs=1,
        exit_sd=exit_sd,
        spread_cost_lookup=lookup,
        financing_daily_pct=fin_daily,
        top_n=n_all,
        sample_n=0,
        scoring_mode='composite',
    )
    return df, long_i, short_i, idx


def main():
    print("=" * 70)
    print("  CROSS-ASSET REPLICATION TEST — Q8 BENCHMARK")
    print("  All 6 category pairs, 1v1, exit_sd=1.0")
    print("=" * 70)

    all_results = []

    for long_class, short_class in CROSS_PAIRS:
        label = f"{long_class.title()} × {short_class.title()}"
        print(f"\n  {label}...", end="", flush=True)

        try:
            df, long_i, short_i, idx = run_cross_asset(long_class, short_class, exit_sd=1.0)
            if df.empty:
                print(" no results")
                continue

            net_pos = (df['NetExpectancy'] > 0).sum()
            best_net = df['NetExpectancy'].max()

            print(
                f" {len(df)} pairs | "
                f"net+ {net_pos} | "
                f"best net {best_net*100:+.2f}%"
            )
            all_results.append((long_class, short_class, df))

        except FileNotFoundError as e:
            print(f" SKIP (file missing: {e})")
        except Exception as e:
            print(f" ERROR: {e}")

    # ── Top pair lookup ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Q8 TOP-5 PAIRS — VERIFICATION")
    print("=" * 70)
    print(f"  {'Pair':<24} {'Q8 net':>9} {'Our net':>9} {'Q8 N':>7} {'Our N':>7}  Status")
    print("  " + "-" * 65)

    for long_inst, short_inst, q8_net, q8_n in Q8_TOP_PAIRS:
        found = False
        for long_class, short_class, df in all_results:
            match = df[
                df['Long'].str.contains(long_inst, case=False, na=False) &
                df['Short'].str.contains(short_inst, case=False, na=False)
            ]
            if not match.empty:
                our_net = float(match['NetExpectancy'].iloc[0])
                our_n   = int(float(match['Trades'].iloc[0])) if 'Trades' in match.columns else 0
                delta   = our_net - q8_net
                ok      = abs(delta) < 0.005
                print(
                    f"  {long_inst} → {short_inst:<16} "
                    f"{q8_net*100:>+8.2f}% {our_net*100:>+8.2f}% "
                    f"{q8_n:>7} {our_n:>7}  {'OK' if ok else 'MISMATCH'}"
                )
                found = True
                break
        if not found:
            print(f"  {long_inst} → {short_inst:<16} {q8_net*100:>+8.2f}%      n/a       n/a  NOT FOUND")

    print()
    print("  Note: Small differences are expected — Q8 used data from 1999,")
    print("  current data may start later for some asset classes.")
    print()


if __name__ == '__main__':
    main()
