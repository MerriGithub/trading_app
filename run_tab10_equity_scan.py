"""
Tab 10 equity grid scan — standalone script (v2).
Fixed params: Entry SD=2.0, Exit SD=2.0, Vol=262, Trend=Both passes (window=262d).
Runs all 132 directional pairs (permutations, both directions).
Filters: Trades_WT >= 8, Aligned% >= 0.10 (equity scalp floor), AvgNet_WT > 0.
Sorted by AvgNet_WT descending.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from itertools import permutations
from pathlib import Path

from engine.backtest import load_asset_prices, prepare_returns, run_backtest, aggregate_trades
from engine.numba_core import COL_ENTRY_IDX, COL_SIDE
from asset_configs import ASSET_CLASSES, get_tradeable_instruments, get_display_name
from account import get_financing_daily_rate

CACHE_DIR   = Path(__file__).parent / 'cache'
CSV_PATH    = CACHE_DIR / 'prices.csv'
ASSET_KEY   = 'equity'
ENTRY_SD    = 2.0
EXIT_SD     = 2.0
VOL_WIN     = 262
TREND_WIN   = 262
BROKER      = 'ig_spreadbet'
MIN_WT      = 8
MIN_ALIGNED = 0.10   # equity scalp regime: holds ~5d, inherently counter-trend; 10% floor
SLB         = min(20, TREND_WIN // 10)  # slope lookback = 20


def main():
    prices, _ = load_asset_prices(CSV_PATH)
    instruments = get_tradeable_instruments(ASSET_KEY)
    instruments = [i for i in instruments if i in prices.columns]

    print(f"Instruments ({len(instruments)}): {instruments}")
    print(f"Price history: {prices.index[0].date()} to {prices.index[-1].date()}  ({len(prices)} days)")

    pairs = list(permutations(instruments, 2))
    print(f"Pairs: {len(pairs)} (both directions)  |  Entry SD={ENTRY_SD}, Exit SD={EXIT_SD}, Vol={VOL_WIN}, Trend={TREND_WIN}d")
    print(f"Filters: Trades_WT >= {MIN_WT}, Aligned% >= {MIN_ALIGNED:.0%} (equity scalp floor), AvgNet_WT > 0\n")

    rows = []
    ac_cfg = ASSET_CLASSES[ASSET_KEY]

    for long_i, short_i in pairs:
        pair_df = prices[[long_i, short_i]].dropna()
        if len(pair_df) < VOL_WIN + 50:
            continue

        # Spread costs
        l_cfg = ac_cfg['instruments'].get(long_i,  {})
        s_cfg = ac_cfg['instruments'].get(short_i, {})
        l_sp = l_cfg.get('spread_pct', 0.001) if isinstance(l_cfg, dict) else 0.001
        s_sp = s_cfg.get('spread_pct', 0.001) if isinstance(s_cfg, dict) else 0.001
        pair_cost = 2.0 * (l_sp + s_sp)

        # Financing
        long_fin  = get_financing_daily_rate(long_i,  ASSET_KEY, 'long',  broker_profile=BROKER)
        short_fin = get_financing_daily_rate(short_i, ASSET_KEY, 'short', broker_profile=BROKER)
        pair_fin  = (long_fin + short_fin) / 2

        # Trend series (raw log spread, rolling mean of cumsum)
        raw_lr       = np.log(pair_df[long_i] / pair_df[short_i]).diff().fillna(0)
        trend_ser    = raw_lr.cumsum().rolling(TREND_WIN, min_periods=10).mean()
        trend_arr    = trend_ser.values

        # Vol-scaled returns + backtest
        try:
            scaled, day_ints, sc_index = prepare_returns(pair_df, [long_i, short_i], vol_window=VOL_WIN)
        except Exception:
            continue
        if scaled.shape[0] < VOL_WIN:
            continue

        spread = scaled[:, 0] - scaled[:, 1]
        bt = run_backtest(spread, day_ints, vol_window=VOL_WIN,
                          xing_sd=ENTRY_SD, exit_sd=EXIT_SD,
                          spread_cost_pct=pair_cost, financing_daily_pct=pair_fin, n_legs=2)

        n_tr = bt['n_trades']
        if n_tr == 0:
            continue

        raw_t  = bt['trades_raw'][:n_tr]
        eidxs  = raw_t[:, COL_ENTRY_IDX].astype(int)
        sides  = raw_t[:, COL_SIDE]
        edates = sc_index[eidxs]

        tipos  = trend_ser.index.get_indexer(edates, method='nearest')
        prev_p = np.maximum(0, tipos - SLB)
        has_tr = (tipos >= SLB) & ~np.isnan(trend_arr[tipos])
        slp    = np.where(has_tr,
                          (trend_arr[tipos] - trend_arr[prev_p]) / SLB,
                          np.nan)
        al     = ((sides > 0) & (slp > 0)) | ((sides < 0) & (slp < 0))
        vld    = ~np.isnan(slp)

        aligned_pct = float(al[vld].mean()) if vld.any() else float('nan')

        wt_f = al & vld
        ct_f = ~al & vld
        n_wt = int(wt_f.sum())
        n_ct = int(ct_f.sum())

        wt_s = aggregate_trades(raw_t[wt_f], n_wt, pair_cost, pair_fin, 2) if n_wt > 0 else {}
        ct_s = aggregate_trades(raw_t[ct_f], n_ct, pair_cost, pair_fin, 2) if n_ct > 0 else {}

        wt_net = float(wt_s.get('avg_net', float('nan')))
        ct_net = float(ct_s.get('avg_net', float('nan')))
        wt_pos = not np.isnan(wt_net) and wt_net > 0
        ct_pos = not np.isnan(ct_net) and ct_net > 0
        best_dir = ('Both' if wt_pos and ct_pos else
                    'WT'   if wt_pos else
                    'CT'   if ct_pos else 'Neither')

        rows.append({
            'Long':       get_display_name(ASSET_KEY, long_i),
            'Short':      get_display_name(ASSET_KEY, short_i),
            '_long':      long_i,
            '_short':     short_i,
            'Trades_WT':  n_wt,
            'NetWR_WT':   float(wt_s.get('net_wr',      float('nan'))),
            'AvgNet_WT':  wt_net,
            'AvgHold_WT': float(wt_s.get('avg_holding', float('nan'))),
            'Trades_CT':  n_ct,
            'NetWR_CT':   float(ct_s.get('net_wr',      float('nan'))),
            'AvgNet_CT':  ct_net,
            'AvgHold_CT': float(ct_s.get('avg_holding', float('nan'))),
            'Aligned%':   aligned_pct,
            'Best Dir':   best_dir,
            'EstCost':    pair_cost,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results.")
        return

    # Apply filters
    filtered = df[
        (df['Trades_WT'] >= MIN_WT) &
        (df['Aligned%']  >= MIN_ALIGNED) &
        (df['AvgNet_WT'] > 0)
    ].sort_values('AvgNet_WT', ascending=False).reset_index(drop=True)

    print(f"Total pairs scanned:  {len(df)}")
    print(f"Pass all filters:     {len(filtered)}\n")

    if filtered.empty:
        print("No pairs pass filters. Full results saved to results/tab10_equity_scan.csv")
    else:
        hdr = (f"{'#':<4} {'Long':<8} {'Short':<8} {'Trades_WT':>10} {'NetWR_WT':>9} "
               f"{'AvgNet_WT':>10} {'AvgHold_WT':>11} {'Trades_CT':>10} {'AvgNet_CT':>10} "
               f"{'Aligned%':>9} {'BestDir':>8}")
        print(hdr)
        print('-' * len(hdr))
        for i, r in filtered.iterrows():
            nwr = r['NetWR_WT']
            print(
                f"{i+1:<4} {r['Long']:<8} {r['Short']:<8} "
                f"{r['Trades_WT']:>10} "
                f"{nwr:>8.1%} "
                f"{r['AvgNet_WT']*100:>+9.3f}% "
                f"{r['AvgHold_WT']:>10.0f}d "
                f"{r['Trades_CT']:>10} "
                f"{r['AvgNet_CT']*100:>+9.3f}% "
                f"{r['Aligned%']:>8.0%} "
                f"{r['Best Dir']:>8}"
            )

    # Save full results
    out_path = Path(__file__).parent / 'results' / 'tab10_equity_scan_v2.csv'
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nFull results ({len(df)} pairs) saved to {out_path}")


if __name__ == '__main__':
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)
    main()
