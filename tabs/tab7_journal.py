from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from core.basket import Basket
from tabs.shared import (
    portfolio, registry, _cached_latest_prices, ALL_INSTRUMENTS, ALL_DISPLAY, _asset_class_of,
)


def _render_pending_entry(p: dict) -> None:
    """Confirm form for a trade received from Stake Calculator via Book Trade."""
    long_str  = " + ".join(ALL_DISPLAY.get(i, i) for i in p["long_legs"])
    short_str = " + ".join(ALL_DISPLAY.get(i, i) for i in p["short_legs"])
    st.info(f"📒 Trade from Stake Calc — review and confirm: **{long_str}** vs **{short_str}**")

    with st.form("tab7_book_trade_form"):
        fc1, fc2 = st.columns(2)
        with fc1:
            _name = st.text_input(
                "Trade name *",
                value=(
                    f"{p['long_legs'][0] if p['long_legs'] else '?'} / "
                    f"{p['short_legs'][0] if p['short_legs'] else '?'}"
                ),
            )
            _direction = st.selectbox(
                "Direction",
                ["long_spread", "short_spread"],
                index=0 if p.get("direction") == "long_spread" else 1,
            )
        with fc2:
            _target_exp = st.number_input(
                "Target exposure (£)",
                value=float(p.get("target_exposure", 500.0)),
                min_value=50.0,
                step=50.0,
            )
            _comments = st.text_area(
                "Comments",
                value=f"Broker: {p.get('broker', 'IG')} | Mode: {p.get('mode', 'SB')}",
                height=68,
            )

        with st.expander("📊 Stake Calc summary", expanded=True):
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Margin", f"£{p.get('margin', 0):,.0f}")
            sm2.metric("Spread cost", f"{p.get('spread_cost_pct', 0):.3%}")
            sm3.metric("Daily financing", f"£{p.get('daily_financing', 0):,.2f}")
            _be = p.get("breakeven_days")
            sm4.metric("Breakeven hold", f"{_be:.0f}d" if _be is not None else "∞")
            st.caption("Stakes (£/pt):")
            for inst, stk in p.get("stakes", {}).items():
                _side = "Long" if stk >= 0 else "Short"
                _px   = p["prices"].get(inst, 0)
                st.caption(
                    f"  {ALL_DISPLAY.get(inst, inst)}: {_side} "
                    f"£{abs(stk):.3f}/pt @ {_px:,.2f}"
                )

        cs, cd = st.columns(2)
        _submitted = cs.form_submit_button(
            "💾 Confirm & Open Position", type="primary", use_container_width=True
        )
        _discarded = cd.form_submit_button("🗑 Discard", use_container_width=True)

    if _submitted:
        if not _name.strip():
            st.error("Trade name is required.")
            st.session_state["tab7_pending_entry"] = p
        else:
            try:
                _basket = Basket(long_legs=p["long_legs"], short_legs=p["short_legs"])
                _basket.validate()
                pos = portfolio.open_position(
                    basket=_basket,
                    direction=_direction,
                    entry_prices=p["prices"],
                    stakes=p["stakes"],
                    target_exposure=_target_exp,
                    name=_name.strip(),
                    comments=_comments,
                )
                st.success(f"✅ Opened **{pos.name}** — navigate to Journal to view.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to open position: {e}")
                st.session_state["tab7_pending_entry"] = p

    if _discarded:
        st.rerun()


def render() -> None:
    st.header("Journal")
    st.caption("Paper trade open/close, position history, and live P&L.")

    _pending = st.session_state.pop("tab7_pending_entry", None)
    if _pending:
        _render_pending_entry(_pending)
        return

    st.subheader(f"Open positions ({len(portfolio.open_positions)})")

    if not portfolio.open_positions:
        st.info("No open positions. Use the form below to open one.")
    else:
        for pos in portfolio.open_positions:
            with st.container(border=True):
                try:
                    current_prices = _cached_latest_prices(tuple(pos.basket.all_instruments))
                except Exception as e:
                    st.warning(f"Could not load prices for {pos.name}: {e}")
                    current_prices = pos.entry_prices

                hcol, mcol1, mcol2, mcol3, mcol4 = st.columns([3, 2, 2, 2, 2])
                hcol.markdown(f"**{pos.name}**  \n_{pos.direction.replace('_', ' ').title()}_")
                hcol.caption(
                    f"Long: {' + '.join(pos.basket.long_legs)}  \n"
                    f"Short: {' + '.join(pos.basket.short_legs)}"
                )
                mcol1.metric("Opened", str(pos.entry_date))
                mcol1.caption(f"{pos.days_held}d held")
                live_pnl = pos.live_pnl(current_prices)
                fin_drag = pos.financing_cost_to_date()
                net_pnl  = pos.net_pnl(current_prices)
                mcol2.metric("Unrealised", f"£{live_pnl:+,.0f}")
                mcol3.metric("Financing",  f"£{-fin_drag:,.0f}")
                mcol4.metric("Net P&L",    f"£{net_pnl:+,.0f}",
                             delta_color="normal" if net_pnl >= 0 else "inverse")

                acol1, acol2, acol3 = st.columns([1, 1, 4])
                if acol1.button("Close", key=f"close_{pos.id}"):
                    st.session_state['pending_close'] = pos.id
                    st.rerun()

                with acol2.popover("Partial close"):
                    pct = st.slider("% to close", 10, 100, 50, 10,
                                    key=f"pct_{pos.id}") / 100.0
                    exit_prices = {}
                    for inst in pos.basket.all_instruments:
                        exit_prices[inst] = st.number_input(
                            f"{ALL_DISPLAY.get(inst, inst)} exit price",
                            value=float(current_prices.get(inst, pos.entry_prices.get(inst, 0.0))),
                            key=f"px_{pos.id}_{inst}",
                            format="%.4f",
                        )
                    if st.button("Confirm partial close", key=f"pc_confirm_{pos.id}"):
                        realised = portfolio.partial_close(
                            pos.id, pct, exit_prices, date.today()
                        )
                        st.success(f"Partial closed {pct:.0%} — realised £{realised:+,.0f}")
                        st.rerun()

                if st.session_state.get('pending_close') == pos.id:
                    with st.container(border=True):
                        st.warning(f"Confirm full close of **{pos.name}**?")
                        c1, c2 = st.columns(2)
                        if c1.button("Yes, close", key=f"close_yes_{pos.id}", type="primary"):
                            realised = portfolio.close_position(
                                pos.id, current_prices, date.today()
                            )
                            st.session_state['pending_close'] = None
                            st.success(f"Closed {pos.name} — realised £{realised:+,.0f}")
                            st.rerun()
                        if c2.button("Cancel", key=f"close_no_{pos.id}"):
                            st.session_state['pending_close'] = None
                            st.rerun()

                with st.expander("Leg detail"):
                    leg_rows = []
                    for inst, stake in pos.stakes.items():
                        entry   = pos.entry_prices.get(inst, 0.0)
                        curr    = current_prices.get(inst, entry)
                        leg_pnl = stake * (curr - entry) * pos.pct_open
                        leg_rows.append({
                            'Instrument': ALL_DISPLAY.get(inst, inst),
                            'Side':       'Long' if stake > 0 else 'Short',
                            'Stake':      f"{stake:+.3f}",
                            'Entry':      f"{entry:,.4f}",
                            'Current':    f"{curr:,.4f}",
                            'Leg P&L':    f"£{leg_pnl:+,.2f}",
                        })
                    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Open new trade")

    nc1, nc2, nc3 = st.columns(3)
    trade_name      = nc1.text_input("Trade name", value="", key="new_name",
                                     placeholder="e.g. Platinum / AUDUSD")
    direction       = nc2.selectbox("Direction", ['long_spread', 'short_spread'], key="new_dir")
    target_exposure = nc3.number_input("Target exposure (£)", value=500.0, step=50.0,
                                       min_value=50.0, key="new_exp")

    nc4, nc5 = st.columns([1, 2])
    entry_date_sel = nc4.date_input("Entry date", value=date.today(), key="new_date")
    comments       = nc5.text_input("Comments", value="", key="new_comments")

    st.markdown("**Legs**")
    lc1, lc2, _ = st.columns([1, 1, 4])
    if lc1.button("+ Add leg"):
        st.session_state['leg_count'] += 1
        st.rerun()
    if lc2.button("− Remove leg") and st.session_state['leg_count'] > 1:
        st.session_state['leg_count'] -= 1
        st.rerun()

    legs: list[dict] = []
    for i in range(st.session_state['leg_count']):
        with st.container(border=True):
            st.markdown(f"**Leg {i + 1}**")
            lcols = st.columns([2, 1, 1, 2, 1, 1])
            buy_inst = lcols[0].selectbox(
                "Buy", ALL_INSTRUMENTS,
                format_func=lambda c: f"{ALL_DISPLAY.get(c, c)} ({_asset_class_of(c)})",
                key=f"leg_buy_{i}",
            )
            try:
                buy_default = float(_cached_latest_prices((buy_inst,)).get(buy_inst, 0.0))
            except Exception:
                buy_default = 0.0
            buy_price = lcols[1].number_input(
                "Buy price", value=buy_default, format="%.4f", key=f"leg_bp_{i}",
            )
            buy_stake = lcols[2].number_input(
                "Buy stake", value=1.0, step=0.1, format="%.3f", key=f"leg_bs_{i}",
            )
            sell_inst = lcols[3].selectbox(
                "Sell", ALL_INSTRUMENTS,
                format_func=lambda c: f"{ALL_DISPLAY.get(c, c)} ({_asset_class_of(c)})",
                index=min(1, len(ALL_INSTRUMENTS) - 1),
                key=f"leg_sell_{i}",
            )
            try:
                sell_default = float(_cached_latest_prices((sell_inst,)).get(sell_inst, 0.0))
            except Exception:
                sell_default = 0.0
            sell_price = lcols[4].number_input(
                "Sell price", value=sell_default, format="%.4f", key=f"leg_sp_{i}",
            )
            sell_stake = lcols[5].number_input(
                "Sell stake", value=1.0, step=0.1, format="%.3f", key=f"leg_ss_{i}",
            )
            legs.append({
                'buy': buy_inst, 'buy_price': buy_price, 'buy_stake': buy_stake,
                'sell': sell_inst, 'sell_price': sell_price, 'sell_stake': sell_stake,
            })

    try:
        prev_basket = Basket(
            long_legs=[l['buy'] for l in legs],
            short_legs=[l['sell'] for l in legs],
        )
        prev_basket.validate()
        sp_cost   = prev_basket.spread_cost(registry)
        daily_fin = prev_basket.financing_cost_daily()
        be_days   = sp_cost / daily_fin if daily_fin > 0 else float('inf')
        st.caption(
            f"Est. spread cost: **{sp_cost:.3%}** round-trip  |  "
            f"Est. daily financing: **£{daily_fin * target_exposure:,.2f}**  |  "
            f"Break-even hold: **{be_days:.0f} days**"
        )
    except Exception as e:
        st.caption(f"_Spread cost preview unavailable: {e}_")

    if st.button("📓 Open position", type="primary", key="new_submit"):
        if not trade_name.strip():
            st.error("Trade name is required.")
        else:
            try:
                basket = Basket(
                    long_legs=[l['buy'] for l in legs],
                    short_legs=[l['sell'] for l in legs],
                )
                basket.validate()
                entry_prices = {}
                stakes = {}
                for l in legs:
                    entry_prices[l['buy']]  = l['buy_price']
                    entry_prices[l['sell']] = l['sell_price']
                    stakes[l['buy']]  = stakes.get(l['buy'],  0.0) + l['buy_stake']
                    stakes[l['sell']] = stakes.get(l['sell'], 0.0) - l['sell_stake']

                pos = portfolio.open_position(
                    basket=basket,
                    direction=direction,
                    entry_prices=entry_prices,
                    stakes=stakes,
                    target_exposure=float(target_exposure),
                    name=trade_name.strip(),
                    comments=comments,
                )
                st.success(f"Opened **{pos.name}** (id={pos.id})")
                st.session_state['leg_count'] = 1
                st.rerun()
            except ValueError as e:
                st.error(f"Validation error: {e}")
            except Exception as e:
                st.error(f"Failed to open position: {e}")

    st.markdown("---")
    st.subheader("Trade history")
    closed = portfolio.closed_positions
    if not closed:
        st.caption("No closed positions yet.")
        return

    total_real = portfolio.total_realised_pnl()
    wins       = sum(1 for p in closed if p.realised_pnl > 0)
    wr         = wins / len(closed) if closed else 0.0
    hm1, hm2, hm3 = st.columns(3)
    hm1.metric("Realised P&L",  f"£{total_real:+,.0f}")
    hm2.metric("Trades closed", f"{len(closed)}")
    hm3.metric("Win rate",      f"{wr:.0%}")

    hist_rows = []
    for p in sorted(closed, key=lambda x: x.exit_date or date.min, reverse=True):
        hist_rows.append({
            'Name':      p.name,
            'Pair':      f"{' + '.join(p.basket.long_legs)} vs {' + '.join(p.basket.short_legs)}",
            'Entry':     str(p.entry_date),
            'Close':     str(p.exit_date),
            'Days held': p.days_held,
            'P&L':       f"£{p.realised_pnl:+,.0f}",
        })
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)


