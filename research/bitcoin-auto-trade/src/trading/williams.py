"""Daily-OHLC Williams stop entries, with explicit intraday uncertainty bounds.

One long entry at open + k * prior high-low, exited at the next open. No
same-day stop/target, short leg, or ambiguous high/low order is inferred.
Funding at the exit open belongs to the carried position, before its exit.
"""
import numpy as np
import pandas as pd

from .data import timestamps
from .directional import DirectionalResult


def uses_williams(profile):
    return profile is not None and "williams" in {profile.bull, profile.side, profile.bear}


def build_williams_setups(bars, regimes, profile):
    """At each open, use only prior-bar ranges and prior-close regime gates."""
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Williams needs sorted unique bars")
    if not regimes.index.is_unique or not regimes.index.is_monotonic_increasing:
        raise ValueError("Williams needs sorted unique regimes")
    choices={0:profile.bear,1:profile.side,2:profile.bull}
    if not set(choices.values()).issubset({"williams","flat"}) or not uses_williams(profile):
        raise ValueError("Williams profiles may only combine williams and flat")
    k=profile.breakout_k
    if not np.isfinite(k) or not 0<k<=1:
        raise ValueError("breakout_k must be in (0, 1]")
    previous=regimes.reindex(bars.index).shift(1)
    prior_range=(bars.high-bars.low).shift(1)
    gate=pd.Series(True,index=bars.index) if set(choices.values())=={"williams"} else previous.map(choices).eq("williams")
    return pd.DataFrame(dict(prior_high=bars.high.shift(1),prior_low=bars.low.shift(1),
        prior_range=prior_range,trigger=bars.open+k*prior_range,
        prior_regime=previous,eligible=gate & prior_range.gt(0)),index=bars.index)


def simulate_williams(bars,setups,funding_events=None,fees=.0005,slippage=.0005,
                      initial_cash=10000.,funding_timing="conservative_daily",final_day="skip_entry"):
    """Model a stop fill at the trigger; daily OHLC cannot resolve jump fills.

    Intraday entry timestamps are unknown and recorded with day precision.
    conservative_daily charges all positive entry-day funding and credits no
    negative entry-day events. favorable_daily does the opposite. These bound
    timing uncertainty for the *same daily-open mark proxy*, not execution risk.
    Entry notional is 1x post-cost equity; BTC quantity stays fixed until exit.
    """
    if len(bars)<2 or not bars.index.equals(setups.index):
        raise ValueError("Need aligned setups and at least two daily bars")
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing or not bars.index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all():
        raise ValueError("Need consecutive sorted daily bars")
    prices=bars[["open","high","low","close"]]
    if not np.isfinite(prices).all().all() or not prices.gt(0).all().all() or (bars.high<prices.max(axis=1)).any() or (bars.low>prices.min(axis=1)).any():
        raise ValueError("Invalid Williams OHLC")
    if not (np.isfinite(initial_cash) and initial_cash>0 and 0<=fees<1 and 0<=slippage<1):
        raise ValueError("Invalid Williams cash or costs")
    if funding_timing not in {"conservative_daily","favorable_daily"} or final_day!="skip_entry":
        raise ValueError("Invalid Williams funding/final-day policy")
    eligible=setups.eligible.fillna(False).to_numpy(dtype=bool)
    trigger=setups.trigger.to_numpy(dtype=float)
    if (eligible & (~np.isfinite(trigger) | (trigger<=bars.open.to_numpy()))).any():
        raise ValueError("Eligible stop triggers must be finite and above today's open")
    funding=pd.DataFrame(columns=["datetime","last_funding_rate","mark_price"])
    if funding_events is not None:
        funding=funding_events.copy()
        funding["datetime"]=timestamps(funding.datetime)
        funding["last_funding_rate"]=pd.to_numeric(funding.last_funding_rate,errors="raise")
        if funding.datetime.duplicated().any() or not np.isfinite(funding.last_funding_rate).all():
            raise ValueError("Invalid Williams funding events")
        if "mark_price" not in funding: funding["mark_price"]=np.nan
        funding["mark_price"]=pd.to_numeric(funding.mark_price,errors="raise")
        supplied=funding.mark_price.notna()
        if not (np.isfinite(funding.loc[supplied,"mark_price"]) & funding.loc[supplied,"mark_price"].gt(0)).all():
            raise ValueError("Invalid Williams funding mark prices")
    groups={day:frame for day,frame in funding.groupby(funding.datetime.dt.floor("D"))} if len(funding) else {}
    equity=float(initial_cash)
    quantity=0.
    last_mark=float(bars.open.iloc[0])
    active=None
    positions,orders,rows=[],[],[]
    minimum=equity
    max_leverage=0.
    ambiguous_events=0

    def mark(price):
        nonlocal equity,last_mark
        pnl=quantity*(price-last_mark)
        equity+=pnl
        if active is not None: active["price_pnl"]+=pnl
        last_mark=float(price)
        return pnl

    def fund(events,proxy,uncertain=False):
        nonlocal equity,ambiguous_events
        if active is None or events is None or events.empty: return 0.
        payments=quantity*events.mark_price.fillna(proxy)*events.last_funding_rate
        if uncertain:
            ambiguous_events+=len(events)
            active["uncertain_funding_events"]+=len(events)
            payments=payments.clip(lower=0) if funding_timing=="conservative_daily" else payments.clip(upper=0)
        amount=float(payments.sum())
        equity-=amount
        active["funding_cost"]+=amount
        return amount

    def order(dt,price,action):
        nonlocal equity
        notional=quantity*price
        fee,impact=notional*fees,notional*slippage
        equity-=fee+impact
        active["fees"]+=fee
        active["slippage"]+=impact
        orders.append(dict(datetime=dt,datetime_precision="day" if action=="entry" else "instant",
            position_id=active["position_id"],action=action,direction=1,
            side="buy" if action=="entry" else "sell",quantity=quantity,reference_price=price,
            notional=notional,fee=fee,slippage=impact,equity_after=equity,
            fill_model="intraday_stop_at_trigger" if action=="entry" else "next_open"))
        return fee+impact

    for i,(dt,bar) in enumerate(bars.iterrows()):
        before=equity
        price_pnl=mark(float(bar.open))
        if equity<=0: raise RuntimeError("Williams equity exhausted across opening gap")
        events=groups.get(dt)
        boundary=events.loc[events.datetime.eq(dt)] if events is not None else None
        later=events.loc[events.datetime.gt(dt)] if events is not None else None
        funding_cost=fund(boundary,float(bar.open))
        cost=0.
        if active is not None:
            cost+=order(dt,float(bar.open),"exit")
            active.update(exit_datetime=dt,exit_price=float(bar.open),status="next_open_exit",exit_equity=equity,
                net_pnl=active["price_pnl"]-active["funding_cost"]-active["fees"]-active["slippage"])
            active["net_return_on_entry_equity"]=active["net_pnl"]/active["entry_equity"]
            positions.append(active)
            active=None
            quantity=0.
        if equity<=0: raise RuntimeError("Williams equity exhausted at exit")
        worst=equity
        if i<len(bars)-1 and eligible[i] and float(bar.high)>=trigger[i]:
            quantity=equity/(trigger[i]*(1+fees+slippage))
            active=dict(position_id=len(positions)+1,direction=1,
                regime={0:"bear",1:"side",2:"bull"}.get(setups.prior_regime.iloc[i],"unavailable"),
                signal_datetime=dt,signal_information="today open and prior completed day only",
                entry_datetime=dt,entry_datetime_precision="day",entry_window_end=dt+pd.Timedelta("1D"),
                entry_price=float(trigger[i]),entry_equity=equity,quantity=quantity,
                price_pnl=0.,funding_cost=0.,fees=0.,slippage=0.,uncertain_funding_events=0)
            last_mark=float(trigger[i])
            cost+=order(dt,float(trigger[i]),"entry")
            before_funding=equity
            funding_cost+=fund(later,float(bar.open),uncertain=True)
            # The day's low can precede entry: use it only as a conservative
            # solvency floor, never as an inferred stop or liquidation fill.
            debits=float((quantity*later.mark_price.fillna(float(bar.open))*later.last_funding_rate).clip(lower=0).sum()) if later is not None else 0.
            worst=before_funding-debits+quantity*(float(bar.low)-trigger[i])
            if worst<=0: raise RuntimeError("Williams daily-low equity floor is nonpositive; liquidation not modeled")
        minimum=min(minimum,worst)
        price_pnl+=mark(float(bar.close))
        if equity<=0: raise RuntimeError("Williams equity exhausted at close")
        leverage=quantity*float(bar.close)/equity
        max_leverage=max(max_leverage,leverage)
        rows.append(dict(datetime=dt,equity=equity,quantity=quantity,position=int(quantity>0),
            funding_cost=funding_cost,transaction_cost=cost,price_pnl=price_pnl,
            long_pnl=equity-before,short_pnl=0.,intrabar_equity_floor=worst,
            close_leverage_before_final=leverage))
    curve=pd.DataFrame(rows).set_index("datetime")
    ledger=pd.DataFrame(positions,columns=["position_id","direction","regime","signal_datetime","signal_information",
        "entry_datetime","entry_datetime_precision","entry_window_end","entry_price","entry_equity","quantity",
        "price_pnl","funding_cost","fees","slippage","uncertain_funding_events","exit_datetime","exit_price",
        "status","exit_equity","net_pnl","net_return_on_entry_equity"])
    orders=pd.DataFrame(orders,columns=["datetime","datetime_precision","position_id","action","direction","side",
        "quantity","reference_price","notional","fee","slippage","equity_after","fill_model"])
    total=equity-initial_cash
    np.testing.assert_allclose([ledger.net_pnl.sum(),curve.price_pnl.sum()-curve.funding_cost.sum()-curve.transaction_cost.sum(),
        curve.long_pnl.sum()],total,rtol=1e-10,atol=1e-7)
    daily=curve.equity.pct_change(fill_method=None)
    daily.iloc[0]=curve.equity.iloc[0]/initial_cash-1
    drawdown=curve.equity/curve.equity.cummax().clip(lower=initial_cash)-1
    stats=dict(initial_cash=float(initial_cash),final_equity=equity,net_pnl=total,total_return=equity/initial_cash-1,
        cagr=(equity/initial_cash)**(365.25/len(bars))-1,max_drawdown=float(drawdown.min()),
        sharpe=float(daily.mean()/daily.std()*np.sqrt(365.25)) if daily.std()>0 else None,
        long_pnl=total,short_pnl=0.,gross_price_pnl=float(curve.price_pnl.sum()),
        transaction_cost=float(curve.transaction_cost.sum()),funding_cost=float(curve.funding_cost.sum()),
        orders=len(orders),positions=len(ledger),winning_positions=int(ledger.net_pnl.gt(0).sum()),
        win_rate=float(ledger.net_pnl.gt(0).mean()) if len(ledger) else None,
        exposure=float(curve.position.ne(0).mean()),long_exposure=float(curve.position.ne(0).mean()),short_exposure=0.,
        min_intrabar_equity=float(minimum),max_close_leverage=float(max_leverage),ledger_reconciled=True,
        execution="Long stop at today's open + k * prior-day range; exit next open; no stop loss or take profit",
        funding_timing=funding_timing,uncertain_funding_events=ambiguous_events,
        funding_mark_proxy="same day's open when event mark unavailable; timing bound is conditional on this proxy",
        final_position="No entry on final evaluation day; prior position exits at final day's open",
        sizing="1x post-cost equity per entry, fixed BTC quantity until next-open exit",
        liquidation="not modeled; reject nonpositive conservative daily-low equity floors",
        fill_assumption="Daily high >= trigger implies stop fill at trigger; intraday jump/slippage beyond configured cost unresolved")
    return DirectionalResult(curve,orders,ledger,stats)
