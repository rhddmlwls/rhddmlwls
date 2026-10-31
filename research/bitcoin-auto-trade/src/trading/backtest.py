"""Daily linear-perpetual long/flat/short execution with a reconciled position ledger."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import timestamps


@dataclass
class DirectionalResult:
    curve: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    stats: dict


def simulate_directional(bars,targets,funding_events=None,fees=.0005,slippage=.0005,
                         initial_cash=10_000.,close_final=True):
    """Use close[t-1] signals at open[t]; charge reversal exit before entry sizing.

    Equity is marked to each close. Signed funding cost is positive for payment,
    negative for income. Costs are explicit reference-price notional deductions.
    Entry notional equals equity after entry costs; quantity is then held fixed.
    """
    target = np.asarray(targets,float)
    if len(bars)<2 or target.shape!=(len(bars),) or not np.isin(target,[-1,0,1]).all():
        raise ValueError("Need aligned -1/0/1 daily targets and at least two bars")
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing or not (bars.index.to_series().diff().dropna()==pd.Timedelta("1D")).all():
        raise ValueError("Need consecutive sorted daily bars")
    if not (0<=fees<1 and 0<=slippage<1 and np.isfinite(initial_cash) and initial_cash>0):
        raise ValueError("Invalid fees, slippage or initial cash")
    funding = pd.DataFrame(columns=["datetime","last_funding_rate","mark_price"])
    if funding_events is not None:
        funding = funding_events.copy()
        funding["datetime"] = timestamps(funding.datetime)
        funding["last_funding_rate"] = pd.to_numeric(funding.last_funding_rate,errors="raise")
        if funding.datetime.duplicated().any() or not np.isfinite(funding.last_funding_rate).all():
            raise ValueError("Invalid funding dates or rates")
        if "mark_price" not in funding:
            funding["mark_price"] = np.nan
        else:
            funding["mark_price"] = pd.to_numeric(funding.mark_price,errors="raise")
            supplied = funding.mark_price.notna()
            if not (np.isfinite(funding.loc[supplied,"mark_price"]) & funding.loc[supplied,"mark_price"].gt(0)).all():
                raise ValueError("Invalid supplied funding mark prices")
        funding = funding.sort_values("datetime")
    groups = {dt:frame for dt,frame in funding.groupby(funding.datetime.dt.floor("D"))} if len(funding) else {}
    equity,quantity,held = float(initial_cash),0.,0
    last_mark = float(bars.open.iloc[0])
    active = None
    ledger,orders,rows = [],[],[]
    side_total = {1:0.,-1:0.}
    min_intrabar_equity = initial_cash
    max_close_leverage = 0.
    cost_rate = fees+slippage

    def record_order(dt,price,delta,action,position):
        nonlocal equity
        notional = abs(delta)*price
        fee,impact = notional*fees,notional*slippage
        equity -= fee+impact
        side_total[position["direction"]] -= fee+impact
        position["fees"] += fee
        position["slippage"] += impact
        orders.append(dict(datetime=dt,position_id=position["position_id"],action=action,
            direction=position["direction"],side="buy" if delta>0 else "sell",quantity=abs(delta),
            reference_price=price,notional=notional,fee=fee,slippage=impact,equity_after=equity))
        return fee+impact

    def mark_price(price):
        nonlocal equity,last_mark
        pnl = quantity*(price-last_mark)
        equity += pnl
        if active is not None:
            active["price_pnl"] += pnl
            side_total[active["direction"]] += pnl
        last_mark = float(price)
        return pnl

    def charge_funding(events,proxy):
        nonlocal equity
        if active is None or events is None or events.empty:
            return 0.
        cost = float((quantity*events.mark_price.fillna(proxy)*events.last_funding_rate).sum())
        equity -= cost
        active["funding_cost"] += cost
        side_total[active["direction"]] -= cost
        return cost

    def finish_position(dt,price,status):
        nonlocal active,quantity,held
        cost = record_order(dt,price,-quantity,"exit" if status=="signal_exit" else "final_exit",active)
        active.update(exit_datetime=dt,exit_price=price,status=status,exit_equity=equity,
            net_pnl=active["price_pnl"]-active["funding_cost"]-active["fees"]-active["slippage"])
        active["net_return_on_entry_equity"] = active["net_pnl"]/active["entry_equity"]
        ledger.append(active)
        active = None
        quantity,held = 0.,0
        return cost

    for i,(dt,bar) in enumerate(bars.iterrows()):
        before_long,before_short = side_total[1],side_total[-1]
        price_pnl = mark_price(float(bar.open))
        if equity<=0:
            raise RuntimeError("Equity exhausted across opening gap")
        events = groups.get(dt)
        boundary = events[events.datetime==dt] if events is not None else None
        later = events[events.datetime>dt] if events is not None else None
        funding_cost = charge_funding(boundary,float(bar.open))
        desired = int(target[i-1]) if i else 0
        transaction_cost = 0.
        if desired!=held:
            if active is not None:
                transaction_cost += finish_position(dt,float(bar.open),"signal_exit")
            if equity<=0:
                raise RuntimeError("Equity exhausted at exit; liquidation model required")
            if desired:
                quantity = desired*equity/(float(bar.open)*(1+cost_rate))
                held = desired
                active = dict(position_id=len(ledger)+1,direction=desired,regime="bull" if desired>0 else "bear",
                    signal_datetime=bars.index[i-1],entry_datetime=dt,entry_price=float(bar.open),
                    entry_equity=equity,quantity=abs(quantity),price_pnl=0.,funding_cost=0.,fees=0.,slippage=0.)
                transaction_cost += record_order(dt,float(bar.open),quantity,"entry",active)
        equity_before_later_funding = equity
        later_debits = 0.
        if later is not None and len(later) and quantity:
            payments = quantity*later.mark_price.fillna(float(bar.open))*later.last_funding_rate
            later_debits = float(payments.clip(lower=0).sum())
        funding_cost += charge_funding(later,float(bar.open))
        # Ignore later funding credits and include all later debits when
        # checking an adverse high/low: their intraday order is unknown.
        adverse = float(bar.low if quantity>0 else bar.high) if quantity else float(bar.open)
        worst = equity_before_later_funding-later_debits+quantity*(adverse-float(bar.open))
        min_intrabar_equity = min(min_intrabar_equity,worst)
        if worst<=0:
            raise RuntimeError("Intrabar equity can be exhausted; no realizable PnL without a liquidation model")
        price_pnl += mark_price(float(bar.close))
        position_before_final = held
        if equity<=0:
            raise RuntimeError("Equity exhausted; liquidation model required")
        leverage = abs(quantity)*float(bar.close)/equity
        max_close_leverage = max(max_close_leverage,leverage)
        if i==len(bars)-1 and active is not None and close_final:
            transaction_cost += finish_position(dt+pd.Timedelta("1D")-pd.Timedelta("1ns"),float(bar.close),"final_exit")
        if equity<=0:
            raise RuntimeError("Equity exhausted by final costs")
        rows.append(dict(datetime=dt,equity=equity,quantity=quantity,position=position_before_final,
            funding_cost=funding_cost,transaction_cost=transaction_cost,price_pnl=price_pnl,
            long_pnl=side_total[1]-before_long,short_pnl=side_total[-1]-before_short,
            intrabar_equity_floor=worst,close_leverage_before_final=leverage))
    if active is not None:
        active.update(exit_datetime=bars.index[-1]+pd.Timedelta("1D")-pd.Timedelta("1ns"),
            exit_price=float(bars.close.iloc[-1]),status="open_marked",exit_equity=equity,
            net_pnl=active["price_pnl"]-active["funding_cost"]-active["fees"]-active["slippage"])
        active["net_return_on_entry_equity"] = active["net_pnl"]/active["entry_equity"]
        ledger.append(active)
    curve = pd.DataFrame(rows).set_index("datetime")
    positions = pd.DataFrame(ledger,columns=["position_id","direction","regime","signal_datetime","entry_datetime",
        "entry_price","entry_equity","quantity","price_pnl","funding_cost","fees","slippage","exit_datetime",
        "exit_price","status","exit_equity","net_pnl","net_return_on_entry_equity"])
    order_frame = pd.DataFrame(orders,columns=["datetime","position_id","action","direction","side","quantity",
        "reference_price","notional","fee","slippage","equity_after"])
    total_pnl = equity-initial_cash
    ledger_pnl = float(positions.net_pnl.sum()) if len(positions) else 0.
    np.testing.assert_allclose([side_total[1]+side_total[-1],ledger_pnl,
        curve.price_pnl.sum()-curve.funding_cost.sum()-curve.transaction_cost.sum()],total_pnl,rtol=1e-10,atol=1e-7)
    daily = curve.equity.pct_change(fill_method=None)
    daily.iloc[0] = curve.equity.iloc[0]/initial_cash-1
    drawdown = curve.equity/curve.equity.cummax().clip(lower=initial_cash)-1
    years = len(bars)/365.25
    stats = dict(initial_cash=float(initial_cash),final_equity=float(equity),net_pnl=float(total_pnl),
        total_return=float(equity/initial_cash-1),cagr=float((equity/initial_cash)**(1/years)-1),
        max_drawdown=float(drawdown.min()),
        sharpe=float(daily.mean()/daily.std()*np.sqrt(365.25)) if daily.std()>0 else None,
        long_pnl=float(side_total[1]),short_pnl=float(side_total[-1]),
        gross_price_pnl=float(curve.price_pnl.sum()),transaction_cost=float(curve.transaction_cost.sum()),
        funding_cost=float(curve.funding_cost.sum()),orders=len(orders),positions=len(positions),
        winning_positions=int(positions.net_pnl.gt(0).sum()) if len(positions) else 0,
        win_rate=float(positions.net_pnl.gt(0).mean()) if len(positions) else None,
        exposure=float(curve.position.ne(0).mean()),long_exposure=float(curve.position.gt(0).mean()),
        short_exposure=float(curve.position.lt(0).mean()),min_intrabar_equity=float(min_intrabar_equity),
        max_close_leverage=float(max_close_leverage),ledger_reconciled=True,
        funding_mark_proxy="same day's open when event mark price unavailable",
        final_position="closed at final close with costs" if close_final else "marked at final close",
        sizing="1x post-cost equity at each entry, fixed BTC quantity until exit; reverse closes first",
        liquidation="not modeled; reject runs with a nonpositive daily high/low equity floor")
    return DirectionalResult(curve,order_frame,positions,stats)


def performance(equity, initial_cash=10_000.):
    """Net return, zero-rate daily Sharpe and drawdown including initial cash."""
    values = pd.Series(equity, dtype=float)
    if (values.empty or not np.isfinite(values).all() or not values.gt(0).all()
            or not np.isfinite(initial_cash) or initial_cash <= 0):
        raise ValueError("Performance requires positive finite equity and initial cash")
    daily = values / values.shift(1, fill_value=initial_cash) - 1
    deviation = daily.std(ddof=1)
    peak = values.cummax().clip(lower=initial_cash)
    return {
        "total_return": float(values.iloc[-1] / initial_cash - 1),
        "sharpe": float(daily.mean() / deviation * np.sqrt(365.25)) if deviation > 0 else None,
        "max_drawdown": float((values / peak - 1).min()),
    }
