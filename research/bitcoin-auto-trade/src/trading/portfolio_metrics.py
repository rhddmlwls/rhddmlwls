"""Supplementary daily-equity metrics, independent of the cached simulation."""
import numpy as np


METRIC_DEFINITIONS = dict(
    sharpe="Mean daily net equity return / sample standard deviation * sqrt(365.25); risk-free rate 0; includes flat days.",
    sharpe_zero="Original Sharpe with risk-free rate 0, includes initial flat day; annualized with sqrt(365.25).",
    sharpe_btc="Mean(strategy close-to-close return - BTC close-to-close price return) / sample standard deviation of that difference * sqrt(365.25). Also called information ratio. Identical dates; omit first close for both; BTC prices exclude fees and funding.",
    btc_annualized_excess_mean="Arithmetic mean of daily strategy-minus-BTC returns * 365.25; not the difference of CAGRs or cumulative returns.",
    btc_tracking_error="Sample standard deviation of daily strategy-minus-BTC returns * sqrt(365.25).",
    max_drawdown="Minimum daily-close equity / prior daily-close high-water mark including initial cash - 1; excludes intraday drawdowns.",
    calmar="Net CAGR / absolute daily-close maximum drawdown; undefined when maximum drawdown is zero.",
    annual_turnover_gross="Sum of absolute executed notional / equity immediately before each order, divided by evaluation days/365.25. Entry and exit both count; reversals count both legs; no division by two; includes final close.",
    annual_entries="Number of entry orders / (evaluation days/365.25); a reversal starts a new position.",
    annual_volatility="Sample standard deviation of daily net equity returns * sqrt(365.25), including flat days.",
    sortino="Mean daily net return / sqrt(mean(min(daily net return,0)^2)) * sqrt(365.25); target return 0; undefined with no downside.",
    max_drawdown_duration_days="Longest consecutive daily closes strictly below the high-water mark; includes an unfinished terminal drawdown.")


def benchmark_relative_metrics(curve, prices):
    """Subtract matched daily BTC returns before computing mean and volatility."""
    if not curve.index.equals(prices.index) or len(curve)<3:
        raise ValueError("Benchmark needs identical closing dates and at least three closes")
    equity=curve.equity.to_numpy(dtype=float)
    price=prices.to_numpy(dtype=float)
    if not np.isfinite(equity).all() or not np.isfinite(price).all() or min(equity.min(),price.min())<=0:
        raise ValueError("Benchmark needs finite positive equity and prices")
    strategy=equity[1:]/equity[:-1]-1
    benchmark=price[1:]/price[:-1]-1
    active=strategy-benchmark
    active_std=active.std(ddof=1)
    ratio=float(active.mean()/active_std*np.sqrt(365.25)) if active_std>10*np.finfo(float).eps else None
    return dict(sharpe_btc=ratio,btc_annualized_excess_mean=float(active.mean()*365.25),
        btc_tracking_error=float(active_std*np.sqrt(365.25)),btc_return_observations=len(active))


def additional_metrics(curve, orders, initial_cash):
    equity=curve.equity.to_numpy(dtype=float)
    if not len(equity) or not np.isfinite(equity).all() or (equity<=0).any() or initial_cash<=0:
        raise ValueError("Metrics require finite positive equity and initial cash")
    years=len(equity)/365.25
    daily=equity/np.r_[initial_cash,equity[:-1]]-1
    peak=np.maximum.accumulate(np.r_[initial_cash,equity])[1:]
    drawdown=equity/peak-1
    cagr=(equity[-1]/initial_cash)**(1/years)-1
    downside=np.sqrt(np.mean(np.minimum(daily,0.)**2))
    longest=current=0
    for underwater in drawdown<0:
        current=current+1 if underwater else 0
        longest=max(longest,current)
    turnover=0.
    entries=0
    if len(orders):
        # Saved equity_after is after this order's fees and slippage only.
        before=orders.equity_after+orders.fee+orders.slippage
        if not np.isfinite(before).all() or not before.gt(0).all():
            raise ValueError("Turnover requires positive pre-order equity")
        turnover=float((orders.notional.abs()/before).sum())
        entries=int(orders.action.eq("entry").sum())
    return dict(annual_turnover_gross=turnover/years,total_turnover_gross=turnover,
        annual_entries=entries/years,annual_volatility=float(np.std(daily,ddof=1)*np.sqrt(365.25)) if len(daily)>1 else None,
        sortino=float(daily.mean()/downside*np.sqrt(365.25)) if downside>0 else None,
        calmar=float(cagr/-drawdown.min()) if drawdown.min()<0 else None,
        max_drawdown_duration_days=longest,drawdown_open_at_end=bool(drawdown[-1]<0))
