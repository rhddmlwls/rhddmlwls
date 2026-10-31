"""Causal technical-indicator targets observed at a bar's close.

These functions produce targets, never executions. The portfolio engine must
execute the resulting target at the following open. All rolling windows end at
the current close, except the explicitly previous-band breakout comparison.
"""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


_DEFAULTS = {
    "ema_cross": dict(fast=10, slow=30, trend_filter=200, direction="long"),
    "donchian": dict(entry_period=20, exit_period=10, direction="long"),
    "rsi_reversion": dict(period=14, lower=30., exit_level=50., trend_filter=0),
    "rsi_trend": dict(period=14, entry_level=55., exit_level=45., trend_filter=0, short=False),
    "macd": dict(fast=12, slow=26, signal=9, trend_filter=0, short=False),
    "momentum": dict(lookback=30, entry_threshold=0., exit_threshold=0., trend_filter=0, short=False),
    "pullback": dict(period=14, lower=40., exit_level=60., fast=10, slow=30, trend_filter=200),
    "bollinger_reversion": dict(window=20, multiplier=2., trend_filter=0),
    "bollinger_breakout": dict(window=20, multiplier=2., trend_filter=0),
    "sma_cross": dict(fast=20, slow=100, trend_filter=0, short=False),
    "rolling_vwap": dict(window=20, trend_filter=0, short=False),
}

TECHNICAL_KINDS = frozenset(_DEFAULTS)
_DIRECTIONAL_KINDS = frozenset({"ema_cross", "donchian", "macd", "rolling_vwap"})


def validate_technical_params(kind: str, params: Mapping | None = None) -> dict:
    """Return a fresh defaults-expanded parameter dictionary or raise.

    RSI thresholds are index points, momentum thresholds are decimal returns
    (0.05 means 5%), and a trend_filter of zero disables the SMA price filter.
    """
    if kind not in TECHNICAL_KINDS:
        raise ValueError(f"Unknown technical strategy: {kind}")
    if params is not None and not isinstance(params, Mapping):
        raise ValueError("Technical strategy params must be a mapping")
    supplied = dict(params or {})
    optional = {"direction"} if kind in _DIRECTIONAL_KINDS else set()
    unknown = set(supplied) - set(_DEFAULTS[kind]) - optional
    if unknown:
        raise ValueError(f"Unknown {kind} parameter(s): {sorted(unknown)}")
    values = dict(_DEFAULTS[kind], **supplied)
    for name in ("period", "fast", "slow", "signal", "lookback", "window", "trend_filter", "entry_period", "exit_period"):
        if name in values:
            value = values[name]
            minimum = 0 if name == "trend_filter" else 1
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{kind}.{name} must be an integer >= {minimum}")
            values[name] = int(value)
    for name in ("lower", "exit_level", "entry_level", "entry_threshold", "exit_threshold", "multiplier"):
        if name in values:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{kind}.{name} must be finite and numeric")
            values[name] = float(value)
    if "short" in values and not isinstance(values["short"], bool):
        raise ValueError(f"{kind}.short must be a boolean")
    if "direction" in values:
        if not isinstance(values["direction"], str) or values["direction"] not in {"long", "short"}:
            raise ValueError(f"{kind}.direction must be 'long' or 'short'")
        if values.get("short", False):
            raise ValueError(f"{kind}.direction cannot be combined with short=true")
    if "fast" in values and values["fast"] >= values["slow"]:
        raise ValueError(f"{kind}.fast must be less than slow")
    if kind in {"rsi_reversion", "pullback"}:
        if not 0 <= values["lower"] < values["exit_level"] <= 100:
            raise ValueError(f"{kind} requires 0 <= lower < exit_level <= 100")
    if kind == "rsi_trend":
        if not 50 < values["entry_level"] <= 100 or not 0 <= values["exit_level"] < values["entry_level"]:
            raise ValueError("rsi_trend requires 50 < entry_level <= 100 and 0 <= exit_level < entry_level")
    if kind == "momentum":
        if values["entry_threshold"] < 0 or values["exit_threshold"] > values["entry_threshold"]:
            raise ValueError("momentum requires entry_threshold >= 0 and exit_threshold <= entry_threshold")
    if "multiplier" in values and values["multiplier"] <= 0:
        raise ValueError(f"{kind}.multiplier must be positive")
    return values


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI with an initial SMA of ``period`` price changes.

    Subsequent averages use (previous * (period - 1) + change) / period.
    A constant-price window has RSI 50; a gain-only window has RSI 100.
    A missing/nonfinite close resets the averages and requires a fresh complete
    warm-up. The first output needs period + 1 consecutive closes.
    """
    if isinstance(period, bool) or not isinstance(period, Integral) or period < 1:
        raise ValueError("RSI period must be an integer >= 1")
    close = pd.Series(close, copy=False)
    prices = close.to_numpy(dtype=float)
    result = np.full(len(prices), np.nan, dtype=float)
    previous = np.nan
    gain_sum = loss_sum = 0.
    average_gain = average_loss = 0.
    changes = 0
    for i, price in enumerate(prices):
        if not np.isfinite(price):
            previous = np.nan
            gain_sum = loss_sum = 0.
            changes = 0
            continue
        if not np.isfinite(previous):
            previous = price
            continue
        change = price - previous
        previous = price
        gain, loss = max(change, 0.), max(-change, 0.)
        if changes < period:
            gain_sum += gain
            loss_sum += loss
            changes += 1
            if changes < period:
                continue
            average_gain, average_loss = gain_sum / period, loss_sum / period
        else:
            average_gain = (average_gain * (period - 1) + gain) / period
            average_loss = (average_loss * (period - 1) + loss) / period
        if average_loss == 0:
            result[i] = 50. if average_gain == 0 else 100.
        else:
            result[i] = 100. - 100. / (1. + average_gain / average_loss)
    return pd.Series(result, index=close.index, name="rsi")


def _hold(index, ready, enter, leave, short_enter=None, short_leave=None):
    """Hysteresis with explicit exits; missing indicators reset to cash."""
    values = np.zeros(len(index), dtype=np.int8)
    state = 0
    for i in range(len(index)):
        if not ready[i]:
            state = 0
            continue
        if state == 1 and leave[i]:
            state = 0
        elif state == -1 and short_leave[i]:
            state = 0
        if state == 0:
            if enter[i]:
                state = 1
            elif short_enter is not None and short_enter[i]:
                state = -1
        values[i] = state
    return pd.Series(values, index=index, name="target")


def _legacy_directional_signal(bars, kind, params):
    """Extend the existing EMA/Donchian rules with a mirrored short direction.

    EMA's SMA filter gates entry only, as in ``strategies.ema_cross``. It does
    not force an exit below/above the filter. Equality holds the current state.
    The shared legacy holding helper retains internal state over unavailable
    indicators and emits flat there. No execution or regime reset occurs here.
    """
    from .strategies import _holdings

    close = bars["close"].astype(float)
    is_short = params["direction"] == "short"
    if kind == "ema_cross":
        fast = close.ewm(span=params["fast"], adjust=False).mean().to_numpy(float)
        slow = close.ewm(span=params["slow"], adjust=False).mean().to_numpy(float)
        ready = np.isfinite(close.to_numpy()) & np.isfinite(fast) & np.isfinite(slow)
        if params["trend_filter"]:
            average = close.rolling(params["trend_filter"]).mean().to_numpy(float)
            ready &= np.isfinite(average)
            allowed = close.to_numpy() < average if is_short else close.to_numpy() > average
        else:
            allowed = np.ones(len(close), dtype=bool)
        enter = (fast < slow if is_short else fast > slow) & allowed
        leave = fast > slow if is_short else fast < slow
    else:
        if not {"high", "low"}.issubset(bars.columns):
            raise ValueError("donchian requires high and low columns")
        if is_short:
            entry = bars.low.rolling(params["entry_period"]).min().shift(1).to_numpy(float)
            exit_ = bars.high.rolling(params["exit_period"]).max().shift(1).to_numpy(float)
            enter, leave = close.to_numpy() < entry, close.to_numpy() > exit_
        else:
            entry = bars.high.rolling(params["entry_period"]).max().shift(1).to_numpy(float)
            exit_ = bars.low.rolling(params["exit_period"]).min().shift(1).to_numpy(float)
            enter, leave = close.to_numpy() > entry, close.to_numpy() < exit_
        ready = np.isfinite(entry) & np.isfinite(exit_) & np.isfinite(close.to_numpy())
    values = _holdings(ready, enter, leave)
    return pd.Series(-values if is_short else values, index=bars.index, name="target")


def technical_signal(bars: pd.DataFrame, kind: str, params: Mapping | None = None) -> pd.Series:
    """Return an int8 -1/0/1 close target, with no execution-time shift.

    MACD EMAs use adjust=False and wait for their span observations; its signal
    EMA waits for its own span of available MACD values. Legacy EMA-cross has
    no extra EMA-span warm-up beyond its optional SMA entry filter. Bollinger bands use
    population standard deviation (ddof=0). Reversion uses the current band;
    breakout compares the close against the *previous* upper band and exits
    below the current mean. Rolling VWAP is sum(actual quote volume) divided
    by sum(actual base volume) over daily bars, not an intraday VWAP or a
    typical-price approximation. Zero-volume windows emit flat.

    Optional ``direction`` on EMA, Donchian, MACD and rolling VWAP requests a
    long-only or short-only signal. Legacy ``short=true`` remains bidirectional
    when direction is omitted. New EMA/Donchian reuse the legacy entry/exit
    rules, including EMA's entry-only price filter, with mirrored short tests.

    Stateful long entries/exits use strict thresholds except momentum's exit
    (<= threshold), and Bollinger reversion's exit (>= mean). Optional shorts
    mirror the long rule. A trend SMA requires price above it for longs and
    below it for shorts, and closes a position when that condition fails.
    """
    p = validate_technical_params(kind, params)
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Technical strategy price history must have a sorted, unique index")
    if "close" not in bars:
        raise ValueError("Technical strategy requires close prices")
    if kind in {"ema_cross", "donchian"}:
        return _legacy_directional_signal(bars, kind, p)
    close = bars["close"].astype(float)
    finite = np.isfinite(close.to_numpy())
    ready = finite.copy()
    trend = p["trend_filter"]
    if trend:
        average = close.rolling(trend, min_periods=trend).mean()
        ready &= np.isfinite(average.to_numpy())
        long_allowed = (close > average).to_numpy()
        short_allowed = (close < average).to_numpy()
    else:
        long_allowed = short_allowed = np.ones(len(close), dtype=bool)
    direction = p.get("direction")
    short = p.get("short", False) or direction == "short"
    short_enter = short_leave = None

    if kind in {"rsi_reversion", "rsi_trend", "pullback"}:
        rsi = rsi_wilder(close, p["period"])
        ready &= np.isfinite(rsi.to_numpy())
        if kind in {"rsi_reversion", "pullback"}:
            enter = (rsi < p["lower"]).to_numpy()
            leave = (rsi > p["exit_level"]).to_numpy()
            if kind == "pullback":
                fast = close.ewm(span=p["fast"], adjust=False, min_periods=p["fast"]).mean()
                slow = close.ewm(span=p["slow"], adjust=False, min_periods=p["slow"]).mean()
                ready &= np.isfinite(fast.to_numpy()) & np.isfinite(slow.to_numpy())
                enter &= (fast > slow).to_numpy()
        else:
            enter = (rsi > p["entry_level"]).to_numpy()
            leave = (rsi < p["exit_level"]).to_numpy()
            if short:
                short_enter = (rsi < 100 - p["entry_level"]).to_numpy()
                short_leave = (rsi > 100 - p["exit_level"]).to_numpy()
    elif kind == "momentum":
        momentum = close / close.shift(p["lookback"]) - 1
        ready &= np.isfinite(momentum.to_numpy())
        enter = (momentum > p["entry_threshold"]).to_numpy()
        leave = (momentum <= p["exit_threshold"]).to_numpy()
        if short:
            short_enter = (momentum < -p["entry_threshold"]).to_numpy()
            short_leave = (momentum >= -p["exit_threshold"]).to_numpy()
    elif kind in {"macd", "sma_cross"}:
        if kind == "macd":
            fast = close.ewm(span=p["fast"], adjust=False, min_periods=p["fast"]).mean()
            slow = close.ewm(span=p["slow"], adjust=False, min_periods=p["slow"]).mean()
            line = fast - slow
            other = line.ewm(span=p["signal"], adjust=False, min_periods=p["signal"]).mean()
        else:
            line = close.rolling(p["fast"], min_periods=p["fast"]).mean()
            other = close.rolling(p["slow"], min_periods=p["slow"]).mean()
        ready &= np.isfinite(line.to_numpy()) & np.isfinite(other.to_numpy())
        enter = (line > other).to_numpy()
        leave = (line <= other).to_numpy()
        if short:
            short_enter = (line < other).to_numpy()
            short_leave = (line >= other).to_numpy()
    elif kind.startswith("bollinger_"):
        mean = close.rolling(p["window"], min_periods=p["window"]).mean()
        deviation = close.rolling(p["window"], min_periods=p["window"]).std(ddof=0)
        if kind == "bollinger_reversion":
            bound = mean - p["multiplier"] * deviation
            enter = (close < bound).to_numpy()
            leave = (close >= mean).to_numpy()
        else:
            bound = (mean + p["multiplier"] * deviation).shift(1)
            enter = (close > bound).to_numpy()
            leave = (close < mean).to_numpy()
        ready &= np.isfinite(mean.to_numpy()) & np.isfinite(bound.to_numpy())
    else:  # rolling_vwap
        if not {"volume", "quote_volume"}.issubset(bars.columns):
            raise ValueError("rolling_vwap requires actual volume and quote_volume columns")
        volume = bars["volume"].astype(float)
        quote = bars["quote_volume"].astype(float)
        if (volume < 0).any() or (quote < 0).any():
            raise ValueError("rolling_vwap volumes must be nonnegative")
        base_sum = volume.rolling(p["window"], min_periods=p["window"]).sum()
        quote_sum = quote.rolling(p["window"], min_periods=p["window"]).sum()
        vwap = quote_sum / base_sum.where(base_sum > 0)
        ready &= np.isfinite(vwap.to_numpy())
        enter = (close > vwap).to_numpy()
        leave = (close <= vwap).to_numpy()
        if short:
            short_enter = (close < vwap).to_numpy()
            short_leave = (close >= vwap).to_numpy()

    enter &= long_allowed
    leave |= ~long_allowed
    if short:
        short_enter &= short_allowed
        short_leave |= ~short_allowed
    result = _hold(bars.index, ready, enter, leave, short_enter, short_leave)
    if direction == "short":
        result = result.clip(upper=0)
    return result
