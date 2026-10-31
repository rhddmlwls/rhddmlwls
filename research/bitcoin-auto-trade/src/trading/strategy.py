"""MACD close signals and their fixed assignment to HMM market regimes."""
import numpy as np
import pandas as pd


def macd_target(close, fast=16, slow=35, signal=9, trend_filter=0, direction="long"):
    """Return close targets; the backtest executes them at the next open.

    MACD EMAs wait for their span observations. Longs require MACD above its
    signal EMA, shorts below it; an optional SMA price filter gates both entry
    and holding. Equality or unavailable indicators mean cash.
    """
    for name, value in {"fast": fast, "slow": slow, "signal": signal, "trend_filter": trend_filter}.items():
        if type(value) is not int or value < (0 if name == "trend_filter" else 1):
            raise ValueError(f"Invalid MACD {name}")
    if fast >= slow or direction not in {"long", "short"}:
        raise ValueError("MACD requires fast < slow and a long/short direction")
    close = close.astype(float)
    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    ready = np.isfinite(close) & np.isfinite(line) & np.isfinite(signal_line)
    direction_sign = 1 if direction == "long" else -1
    active = line.gt(signal_line) if direction_sign == 1 else line.lt(signal_line)
    if trend_filter:
        average = close.rolling(trend_filter, min_periods=trend_filter).mean()
        ready &= np.isfinite(average)
        active &= close.gt(average) if direction_sign == 1 else close.lt(average)
    return (direction_sign * (ready & active)).astype(np.int8).rename("target")


def build_targets(bars, regimes, settings):
    """Apply the configured rule for each close's bear(0), side(1), bull(2).

    Indicators run over all supplied past prices before regime gating. Changing
    regime therefore does not restart MACD or its warm-up. Unknown regimes
    remain cash. The long-hold benchmark is an independent all-ones target.
    """
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Strategy prices must have sorted unique dates")
    if not isinstance(regimes, pd.Series):
        regimes = pd.Series(regimes, index=bars.index)
    if not regimes.index.is_unique or not regimes.index.isin(bars.index).all():
        raise ValueError("Regimes must have unique dates in the price history")
    parameters = {key: settings.get(key, default) for key, default in
                  {"fast": 16, "slow": 35, "signal": 9}.items()}
    signals = {
        "long": pd.Series(1, index=bars.index, dtype=np.int8),
        "short": pd.Series(-1, index=bars.index, dtype=np.int8),
        "flat": pd.Series(0, index=bars.index, dtype=np.int8),
        "macd_long": macd_target(bars.close, **parameters, trend_filter=settings.get("trend_filter", 200)),
        "macd_short": macd_target(bars.close, **parameters, direction="short"),
    }
    choices = [settings.get("bear", "macd_short"), settings.get("side", "macd_long"), settings.get("bull", "long")]
    if any(choice not in signals for choice in choices):
        raise ValueError("Unknown regime strategy")
    target = pd.Series(0, index=regimes.index, dtype=np.int8, name="target")
    for regime, choice in enumerate(choices):
        eligible = regimes.eq(regime)
        target.loc[eligible] = signals[choice].reindex(regimes.index).loc[eligible]
    return target
