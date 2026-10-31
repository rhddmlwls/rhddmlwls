"""Gate existing close signals by regime; execution remains the engine's job."""

import numpy as np
import pandas as pd

from .strategies import donchian, ema_cross


STRATEGY_IDS = frozenset({"long", "short", "flat", "ema", "donchian", "williams"})


def build_strategy_targets(bars: pd.DataFrame, regimes: pd.Series, profile, registered_signals=None) -> pd.Series:
    """Return -1/0/1 close targets on ``regimes.index`` without shifting.

    EMA and Donchian are the existing long/flat strategies. Their indicators and
    internal position state run continuously over the supplied price history;
    regime gates only determine whether that signal is eligible at each close.
    A gate opening can therefore enter an already-active strategy signal.
    Unavailable regimes or dates without a price bar remain flat.
    """
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("Strategy price history must have a sorted, unique index")
    if not isinstance(regimes, pd.Series):
        regimes = pd.Series(regimes, index=bars.index)
    if not regimes.index.is_unique:
        raise ValueError("Strategy regimes must have a unique index")
    choices = {0: profile.bear, 1: profile.side, 2: profile.bull}
    if "williams" in choices.values():
        raise ValueError("Williams uses intraday setups, not close targets; use build_williams_setups")
    registered_signals=registered_signals or {}
    unknown = set(choices.values()) - STRATEGY_IDS - set(registered_signals)
    if unknown:
        raise ValueError(f"Unknown regime strategy: {sorted(unknown)}")

    signals = {}
    for strategy in set(choices.values()):
        if strategy in registered_signals:
            values=registered_signals[strategy]
            if not isinstance(values,pd.Series) or not values.index.equals(bars.index) or not values.isin([-1,0,1]).all():
                raise ValueError("Registered signals need aligned finite -1/0/1 close targets")
        elif strategy == "ema":
            values = ema_cross(
                bars,
                fast_ema=getattr(profile, "ema_fast", 10),
                slow_ema=getattr(profile, "ema_slow", 30),
                ma_filter=getattr(profile, "ema_filter", 200),
            )
        elif strategy == "donchian":
            values = donchian(
                bars,
                entry_period=getattr(profile, "donchian_entry", 20),
                exit_period=getattr(profile, "donchian_exit", 10),
            )
        else:
            values = {"long": 1, "short": -1, "flat": 0}[strategy]
        signals[strategy] = pd.Series(values, index=bars.index).reindex(regimes.index).fillna(0)

    targets = np.zeros(len(regimes), dtype=np.int8)
    for label, strategy in choices.items():
        eligible = regimes.eq(label).fillna(False).to_numpy(dtype=bool)
        targets[eligible] = signals[strategy].to_numpy(dtype=np.int8)[eligible]
    return pd.Series(targets, index=regimes.index, name="target")
