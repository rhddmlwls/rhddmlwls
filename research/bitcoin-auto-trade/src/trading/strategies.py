"""각 봉의 목표 보유 상태를 반환한다: 1=BTC 보유, 0=현금."""

import numpy as np
import pandas as pd


def _holdings(ready, enter, exit):
    """Keep position state across unavailable indicators, emitting flat there."""
    signals = np.zeros(len(ready), dtype=np.int8)
    position = 0
    for i in range(len(ready)):
        if not ready[i]:
            continue
        if position == 0 and enter[i]:
            position = 1
        elif position == 1 and exit[i]:
            position = 0
        signals[i] = position
    return signals


def ema_cross(df: pd.DataFrame, fast_ema=10, slow_ema=30, ma_filter=600) -> np.ndarray:
    """단기 EMA가 장기 EMA보다 높고 가격이 MA 위일 때 진입, 역전되면 청산."""
    close = df["close"]
    fast = close.ewm(span=fast_ema, adjust=False).mean().to_numpy(dtype=float)
    slow = close.ewm(span=slow_ema, adjust=False).mean().to_numpy(dtype=float)
    average = close.rolling(ma_filter).mean().to_numpy(dtype=float)
    prices = close.to_numpy(dtype=float)
    return _holdings(~np.isnan(average), (fast > slow) & (prices > average), fast < slow)


def donchian(df: pd.DataFrame, entry_period=120, exit_period=60) -> np.ndarray:
    """직전 봉까지의 최고가 돌파 시 진입, 최저가 이탈 시 청산."""
    # shift(1)은 현재 봉의 고가·저가가 비교 기준에 섞이지 않게 한다.
    upper = df["high"].rolling(entry_period).max().shift(1).to_numpy(dtype=float)
    lower = df["low"].rolling(exit_period).min().shift(1).to_numpy(dtype=float)
    prices = df["close"].to_numpy(dtype=float)
    return _holdings(~(np.isnan(upper) | np.isnan(lower)), prices > upper, prices < lower)
