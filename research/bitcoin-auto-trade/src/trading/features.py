"""시장 국면 판단에 쓰는 가격 및 파생지표 피처."""

import numpy as np
import pandas as pd


FEATURE_COLS = ["cum_return", "volatility", "adx_mean", "r2_mean", "up_candle_ratio"]
def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """방향성 강도를 계산한다. 변동이 없으면 0을 반환한다."""
    high, low = df["high"], df["low"]

    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    overlap = (dm_plus > 0) & (dm_minus > 0)
    plus_wins = dm_plus > dm_minus
    minus_wins = dm_minus > dm_plus
    dm_plus = dm_plus.where(~overlap | plus_wins, 0)
    dm_minus = dm_minus.where(~overlap | minus_wins, 0)

    dp = dm_plus.ewm(span=period, adjust=False).mean()
    dm = dm_minus.ewm(span=period, adjust=False).mean()
    di_sum = (dp + dm).replace(0, np.nan)
    dx = ((dp - dm).abs() / di_sum * 100).fillna(0)
    return dx.ewm(span=period, adjust=False).mean()


def compute_r2(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """로그 종가가 직선 추세에 얼마나 가까운지 계산한다."""
    log_close = np.log(df["close"].clip(lower=1e-10))
    x = np.arange(period, dtype=float)
    x -= x.mean()

    def _r2(y):
        if len(y) < period:
            return np.nan
        y = y - y.mean()
        ss_tot = (y ** 2).sum()
        if ss_tot == 0:
            return 1.0
        slope = (x * y).sum() / (x ** 2).sum()
        ss_res = ((y - slope * x) ** 2).sum()
        return float(1 - ss_res / ss_tot)

    return log_close.rolling(period).apply(_r2, raw=True)


def compute_cum_return(df: pd.DataFrame, window_size: int = 60) -> pd.Series:
    """Compute rolling log return over the same window used by the PV feature."""
    ret = df["close"].pct_change()
    log_ret = np.log1p(ret)
    return log_ret.rolling(window_size, min_periods=window_size).sum()


def compute_window_features(
    df: pd.DataFrame,
    window_size: int = 60,
    adx_period: int = 14,
    r2_period: int = 20,
    use_ewm: bool = True,
    ewm_span: int = 20,
) -> pd.DataFrame:
    """각 봉의 종가까지 관측한 가격 피처 5개를 만든다."""
    df = df.copy().reset_index(drop=True)
    ret = df["close"].pct_change()
    adx = compute_adx(df, period=adx_period)
    r2 = compute_r2(df, period=r2_period)
    up = (df["close"] >= df["open"]).astype(float)

    def _roll_mean(s):
        if use_ewm:
            return s.ewm(span=ewm_span, min_periods=ewm_span // 2, adjust=False).mean()
        return s.rolling(window_size, min_periods=window_size // 2).mean()

    def _roll_std(s):
        if use_ewm:
            return s.ewm(span=ewm_span, min_periods=ewm_span // 2, adjust=False).std()
        return s.rolling(window_size, min_periods=window_size // 2).std()

    cum_return = compute_cum_return(df, window_size=window_size)
    features = pd.DataFrame(
        {
            "cum_return": cum_return,
            "volatility": _roll_std(ret),
            "adx_mean": _roll_mean(adx),
            "r2_mean": _roll_mean(r2),
            "up_candle_ratio": _roll_mean(up),
        },
        index=df.index,
    )
    if "datetime" in df.columns:
        features.insert(0, "datetime", df["datetime"])
    return features
