"""Observation feature registry; dimensions and windows come from settings."""
import numpy as np
import pandas as pd

from .data import align_daily


def compute_feature(bars,kind,window,funding=None,metrics=None):
    """Compute one registered observation from the current close and its past."""
    close=bars.close
    if kind=="log_return":
        return np.log(close/close.shift(window))
    if kind=="efficiency":
        path=close.diff().abs().rolling(window,min_periods=window).sum()
        return (close.diff(window).abs()/path.replace(0.,np.nan)).mask(path.eq(0.),0.).clip(0,1)
    if kind=="taker_imbalance":
        volume=bars.volume.rolling(window,min_periods=window).sum()
        return 2*bars.taker_buy_volume.rolling(window,min_periods=window).sum()/volume.replace(0.,np.nan)-1
    if kind=="volatility":
        return np.log(close).diff().rolling(window,min_periods=window).std(ddof=0)
    if kind=="volume_z":
        mean=bars.volume.rolling(window,min_periods=window).mean()
        std=bars.volume.rolling(window,min_periods=window).std(ddof=0)
        return ((bars.volume-mean)/std.replace(0.,np.nan)).mask(std.eq(0.),0.)
    if kind=="funding_mean":
        if funding is None:
            raise ValueError("funding_mean needs daily funding data")
        frame=funding.copy()
        frame["derived"]=pd.to_numeric(frame.funding_mean).rolling(window,min_periods=window).mean()
        return align_daily(bars,frame,"derived")
    if kind=="oi_change":
        if metrics is None:
            raise ValueError("oi_change needs daily OI data")
        frame=metrics.copy()
        oi=pd.to_numeric(frame.oi)
        oi=oi.where(np.isfinite(oi)&oi.gt(0))
        frame["derived"]=(oi/oi.shift(window)-1).where(oi.rolling(window+1,min_periods=window+1).count().eq(window+1))
        return align_daily(bars,frame,"derived")
    raise ValueError(f"Unknown feature kind: {kind}")


def build_feature_set(bars,spec,funding=None,metrics=None):
    return pd.DataFrame({f"{f.kind}_{f.window}":compute_feature(bars,f.kind,f.window,funding,metrics)
                         for f in spec.features},index=bars.index)
