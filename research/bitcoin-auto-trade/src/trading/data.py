"""Strict UTC bar loading and availability-aware daily derivative alignment."""

from pathlib import Path

import numpy as np
import pandas as pd


def timestamps(values):
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True, errors="raise")
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        magnitude = numeric.abs().median()
        unit = "ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14 else "ms" if magnitude >= 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="raise")
    return pd.to_datetime(values, utc=True, errors="raise", format="mixed")


def validate_funding_coverage(funding,start,end_exclusive):
    """Require complete event coverage while allowing declared interval changes.

    Official archives can record the new interval on either side of a change.
    Consequently a boundary day's sum of row intervals need not be 24 hours.
    Stable intervals must still have every event; no rates are synthesized.
    """
    start,end=pd.Timestamp(start),pd.Timestamp(end_exclusive)
    start=start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end=end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start!=start.normalize() or end!=end.normalize() or start>=end:
        raise ValueError("Funding coverage boundaries must be complete UTC days")
    times=pd.DatetimeIndex(timestamps(funding.datetime))
    hours=pd.to_numeric(funding.funding_interval_hours,errors="raise").to_numpy(dtype=float)
    if (not len(times) or not times.is_unique or not times.is_monotonic_increasing
            or not np.isfinite(hours).all() or not (hours>0).all()):
        raise ValueError("Invalid funding coverage events or intervals")
    tolerance=pd.Timedelta(seconds=1)
    if times[0]>start+tolerance or times[-1]+pd.Timedelta(hours=float(hours[-1]))<end-tolerance:
        raise ValueError("Incomplete daily funding coverage at the requested boundaries")
    gaps=np.diff(times.asi8)/1e9
    changed=hours[1:]!=hours[:-1]
    matches_current=np.isclose(gaps,hours[1:]*3600,rtol=0,atol=1.)
    matches_previous=changed&np.isclose(gaps,hours[:-1]*3600,rtol=0,atol=1.)
    relevant=(times[1:]>start+tolerance)&(times[:-1]<end-tolerance)
    if (~(matches_current|matches_previous)&relevant).any():
        bad=times[1:][~(matches_current|matches_previous)&relevant]
        raise ValueError(f"Incomplete daily funding coverage: event gaps before {list(map(str,bad[:5]))}")
    selected=(times>=start)&(times<end)
    sums=pd.Series(hours[selected],index=times[selected].floor("D")).groupby(level=0).sum()
    exceptions={str(day.date()):float(value) for day,value in sums.items() if not np.isclose(value,24.)}
    return dict(start=str(start),end_exclusive=str(end),events=int(selected.sum()),
        interval_changes=int((changed&relevant).sum()),interval_sum_exception_days=exceptions,
        timestamp_tolerance_seconds=1.,
        gap_rule="Each consecutive event gap matches its stated interval; at declared interval changes, either adjacent interval is accepted. Boundary coverage is required; missing events are never filled.")


def load_bars(path: str | Path, timeframe="1D"):
    step = pd.Timedelta(timeframe)
    if step <= pd.Timedelta(0):
        raise ValueError("timeframe must be positive")
    frame = pd.read_csv(path).rename(columns=str.lower)
    key = next((c for c in ("datetime", "open_time", "timestamp", "date") if c in frame), None)
    if key is None:
        raise ValueError("CSV requires datetime/open_time/timestamp/date")
    frame.index = pd.DatetimeIndex(timestamps(frame[key]), name="datetime")
    frame = frame.sort_index()
    if frame.empty or not frame.index.is_unique:
        raise ValueError("Empty or duplicate candle timestamps")
    cols = ["open", "high", "low", "close", "volume"]
    frame[cols] = frame[cols].apply(pd.to_numeric, errors="raise")
    a = frame[cols]
    if not np.isfinite(a).all().all() or (a[["open", "high", "low", "close"]] <= 0).any().any() or (a.volume < 0).any():
        raise ValueError("Invalid OHLCV values")
    if ((a.high < a[["open", "close", "low"]].max(axis=1)) | (a.low > a[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC price bounds")
    if (frame.index != frame.index.floor(step)).any() or not (frame.index.to_series().diff().dropna() == step).all():
        raise ValueError("Candles must match timeframe with no missing bars; resample explicitly first")
    if (frame.index + step > pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("CSV contains unclosed/future bars")
    return frame[cols + [c for c in ("quote_volume", "taker_buy_volume", "taker_buy_quote_volume") if c in frame]]


def align_daily(bars, source, value):
    """Join a daily value available at each candle close without stale carry."""
    frame = source[["available_at", value]].copy()
    frame["available_at"] = timestamps(frame.available_at)
    # Reindexed missing days already invalidate rolling features; they cannot
    # participate in the as-of join because their publication time is unknown.
    frame = frame.dropna(subset=["available_at"])
    if not frame.available_at.is_unique:
        raise ValueError("Duplicate daily availability timestamps")
    decision = pd.DataFrame({"decision_time": bars.index+pd.Timedelta("1D")})
    joined = pd.merge_asof(decision, frame.sort_values("available_at"), left_on="decision_time",
                           right_on="available_at", direction="backward", tolerance=pd.Timedelta("26h"))
    return pd.Series(joined[value].to_numpy(dtype=float), index=bars.index, name=value)


def daily_source(path):
    """Load daily observations, preserving missing calendar days as gaps."""
    frame = pd.read_csv(path)
    frame.index = pd.DatetimeIndex(timestamps(frame.datetime))
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Daily derivative dates must be sorted and unique")
    return frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="D"))


def derivative_features(bars, metrics_path, funding_path, timeframe="1D"):
    """Inputs use actual availability timestamps; no backward fill or stale carry."""
    decision = pd.DataFrame({"decision_time": bars.index + pd.Timedelta(timeframe)})
    result = pd.DataFrame(index=bars.index)
    for path, columns in ((metrics_path, ["oi_pct_change", "top_ls", "taker_ls", "account_ls"]),
                          (funding_path, ["funding_mean"])):
        if not path:
            continue
        source = pd.read_csv(path)
        source["available_at"] = timestamps(source["available_at"])
        source = source.sort_values("available_at")
        if not source.available_at.is_unique:
            raise ValueError("Duplicate derivative availability timestamps")
        source[columns] = source[columns].apply(pd.to_numeric, errors="raise")
        joined = pd.merge_asof(decision, source[["available_at", *columns]],
                               left_on="decision_time", right_on="available_at",
                               direction="backward", tolerance=pd.Timedelta("26h"))
        for col in columns:
            result[col] = joined[col].to_numpy()
    if result.empty:
        raise ValueError("deriv mode requires metrics and/or funding daily CSV")
    return result
