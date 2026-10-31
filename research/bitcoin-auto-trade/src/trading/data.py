"""Load UTC market data and calculate the five past-only HMM observations."""
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


def validate_funding_coverage(funding, start, end_exclusive):
    """Require actual events; allow either interval at a declared interval change."""
    start, end = pd.Timestamp(start), pd.Timestamp(end_exclusive)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start != start.normalize() or end != end.normalize() or start >= end:
        raise ValueError("Funding coverage boundaries must be complete UTC days")
    times = pd.DatetimeIndex(timestamps(funding.datetime))
    hours = pd.to_numeric(funding.funding_interval_hours, errors="raise").to_numpy(dtype=float)
    if (not len(times) or not times.is_unique or not times.is_monotonic_increasing
            or not np.isfinite(hours).all() or not (hours > 0).all()):
        raise ValueError("Invalid funding coverage events or intervals")
    tolerance = pd.Timedelta(seconds=1)
    if times[0] > start + tolerance or times[-1] + pd.Timedelta(hours=float(hours[-1])) < end - tolerance:
        raise ValueError("Incomplete daily funding coverage at the requested boundaries")
    gaps = np.diff(times.asi8) / 1e9
    changed = hours[1:] != hours[:-1]
    current = np.isclose(gaps, hours[1:] * 3600, rtol=0, atol=1.)
    previous = changed & np.isclose(gaps, hours[:-1] * 3600, rtol=0, atol=1.)
    relevant = (times[1:] > start + tolerance) & (times[:-1] < end - tolerance)
    if (~(current | previous) & relevant).any():
        raise ValueError("Incomplete daily funding coverage: missing event(s)")
    return {"events": int(((times >= start) & (times < end)).sum()),
            "interval_changes": int((changed & relevant).sum())}


def load_inputs(data_dir, symbol, start, end):
    """Validate all input files and the complete requested evaluation period."""
    folder = Path(data_dir)
    bars = pd.read_csv(folder / f"{symbol}_1d.csv").rename(columns=str.lower)
    key = next((name for name in ("datetime", "open_time", "timestamp", "date") if name in bars), None)
    if key is None:
        raise ValueError("Price CSV requires a datetime column")
    bars.index = pd.DatetimeIndex(timestamps(bars[key]), name="datetime")
    bars = bars.sort_index()
    columns = ["open", "high", "low", "close", "volume", "taker_buy_volume"]
    bars = bars[columns].apply(pd.to_numeric, errors="raise")
    prices = bars[["open", "high", "low", "close"]]
    if bars.empty or not bars.index.is_unique or not np.isfinite(bars).all().all():
        raise ValueError("Empty, duplicate or nonfinite price data")
    if (prices <= 0).any().any() or (bars.volume < 0).any():
        raise ValueError("Invalid OHLCV values")
    if (bars.high < prices.max(axis=1)).any() or (bars.low > prices.min(axis=1)).any():
        raise ValueError("Invalid OHLC price bounds")
    day = pd.Timedelta("1D")
    if (bars.index != bars.index.floor("D")).any() or not bars.index.to_series().diff().dropna().eq(day).all():
        raise ValueError("Daily candles must be consecutive complete UTC dates")
    if (bars.index + day > pd.Timestamp.now(tz="UTC")).any():
        raise ValueError("Price CSV contains unclosed/future candles")
    first, last = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if first > last or first not in bars.index or last not in bars.index:
        raise ValueError("Price data must cover every requested evaluation date")

    funding = pd.read_csv(folder / f"{symbol}_funding_events.csv")
    funding["datetime"] = timestamps(funding.datetime)
    funding["last_funding_rate"] = pd.to_numeric(funding.last_funding_rate, errors="raise")
    if not np.isfinite(funding.last_funding_rate).all():
        raise ValueError("Funding rates must be finite")
    if "mark_price" in funding:
        funding["mark_price"] = pd.to_numeric(funding.mark_price, errors="raise")
        supplied = funding.mark_price.dropna()
        if not (np.isfinite(supplied) & supplied.gt(0)).all():
            raise ValueError("Invalid supplied funding mark prices")
    validate_funding_coverage(funding, first, last + day)

    daily = pd.read_csv(folder / f"{symbol}_funding_daily.csv")
    daily.index = pd.DatetimeIndex(timestamps(daily.datetime))
    if daily.empty or not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("Daily funding dates must be sorted and unique")
    if (daily.index != daily.index.floor("D")).any():
        raise ValueError("Daily funding dates must be UTC midnights")
    daily["funding_mean"] = pd.to_numeric(daily.funding_mean, errors="raise")
    daily["available_at"] = timestamps(daily.available_at)
    if not daily.available_at.dropna().is_unique:
        raise ValueError("Duplicate daily funding availability timestamps")
    daily = daily.reindex(pd.date_range(daily.index[0], daily.index[-1], freq="D"))
    return bars, funding, daily


def build_features(bars, daily_funding, settings):
    """Funding enters observations only after its recorded availability time."""
    close = bars.close
    windows = settings
    distance = close.diff(windows["efficiency_window"]).abs()
    path = close.diff().abs().rolling(windows["efficiency_window"]).sum()
    volume = bars.volume.rolling(windows["imbalance_window"]).sum()
    taker = bars.taker_buy_volume.rolling(windows["imbalance_window"]).sum()
    daily = daily_funding[["available_at", "funding_mean"]].copy()
    daily["mean"] = daily.funding_mean.rolling(windows["funding_window"]).mean()
    known = daily.dropna(subset=["available_at"]).sort_values("available_at")
    decision = pd.DataFrame({"decision_time": bars.index + pd.Timedelta("1D")})
    aligned = pd.merge_asof(decision, known[["available_at", "mean"]],
                            left_on="decision_time", right_on="available_at",
                            direction="backward", tolerance=pd.Timedelta("26h"))
    return pd.DataFrame({
        "log_return": np.log(close / close.shift(windows["return_window"])),
        "efficiency": (distance / path.replace(0., np.nan)).mask(path.eq(0.), 0.).clip(0, 1),
        "taker_imbalance": 2 * taker / volume.replace(0., np.nan) - 1,
        "volatility": np.log(close).diff().rolling(windows["volatility_window"]).std(ddof=0),
        "funding_mean": aligned["mean"].to_numpy(),
    }, index=bars.index)
