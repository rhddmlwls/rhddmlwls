"""TOML -> past-only regimes -> fixed strategy -> next-open portfolio."""
import argparse
from datetime import date
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import tomllib

import numpy as np
import pandas as pd

from .backtest import performance, simulate_directional
from .cache import atomic_json, commit_manifest, file_digest, fingerprint, frame_digest, read_manifest, save_frame
from .data import build_features, load_inputs
from .regime import compute_regimes, trailing_average
from .strategy import build_targets


DEFAULTS = {
    "data": {"symbol": "BTCUSDT"},
    "paths": {"data_dir": "../data/extended", "output_root": "../reports/experiments", "cache_dir": "../.interview-cache"},
    "features": {"return_window": 120, "efficiency_window": 120, "imbalance_window": 120, "volatility_window": 20, "funding_window": 20},
    "model": {"train_window": 365, "retrain_every": 30, "naming_window": 60, "decode_window": 61, "iterations": 100, "restarts": 10, "seeds": [42], "smoothing_window": 10},
    "strategy": {"fast": 16, "slow": 35, "signal": 9, "trend_filter": 200, "bull": "long", "side": "macd_long", "bear": "macd_short"},
    "trading": {"initial_cash": 10000.0, "fees": 0.0005, "slippage": 0.0005, "close_final": True},
    "evaluation": {"start": "2021-05-09", "end": "2026-09-30"},
}


def load_config(path):
    path = Path(path).resolve()
    source = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(source) - {"name", *DEFAULTS}:
        raise ValueError("Unknown configuration sections")
    name = source.get("name", path.stem)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Invalid experiment name")
    config = {"name": name}
    for section, defaults in DEFAULTS.items():
        values = source.get(section, {})
        if not isinstance(values, dict) or set(values) - set(defaults):
            raise ValueError(f"Unknown fields in {section}")
        config[section] = defaults | values
        for key, value in config[section].items():
            default = defaults[key]
            if key in {"start", "end"}:
                config[section][key] = date.fromisoformat(str(value)).isoformat()
            elif type(default) is int and (type(value) is not int or value < (0 if key == "trend_filter" else 1)):
                raise ValueError(f"Invalid positive integer: {section}.{key}")
            elif type(default) is float and (type(value) not in {int, float} or not np.isfinite(value)):
                raise ValueError(f"Invalid finite number: {section}.{key}")
            elif type(default) is bool and type(value) is not bool:
                raise ValueError(f"Invalid boolean: {section}.{key}")
            elif isinstance(default, str) and (not isinstance(value, str) or not value):
                raise ValueError(f"Invalid string: {section}.{key}")
    model, strategy, trading = (config[k] for k in ["model", "strategy", "trading"])
    seeds = model["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s <= 2**32 - model["restarts"] for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain unique valid integers")
    if model["train_window"] < 60 or model["decode_window"] > model["train_window"] + 1:
        raise ValueError("Invalid HMM training or decoding window")
    if strategy["fast"] >= strategy["slow"] or any(strategy[k] not in {"long", "short", "flat", "macd_long", "macd_short"} for k in ["bull", "side", "bear"]):
        raise ValueError("Invalid MACD parameters or regime strategy")
    if trading["initial_cash"] <= 0 or not all(0 <= trading[k] < 1 for k in ["fees", "slippage"]):
        raise ValueError("Invalid trading costs or initial cash")
    if config["evaluation"]["start"] >= config["evaluation"]["end"]:
        raise ValueError("Evaluation requires at least two days")
    if not re.fullmatch(r"[A-Z0-9]+USDT", config["data"]["symbol"]):
        raise ValueError("Invalid USDT symbol")
    for key, value in config["paths"].items():
        config["paths"][key] = str((path.parent / value).resolve())
    return config


def _read_frame(path, index=True):
    return pd.read_csv(path, index_col=0 if index else None, parse_dates=index, float_precision="round_trip")


def _probabilities(features, naming, model, seed, cache, versions):
    definition = {k: v for k, v in model.items() if k not in {"seeds", "smoothing_window"}}
    key = fingerprint({"features": frame_digest(features), "naming": frame_digest(naming.to_frame()), "model": definition,
                       "seed": seed, "versions": versions, "engine": file_digest(Path(__file__).with_name("regime.py"))})
    folder, files = cache / "probabilities" / key, ["probabilities.csv", "fits.csv", "normalization.csv"]
    if read_manifest(folder, key, files) is None:
        probabilities, fits, normalization = compute_regimes(features, naming, model, seed, cache)
        for name, frame in zip(files, [probabilities, fits, normalization]):
            save_frame(folder / name, frame, index=name == "probabilities.csv")
        commit_manifest(folder, key, files)
    return _read_frame(folder / files[0]), _read_frame(folder / files[1], False), _read_frame(folder / files[2], False)


def _portfolio(bars, targets, funding, trading, cache):
    key = fingerprint({"bars": frame_digest(bars), "targets": frame_digest(targets.to_frame()), "funding": frame_digest(funding),
                       "trading": trading, "engine": file_digest(Path(__file__).with_name("backtest.py"))})
    folder, files = cache / "backtests" / key, ["equity.csv", "orders.csv", "positions.csv", "stats.json"]
    if read_manifest(folder, key, files) is None:
        result = simulate_directional(bars, targets.to_numpy(), funding, **trading)
        for name, frame in zip(files, [result.curve, result.orders, result.positions]):
            save_frame(folder / name, frame, index=name == "equity.csv")
        atomic_json(folder / "stats.json", result.stats)
        commit_manifest(folder, key, files)
    return folder, json.loads((folder / "stats.json").read_text(encoding="utf-8"))


def run_experiment(config_path, check=False):
    config = load_config(config_path)
    model, trading, dates = (config[k] for k in ["model", "trading", "evaluation"])
    bars, funding, daily = load_inputs(Path(config["paths"]["data_dir"]), config["data"]["symbol"], dates["start"], dates["end"])
    features = build_features(bars, daily, config["features"])
    naming = np.log(bars.close / bars.close.shift(model["naming_window"]))
    finite = np.isfinite(features).all(axis=1) & np.isfinite(naming)
    available = finite & finite.shift(1, fill_value=False).rolling(model["train_window"], min_periods=model["train_window"]).sum().eq(model["train_window"])
    available = available.rolling(model["smoothing_window"], min_periods=model["smoothing_window"]).sum().eq(model["smoothing_window"])
    evaluation = bars.loc[dates["start"]:dates["end"]]
    expected = pd.date_range(dates["start"], dates["end"], freq="D", tz="UTC")
    if not evaluation.index.equals(expected.rename(evaluation.index.name)) or not available.loc[evaluation.index].all():
        raise ValueError("Every requested evaluation day needs complete past training and smoothing windows")
    report = {"config": config, "evaluation_days": len(evaluation), "failures": [], "results": []}
    if check:
        print(f"Validated {config['data']['symbol']}: {len(evaluation)} days, {len(model['seeds'])} seed groups, no training")
        return report
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "hmmlearn"]}
    engine = {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")}
    run_id = fingerprint({"config": config, "bars": frame_digest(bars), "funding": frame_digest(funding), "daily": frame_digest(daily), "engine": engine, "versions": versions})[:12]
    output = Path(config["paths"]["output_root"]) / f"{config['name']}-{run_id}"
    cache = Path(config["paths"]["cache_dir"])
    output.mkdir(parents=True, exist_ok=True)
    report.update(output=str(output), run_id=run_id, versions=versions, engine=engine,
                  assumptions={"funding": "actual signed event timing", "sizing": "1x post-cost equity then fixed quantity", "execution": "next open; evaluation first day starts in cash", "final_exit": trading["close_final"], "margin_liquidation": "not modeled; reject insolvency"})
    shutil.copyfile(config_path, output / "config.toml")
    for seed in model["seeds"]:
        try:
            probabilities, fits, normalization = _probabilities(features, naming, model, seed, cache, versions)
            regimes, averaged = trailing_average(probabilities, model["smoothing_window"])
            if regimes.loc[evaluation.index].lt(0).any():
                raise ValueError("Model produced unavailable evaluation days")
            targets = build_targets(bars, regimes, config["strategy"]).loc[evaluation.index]
            for name, frame in [("probabilities", probabilities), ("fits", fits), ("normalization", normalization), ("mean_probabilities", averaged)]:
                save_frame(output / f"{name}_{seed}.csv", frame, index=name not in {"fits", "normalization"})
        except (ValueError, RuntimeError) as exc:
            report["failures"].append({"seed": seed, "stage": "HMM", "error": str(exc)})
            continue
        cases = {"regime_mix": targets, "long_hold": pd.Series(1, index=evaluation.index, name="target")}
        for name, target in cases.items():
            try:
                folder, stats = _portfolio(evaluation, target, funding, trading, cache)
                destination = output / f"{name}_{seed}"
                destination.mkdir(exist_ok=True)
                for source in folder.iterdir():
                    if source.name != "manifest.json":
                        shutil.copyfile(source, destination / source.name)
                save_frame(destination / "targets.csv", target.to_frame())
                report["results"].append({"strategy": name, "seed": seed, **stats})
            except RuntimeError as exc:
                report["failures"].append({"strategy": name, "seed": seed, "stage": "backtest", "error": str(exc)})
    report["status"] = "complete" if not report["failures"] else "partial" if report["results"] else "failed"
    save_frame(output / "summary.csv", pd.DataFrame(report["results"]), index=False)
    atomic_json(output / "report.json", report)
    atomic_json(output.parent / f"{config['name']}-latest.json", {"output": str(output), "run_id": run_id, "status": report["status"]})
    print(f"{report['status']}: {output}")
    return report


def verify_results(root):
    summary = pd.read_csv(root / "results/summary.csv")
    curves = pd.read_csv(root / "results/equity.csv", index_col="date", parse_dates=True)
    if not summary.case.is_unique or set(curves) != set(summary.case) or not curves.index.is_unique:
        raise ValueError("Published cases and dates must be unique and match")
    for row in summary.itertuples():
        start, end = pd.Timestamp(row.start), pd.Timestamp(row.end)
        series = curves.loc[start:end, row.case]
        if not series.index.equals(pd.date_range(start, end, name=curves.index.name)) or not series.gt(0).all() or not np.isfinite(series).all():
            raise ValueError(f"Incomplete or invalid published equity: {row.case}")
        if curves.loc[(curves.index < start) | (curves.index > end), row.case].notna().any():
            raise ValueError(f"Equity outside the declared period: {row.case}")
        actual = performance(series, 10000.0)
        for key in ["total_return", "sharpe", "max_drawdown"]:
            if not np.isclose(actual[key], getattr(row, key), rtol=1e-10, atol=1e-10):
                raise ValueError(f"Published metric mismatch: {row.case}/{key}")
    print(f"Verified {len(summary)} published equity curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/btc.toml")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-results", action="store_true")
    args = parser.parse_args()
    if args.verify_results:
        verify_results(Path(__file__).resolve().parents[2])
    else:
        report = run_experiment(args.config, args.check)
        if report["failures"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
