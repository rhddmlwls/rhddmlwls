"""Verify published portfolio metrics from saved daily equity; no model training."""
import csv
from datetime import date
import math
from pathlib import Path
import statistics
import sys


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite number: {value}")
    return number


def read_summary(path):
    cases = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "symbol", "seed", "start", "end", "total_return", "sharpe", "max_drawdown"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Summary columns are missing")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Duplicate summary columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed summary row")
            case = row["case"].strip()
            if not case or case in cases or not row["symbol"].strip() or not row["seed"].strip():
                raise ValueError(f"Missing or duplicate case identity: {case}")
            start, end = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
            if start > end:
                raise ValueError(f"Reversed evaluation dates: {case}")
            expected = {key: finite(row[key]) for key in ["total_return", "max_drawdown"]}
            expected["sharpe"] = finite(row["sharpe"]) if row["sharpe"].strip() else None
            cases[case] = (start, end, expected)
    if not cases:
        raise ValueError("Summary has no cases")
    return cases


def read_curves(path, cases):
    curves = {case: [] for case in cases}
    first = previous = None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or set(fields) != {"date", *cases}:
            raise ValueError("Equity columns must match summary cases exactly")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed equity row")
            day = date.fromisoformat(row["date"])
            if previous is not None and (day - previous).days != 1:
                raise ValueError(f"Duplicate, unsorted or missing equity date: {day}")
            first = first or day
            previous = day
            for case, (start, end, _) in cases.items():
                value = row[case].strip()
                if start <= day <= end:
                    if not value:
                        raise ValueError(f"Missing equity inside evaluation: {case}, {day}")
                    number = finite(value)
                    if number <= 0:
                        raise ValueError(f"Nonpositive equity: {case}, {day}")
                    curves[case].append(number)
                elif value:
                    raise ValueError(f"Equity outside evaluation: {case}, {day}")
    if first != min(c[0] for c in cases.values()) or previous != max(c[1] for c in cases.values()):
        raise ValueError("Equity dates do not cover the declared evaluation range")
    for case, (start, end, _) in cases.items():
        if len(curves[case]) != (end - start).days + 1:
            raise ValueError(f"Incomplete evaluation period: {case}")
    return curves


def metrics(equity, initial_cash=10000.0):
    if not equity or initial_cash <= 0 or any(not math.isfinite(x) or x <= 0 for x in equity):
        raise ValueError("Metrics require positive finite equity")
    previous = peak = initial_cash
    returns, drawdown = [], 0.0
    for value in equity:
        daily = value / previous - 1
        if not math.isfinite(daily):
            raise ValueError("Nonfinite daily return")
        returns.append(daily)
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1)
        previous = value
    deviation = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sharpe = statistics.mean(returns) / deviation * math.sqrt(365.25) if deviation else None
    return dict(total_return=equity[-1] / initial_cash - 1, sharpe=sharpe, max_drawdown=drawdown)


def verify(root):
    cases = read_summary(root / "results/summary.csv")
    curves = read_curves(root / "results/equity.csv", cases)
    for case, (_, _, expected) in cases.items():
        actual = metrics(curves[case])
        for key, saved in expected.items():
            value = actual[key]
            if (saved is None or value is None) and saved is not value:
                raise ValueError(f"Undefined metric differs: {case}, {key}")
            if saved is not None and not math.isclose(value, saved, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Metric differs: {case}, {key}: {value} != {saved}")
        sharpe = "undefined" if actual["sharpe"] is None else f"{actual['sharpe']:.3f}"
        print(f"{case}: return {actual['total_return']:+.2%} | Sharpe {sharpe} | MDD {actual['max_drawdown']:.2%}")
    print(f"Verified {len(cases)} cases; Rf=0, sample deviation, 365.25 days/year, initial cash 10000.")


if __name__ == "__main__":
    try:
        verify(Path(__file__).resolve().parent)
    except (OSError, ValueError, csv.Error, KeyError, OverflowError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        sys.exit(1)
