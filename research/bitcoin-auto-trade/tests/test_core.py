"""Behavior checks for the causal model, execution engine and small pipeline."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from trading.backtest import performance, simulate_directional
from trading.data import build_features, load_inputs, validate_funding_coverage
from trading.regime import compute_regimes, trailing_average
from trading.run import run_experiment, verify_results
from trading.strategy import build_targets


def market(days=170):
    rng = np.random.default_rng(6)
    drift = np.resize(np.repeat([-.01, 0., .012], 15), days)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, .003, days)))
    opening = np.r_[close[0], close[:-1]]
    index = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC", name="datetime")
    bars = pd.DataFrame({"open": opening, "close": close,
        "high": np.maximum(opening, close) * 1.01, "low": np.minimum(opening, close) * .99,
        "volume": rng.uniform(100, 200, days), "taker_buy_volume": rng.uniform(40, 80, days)}, index=index)
    event_times = pd.date_range(index[0], periods=days * 3, freq="8h")
    events = pd.DataFrame({"datetime": event_times, "last_funding_rate": .0001, "funding_interval_hours": 8})
    daily = pd.DataFrame({"datetime": index, "funding_mean": .0001,
                          "available_at": index + pd.Timedelta(days=1, minutes=5)}, index=index)
    return bars, events, daily


def flat_bars(days=4):
    index = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC")
    return pd.DataFrame({"open": 100., "close": 100., "high": 101., "low": 99.}, index=index)


FEATURES = {"return_window": 5, "efficiency_window": 5, "imbalance_window": 5,
            "volatility_window": 5, "funding_window": 5}
MODEL = {"train_window": 60, "retrain_every": 30, "naming_window": 10,
         "decode_window": 15, "iterations": 75, "restarts": 2, "smoothing_window": 3, "seeds": [42]}


class CausalInputsAndModel(unittest.TestCase):
    def test_funding_availability_and_feature_prefix(self):
        bars, _, daily = market()
        daily["funding_mean"] = np.arange(len(daily)) * .0001
        settings = dict.fromkeys(FEATURES, 2)
        features = build_features(bars, daily, settings)
        self.assertAlmostEqual(features.funding_mean.iloc[5], .00035)
        prefix = build_features(bars.iloc[:12], daily, settings)
        np.testing.assert_allclose(features.iloc[:12], prefix, equal_nan=True)

    def test_trailing_average_uses_ten_complete_days_and_resets_at_gaps(self):
        dates = pd.date_range("2020-01-01", periods=12, tz="UTC")
        probabilities = pd.DataFrame([[.15, .2, .65]] * 12, index=dates, columns=["bear", "side", "bull"])
        labels, _ = trailing_average(probabilities, 10)
        self.assertEqual(labels.iloc[:9].tolist(), [-1] * 9)
        self.assertEqual(labels.iloc[9:].tolist(), [2] * 3)
        probabilities.iloc[10] = np.nan
        labels, _ = trailing_average(probabilities, 10)
        self.assertEqual(labels.iloc[10:].tolist(), [-1, -1])
        tie = pd.DataFrame([[.5, .5, 0.]], index=dates[:1], columns=probabilities.columns)
        self.assertEqual(trailing_average(tie, 1)[0].iloc[0], 0)

    def test_hmm_training_excludes_current_observation_and_future_changes(self):
        bars, _, daily = market(140)
        features = build_features(bars, daily, FEATURES)
        naming = np.log(bars.close / bars.close.shift(10))
        with TemporaryDirectory() as folder:
            before, fits, normalization = compute_regimes(features, naming, MODEL, 42, Path(folder))
            changed = features.copy()
            changed.iloc[120:] *= 3
            after, _, _ = compute_regimes(changed, naming, MODEL, 42, Path(folder))
        pd.testing.assert_frame_equal(before.iloc[:120], after.iloc[:120])
        self.assertTrue((fits.train_end < fits.fit_at).all())
        for row in normalization.itertuples():
            expected = features[row.feature].iloc[row.train_start:row.train_end + 1].mean()
            self.assertAlmostEqual(row.mean, expected)

    def test_funding_changes_and_missing_event_rejection(self):
        times = pd.to_datetime(["2020-01-01T00:00Z", "2020-01-01T08:00Z", "2020-01-01T12:00Z",
                                "2020-01-01T16:00Z", "2020-01-01T20:00Z"])
        events = pd.DataFrame({"datetime": times, "funding_interval_hours": [8, 4, 4, 4, 4]})
        validate_funding_coverage(events, "2020-01-01", "2020-01-02")
        with self.assertRaisesRegex(ValueError, "missing event"):
            validate_funding_coverage(events.drop(index=2), "2020-01-01", "2020-01-02")


class PortfolioBehavior(unittest.TestCase):
    def test_first_day_cash_and_next_open_fixed_quantity(self):
        bars = flat_bars(3)
        bars["open"], bars["close"] = [100., 100., 80.], [100., 90., 85.]
        bars["high"], bars["low"] = [101., 101., 86.], [99., 89., 79.]
        result = simulate_directional(bars, [-1, 0, 0], fees=0, slippage=0)
        np.testing.assert_allclose(result.curve.equity, [10000, 11000, 12000])

    def test_reversal_pays_both_sides_before_resizing(self):
        cost = .003
        result = simulate_directional(flat_bars(), [1, -1, 0, 0], fees=.001, slippage=.002)
        first_notional = 10000 / (1 + cost)
        remaining = 10000 - 2 * cost * first_notional
        expected = remaining - 2 * cost * remaining / (1 + cost)
        self.assertEqual(result.orders.action.tolist(), ["entry", "exit", "entry", "exit"])
        self.assertAlmostEqual(result.stats["final_equity"], expected)

    def test_funding_boundary_sign_and_final_close(self):
        bars = flat_bars()
        events = pd.DataFrame({"datetime": [bars.index[1], bars.index[1] + pd.Timedelta("8h"), bars.index[2]],
                               "last_funding_rate": [.01, .01, -.02], "mark_price": [150.] * 3})
        result = simulate_directional(bars, [-1, 0, 0, 0], events, fees=0, slippage=0)
        np.testing.assert_allclose(result.curve.funding_cost, [0, -150, 300, 0])
        final = simulate_directional(bars, [-1] * 4, fees=.001, slippage=0)
        self.assertEqual(final.positions.status.tolist(), ["final_exit"])
        self.assertAlmostEqual(final.stats["final_equity"], 10000 - 2 * 10000 * .001 / 1.001)

    def test_insolvency_is_rejected_and_initial_loss_counts_in_drawdown(self):
        bars = flat_bars(2)
        bars.loc[bars.index[1], "high"] = 201
        with self.assertRaisesRegex(RuntimeError, "Intrabar equity"):
            simulate_directional(bars, [-1, 0], fees=0, slippage=0)
        self.assertAlmostEqual(performance(pd.Series([9000., 9500.]))["max_drawdown"], -.1)

    def test_signals_keep_history_and_do_not_change_on_future_prices(self):
        bars, _, _ = market(250)
        regimes = pd.Series(np.resize([0, 1, 2], len(bars)), index=bars.index)
        settings = {"fast": 16, "slow": 35, "signal": 9, "trend_filter": 200}
        target = build_targets(bars, regimes, settings)
        prefix = build_targets(bars.iloc[:225], regimes.iloc[:225], settings)
        pd.testing.assert_series_equal(target.iloc[:225], prefix)
        self.assertTrue(target.loc[regimes.eq(2)].eq(1).all())


class PipelineAndResults(unittest.TestCase):
    def test_small_pipeline_check_cache_and_cost_change(self):
        bars, funding, daily = market()
        template = Path(__file__).parents[1] / "configs/btc.toml"
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bars.to_csv(root / "BTCUSDT_1d.csv")
            funding.to_csv(root / "BTCUSDT_funding_events.csv", index=False)
            daily.to_csv(root / "BTCUSDT_funding_daily.csv", index=False)
            text = template.read_text(encoding="utf-8")
            replacements = {"../data/extended": str(root).replace("\\", "/"),
                "../reports/experiments": "reports", "../.interview-cache": "cache",
                "2021-05-09": "2020-04-10", "2026-09-30": "2020-06-18"}
            for old, new in replacements.items():
                text = text.replace(old, new)
            for name, value in FEATURES.items():
                text = text.replace(f"{name} = 120", f"{name} = {value}").replace(f"{name} = 20", f"{name} = {value}")
            for name, old in {"train_window": 365, "retrain_every": 30, "naming_window": 60,
                              "decode_window": 61, "iterations": 100, "restarts": 10, "smoothing_window": 10}.items():
                text = text.replace(f"{name} = {old}", f"{name} = {MODEL[name]}")
            config = root / "small.toml"
            config.write_text(text, encoding="utf-8")
            funding.assign(mark_price=-1).to_csv(root / "BTCUSDT_funding_events.csv", index=False)
            with self.assertRaisesRegex(ValueError, "funding mark"):
                run_experiment(config, check=True)
            funding.to_csv(root / "BTCUSDT_funding_events.csv", index=False)
            with patch("trading.run.compute_regimes", side_effect=AssertionError("CheckOnly trained")):
                checked = run_experiment(config, check=True)
            self.assertEqual(checked["evaluation_days"], 70)
            first = run_experiment(config)
            self.assertEqual(first["status"], "complete")
            self.assertEqual(len(first["results"]), 2)
            config.write_text(text.replace("fees = 0.0005", "fees = 0.001"), encoding="utf-8")
            with patch("trading.run.compute_regimes", side_effect=AssertionError("Fee change retrained")):
                run_experiment(config)

    def test_saved_fifteen_results_match_their_equity(self):
        verify_results(Path(__file__).parents[1])


if __name__ == "__main__":
    unittest.main()
