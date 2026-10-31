"""Known-value, state-transition and causality tests for indicator targets."""

import unittest

import numpy as np
import pandas as pd

from trading.technical_strategies import (
    TECHNICAL_KINDS, rsi_wilder, technical_signal, validate_technical_params,
)


def bars_for(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {"close": close, "high": close, "low": close,
         "volume": np.ones(len(close)), "quote_volume": close},
        index=pd.date_range("2020-01-01", periods=len(close), freq="D", tz="UTC"),
    )


class WilderRSITests(unittest.TestCase):
    def test_initial_sma_and_hand_computed_wilder_recurrence(self):
        # First 3 changes: +2,-1,+3. Average gain=5/3, loss=1/3.
        # After -1: gain=10/9, loss=5/9. After +3: 47/27 and 10/27.
        result = rsi_wilder(pd.Series([10., 12., 11., 14., 13., 16.]), 3)
        self.assertTrue(result.iloc[:3].isna().all())
        np.testing.assert_allclose(result.iloc[3:], [100 * 5 / 6, 100 * 10 / 15, 100 * 47 / 57])

    def test_standard_fourteen_period_worked_example(self):
        close = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
                           45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
                           46.28, 46.00, 46.03])
        rsi = rsi_wilder(close)
        np.testing.assert_allclose(rsi.iloc[14:], [70.464135, 66.249619, 66.480942], atol=1e-6)

    def test_zero_gains_losses_flat_and_missing_reset(self):
        for close, expected in [([3, 3, 3, 3], 50), ([1, 2, 3, 4], 100), ([4, 3, 2, 1], 0)]:
            with self.subTest(close=close):
                np.testing.assert_array_equal(rsi_wilder(pd.Series(close), 2).iloc[2:], expected)
        result = rsi_wilder(pd.Series([1., 2., 3., np.nan, 3., 2., 1.]), 2)
        np.testing.assert_array_equal(result.isna(), [True, True, False, True, True, True, False])
        self.assertEqual(result.iloc[-1], 0)
        with self.assertRaises(ValueError):
            rsi_wilder(pd.Series([1, 2]), True)


class TechnicalSignalTests(unittest.TestCase):
    def test_rsi_reversion_strict_entry_and_exit_with_hysteresis(self):
        signal = technical_signal(bars_for([10, 9, 8, 9, 10, 9, 8, 9, 10]),
                                  "rsi_reversion", {"period": 2, "lower": 30, "exit_level": 50})
        # RSI=50 at the fourth close does not exit a strict >50 rule.
        self.assertEqual(signal.tolist(), [0, 0, 1, 1, 0, 0, 1, 0, 0])
        self.assertEqual(signal.dtype, np.dtype("int8"))
        self.assertEqual(signal.name, "target")

    def test_rsi_trend_symmetric_short_and_reversal(self):
        signal = technical_signal(bars_for([10, 11, 12, 11, 10, 9, 10, 11]),
                                  "rsi_trend", {"period": 2, "entry_level": 60, "exit_level": 45, "short": True})
        self.assertEqual(signal.tolist(), [0, 0, 1, 1, -1, -1, 0, 1])

    def test_momentum_holds_between_thresholds_and_exits_at_equality(self):
        signal = technical_signal(bars_for([100, 106, 107, 106, 100, 94, 94, 100]),
                                  "momentum", {"lookback": 1, "entry_threshold": .05, "exit_threshold": 0, "short": True})
        self.assertEqual(signal.tolist(), [0, 1, 1, 0, -1, -1, 0, 1])

    def test_sma_cross_exact_values_and_warmup(self):
        signal = technical_signal(bars_for([1, 2, 3, 3, 2, 1, 1]),
                                  "sma_cross", {"fast": 1, "slow": 3, "short": True})
        self.assertEqual(signal.tolist(), [0, 0, 1, 1, -1, -1, -1])

    def test_macd_signal_ema_warmup_and_known_values(self):
        bars = bars_for([1, 2, 3, 2, 1, 2, 3, 2])
        signal = technical_signal(bars, "macd", {"fast": 1, "slow": 2, "signal": 2, "short": True})
        # First MACD value is at index 1; signal needs two available MACDs.
        # MACD at indices 1,2,3: 1/3,4/9,-5/27.
        # Signal at indices 2,3: 11/27,1/81.
        self.assertEqual(signal.iloc[:4].tolist(), [0, 0, 1, -1])
        line = bars.close.ewm(span=1, adjust=False).mean() - bars.close.ewm(span=2, adjust=False).mean()
        other = line.iloc[1:].ewm(span=2, adjust=False, min_periods=2).mean()
        expected = np.sign(line.iloc[2:] - other.iloc[1:]).astype("int8")
        np.testing.assert_array_equal(signal.iloc[2:], expected)

    def test_trend_filter_forces_exit_and_does_not_wait_for_indicator_nan(self):
        bars = bars_for([100, 102, 104, 100, 99, 102])
        # RSI=0 does not satisfy the strict exit rule <0. The trend filter
        # must therefore force the exit when close drops below SMA3.
        signal = technical_signal(bars, "rsi_trend",
                                  {"period": 1, "entry_level": 60, "exit_level": 0, "trend_filter": 3})
        self.assertEqual(signal.tolist(), [0, 0, 1, 0, 0, 1])

    def test_pullback_requires_trend_and_exits_below_trend(self):
        bars = bars_for([100, 105, 110, 115, 120, 118, 112, 108, 120])
        signal = technical_signal(bars, "pullback",
                                  {"period": 1, "lower": 40, "exit_level": 70,
                                   "fast": 2, "slow": 4, "trend_filter": 4})
        # Day 5 pullback remains above SMA4 with EMA2 > EMA4; day 6
        # drops below SMA4, forcing exit even while RSI stays oversold.
        self.assertEqual(signal.tolist(), [0, 0, 0, 0, 0, 1, 0, 0, 0])

    def test_bollinger_reversion_current_band_then_exit_at_mean(self):
        signal = technical_signal(bars_for([10, 10, 8, 8, 10]),
                                  "bollinger_reversion", {"window": 3, "multiplier": 1})
        self.assertEqual(signal.tolist(), [0, 0, 1, 1, 0])

    def test_bollinger_breakout_uses_prior_band_and_current_mean_exit(self):
        # A current-window upper band with multiplier 2 would prevent the
        # first breakout; the previous [10,10,10] upper band is exactly 10.
        signal = technical_signal(bars_for([10, 10, 10, 11, 11, 9]),
                                  "bollinger_breakout", {"window": 3, "multiplier": 2})
        self.assertEqual(signal.tolist(), [0, 0, 0, 1, 1, 0])

    def test_rolling_vwap_uses_actual_quote_base_volume_not_close_proxy(self):
        bars = bars_for([11, 11, 11, 11])
        bars["volume"] = [1., 3., 1., 1.]
        bars["quote_volume"] = [10., 42., 10., 10.]
        # VWAP=52/4=13 at day 1 and day 2; day 3 VWAP=20/2=10.
        signal = technical_signal(bars, "rolling_vwap", {"window": 2, "short": True})
        self.assertEqual(signal.tolist(), [0, -1, -1, 1])
        missing = bars.drop(columns="quote_volume")
        with self.assertRaisesRegex(ValueError, "actual volume and quote_volume"):
            technical_signal(missing, "rolling_vwap")
        bars[["volume", "quote_volume"]] = 0
        self.assertEqual(technical_signal(bars, "rolling_vwap", {"window": 2}).tolist(), [0] * 4)
        bars.loc[bars.index[0], "quote_volume"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            technical_signal(bars, "rolling_vwap")

    def test_all_kinds_are_prefix_invariant_and_do_not_mutate_bars(self):
        n = 400
        close = 100 * np.exp(np.cumsum(.003 + .02 * np.sin(np.arange(n) / 8)))
        bars = bars_for(close)
        bars["volume"] = 1 + (np.arange(n) % 7)
        bars["quote_volume"] = bars.volume * bars.close * (1 + .001 * np.cos(np.arange(n)))
        original = bars.copy(deep=True)
        cut = 285
        changed = bars.copy()
        changed.iloc[cut:, changed.columns.get_loc("close")] *= .01
        changed.iloc[cut:, changed.columns.get_loc("quote_volume")] *= 100
        for kind in sorted(TECHNICAL_KINDS):
            with self.subTest(kind=kind):
                full = technical_signal(bars, kind)
                pd.testing.assert_series_equal(technical_signal(bars.iloc[:cut], kind), full.iloc[:cut])
                pd.testing.assert_series_equal(technical_signal(changed, kind).iloc[:cut], full.iloc[:cut])
                self.assertTrue(full.isin([-1, 0, 1]).all())
                pd.testing.assert_index_equal(full.index, bars.index)
        pd.testing.assert_frame_equal(bars, original)

    def test_missing_ready_indicator_resets_position(self):
        bars = bars_for([1, 2, 3, np.nan, 4, 3, 2, 3])
        signal = technical_signal(bars, "rsi_reversion", {"period": 2, "lower": 30, "exit_level": 80})
        self.assertEqual(signal.tolist(), [0, 0, 0, 0, 0, 0, 1, 1])

    def test_config_validation_rejects_typos_nonfinite_wrong_types_and_bad_order(self):
        invalid = [
            ("unknown", {}), ("macd", {"fast": 30, "slow": 10}),
            ("macd", {"singal": 9}), ("rsi_reversion", {"period": True}),
            ("rsi_reversion", {"period": 14.5}), ("rsi_reversion", {"lower": 60}),
            ("rsi_trend", {"entry_level": 45}), ("sma_cross", {"short": 1}),
            ("momentum", {"entry_threshold": -.1}), ("momentum", {"entry_threshold": float("nan")}),
            ("momentum", {"entry_threshold": 0, "exit_threshold": .1}),
            ("bollinger_reversion", {"multiplier": 0}), ("rolling_vwap", {"window": 0}),
            ("sma_cross", {"trend_filter": -1}),
        ]
        for kind, params in invalid:
            with self.subTest(kind=kind, params=params), self.assertRaises(ValueError):
                validate_technical_params(kind, params)
        supplied = {"period": 2}
        normalized = validate_technical_params("rsi_reversion", supplied)
        normalized["lower"] = 40
        self.assertEqual(supplied, {"period": 2})
        self.assertEqual(validate_technical_params("rsi_reversion")["lower"], 30)


if __name__ == "__main__":
    unittest.main()
