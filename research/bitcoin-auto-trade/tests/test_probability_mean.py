"""Causal trailing probability means as existing strategy decoder inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from trading.decisions import probability_hysteresis, probability_moving_average
from trading.experiment_config import DecisionSpec, load_experiment


BASE = '''name="smoothing_decisions"
[[observations]]
name="test"
features=[{kind="log_return",window=5}]
[decisions]
ratios=[]
include_viterbi=false
smoothing_windows=VALUE
'''


def probabilities(values):
    return pd.DataFrame(values, columns=["bear", "side", "bull"],
        index=pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC"))


class MovingAverageDecisionTests(unittest.TestCase):
    def test_ten_day_hand_average_step_tie_and_close_availability(self):
        p = probabilities([[0., 0., 1.]] * 15 + [[1., 0., 0.]] * 15)
        labels, audit = probability_moving_average(p, 10)
        self.assertTrue(labels.iloc[:9].eq(-1).all())
        self.assertTrue(labels.iloc[9:19].eq(2).all())
        self.assertTrue(labels.iloc[19:].eq(0).all())
        # The fifth bear observation makes a 5/5 tie; state index0 is fixed.
        self.assertTrue(audit.iloc[19].tie)
        self.assertAlmostEqual(audit.iloc[19].mean_bear, .5)
        self.assertAlmostEqual(audit.iloc[19].mean_bull, .5)
        self.assertTrue(audit.iloc[19].switched)
        self.assertFalse(audit.iloc[9].switched)
        self.assertEqual(audit.iloc[9].window_start, p.index[0])
        self.assertEqual(audit.iloc[9].available_at,
            p.index[9] + pd.Timedelta("1D") - pd.Timedelta("1ns"))
        self.assertEqual(len(audit), len(p))

    def test_gap_requires_a_new_complete_ten_day_window(self):
        p = probabilities([[.1, .2, .7]] * 35)
        p.iloc[12] = np.nan
        labels, audit = probability_moving_average(p, 10)
        self.assertTrue(labels.iloc[9:12].eq(2).all())
        self.assertTrue(labels.iloc[12:22].eq(-1).all())
        self.assertTrue(labels.iloc[22:].eq(2).all())
        self.assertFalse(audit.iloc[22].switched)
        self.assertTrue(audit.iloc[12:22][["mean_bear", "mean_side", "mean_bull"]].isna().all().all())

    def test_prefix_and_future_perturbation_leave_past_labels_unchanged(self):
        rng = np.random.default_rng(25)
        p = probabilities(rng.dirichlet([1., 2., 1.], size=60))
        original = p.copy(deep=True)
        whole, audit = probability_moving_average(p, 10)
        prefix, prefix_audit = probability_moving_average(p.iloc[:37], 10)
        pd.testing.assert_series_equal(whole.iloc[:37], prefix)
        pd.testing.assert_frame_equal(audit.iloc[:37], prefix_audit)
        altered = p.copy()
        altered.iloc[37:] = [1., 0., 0.]
        altered_labels, altered_audit = probability_moving_average(altered, 10)
        pd.testing.assert_series_equal(whole.iloc[:37], altered_labels.iloc[:37])
        pd.testing.assert_frame_equal(audit.iloc[:37], altered_audit.iloc[:37])
        pd.testing.assert_frame_equal(p, original)

    def test_window_one_is_pure_argmax_and_keeps_legacy_tie_rule_separate(self):
        p = probabilities([[0., 0., 1.], [.5, 0., .5], [1., 0., 0.]])
        labels, audit = probability_moving_average(p, 1)
        np.testing.assert_array_equal(labels, [2, 0, 0])
        legacy, _ = probability_hysteresis(p, 1.)
        np.testing.assert_array_equal(legacy, [2, 2, 0])
        self.assertTrue(audit.iloc[1].tie)

    def test_invalid_daily_grid_simplex_and_windows_are_rejected(self):
        p = probabilities([[.2, .3, .5]] * 12)
        for window in [0, -1, True, 10., "10"]:
            with self.subTest(window=window), self.assertRaises(ValueError):
                probability_moving_average(p, window)
        bad_frames = [p.drop(p.index[5]), p.iloc[::-1], p.reset_index(drop=True),
            p.drop(columns="bull")]
        for row in [[np.nan, 0., 1.], [np.inf, 0., 0.], [-.1, .5, .6], [.2, .3, .6]]:
            bad = p.copy()
            bad.iloc[5] = row
            bad_frames.append(bad)
        for bad in bad_frames:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                probability_moving_average(bad, 10)

    def test_unavailable_or_short_history_is_preserved_without_filling(self):
        p = probabilities([[np.nan, np.nan, np.nan]] * 12)
        labels, audit = probability_moving_average(p, 10)
        self.assertTrue(labels.eq(-1).all())
        self.assertFalse(audit.available.any())
        short = probabilities([[.2, .3, .5]] * 9)
        labels, audit = probability_moving_average(short, 10)
        self.assertTrue(labels.eq(-1).all())
        self.assertEqual(len(audit), 9)
        empty, empty_audit = probability_moving_average(short.iloc[:0], 10)
        self.assertEqual(len(empty), 0)
        self.assertIn("mean_bear", empty_audit)


if __name__ == '__main__': unittest.main()
