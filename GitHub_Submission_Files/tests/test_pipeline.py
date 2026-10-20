import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from project_config import (
    EXPECTED_PARTICIPANTS,
    EXPECTED_POSITIVES,
    EXPECTED_SAMPLES,
    HORIZON_MINUTES,
    SAMPLE_MINUTES,
)
from train_cgmacros_forecast import (
    EXTRACT_DIR,
    build_samples,
    grouped_splits,
    make_features,
    make_target,
)


class PipelineContractTests(unittest.TestCase):
    def test_participant_groups_do_not_overlap_across_folds(self):
        group_ids = np.repeat(np.arange(EXPECTED_PARTICIPANTS), 10)
        X = pd.DataFrame({"x": np.arange(len(group_ids), dtype=float)})
        y = np.tile(np.array([0] * 8 + [1] * 2), EXPECTED_PARTICIPANTS)
        held_out = []

        for train_idx, test_idx in grouped_splits(X, y, group_ids):
            train_groups = set(group_ids[train_idx])
            test_groups = set(group_ids[test_idx])
            self.assertFalse(train_groups & test_groups)
            self.assertEqual(len(test_groups), EXPECTED_PARTICIPANTS // 5)
            held_out.extend(test_groups)

        self.assertEqual(len(held_out), EXPECTED_PARTICIPANTS)
        self.assertEqual(set(held_out), set(range(EXPECTED_PARTICIPANTS)))

    def test_features_at_t_are_unchanged_when_future_rows_are_removed(self):
        index = pd.date_range("2025-01-01", periods=16, freq=f"{SAMPLE_MINUTES}min")
        raw = pd.DataFrame(
            {
                "glucose": np.arange(16, dtype=float) + 100,
                "heart_rate": np.arange(16, dtype=float) + 60,
                "activity_level": np.arange(16, dtype=float),
                "activity_calories": np.arange(16, dtype=float),
                "carbs": [0.0] * 5 + [30.0] + [0.0] * 10,
                "food_calories": [0.0] * 5 + [120.0] + [0.0] * 10,
                "protein": [0.0] * 16,
                "fat": [0.0] * 16,
                "fiber": [0.0] * 16,
                "meal_event": [False] * 5 + [True] + [False] * 10,
            },
            index=index,
        )
        cutoff = index[8]
        full_features = make_features(raw, index)
        truncated_features = make_features(raw.loc[:cutoff], index[:9])
        assert_series_equal(
            full_features.loc[cutoff],
            truncated_features.loc[cutoff],
            check_names=False,
            check_dtype=False,
        )

    def test_forward_fill_expires_after_configured_tolerance(self):
        index = pd.DatetimeIndex([pd.Timestamp("2025-01-01 00:00")])
        raw = pd.DataFrame(
            {
                "glucose": [100.0],
                "heart_rate": [70.0],
                "activity_level": [1.0],
                "activity_calories": [2.0],
                "carbs": [0.0],
                "food_calories": [0.0],
                "protein": [0.0],
                "fat": [0.0],
                "fiber": [0.0],
                "meal_event": [False],
            },
            index=index,
        )
        grid = pd.date_range(index[0], periods=4, freq=f"{SAMPLE_MINUTES}min")
        features = make_features(raw, grid)
        self.assertEqual(features.loc[grid[2], "glucose_now"], 100.0)
        self.assertTrue(pd.isna(features.loc[grid[3], "glucose_now"]))

    def test_target_uses_exact_t_plus_120_glucose(self):
        index = pd.date_range("2025-01-01", periods=16, freq=f"{SAMPLE_MINUTES}min")
        glucose = pd.Series(np.arange(16, dtype=float) + 100, index=index)
        target = make_target(glucose, index[:4])
        self.assertEqual(target.iloc[0], glucose.loc[index[0] + pd.Timedelta(minutes=HORIZON_MINUTES)])
        self.assertEqual(target.iloc[1], glucose.loc[index[1] + pd.Timedelta(minutes=HORIZON_MINUTES)])

    def test_full_dataset_participant_sample_and_event_counts(self):
        participant_files = list(EXTRACT_DIR.glob("CGMacros-*/CGMacros-*.csv"))
        self.assertEqual(len(participant_files), EXPECTED_PARTICIPANTS)
        samples, _, _ = build_samples()
        self.assertEqual(samples["participant_id"].nunique(), EXPECTED_PARTICIPANTS)
        self.assertEqual(len(samples), EXPECTED_SAMPLES)
        self.assertEqual(int(samples["label"].sum()), EXPECTED_POSITIVES)


if __name__ == "__main__":
    unittest.main()
