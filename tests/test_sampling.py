"""仅用人工数据测试采样约束和 Pipeline 行为。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402
from training_pipeline import (  # noqa: E402
    CONTACTED,
    DAYS,
    ConsistentSMOTENC,
    build_training_pipeline,
)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            **{c: ["known", "unknown"] * 9 for c in CATEGORICAL_FEATURES},
            **{c: np.arange(18, dtype=float) for c in NUMERIC_FEATURES},
            CONTACTED: [0, 1] * 9,
        })
        self.X[DAYS] = np.where(self.X[CONTACTED].eq(0), 0.0, 5.0)
        self.y = pd.Series([0] * 12 + [1] * 6)

    def test_only_synthetic_records_are_repaired(self):
        fake = pd.concat([self.X, self.X.iloc[[0]]], ignore_index=True)
        fake.loc[len(self.X), DAYS] = 3.0
        fake_y = pd.concat([self.y, pd.Series([1])], ignore_index=True)
        original = self.X.copy(deep=True)
        with patch("training_pipeline.SMOTENC.fit_resample", return_value=(fake, fake_y)):
            sampler = ConsistentSMOTENC()
            result, _ = sampler.fit_resample(self.X, self.y)
        self.assertEqual(sampler.n_adjusted_, 1)
        self.assertEqual(result.iloc[-1][DAYS], 0)
        pd.testing.assert_frame_equal(result.iloc[:len(original)], original)
        pd.testing.assert_frame_equal(self.X, original)

    def test_prediction_skips_sampling_and_preserves_rows(self):
        pipeline = build_training_pipeline(DummyClassifier(strategy="prior"))
        fresh = clone(pipeline)
        self.assertFalse(hasattr(fresh.named_steps["sample"], "sampler_"))
        original = self.X.copy(deep=True)
        pipeline.fit(self.X, self.y)
        self.assertEqual(pipeline.named_steps["sample"].n_synthetic_, 6)
        with patch.object(
            pipeline.named_steps["sample"], "fit_resample",
            side_effect=AssertionError("预测时不得采样"),
        ):
            probabilities = pipeline.predict_proba(self.X.iloc[:3])
        self.assertEqual(probabilities.shape, (3, 2))
        pd.testing.assert_frame_equal(self.X, original)

    def test_real_conflicting_placeholders_are_rejected(self):
        invalid = self.X.copy()
        invalid.loc[0, DAYS] = 2
        with self.assertRaises(ValueError):
            ConsistentSMOTENC().fit_resample(invalid, self.y)


if __name__ == "__main__":
    unittest.main()
