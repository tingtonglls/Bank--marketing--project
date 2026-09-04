"""使用人工数据检查公共预处理规则，不接触真实测试集。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame({
            **{field: ["known", "unknown"] for field in CATEGORICAL_FEATURES},
            **{field: [20, 50] for field in NUMERIC_FEATURES},
            "previously_contacted": [0, 1],
        })
        self.processor = build_preprocessor()
        self.result = self.processor.fit_transform(self.features)

    def test_unknown_is_retained_and_unseen_is_zero(self):
        encoder = self.processor.named_transformers_["categorical"]
        unknown_index = list(encoder.categories_[0]).index("unknown")
        self.assertEqual(self.result[1, unknown_index], 1)
        probe = self.features.iloc[[1]].copy()
        probe["job"] = "new_job"
        result = self.processor.transform(probe)
        np.testing.assert_array_equal(result[0, :2], [0, 0])
        np.testing.assert_array_equal(result[0, 2:], self.result[1, 2:])

    def test_scaling_uses_training_bounds_without_clipping(self):
        probe = self.features.iloc[[0]].copy()
        probe[NUMERIC_FEATURES] = 80
        before = self.processor.named_transformers_["numeric"].data_max_.copy()
        result = self.processor.transform(probe)
        numeric_slice = self.processor.output_indices_["numeric"]
        np.testing.assert_allclose(self.result[0, numeric_slice], 0)
        np.testing.assert_allclose(self.result[1, numeric_slice], 1)
        np.testing.assert_allclose(result[0, numeric_slice], 2)
        np.testing.assert_array_equal(
            before, self.processor.named_transformers_["numeric"].data_max_
        )

    def test_shape_binary_passthrough_and_no_input_mutation(self):
        original = self.features.copy(deep=True)
        self.processor.transform(self.features)
        pd.testing.assert_frame_equal(self.features, original)
        self.assertEqual(self.result.shape, (2, 30))
        np.testing.assert_array_equal(self.result[:, -1], [0, 1])
        self.assertEqual(len(self.processor.get_feature_names_out()), 30)

    def test_factory_and_clone_produce_unfitted_instances(self):
        for processor in (build_preprocessor(), clone(self.processor)):
            self.assertFalse(hasattr(processor, "transformers_"))


if __name__ == "__main__":
    unittest.main()
