"""训练专用公共流程：缩放 → SMOTENC + 合成记录约束 → 编码 → 模型。"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from preprocessing import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

CONTACTED = "previously_contacted"
DAYS = "days_since_previous_contact"


class ConsistentSMOTENC(BaseEstimator):
    """接收缩放后的混合 DataFrame；只约束新增记录，不改变真实记录。

    fit_resample 是训练阶段操作；imblearn Pipeline 在预测时跳过此步骤。
    """

    def __init__(self, sampling_strategy=1.0, k_neighbors=5, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SMOTENC 需要带字段名的混合类型 DataFrame。")
        if X.isna().any().any() or not X[CONTACTED].isin([0, 1]).all():
            raise ValueError("输入有空值或此前联系标记非法。")
        categorical = CATEGORICAL_FEATURES + BINARY_FEATURES
        never_contacted = X[CONTACTED].eq(0)
        placeholders = X.loc[never_contacted, DAYS].unique()
        if len(placeholders) > 1:
            raise ValueError("真实未联系记录的间隔占位值不一致，不能自动修改。")
        # 取训练折中真实 0 天经缩放后的值，而非假定缩放后仍为 0。
        placeholder = placeholders[0] if len(placeholders) else None
        self.sampler_ = SMOTENC(
            categorical_features=categorical,
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        sampled, sampled_y = self.sampler_.fit_resample(X, y)
        sampled = sampled.reset_index(drop=True)
        self.n_original_ = len(X)
        self.n_synthetic_ = len(sampled) - len(X)
        # 校验库返回顺序，保证下面的约束不会改到真实样本。
        pd.testing.assert_frame_equal(
            sampled.iloc[:len(X)].reset_index(drop=True),
            X.reset_index(drop=True),
            check_dtype=False,
        )
        np.testing.assert_array_equal(np.asarray(sampled_y)[:len(X)], np.asarray(y))
        synthetic = np.arange(len(sampled)) >= len(X)
        needs_placeholder = synthetic & sampled[CONTACTED].eq(0).to_numpy()
        self.n_adjusted_ = 0
        if needs_placeholder.any():
            if placeholder is None:
                raise ValueError("采样器产生了训练数据不存在的未联系类别。")
            self.n_adjusted_ = int(
                sampled.loc[needs_placeholder, DAYS].ne(placeholder).sum()
            )
            sampled.loc[needs_placeholder, DAYS] = placeholder
        for field in categorical:
            if not sampled[field].isin(X[field].unique()).all():
                raise ValueError(f"合成数据出现未见分类值：{field}")
        return sampled, sampled_y


def build_sampling_steps():
    """每次创建全新实例；可用于检查，也可接入正式模型。"""
    scaler = ColumnTransformer(
        [
            ("numeric", MinMaxScaler(clip=False), NUMERIC_FEATURES),
            ("categorical", "passthrough", CATEGORICAL_FEATURES + BINARY_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    encoder = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ]
    )
    return [
        ("scale", scaler),
        ("sample", ConsistentSMOTENC()),
        ("encode", encoder),
    ]


def build_training_pipeline(model):
    """传入尚未拟合的模型；交叉验证需将整个 Pipeline 交给搜索器。"""
    return Pipeline(build_sampling_steps() + [("model", model)])
