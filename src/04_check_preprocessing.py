"""仅在训练特征上试运行公共预处理；不读取测试集，不保存转换数据。
调用preprocessiong.py中的公共规则，在训练集上试运行，检查编码和缩放是否正确
不保存转换结果和拟合参数
"""

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv")
    expected = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES
    if not features.columns.is_unique or set(features.columns) != set(expected):
        raise ValueError("训练输入应恰好包含约定的 20 个特征，不含标签或编号。")
    if features.isna().any().any():
        raise ValueError("存在意外缺失值，请检查；本流程不自动填补。")
    if not features["previously_contacted"].isin([0, 1]).all():
        raise ValueError("previously_contacted 只能为 0 或 1。")

    processor = build_preprocessor()
    transformed = processor.fit_transform(features)
    names = processor.get_feature_names_out()
    if transformed.shape != (len(features), len(names)):
        raise ValueError("转换后行数或字段名数量异常。")
    if not np.isfinite(transformed).all():
        raise ValueError("转换结果存在 NaN 或无穷值。")

    encoder = processor.named_transformers_["categorical"]
    category_count = sum(len(values) for values in encoder.categories_)
    print(f"训练输入：{features.shape[0]} 行，{features.shape[1]} 列")
    print(f"One-hot 后分类字段：{category_count} 列")
    print(f"Min-Max 数值字段：{len(NUMERIC_FEATURES)} 列")
    print(f"保留 0/1 字段：{len(BINARY_FEATURES)} 列")
    print(f"转换后训练输入：{transformed.shape[0]} 行，{transformed.shape[1]} 列")

    print("\n训练数据中的 unknown 独立编码列：")
    for field, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        if "unknown" in categories:
            print(f"  {field}_unknown")

    # 使用人为构造的职业类别验证行为，不读取真实测试集。
    unseen_job = "__unseen_job_for_check__"
    if unseen_job in encoder.categories_[0]:
        raise ValueError("检查用类别已存在，请更换测试值。")
    probe = features.iloc[[0]].copy()
    probe["job"] = unseen_job
    encoded_probe = processor.transform(probe)
    job_columns = len(encoder.categories_[0])
    if not np.all(encoded_probe[0, :job_columns] == 0):
        raise ValueError("未见类别未正确编码为对应字段全 0。")
    print("\n验证通过：未见职业类别对应的 One-hot 块全为 0，不报错。")
    print("仅试运行：未读取测试集，未保存拟合器或转换数据，未执行 SMOTE 或训练。")
    print("正式交叉验证时必须在各训练折内重新拟合，不复用本次拟合状态。")


if __name__ == "__main__":
    main()
