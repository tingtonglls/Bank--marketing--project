"""检查指定折或全部 5 折的采样流程，不训练模型或保存合成数据。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from training_pipeline import CONTACTED, DAYS, build_sampling_steps

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_fold(fold):
    features = pd.read_csv(PROJECT_ROOT / "data/processed/X_train.csv")
    labels = pd.read_csv(PROJECT_ROOT / "data/processed/y_train.csv")["y"]
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/cv_folds.csv")
    indices = pd.read_csv(PROJECT_ROOT / "data/splits/train_indices.csv")
    if len(features) != len(folds) or not np.array_equal(
        folds["train_position"], np.arange(len(features))
    ):
        raise ValueError("交叉验证行序与训练特征不对应。")
    pd.testing.assert_frame_equal(folds[["row_id", "group_id", "y"]], indices)
    np.testing.assert_array_equal(folds["y"], labels)
    if set(folds["fold"]) != {1, 2, 3, 4, 5}:
        raise ValueError("交叉验证文件必须包含完整的 1 至 5 折。")
    if set(features) != set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES):
        raise ValueError("特征字段与公共配置不一致。")
    if not features.loc[features[CONTACTED].eq(0), DAYS].eq(0).all():
        raise ValueError("真实未联系记录的间隔不是 0，需先检查数据。")
    train_positions = np.flatnonzero(folds["fold"].ne(fold))
    valid_positions = np.flatnonzero(folds["fold"].eq(fold))
    if set(folds.iloc[train_positions]["group_id"]) & set(
        folds.iloc[valid_positions]["group_id"]
    ):
        raise ValueError("训练折和验证折存在交叉分组。")
    train = features.iloc[train_positions].reset_index(drop=True)
    train_y = labels.iloc[train_positions].reset_index(drop=True)
    valid = features.iloc[valid_positions].reset_index(drop=True)
    steps = dict(build_sampling_steps())
    scaled_train = steps["scale"].fit_transform(train)
    sampled, sampled_y = steps["sample"].fit_resample(scaled_train, train_y)
    encoded_train = steps["encode"].fit_transform(sampled)
    # 验证数据只转换，绝不调用 fit 或 fit_resample。
    encoded_valid = steps["encode"].transform(steps["scale"].transform(valid))
    if not np.isfinite(encoded_train).all() or not np.isfinite(encoded_valid).all():
        raise ValueError("编码结果包含非法数值。")
    if encoded_valid.shape != (len(valid), encoded_train.shape[1]):
        raise ValueError("验证行数或输出列数错误。")
    counts = pd.Series(sampled_y).value_counts()
    if counts[0] != counts[1]:
        raise ValueError("采样结果未达到已确认的 1:1 比例。")
    placeholder = scaled_train.loc[scaled_train[CONTACTED].eq(0), DAYS].iloc[0]
    if not sampled.loc[sampled[CONTACTED].eq(0), DAYS].eq(placeholder).all():
        raise ValueError("未联系记录的间隔约束未满足。")
    print(f"第 {fold} 折训练：{len(train)} 条；采样后：{len(sampled)} 条")
    print(f"采样后未订购：{counts[0]} 条；订购：{counts[1]} 条")
    print(f"新增合成样本：{steps['sample'].n_synthetic_} 条")
    print(f"修正未联系间隔占位值的合成样本：{steps['sample'].n_adjusted_} 条")
    print(f"编码后训练形状：{encoded_train.shape}")
    print(f"编码后验证形状：{encoded_valid.shape}（未重采样）")
    print("验证通过：真实训练记录未修改、分类取值合法、指定间隔约束满足。")
    print("未读取外部测试集，未保存合成数据或拟合器，未训练模型。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    options.add_argument("--all-folds", action="store_true", help="检查全部 5 折")
    args = parser.parse_args()
    selected = range(1, 6) if args.all_folds else [args.fold]
    for fold in selected:
        check_fold(fold)
    if args.all_folds:
        print("\n全部 5 折采样与编码检查通过。")


if __name__ == "__main__":
    main()
