"""仅在已保存的训练集内部创建六种模型共用的 5 折划分。
保持同组（group_id相同的记录）不跨折
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
N_SPLITS = 5
RANDOM_STATE = 42


def build_cv_folds(indices):
    """返回每条训练记录作为验证样本时的折号（1 至 5）。"""
    if set(indices.columns) != {"row_id", "group_id", "y"}:
        raise ValueError("训练划分必须包含 row_id、group_id、y 三列。")
    if indices.empty or indices.isna().any().any():
        raise ValueError("训练划分为空或存在缺失值。")
    if not all(pd.api.types.is_integer_dtype(indices[c]) for c in indices):
        raise ValueError("行号、组号、标签必须为整数。")
    if not indices["row_id"].is_unique or not indices["y"].isin([0, 1]).all():
        raise ValueError("训练行号重复或标签不合法。")
    if set(indices["y"]) != {0, 1} or indices["group_id"].nunique() < N_SPLITS:
        raise ValueError("需要两种标签以及至少 5 个独立分组。")

    folds = indices.reset_index(drop=True).copy()
    # train_position 对应 X_train、y_train 的行位置，不是原始 CSV 的 row_id。
    folds.insert(0, "train_position", np.arange(len(folds)))
    fold_ids = np.zeros(len(folds), dtype=int)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    for fold, (_, valid_positions) in enumerate(
        splitter.split(
            np.zeros((len(folds), 1)), folds["y"], groups=folds["group_id"]
        ),
        start=1,
    ):
        if np.any(fold_ids[valid_positions] != 0):
            raise ValueError("同一训练记录被重复分配为验证样本。")
        fold_ids[valid_positions] = fold
    folds["fold"] = fold_ids
    validate_cv_folds(folds)
    return folds


def validate_cv_folds(folds):
    if not np.array_equal(folds["train_position"], np.arange(len(folds))):
        raise ValueError("训练行位置不连续或顺序已改变。")
    if set(folds["fold"]) != set(range(1, N_SPLITS + 1)):
        raise ValueError("折号不完整或存在未分配记录。")
    if not folds["row_id"].is_unique:
        raise ValueError("原始行号重复。")
    if folds.groupby("group_id")["fold"].nunique().max() != 1:
        raise ValueError("同一个 group_id 被拆到不同折。")
    for fold in range(1, N_SPLITS + 1):
        train = folds.loc[folds["fold"].ne(fold)]
        valid = folds.loc[folds["fold"].eq(fold)]
        if set(train["group_id"]) & set(valid["group_id"]):
            raise ValueError("训练折与验证折存在分组交叉。")
        if set(train["y"]) != {0, 1} or set(valid["y"]) != {0, 1}:
            raise ValueError("每轮训练与验证都应包含两类标签。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只验证，不保存")
    args = parser.parse_args()
    output_path = SPLIT_DIR / "cv_folds.csv"
    if not args.dry_run and output_path.exists():
        raise FileExistsError("cv_folds.csv 已存在，不会覆盖已共享的交叉验证划分。")

    indices = pd.read_csv(SPLIT_DIR / "train_indices.csv")
    folds = build_cv_folds(indices)
    print(f"训练集：{len(folds)} 条；随机种子：{RANDOM_STATE}")
    for fold in range(1, N_SPLITS + 1):
        valid = folds.loc[folds["fold"].eq(fold)]
        print(
            f"第 {fold} 折：训练 {len(folds) - len(valid)} 条，验证 {len(valid)} 条；"
            f"验证订购 {valid['y'].sum()} 条，占 {valid['y'].mean():.4%}"
        )
    print("验证通过：每条记录恰好验证一次，所有轮次的训练/验证组均不交叉。")
    print("未读取测试集，未执行编码、缩放、SMOTE 或模型训练。")
    if args.dry_run:
        print("仅验证：未保存交叉验证划分。")
        return
    folds.to_csv(output_path, index=False, mode="x")
    print(f"已保存：{output_path}")


if __name__ == "__main__":
    main()
