"""按已保存的划分准备原始特征；
按已有划分取数据，排除 duration，把 pdays 拆成两个字段，分离标签 y
保存未编码、未缩放的 X_train、X_test、y_train、y_test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def transform_features(rows):
    """仅执行确定性转换，不从数据学习参数，也不修改输入。"""
    features = rows.drop(columns=["duration", "y"]).copy()
    pdays = features.pop("pdays")
    contacted = pdays.ne(999)
    features["previously_contacted"] = contacted.astype(int)
    features["days_since_previous_contact"] = pdays.where(contacted, 0)
    return features.reset_index(drop=True)


def prepare_datasets(raw, train_indices, test_indices):
    """保持划分文件的行顺序，返回独立的 X 和 y。"""
    if raw.shape[1] != 21 or not {"duration", "pdays", "y"}.issubset(raw.columns):
        raise ValueError("原始 CSV 的结构不符合预期。")
    if raw.isna().any().any() or not raw["y"].isin(["no", "yes"]).all():
        raise ValueError("原始 CSV 包含意外的空值或未知目标标签，请先检查。")

    for name, indices in [("训练集", train_indices), ("测试集", test_indices)]:
        if set(indices.columns) != {"row_id", "group_id", "y"}:
            raise ValueError(f"{name}划分必须包含 row_id、group_id、y 三列。")
        if indices.empty or indices.isna().any().any():
            raise ValueError(f"{name}划分为空或存在缺失信息。")
        if not all(pd.api.types.is_integer_dtype(indices[c]) for c in indices):
            raise ValueError(f"{name}划分中的行号、组号和标签必须是整数。")
        if not indices["row_id"].between(0, len(raw) - 1).all():
            raise ValueError(f"{name}包含超出原始 CSV 范围的行号。")
        if not indices["row_id"].is_unique:
            raise ValueError(f"{name}存在重复行号。")
        expected_y = raw.iloc[indices["row_id"]]["y"].map({"no": 0, "yes": 1})
        if expected_y.tolist() != indices["y"].tolist():
            raise ValueError(f"{name}标签与原始数据不对应。")

    train_ids = set(train_indices["row_id"])
    test_ids = set(test_indices["row_id"])
    retained_ids = set(np.flatnonzero(~raw.duplicated().to_numpy()))
    if train_ids & test_ids or train_ids | test_ids != retained_ids:
        raise ValueError("划分存在交叉或遗漏，或不符合完整去重保留第一条的规则。")
    if set(train_indices["group_id"]) & set(test_indices["group_id"]):
        raise ValueError("训练集与测试集存在跨集合分组。")

    outputs = {}
    for name, indices in [("train", train_indices), ("test", test_indices)]:
        rows = raw.iloc[indices["row_id"]]
        features = transform_features(rows)
        if features.shape != (len(indices), 20):
            raise ValueError("转换后的输入特征应为 20 列，且不改变行数。")
        if {"row_id", "group_id", "y", "duration", "pdays"} & set(features.columns):
            raise ValueError("模型输入中混入了不应保留的字段。")
        unchanged_columns = [c for c in raw if c not in ["duration", "y", "pdays"]]
        pd.testing.assert_frame_equal(
            features[unchanged_columns], rows[unchanged_columns].reset_index(drop=True)
        )
        # y 和 X 的第 i 行始终对应划分文件的第 i 行。
        outputs[f"X_{name}.csv"] = features
        outputs[f"y_{name}.csv"] = indices[["y"]].reset_index(drop=True).copy()

    train_features = outputs["X_train.csv"]
    test_features = outputs["X_test.csv"]
    if not train_features.drop_duplicates().merge(
        test_features.drop_duplicates(), on=list(train_features.columns), how="inner"
    ).empty:
        raise ValueError("转换后出现跨集合相同输入组合。")
    return outputs


def save_outputs(outputs, output_dir):
    """所有目标文件均不存在时才写入，防止覆盖共享数据。"""
    existing = [name for name in outputs if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"处理结果已存在，不会覆盖：{', '.join(existing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        data.to_csv(output_dir / name, index=False, mode="x")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只检查，不生成文件")
    args = parser.parse_args()
    raw = pd.read_csv(DATA_PATH, sep=";")
    train_indices = pd.read_csv(SPLIT_DIR / "train_indices.csv")
    test_indices = pd.read_csv(SPLIT_DIR / "test_indices.csv")
    outputs = prepare_datasets(raw, train_indices, test_indices)

    for name, data in outputs.items():
        print(f"{name}：{data.shape[0]} 行，{data.shape[1]} 列")
    print("验证通过：划分及标签对应正确，其他字段含 unknown 保持不变。")
    print("已排除 duration；pdays 已拆为两个字段；转换后跨集合相同输入组合为 0。")
    print("尚未执行 One-hot 编码、Min-Max 缩放、SMOTE 或模型训练。")
    if args.dry_run:
        print("仅验证：未写入处理结果。")
        return
    save_outputs(outputs, OUTPUT_DIR)
    print(f"已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
