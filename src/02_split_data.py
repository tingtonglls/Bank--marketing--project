"""完整去重后，按输入特征组合进行约 7:3 的分层分组划分。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
RANDOM_STATE = 42


def build_split(raw):
    """返回保留原始行号的划分；不修改 raw，不写文件。"""
    if raw.shape[1] != 21 or not {"y", "duration"}.issubset(raw.columns):
        raise ValueError("需要原始 bank-additional-full.csv：21 列，包含 y、duration。")
    if raw.isna().any().any():
        raise ValueError("发现意外的空值，请先检查输入；文本 unknown 应保持原样。")
    if not raw["y"].isin(["no", "yes"]).all():
        raise ValueError("目标变量只能包含 no 和 yes。")

    # 先比较原始全部 21 列，保留第一条；row_id 不能加入去重字段。
    keep = ~raw.duplicated(keep="first")
    row_ids = np.flatnonzero(keep.to_numpy())
    clean = raw.iloc[row_ids].copy()
    features = clean.drop(columns=["duration", "y"])

    # 按实际字段值分组，不包含 y、duration 或 row_id。
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(features), sort=False)
    records = pd.DataFrame({
        "row_id": row_ids,
        "group_id": groups,
        "y": clean["y"].map({"no": 0, "yes": 1}).to_numpy(),
    })
    if records["group_id"].nunique() < 10:
        raise ValueError("独立特征组合不足 10 个，无法使用当前划分方法。")

    # 10 份仅用于构造 70%/30% 留出集，不是模型交叉验证设置。
    # 固定取第 0、1、2 份作测试集，不按模型成绩挑选划分。
    splitter = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=RANDOM_STATE
    )
    test_mask = np.zeros(len(records), dtype=bool)
    for fold, (_, positions) in enumerate(
        splitter.split(features, records["y"], groups=records["group_id"])
    ):
        if fold < 3:
            test_mask[positions] = True

    train = records.loc[~test_mask].reset_index(drop=True)
    test = records.loc[test_mask].reset_index(drop=True)
    validate_split(raw, records, train, test)
    return records, train, test


def validate_split(raw, records, train, test):
    """验证去重范围、行号、标签以及真实输入组合不跨集合。"""
    if train.empty or test.empty:
        raise ValueError("训练集和测试集均不得为空。")
    combined = pd.concat([train, test], ignore_index=True)
    expected_ids = set(np.flatnonzero(~raw.duplicated().to_numpy()))
    if not combined["row_id"].is_unique or set(combined["row_id"]) != expected_ids:
        raise ValueError("划分存在行号重复、遗漏，或去重范围错误。")
    if len(combined) != len(records):
        raise ValueError("划分后的总记录数不正确。")
    if set(train["group_id"]) & set(test["group_id"]):
        raise ValueError("特征组跨越训练集与测试集。")
    for part in (train, test):
        expected_y = raw.iloc[part["row_id"]]["y"].map({"no": 0, "yes": 1})
        if expected_y.tolist() != part["y"].tolist():
            raise ValueError("保存的标签与原始记录不一致。")
        if set(part["y"]) != {0, 1}:
            raise ValueError("每个集合都应包含两种标签。")

    # 独立检查实际特征值，而不只依赖 group_id。
    columns = [c for c in raw.columns if c not in ["duration", "y"]]
    train_values = raw.iloc[train["row_id"]][columns].drop_duplicates()
    test_values = raw.iloc[test["row_id"]][columns].drop_duplicates()
    if not train_values.merge(test_values, on=columns, how="inner").empty:
        raise ValueError("检测到跨集合相同输入特征。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="只计算和验证，不生成或覆盖文件"
    )
    args = parser.parse_args()
    train_path = OUTPUT_DIR / "train_indices.csv"
    test_path = OUTPUT_DIR / "test_indices.csv"
    if not args.dry_run and (train_path.exists() or test_path.exists()):
        raise FileExistsError("划分文件已存在，请先检查；脚本不会覆盖已有划分。")

    raw = pd.read_csv(DATA_PATH, sep=";")
    records, train, test = build_split(raw)
    print(f"原始记录：{len(raw)}")
    print(f"删除完整重复记录：{len(raw) - len(records)}")
    print(f"去重后记录：{len(records)}")
    print(f"去重后订购比例：{records['y'].mean():.4%}")
    print(f"输入特征组合数：{records['group_id'].nunique()}")
    for name, part in [("训练集", train), ("测试集", test)]:
        print(
            f"{name}：{len(part)} 条，占 {len(part) / len(records):.4%}；"
            f"订购 {part['y'].sum()} 条，订购比例 {part['y'].mean():.4%}"
        )
    print("验证通过：无记录遗漏或交叉、标签正确、跨集合相同特征组合为 0。")

    if args.dry_run:
        print("仅验证：未写入任何划分文件。")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False, mode="x")
    test.to_csv(test_path, index=False, mode="x")
    print(f"已保存：{train_path}")
    print(f"已保存：{test_path}")


if __name__ == "__main__":
    main()
