"""数据检查脚本"""

from pathlib import Path

import pandas as pd

# 根据脚本位置定位项目，不依赖终端当前所在目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"


def main():
    df = pd.read_csv(DATA_PATH, sep=";")

    print("=== 1. 数据规模 ===")
    print(f"行数：{df.shape[0]}")
    print(f"列数：{df.shape[1]}")

    print("\n=== 2. 每列的数据类型与缺失情况 ===")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique_count": df.nunique(dropna=False),
        "missing_count": df.isna().sum(),
        "unknown_count": df.eq("unknown").sum(),
    })
    print(summary.to_string())

    print("\n=== 3. 完全相同的重复行数量 ===")
    print(df.duplicated().sum())

    print("\n=== 4. 目标变量分布 ===")
    target_summary = pd.DataFrame({
        "count": df["y"].value_counts(),
        "percentage": df["y"].value_counts(normalize=True).mul(100),
    })
    print(target_summary.round(2).to_string())

    print("\n=== 5. 数值字段统计 ===")
    print(df.describe().round(2).to_string())

    print("\n=== 6. 特殊取值检查 ===")
    print(f"pdays = 999 的行数：{df['pdays'].eq(999).sum()}")
    print(f"duration = 0 的行数：{df['duration'].eq(0).sum()}")


if __name__ == "__main__":
    main()