# SC6122 Bank Marketing Group Project

目标：在本次营销电话开始前，预测客户是否订购定期存款。正类 `y=1` 表示订购，`y=0` 表示未订购。假设本次联系渠道、日期及包含本次联系的次数已知；不使用通话结束后才能知道的 `duration`。

## 从这里开始

当前已完成公共数据清洗、训练/测试划分、5 折分组划分，以及编码、缩放、SMOTENC 流程。全部 5 折采样检查和 7 项单元测试已通过。

本仓库仅作为统一数据、划分和实验规则的发布入口

当前尚未提供六种模型的正式训练脚本。各成员自行设计并记录自己模型的候选参数；朴素贝叶斯变体需明确说明，最终分类阈值口径仍待统一。不要把检查脚本当作模型训练结果。

阅读顺序：

1. 本 README：如何使用统一数据及实验规则。
2. [数据检查与实验设定](docs/data_audit.md)：处理理由、模板设定与本项目调整。
3. [公共训练流程](src/training_pipeline.py)：正式训练必须调用的入口。
4. [字段配置](src/preprocessing.py)：分类、数值、二元字段名单。
5. [全折采样检查](src/06_check_sampling.py)：如何按折取数据、只在训练折拟合和采样。

## 三人分工

| 成员 | 模型 | 其他职责 |
|---|---|---|
| A（公共数据负责人） | 逻辑回归、决策树 | 统一数据与划分，维护公共流程，汇总实验口径 |
| B | 随机森林、KNN | 负责各自模型的调参、评估、解释和报告内容 |
| C | XGBoost、朴素贝叶斯 | 负责各自模型的调参、评估、解释和报告内容 |


## 建立环境与验证

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后下载仓库 ZIP 并解压，或克隆仓库，进入包含 `pyproject.toml` 的项目根目录。以下所有命令均在根目录运行。不使用 Git 也可以完成所有数据检查和训练。

```bash
uv sync --locked
uv run --locked python --version
uv run --locked python -m unittest discover -s tests
uv run --locked python src/06_check_sampling.py --all-folds
```

- 使用 `.python-version` 指定的 Python 3.12 和 `uv.lock` 锁定的依赖；不复制别人的 `.venv`。
- `--locked` 用于避免意外更新锁文件。依赖报错时先反馈，不各自升级或降级。
- 采样检查不读取外部测试集、不训练模型，也不保存合成数据。
- **当前依赖尚未包含 XGBoost。** C 可先运行公共检查；正式实现 XGBoost 时，需统一加入依赖并提交更新后的 `pyproject.toml` 与 `uv.lock`。
- 不默认使用 GPU，不同时在外层调参和模型内部开启全部 CPU 核心。

## 训练用哪些数据？

| 文件 | 内容 | 使用时机 |
|---|---|---|
| [X_train.csv](data/processed/X_train.csv) | 28,822 行、20 个输入字段，未编码、未缩放 | 模型调参和最终训练 |
| [y_train.csv](data/processed/y_train.csv) | 同序的 0/1 标签 | 模型调参和最终训练 |
| [cv_folds.csv](data/splits/cv_folds.csv) | 六种模型共用的 5 折分配 | 构造训练/验证行位置 |
| [train_indices.csv](data/splits/train_indices.csv) | 训练记录的原始行号、组号和标签 | 核验对应关系 |
| [X_test.csv](data/processed/X_test.csv) | 12,354 行、20 个输入字段 | 仅在方案冻结后的最终评估使用 |
| [y_test.csv](data/processed/y_test.csv) | 同序测试标签 | 仅在方案冻结后的最终评估使用 |
| [test_indices.csv](data/splits/test_indices.csv) | 测试记录的原始行号、组号和标签 | 最终预测结果对应回原始记录 |
| [原始数据](data/raw/bank-additional-full.csv) | 未修改的 41,188 行、21 列 CSV | 来源留档，不直接作为各自的训练入口 |
| [官方字段说明](data/raw/bank-additional-names.txt) | 字段含义及特殊值定义 | 理解数据与解释结果 |

`X_train`、`y_train`、`train_indices`、`cv_folds` 的第 i 行对应同一条训练记录。测试集的 X、y、indices 也保持相同行序。不要单独排序、删行或重置其中某个文件的记录顺序。

### 三种编号的区别

- `row_id`：原始 CSV 从 0 开始的数据行位置；完整去重后保留原编号。
- `train_position`：当前 X_train / y_train 中从 0 开始的行位置，适用于 `.iloc`。
- `group_id`：排除 duration、y 后，19 个原始输入完全相同的记录所属组。
- `fold`：1 至 5；第 k 轮用 `fold=k` 的记录验证，其余记录训练。

**这些编号和 fold 都不能作为模型输入。不能拿 row_id 去对 X_train 做 `.iloc`。**

## 已固定的实验规则

| 项目 | 规则 |
|---|---|
| 完整去重 | 原始全部 21 列相同的记录保留第一条；去掉 12 条，剩余 41,176 条 |
| 主实验特征 | 排除 duration；不输入 y 或编号 |
| unknown | 保留为分类状态，不删行、不填成 no 或众数 |
| pdays | 替换为 previously_contacted 和 days_since_previous_contact；未联系时为 (0, 0) |
| 外部划分 | 约 70%/30%，输入组合不跨集合，种子 42；不再重新划分 |
| 调参划分 | 固定 5 折分层分组，使用 cv_folds.csv；不改成普通 cv=5 |
| 主调参指标 | 五折平均 AP，`scoring="average_precision"`，越高越好 |
| 缩放 | 每个原始训练折内拟合 Min-Max；范围 (0,1)，不裁剪验证/测试超范围值 |
| 采样 | SMOTENC，sampling_strategy=1.0、k_neighbors=5、random_state=42；仅训练折执行 |
| 合成记录约束 | 合成记录若此前未联系，间隔设为原始 0 对应的缩放值；统计调整数，不改真实记录 |
| 编码 | One-hot 保留全部类别列；训练时未见的类别编码为该字段全 0 |
| 最终评估 | AP、ROC-AUC，及模板的 Precision、Recall、F1、Accuracy、RMSE、混淆矩阵；阈值规则待确认 |

当前每折检查输出 63 个编码后字段，但不要在模型代码里硬编码 63；输出列数应从实际拟合的转换器获取。

## 模型代码怎么接公共流程？

正式入口是 `training_pipeline.build_training_pipeline(model)`，流程为：

```text
未编码的 20 列输入
  → Min-Max（只在原始训练折拟合）
  → SMOTENC + 合成记录约束（仅训练阶段）
  → One-hot
  → 本人负责的模型
```

将模型脚本放在 `src/`，可直接 `from training_pipeline import build_training_pipeline`。输入 `model` 应是尚未拟合的模型实例。

构造交叉验证时使用下面的行位置逻辑。此片段只读取训练数据，可写进各自的 `src/` 脚本；它不是完整的调参程序。

```python
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
X = pd.read_csv(ROOT / "data/processed/X_train.csv")
y = pd.read_csv(ROOT / "data/processed/y_train.csv")["y"]
folds = pd.read_csv(ROOT / "data/splits/cv_folds.csv")
indices = pd.read_csv(ROOT / "data/splits/train_indices.csv")

assert len(X) == len(y) == len(folds)
assert np.array_equal(folds["train_position"], np.arange(len(X)))
assert np.array_equal(folds["y"], y)
pd.testing.assert_frame_equal(folds[["row_id", "group_id", "y"]], indices)
assert set(folds["fold"]) == {1, 2, 3, 4, 5}
assert folds.groupby("group_id")["fold"].nunique().eq(1).all()

cv_splits = [
    (
        np.flatnonzero(folds["fold"].ne(k)),
        np.flatnonzero(folds["fold"].eq(k)),
    )
    for k in range(1, 6)
]
```

随后将完整 Pipeline 交给 GridSearchCV 等搜索器，设置 `cv=cv_splits`、`scoring="average_precision"`。模型参数名称加 `model__` 前缀。具体搜索范围由各模型负责人设计并记录，不要求六种模型使用相同参数网格，不根据外部测试结果调整。

- 不先拟合整个训练集的编码器或缩放器，再对转换结果做交叉验证。
- 不提前对整个训练集做 SMOTENC 再分折，也不把合成记录分配 fold。
- 不在 `build_training_pipeline()` 之前再次调用 `build_preprocessor()`，否则会重复转换。
- 不用普通 sklearn Pipeline 替换内部 imblearn Pipeline；预测时采样步骤必须被跳过。
- AP / ROC-AUC 使用连续分数或正类概率，不使用已经阈值化的 0/1 预测值。
- 最佳参数确定后，在全部训练集上重新拟合完整流程；拟合器与模型应一起保存，不仅保存裸模型。
- 基于重采样训练产生的概率不自动等于真实订购概率，不直接用于收益承诺。

## 脚本分别做什么？

| 文件 | 用途 | 新组员是否需要运行 |
|---|---|---|
| [01_check_data.py](src/01_check_data.py) | 检查原始数据 | 不必重跑，先读审计记录 |
| [02_split_data.py](src/02_split_data.py) | 完整去重并生成外部划分 | 不要重新生成共享划分 |
| [03_prepare_features.py](src/03_prepare_features.py) | 排除 duration、转换 pdays、保存 X/y | 已提供输出，不必重跑 |
| [04_check_preprocessing.py](src/04_check_preprocessing.py) | 无采样编码/缩放检查，不保存结果 | 可选 |
| [05_make_cv_folds.py](src/05_make_cv_folds.py) | 生成公共 5 折文件 | 不要重新生成共享折号 |
| [06_check_sampling.py](src/06_check_sampling.py) | 训练折采样与编码检查 | 运行 `--all-folds` |
| [preprocessing.py](src/preprocessing.py) | 字段名单及无采样基础流程 | 阅读；不作为含采样训练入口 |
| [training_pipeline.py](src/training_pipeline.py) | 正式训练公共流程 | 必须复用 |
| [tests/](tests/) | 人工数据的行为测试 | 运行 unittest |

## 每人最后交回什么？

每种模型至少交回以下材料，文件名带模型名，避免覆盖他人结果：

- 可复现的模型脚本，说明运行命令、候选参数和实际依赖。
- 候选参数及各折 AP、平均 AP、标准差和训练耗时；可导出搜索器的 `cv_results_`。
- 最佳参数和选择理由。不能只给一个最高分而不保留搜索过程。
- 最终评估阶段的指标、ROC/PR 曲线、混淆矩阵及解释；分类阈值统一后再生成对应结果。
- 必要时保存预测分数，带原始 row_id 方便核对；正式测试评估前不生成测试结果。
- 训练完成的完整 Pipeline、对应训练脚本、依赖版本及所用公共数据版本；只加载可信组员生成的模型文件。
- 对应报告段落、模型局限性、业务解读与个人贡献说明。

小型指标表可放 `results/metrics/`，图可放 `results/figures/`，模型放本地 `results/models/`。空目录不会被 Git 自动保存，写结果时自行创建。模型文件默认不入 Git，后续单独约定共享方式。



