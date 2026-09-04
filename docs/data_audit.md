# 数据检查记录

## 数据来源与版本
- 来源：UCI Bank Marketing
- 使用文件：bank-additional-full.csv
- 字段说明：bank-additional-names.txt
- CSV 分隔符：分号
- 原始规模：41188 行、20 个输入特征、1 个目标变量
- 原始文件保持不变

## 目标变量
- y=yes：4640 条，占 11.27%
- y=no：36548 条，占 88.73%
- 任务：预测客户是否订购定期存款
- 后续编码：yes=1，no=0
- 类别不平衡，不能仅使用准确率评价模型

## 数据质量检查
- 所有字段的 NaN 数量均为 0
- unknown 数量：
  - job：330
  - marital：80
  - education：1,731
  - default：8,597
  - housing：990
  - loan：990
- 与先前记录完全相同的额外行：12
- pdays=999：39673 行
- duration=0：4 行

## 数据处理原则
- job、marital、education、default、housing、loan 中的 unknown 均保留
- 后续通过 One-hot 编码将 unknown 表示为独立类别
- 在排除任何字段前，按原始全部 21 列删除额外的完全重复记录，每组保留第一条
- 原始检查发现 12 条额外重复记录，预期去重后为 41176 条，以脚本验证为准
- 没有客户 ID，不能证明这些记录一定是重复录入；完整去重是本项目采用的清洗假设
- pdays=999 表示此前未被联系，后续拆为 previously_contacted=0 和 days_since_previous_contact=0
- pdays 不为 999 时，previously_contacted=1，days_since_previous_contact 保留原始天数
- 未联系时的 0 只是占位值，必须与是否联系过的字段一起使用
- 主实验排除 duration，因为通话前无法获取最终通话时长
- 不仅凭数值极端就删除记录
- 原始 CSV 不改写；row_id 始终表示原始 CSV 从 0 开始的数据行位置，去重后不重新编号
- 编码器、填补器和缩放器等需要学习参数的步骤，
  只在训练数据上拟合；交叉验证时在每个训练折内拟合

## 主实验设定

### 预测场景
- 在本次营销电话开始前预测客户是否订购定期存款
- 排除 duration
- 假设本次联系的渠道、日期和联系次数已知
- campaign 包含本次联系，需要按该场景解释
- 不将实验结果直接解释为实际部署效果

### 数据划分与评估
- 模板依据：Sample Final Report 第 3.2 节采用 70% 训练集、30% 测试集，Min-Max 缩放和训练数据 SMOTE
- 本项目补充：完整去重、按输入组合分组、近似分层和随机种子 42，并非模板明确指定的全部参数
- 先按全部 21 列去重，再分离 y、排除 duration，按剩余 19 个输入特征建立 group_id
- 分组不包含 y、duration 或 row_id；即使 y 不同，相同输入组合仍属于同组
- 同组记录整体进入训练集或测试集；允许集合内部相同输入，不允许跨集合相同输入
- 以约 70% 训练集、30% 测试集为目标；在不拆组的前提下尽量保持去重后数据的类别比例
- 由于组大小不同，记录数比例和订购比例可能有小幅偏差，不声称精确分层
- 随机种子为 42
- 实现方法：StratifiedGroupKFold 将全部组分配到 10 份，固定取第 0、1、2 份作为测试集，其余作为训练集
- 上述 10 份仅用于实现约 7:3 的留出划分，不是模型的交叉验证折数；不按模型成绩挑选份数或随机种子
- 模型调参统一采用 5 折分层分组交叉验证，随机种子 42，并遵守相同的分组约束（已由用户确认）
- 三位成员共享生成的 train_indices.csv 和 test_indices.csv，不各自重新划分
- 划分文件字段：row_id（原始行位置）、group_id（输入组合分组编号）、y（no=0，yes=1）
- row_id 和 group_id 仅用于追踪与划分，不能作为模型输入
- 预处理和重采样仅在训练折内拟合或执行
- unknown 保留为独立类别；编码、Min-Max 缩放和 SMOTE 不在本划分脚本中执行
- 测试集不做 SMOTE，保留划分后的自然类别比例
- 测试集不参与调参或模型选择
- 随机分组划分衡量对未见输入组合的预测表现，不代表未来时期或客户层面独立的表现
- 缺少客户 ID，即使输入组合完全隔离，仍不能保证同一客户不跨集合


## 划分脚本与运行方式
- 脚本：src/02_split_data.py
- uv run python src/02_split_data.py --dry-run：仅计算和验证，不保存文件
- uv run python src/02_split_data.py：验证后保存新版划分；已有任一划分文件时拒绝覆盖
- 验证项目：完整去重范围、原始行号无交叉无遗漏、标签对应正确、真实输入组合跨集合重合数为 0
- 原始检查统计仍描述 41188 条数据；新版划分应使用去重后的统计，不能混用

### 已生成的正式划分结果
- 完整去重删除 12 条，保留 41176 条；其中订购 4639 条，占 11.2663%
- 19 维输入特征共有 39167 种组合
- 训练集：28822 条，占 69.9971%；订购 3247 条，占 11.2657%
- 测试集：12354 条，占 30.0029%；订购 1392 条，占 11.2676%
- 跨集合相同输入特征组合：0
- 已运行脚本生成正式划分文件，并与上述统计核对一致

## 确定性特征处理
- 脚本：src/03_prepare_features.py
- 严格按照已保存划分的 row_id 顺序读取原始记录，不重新划分或改变记录数量
- 将 y 单独保存；从输入中排除 duration，将 pdays 替换为此前是否联系过和间隔天数两个字段
- 保留 unknown 和其他字段原值；此步骤不执行编码、缩放、填补或重采样
- 训练特征为 28,822 行、20 列；测试特征为 12,354 行、20 列
- 输出位置：data/processed；文件为 X_train.csv、y_train.csv、X_test.csv、y_test.csv
- X 文件仅含 20 个输入字段，y 文件仅含 y；两者第 i 行与对应 indices 文件的第 i 行一一对应
- row_id、group_id 仍从 data/splits 中读取，不加入 X 文件，不作为模型输入
- uv run python src/03_prepare_features.py --dry-run：仅验证，不保存文件
- uv run python src/03_prepare_features.py：验证后保存四个文件；任何目标文件已存在时拒绝覆盖
- 已由用户运行脚本生成正式处理文件，行列数与上述结果一致
- 后续编码器和缩放器只在训练数据上拟合；交叉验证时必须在每个训练折内部拟合

### pdays 字段替换与特征数量变化

依据原始说明文件 bank-additional-names.txt，pdays=999 表示此前未被联系，并非真实间隔 999 天。为避免模型将该特殊编码当作普通天数，本项目将一个 pdays 字段替换为两个字段；这是针对 Bank Marketing 的处理选择，不是示例报告规定的设定。

| 原始 pdays | previously_contacted（此前是否联系过） | days_since_previous_contact（间隔天数） |
|---|---:|---:|
| 999 | 0 | 0（占位值） |
| 非 999 的有效天数 d | 1 | d（保留原值） |

- 未联系时的 0 仅作占位，不是推断真实间隔为 0 天；必须结合 previously_contacted 解读。
- 例如，原始 pdays=0 转换为 (1, 0)，原始 pdays=999 转换为 (0, 0)，两种情况仍可区分。
- 两个新字段替换原字段，模型输入中不再保留 pdays；原始 CSV 不修改。
- 输入特征数量变化：原始 20 个输入特征 → 排除 duration 后 19 个 → 移除 pdays 并增加两个字段后 20 个。
- 这里的 20 个特征是 One-hot 编码之前的数量；y、row_id 和 group_id 均不计入模型输入。
- 转换只依赖每条记录自身的 pdays 值，不学习统计参数、不改变行数，也不会把 unknown 替换成其他类别。

## 公共编码与缩放规则

- 模板采用类别编码与 Min-Max 缩放；以下字段分配及未见类别策略是本项目确认的具体实现。
- One-hot 字段：job、marital、education、default、housing、loan、contact、month、day_of_week、poutcome。
- Min-Max 字段：age、campaign、previous、emp.var.rate、cons.price.idx、cons.conf.idx、euribor3m、nr.employed、days_since_previous_contact。
- previously_contacted 已是 0/1，直接保留；不将 y、row_id、group_id 输入预处理器。
- OneHotEncoder 使用 categories="auto"、drop=None、handle_unknown="ignore"，保留训练中已出现的每个类别，不自动删除参考类别列，不合并稀有类别。
- 若 unknown 在本次训练数据中出现，它拥有独立编码列；不会被替换成 no 或众数。
- 预测时遇到训练中未见的类别，该字段对应的全部 One-hot 列为 0；其他字段正常转换，不报错，也不将新类别改写为 unknown。
- 如果某训练折没有出现 unknown，那么该折遇到 unknown 时也按未见类别处理；不能为获取类别信息查看验证集或测试集。
- MinMaxScaler 使用 feature_range=(0, 1)、clip=False；最小值、最大值仅从本次训练数据学习。
- 验证集或测试集超出训练范围时，允许缩放值小于 0 或大于 1，不重新拟合或裁剪。
- 为方便六种模型共用，当前输出为稠密数值数组（sparse_output=False）；列名通过 get_feature_names_out() 获取，不手工猜测顺序。
- 编码后列数由本次训练数据实际出现的类别决定；不同交叉验证训练折可能不同，同一个拟合器的输出列始终固定。
- 公共代码：src/preprocessing.py 中的 build_preprocessor() 每次返回全新、未拟合的转换器。
- 后续将预处理器与各自模型组合进 Pipeline；交叉验证时每个训练折单独拟合，不能提前转换完整训练集后直接做交叉验证。
- 检查命令：uv run python src/04_check_preprocessing.py；仅使用 X_train.csv 试运行及人工未见类别测试，不读取真实测试集，不保存拟合状态或转换文件。
- 单元测试：uv run python -m unittest discover -s tests -p 'test_preprocessing.py'。
- 本阶段不执行 SMOTE，也不训练模型；data/processed 中的数据继续保持未编码、未缩放的版本。

## 六种模型共用的交叉验证划分

- 模型：逻辑回归、决策树、随机森林、KNN、XGBoost、朴素贝叶斯；三位成员各负责两种。
- 仅在已保存的 28,822 条训练记录内部使用 StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)。
- 统一 5 折是本项目确认的设置，不是模板明确规定的折数。
- 每轮使用 4 份训练、1 份验证，每条训练记录恰好参与一次验证；同一 group_id 不能跨折。
- 在整组不拆分的前提下尽量保持类别比例接近，各折大小和订购比例允许小幅偏差。
- 六种模型及其各组候选参数使用同一份保存的折号，不各自随机划分。
- 在每一轮的训练折上拟合编码器、缩放器；若使用 SMOTE，也只在训练折内执行，不对验证折重采样。
- 用交叉验证结果选择参数后，再用全部训练集重新拟合最终流程；外部测试集不参与交叉验证、调参或模型选择。
- 脚本：src/05_make_cv_folds.py；输入仅为 data/splits/train_indices.csv，不读取测试集。
- 输出：data/splits/cv_folds.csv，包含 train_position、row_id、group_id、y、fold。
- train_position 是 X_train.csv / y_train.csv 的从 0 开始的行位置；row_id 是原始 CSV 的行位置，两者不能混用。
- fold 为 1 至 5；运行第 k 轮时，fold=k 的行为验证数据，其余行为训练数据。
- fold 和所有编号都不是模型输入。正式训练前应检查 X_train、y_train 与保存划分的顺序一致。
- uv run python src/05_make_cv_folds.py --dry-run：仅验证；不带 --dry-run 时保存文件，已有文件时拒绝覆盖。
- 交叉验证折号已由用户运行脚本保存；本阶段不训练模型，各模型候选参数仍需后续确认。

## 调参主指标

- 六种模型统一使用五折验证集 Average Precision（AP）的算术平均值选择参数，越高越好。
- 实现时使用 scoring="average_precision"，正类为订购客户 y=1；输入应为连续预测分数或正类概率，不能先转成 0/1 预测标签。
- 选择理由：订购客户仅约 11%，希望同时关注找回订购客户的能力及所选客户的准确性；目前未指定营销成本、收益或联系预算。
- AP 是针对本项目补充的主指标，不是模板明确指定的设置；它与梯形法计算的 PR 曲线面积不完全相同。
- ROC-AUC 为辅助指标，最终保留模板的 Precision、Recall、F1、Accuracy、RMSE、混淆矩阵等评价内容；阈值规则后续确定。
- 所有参数选择只使用训练集内部的交叉验证结果，不依据外部测试集成绩修改参数或选择划分。

## 合成过采样方案（已确认参数）

- 采用适合数值和分类混合输入的 SMOTENC 作为调整
- 训练折处理顺序：在训练折拟合数值 Min-Max 缩放 → 对混合字段执行 SMOTENC → 约束新增记录 → One-hot 编码 → 模型训练。
- 原始分类字段及 previously_contacted 按类别处理，避免将类别指示值当作连续量插值。
- 验证折和测试集只应用训练折学到的转换，不执行过采样；合成样本不写回原始数据、划分文件或共享的未编码特征文件。
- SMOTENC 不保证提高模型表现，也不保证字段之间的业务关系自动成立。
- 参数：sampling_strategy=1.0（训练折中订购与未订购样本补至 1:1）、k_neighbors=5、random_state=42；均已由用户确认。
- 1:1 对齐示例平衡类别的做法；邻居数 5 是本项目采用的起点
- 合成记录约束：previously_contacted=0 时，将 days_since_previous_contact 设为原始尺度的 0 所对应的缩放值。
- 代码从当前训练折真实未联系记录提取该占位值，只调整新增合成记录并记录 n_adjusted_；不假定所有缩放情形下原始 0 都映射为数值 0。
- 如果真实未联系记录的间隔值不一致，报错检查，不自动修改真实数据。
- 约束仅保障上述两字段的关系，不保证所有业务关系成立；计数、年龄和宏观指标插值后的业务合理性仍是合成样本的限制。
- src/preprocessing.py 保留为不含采样器的基础规则和 04 检查流程；正式含采样训练改用 src/training_pipeline.py 的 build_training_pipeline(model)，不能再在它之前重复编码或缩放。
- build_training_pipeline 返回 imblearn Pipeline，将全部步骤与模型一起交给交叉验证搜索器；预测时自动跳过采样步骤。
- 数值缩放器只在原始训练折上拟合；采样后新增分类值须来自该训练折已有类别，随后 One-hot 编码使用已确认的未见类别全 0 规则。
- 输入继续使用 data/processed 中未编码、未缩放的 20 列特征；六种模型共用同一流程。
- 依赖包为 imbalanced-learn（Python 导入名为 imblearn）。
- 检查脚本：uv run python src/06_check_sampling.py，仅使用第 1 个 CV 训练折采样，验证折只转换，不读取外部测试集，不保存合成数据，不训练模型。
- 全部折检查：uv run python src/06_check_sampling.py --all-folds；每折分别创建并拟合新的缩放器、采样器和编码器，不跨折复用拟合参数。
- 测试文件：tests/test_sampling.py，使用人工数据验证仅修正合成记录、真实记录不变、预测阶段跳过采样及 Pipeline 可克隆。
