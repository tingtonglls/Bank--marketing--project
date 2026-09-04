"""公共预处理配置：每个模型/交叉验证训练折都创建并拟合自己的实例。
定义公共规则：哪些列做 One-hot、哪些做 Min-Max，以及怎么处理未见类别
供04.py调用
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUMERIC_FEATURES = [
    "age", "campaign", "previous", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "days_since_previous_contact",
]
BINARY_FEATURES = ["previously_contacted"]


def build_preprocessor():
    """返回全新、未拟合的转换器；不读取数据，不保存拟合参数。

    unknown 若在训练数据中出现，则有独立编码列；未见类别的对应块全为 0。
    Min-Max 参数只从 fit 的数据学习；超出训练范围的值不裁剪。
    在交叉验证中应把此转换器放入模型 Pipeline，不能提前拟合全训练集。
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    categories="auto",
                    drop=None,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", MinMaxScaler(feature_range=(0, 1), clip=False), NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
