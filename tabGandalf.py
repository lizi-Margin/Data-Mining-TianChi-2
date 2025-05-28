import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pre import *
from rich import print
import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_data_A = pd.read_csv('used_car_testA_20200313.csv', sep=' ')
test_data_B = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
# test_data = pd.concat([test_data_A, test_data_B], axis=0, ignore_index=True)
test_data = test_data_B
## 输出数据的大小信息
print('Train data shape: ', train_data.shape)
print('Test data shape: ', test_data.shape)

info = {
    'NJ': 16,
    'verbose': True,
    'search': False,

    'col_a': 'SaleID',
    'col_b': 'price',
}

## 数据探索
## 数据探索
train_data, test_data = 数据特征(train_data, test_data, **info)
train_data, test_data = 数据分布(train_data, test_data, **info)
train_data, test_data = 数据质量(train_data, test_data, **info)
train_data, test_data = 数据清洗(train_data, test_data, **info)
train_data, test_data = 数据转换(train_data, test_data, **info)
## 特征工程
# train_data, test_data =      特征处理(train_data, test_data, **info)
assert not train_data.isna().any().any(); assert not np.isinf(train_data.select_dtypes(include=np.number)).any().any()
assert not test_data .isna().any().any(); assert not np.isinf(test_data .select_dtypes(include=np.number)).any().any()

numeric_features, categorical_features = get_nc(train_data)
numeric_features.remove('price')
train_data, eval_data = train_test_split(train_data, test_size=0.2, random_state=1)
X_eval, y_eval = eval_data.drop('price', axis=1), eval_data['price']
print("min max of y_eval: ", min(y_eval), max(y_eval))


from pytorch_tabular.models import (
    TabTransformerConfig,
    GANDALFConfig,
    FTTransformerConfig
)
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

data_config = DataConfig(
    target=[
        'price'
    ],  # target should always be a list
    continuous_cols=numeric_features,
    categorical_cols=categorical_features,
)
trainer_config = TrainerConfig(
    # batch_size=1024,
    batch_size=512,
    max_epochs=100,
    accelerator='gpu',
    early_stopping=None
)
optimizer_config = OptimizerConfig(
    optimizer='Adam'
)
model_config = GANDALFConfig(
    task="regression",
    gflu_stages=6,
    gflu_feature_init_sparsity=0.3,
    gflu_dropout=0.0,
    learning_rate=8e-3,
)
# model_config = TabTransformerConfig(
#     task="regression",
#     learning_rate=8e-3
# )
# model_config = FTTransformerConfig(
#     task="regression",
#     learning_rate=8e-3
# )
from pytorch_tabular import TabularModel
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    verbose=True
)

tabular_model.fit(train=train_data, validation=eval_data)
# print(tabular_model.evaluate(eval_data))
# Predict on the test set
print("evaluating...")
eval_predictions = tabular_model.predict(X_eval)['price_prediction']
mae = mean_absolute_error(decode(y_eval), decode(eval_predictions))
r2 = r2_score(decode(y_eval), decode(eval_predictions))
# mae = mean_absolute_error(y_eval, eval_predictions)
# r2 = r2_score(y_eval, eval_predictions)
# print("min max of predictions: ", min(eval_predictions), max(eval_predictions))
# print("min max of y_eval: ", min(y_eval), max(y_eval))
print("min max of predictions: ", min(decode(eval_predictions)), max(decode(eval_predictions)))
print("min max of y_eval: ", min(decode(y_eval)), max(decode(y_eval)))
print("MAE: ", mae)
print("R² Score:", r2)

pred_df = tabular_model.predict(test_data)
# save to csv
pred_df.to_csv('tabGandalf_predictions.csv', index=False)


