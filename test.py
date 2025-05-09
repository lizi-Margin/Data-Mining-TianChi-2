import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pre import *
from model import *
from eval import *
from output import *


import warnings
warnings.filterwarnings('ignore')



## 通过Pandas对于数据进行读取
train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_data_A = pd.read_csv('used_car_testA_20200313.csv', sep=' ')
test_data_B = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
# test_data = pd.concat([test_data_A, test_data_B], axis=0, ignore_index=True)
test_data = test_data_B
## 输出数据的大小信息
print('Train data shape: ', train_data.shape)
print('Test data shape: ', test_data.shape)
# print(train_data.head(n=15))

info = {
    'NJ': 16,
    'verbose': True,
    'search': False,

    'col_a': 'SaleID',
    'col_b': 'price',
}

## 数据探索
train_data, test_data = 数据特征(train_data, test_data, **info)
train_data, test_data = 数据分布(train_data, test_data, **info)
train_data, test_data = 数据质量(train_data, test_data, **info)
train_data, test_data = 数据清洗(train_data, test_data, **info)
train_data, test_data = 数据转换(train_data, test_data, **info)

## 特征工程
train_data, test_data =      特征处理(train_data, test_data, **info)
train_data, test_data = 特征选择与构造(train_data, test_data, **info)

assert not train_data.isna().any().any(); assert not np.isinf(train_data.select_dtypes(include=np.number)).any().any()
assert not test_data .isna().any().any(); assert not np.isinf(test_data .select_dtypes(include=np.number)).any().any()

## 模型训练
## 评估指标：MAE，RMSE，R2

# train_data, eval_data
train_data, eval_data = train_test_split(train_data, test_size=0.2, random_state=1)

# 模型训练与数据调优
print("开始训练")
lr_model = get_best_lr_model(train_data, **info); eval_metrics(lr_model, eval_data, **info)
rf_model = get_best_rf_model(train_data, **info); eval_metrics(rf_model, eval_data, **info)
xgb_model = get_best_xgb_model(train_data, **info); eval_metrics(xgb_model, eval_data, **info)

# 模型融合
soft_voting_of_best_model = get_soft_voting_model([xgb_model, rf_model], train_data, **info); eval_metrics(soft_voting_of_best_model, eval_data, **info)
final_model = soft_voting_of_best_model


# 模型预测
print('final MAE: ', get_mae(final_model, eval_data, **info))
# predict test data
output(final_model, test_data, **info)
