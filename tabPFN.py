from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor  
def batch_predict(regressor, X, batch_size=10000):
    # Convert to numpy array if it's a DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch = X[i:batch_end]
        
        # Predict on the current batch
        batch_pred = regressor.predict(batch)
        predictions[i:batch_end] = batch_pred
        
        # Progress update
        print(f"Processed samples {i} to {batch_end-1} of {n_samples}")
    
    return predictions


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

# Train-test split
train_data, eval_data = train_test_split(train_data, test_size=0.75, random_state=1)
print("train_data.shape: ", train_data.shape)

# train_data = train_data.sample(frac=1, random_state=1)
# train_data, eval_data = train_data.iloc[:9900, :], train_data.iloc[15000:30000, :]

eval_data = eval_data.iloc[:9000, :]

# X_train, y_train = train_data.drop(['SaleID', 'price'], axis=1), train_data['price']
# X_eval, y_eval = eval_data.drop(['SaleID','price'], axis=1), eval_data['price']
# X_test = test_data.drop(['SaleID', 'price'], axis=1, errors='ignore')
X_train, y_train = train_data.drop(['price'], axis=1), train_data['price']
X_eval, y_eval = eval_data.drop(['price'], axis=1), eval_data['price']
X_test = test_data.drop(['price'], axis=1, errors='ignore')

# Initialize the regressor
regressor = TabPFNRegressor(
    n_jobs=info['NJ'],
    device='cuda:0',
    inference_precision="autocast",
    ignore_pretraining_limits=True,
)
print("trainning...")
regressor.fit(X_train, encode(y_train))

# Predict on the test set
print("evaluating...")
eval_predictions = decode(batch_predict(regressor, X_eval, batch_size=10000))

# Evaluate the model
mae = mean_absolute_error(y_eval, eval_predictions)
r2 = r2_score(y_eval, eval_predictions)

print("MAE: ", mae)
print("R² Score:", r2)

# Predict on the test set
print("predicting...")
test_predictions = decode(batch_predict(regressor, X_test, batch_size=10000))
# save .np
np.save('tabpfn_predictions.npy', test_predictions)