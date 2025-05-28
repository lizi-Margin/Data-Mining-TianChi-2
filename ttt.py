from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor  
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from tabpfn_extensions.rf_pfn import RandomForestTabPFNRegressor
import numpy as np
import pandas as pd
from mapping import *
import os

DIR='./tabpfn_ensem/'

X_eval = pd.read_csv(f'{DIR}/eval_X.csv')
y_eval = pd.read_csv(f'{DIR}/eval_y.csv')

eval_pred = []
dir_files = os.listdir(path=DIR)
for f in dir_files:
    if f.endswith(".npy"):
        this_pred = np.load(f"{DIR}/{f}", allow_pickle=True)
        eval_pred.append(this_pred)
        mae = mean_absolute_error(decode(y_eval), this_pred)
        r2 = r2_score(decode(y_eval), this_pred)

        print("MAE: ", mae)
        print("R² Score:", r2)

eval_pred = np.stack(eval_pred)
eval_pred = np.mean(eval_pred, axis=0)

# Evaluate the model
mae = mean_absolute_error(decode(y_eval), eval_pred)
r2 = r2_score(decode(y_eval), eval_pred)

print("MAE: ", mae)
print("R² Score:", r2)