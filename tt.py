from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor  
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from tabpfn_extensions.rf_pfn import RandomForestTabPFNRegressor
import numpy as np
import pandas as pd
from mapping import *
eval_predictions = np.load('tabpfn_rf_eval_pred.npy', allow_pickle=True)
X_eval = pd.read_csv('tabpfn_rf_eval_X.csv')
y_eval = pd.read_csv('tabpfn_rf_eval_y.csv')

# Evaluate the model
mae = mean_absolute_error(decode(y_eval), eval_predictions)
r2 = r2_score(decode(y_eval), eval_predictions)

print("MAE: ", mae)
print("R² Score:", r2)