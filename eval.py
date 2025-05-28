# compute eval metrics for a model, in this regression task, we use MAE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mapping import *

def get_mae(model, eval_data, **kwargs):
    col_a = kwargs['col_a']  # id
    col_b = kwargs['col_b']  # price
    X = eval_data.drop(columns=[col_a, col_b])
    y = eval_data[col_b]
    model_name = model.__class__.__name__
    print(f'Model: {model_name}')
    y_pred = model.predict(X)
    mae = mean_absolute_error(decode(y), decode(y_pred))
    print(f'MAE: {mae:.4f}')
    return mae

def eval_metrics(model, eval_data, **kwargs):
    col_a = kwargs['col_a']  # id
    col_b = kwargs['col_b']  # price
    X = eval_data.drop(columns=[col_a, col_b])
    y = eval_data[col_b]

    model_name = model.__class__.__name__
    print(f'Model: {model_name}')
    y_pred = model.predict(X)
    y = decode(y)
    y_pred = decode(y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # output and visualize
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R2: {r2:.4f}')
    # visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'True vs Predicted ({model_name})')
    plt.savefig(f'True vs Predicted ({model_name}).png')
    plt.clf()