import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def output(final_model, test_data: pd.DataFrame, **kwargs):
    col_a = kwargs['col_a']  # id
    col_b = kwargs['col_b']  # price
    X = test_data.drop(columns=[col_a])
    y_pred = final_model.predict(X)
    # output to csv
    test_data[col_b] = y_pred
    test_data[col_b] = np.expm1(test_data[col_b])
    test_data[[col_a, col_b]].to_csv('result.csv', index=False)
    # Price Distribution (Predicted)
    plt.hist(test_data[col_b], bins=50, color='blue')
    plt.title('Prediction Distribution')
    plt.savefig("Price Distribution (Predicted)")
    plt.clf()
    # Price Distribution with KDE (Predicted)
    plt.hist(test_data[col_b], bins=50, color='blue', density=True)
    test_data[col_b].plot(kind='kde', color='red')
    plt.title('Prediction Distribution with KDE')
    plt.savefig("Price Distribution with KDE (Predicted)")
    plt.clf()