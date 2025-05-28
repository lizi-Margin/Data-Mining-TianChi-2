import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


test_data_A = pd.read_csv('../used_car_testA_20200313.csv', sep=' ')
test_data_B = pd.read_csv('../used_car_testB_20200421.csv', sep=' ')
test_data = test_data_B
col_a = 'SaleID'
col_b = 'price'

print('Test data shape: ', test_data.shape)
test_data[col_b] = np.load('./tabpfn_rf_predictions.npy', allow_pickle=True)
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

old_res = pd.read_csv('../result.csv')
print(mean_absolute_error(old_res[col_b], test_data[col_b]))