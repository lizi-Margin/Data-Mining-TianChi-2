import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')



## 通过Pandas对于数据进行读取
Train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
Test_data_A = pd.read_csv('used_car_testA_20200313.csv', sep=' ')
Test_data_B = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
Test_data = pd.concat([Test_data_A, Test_data_B], axis=0, ignore_index=True)
print(Test_data)
## 输出数据的大小信息
print('Train data shape: ', Train_data.shape)
print('Test data shape: ', Test_data.shape)

## 通过.head()简要浏览读取数据的形式
Train_data.head()


Train_data.isnull().sum()


#缺失数据可视化
missing = Train_data.isnull().sum()
missing= missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()

Train_data.hist(bins=50,figsize=(20,15))
plt.cla()


Train_data['offerType'].value_counts()
Train_data['seller'].value_counts()
Train_data['creatDate'].value_counts()

# 自动生成 categorical_features 和 numeric_features
categorical_features = Train_data.select_dtypes(include=['object']).columns.tolist()
numeric_features = Train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical Features:", categorical_features)
print("Numeric Features:", numeric_features)


# 特征nunique分布
for cat_fea in categorical_features:
    print(cat_fea + "的特征分布如下:")
    print("{}特征有{}个不同的值".format(cat_fea, Train_data[cat_fea].nunique()))
    print(Train_data[cat_fea].value_counts())



# 1) 相关性分析
# 选择数值型特征
price_numeric = Train_data[numeric_features]
# 计算相关系数矩阵
correlation = price_numeric.corr()
# 对与price的相关系数进行排序并输出
correlation['price'].sort_values()



## 查看目标变量的具体频数
## 绘制标签的统计图，查看标签分布
plt.hist(Train_data['price'], 
         orientation='vertical', 
         histtype='bar', 
         color='red')  # 修复了未闭合的引号和括号
plt.savefig('绘制标签的统计图，查看标签分布.png')
plt.show()




# 目标变量进行对数变换（使其更接近正态分布）
Train_data['price'] = np.log1p(Train_data['price'])

# 绘制对数变换后的价格分布直方图
plt.hist(Train_data['price'], 
         orientation='vertical', 
         histtype='bar', 
         color='red') 
plt.title('Log-Transformed Price Distribution')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')
plt.savefig('./Log-Transformed Price Distribution.png')
plt.show()



# 使用seaborn绘制分布图（包含KDE曲线）
sns.distplot(Train_data['price'])  # 修正了括号和分号
plt.title('Price Distribution with KDE')
plt.savefig('Price Distribution with KDE.png')
plt.show()


# 计算并打印偏度和峰度
print("Skewness: %f" % Train_data['price'].skew())  # 偏度（衡量分布不对称性）
print("Kurtosis: %f" % Train_data['price'].kurt())  # 峰度（修正了EE为kurt，衡量分布尖锐程度）








# 对numeric类型col中的nan进行填midean
for col in numeric_features:
    if col != 'price':
        Train_data[col].fillna(Train_data[col].median(), inplace=True)
        Test_data[col].fillna(Test_data[col].median(), inplace=True)




if False:
    # columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
    # columns = Train_data.columns
    sns.pairplot(Train_data[numeric_features], height=2, diag_kind = None)  # Added pairplot visualization
    plt.savefig('pairplot.png')
    # plt.show()

    # print('pairplot.png saved')

    sys.exit(0)


# 对分类特征进行独热编码
def onehot_encode(data, feature):
    onehot = pd.get_dummies(data[feature], prefix=feature)
    data = pd.concat([data, onehot], axis=1)
    data.drop(feature, axis=1, inplace=True)
    return data
for cat_fea in categorical_features:
    Train_data = onehot_encode(Train_data, cat_fea)
    Test_data = onehot_encode(Test_data, cat_fea)














NJ = 30

# 定义必要的库和模型
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import mean_absolute_error

def perform_search(model, param_grid, X_train, y_train, X_val, y_val):
    print(f"Performing search for {model.__class__.__name__}...")
    search = BayesSearchCV(model, param_grid, n_iter=20, cv=3, scoring='neg_mean_absolute_error', n_jobs=NJ)
    search.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {search.best_params_}")
    
    # 使用最佳参数预测验证集
    best_model = search.best_estimator_
    val_predictions = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)
    print(f"MAE for {model.__class__.__name__}: {mae}")
    # 保存搜索结果到文件
    with open(f'{model.__class__.__name__}_search_results.txt', 'w') as f:
        f.write(f"Best parameters: {search.best_params_}\n")
        f.write(f"MAE: {mae}\n")
    # 绘制特征重要性图
    feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    feature_importances.nlargest(15).plot(kind='barh')
    plt.title(f'Feature Importances for {model.__class__.__name__}')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig(f'{model.__class__.__name__}_feature_importances.png')
    # plt.show()
    
    return best_model, mae

# 定义参数网格
param_lgb = {
    'num_leaves': Integer(31, 50),
    # 'learning_rate': Real(0.05, 0.1),
    'n_estimators': Integer(100, 200)
}
param_xgb = {
    'max_depth': Integer(5, 15),
    # 'learning_rate': Real(0.05, 0.1),
    'n_estimators': Integer(50, 150),
    # 'subsample': Real(0.5, 1.0),
    # 'colsample_bytree': Real(0.5, 1.0),
    # 'gamma': Real(0.0, 0.5),
}
param_rf = {
    'n_estimators': Integer(300, 400),
    'max_depth': Integer(30, 60),
    'min_samples_split': Integer(2, 3),
    # 'min_samples_leaf': Integer(1, 10),
    # 'min_weight_fraction_leaf': Real(0.0, 0.5),
    # 'max_features': Real(0.1, 1.0),
    # 'max_leaf_nodes': Integer(1, 20),
    'criterion': Categorical(['absolute_error']),
    # 'criterion': Categorical(['absolute_error', 'poisson', 'squared_error', 'friedman_mse']),
    
    # 'bootstrap': Categorical([True, False]),
    # 'max_samples': Real(0.1, 1.0),

    # 'oob_score': Categorical([True, False]),
}

# 定义特征和目标变量
X = Train_data.drop(columns=['price'])  # 假设 'price' 是目标变量
# X['notRepairedDamage'] = X['notRepairedDamage'].astype('category') 
y = Train_data['price']

# 将数据分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42)

# 对每个模型进行超参数搜索和评估
# model_rf, MAE_rf = perform_search(RandomForestRegressor(random_state=42, n_jobs=NJ), param_rf, X_train, y_train, X_val, y_val)
model_xgb, MAE_xgb = perform_search(XGBRegressor(random_state=42, n_jobs=1), param_xgb, X_train, y_train, X_val, y_val)
# model_lgb, MAE_lgb = perform_search(LGBMRegressor(random_state=42, n_jobs=NJ), param_lgb, X_train, y_train, X_val, y_val)



# models = [model_lgb, model_xgb, model_rf]
# MAEs = [MAE_lgb, MAE_xgb, MAE_rf]
# vals = [model.predict(X_val) for model in models]

# # 对所有模型进行加权融合
# weights = [1 - MAE / sum(MAEs) for MAE in MAEs]
# val_Weighted = sum(weight * val for weight, val in zip(weights, vals))
# val_Weighted[val_Weighted < 0] = 10
# print('MAE of val with Weighted ensemble:', mean_absolute_error(y_val, val_Weighted))

# # 对测试集进行预测
# subs = [model.predict(Test_data) for model in models]
# sub_Weighted = sum(weight * sub for weight, sub in zip(weights, subs))
# sub_Weighted[sub_Weighted < 0] = 10

# # 保存预测结果
# submission = pd.DataFrame({'SaleID': Test_data['SaleID'], 'price': sub_Weighted})
# submission.to_csv('submission.csv', index=False)

# # 查看预测值的分布
# plt.hist(sub_Weighted, bins=50, color='blue')
# plt.title('Prediction Distribution')
# plt.show()