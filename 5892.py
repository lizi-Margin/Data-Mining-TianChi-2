import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')



## 通过Pandas对于数据进行读取
Train_data =pd.read_csv('used_car_train_20200313.csv', sep=' ')
Test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
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


# columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
# columns = Train_data.columns
# sns.pairplot(Train_data[numeric_features], height=2, diag_kind = None)  # Added pairplot visualization
# plt.savefig('pairplot.png')
# plt.show()

# print('pairplot.png saved')



# 对分类特征进行独热编码
def onehot_encode(data, feature):
    onehot = pd.get_dummies(data[feature], prefix=feature)
    data = pd.concat([data, onehot], axis=1)
    data.drop(feature, axis=1, inplace=True)
    return data
for cat_fea in categorical_features:
    Train_data = onehot_encode(Train_data, cat_fea)
    Test_data = onehot_encode(Test_data, cat_fea)



# 对numeric类型col中的nan进行填0
for col in numeric_features:
    if col != 'price':
        Train_data[col].fillna(0, inplace=True)
        Test_data[col].fillna(0, inplace=True)












# 定义必要的库和模型
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor



# 定义特征和目标变量
X = Train_data.drop(columns=['price'])  # 假设 'price' 是目标变量
# X['notRepairedDamage'] = X['notRepairedDamage'].astype('category') 
y = Train_data['price']

# 将数据分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义模型
model_lgb = LGBMRegressor(random_state=42, n_jobs=24)
model_xgb = XGBRegressor(random_state=42, n_jobs=24)


# 训练模型
cols = X_train.columns.astype(str)
for col in cols:
    print(col, ':', X_train[col].dtype)
# print(X_train, y_train)
model_lgb.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)



# 预测验证集
val_lgb = model_lgb.predict(X_val)
val_xgb = model_xgb.predict(X_val)



# 计算验证集的 MAE
MAE_lgb = mean_absolute_error(y_val, val_lgb)
MAE_xgb = mean_absolute_error(y_val, val_xgb)

print(f'MAE of val with LGB:', MAE_lgb)
print(f'MAE of val with XGB:', MAE_xgb)

##这里我们采取了简单的加权融合的方式
val_Weighted = (1 - MAE_lgb / (MAE_xgb + MAE_lgb)) * val_lgb + (1 - MAE_xgb / (MAE_xgb + MAE_lgb)) * val_xgb
val_Weighted[val_Weighted<0]=10 # 由于我们发现预测的最小值有负数，而真实情况下，pric
print('MAE of val with Weighted ensemble:', mean_absolute_error(y_val, val_Weighted))



# 对测试集进行预测
sub_lgb = model_lgb.predict(Test_data)
sub_xgb = model_xgb.predict(Test_data)

# sub_Weighted=(1-MAE_lgb/(MAE_xgb+MAE_lgb))*sub_lgb+ (1-MAE_xgb/(MAE_xgb+MAE_lgb))
sub_Weighted = (1 - MAE_lgb / (MAE_xgb + MAE_lgb)) * sub_lgb + (1 - MAE_xgb / (MAE_xgb + MAE_lgb)) * sub_xgb
sub_Weighted[sub_Weighted<0]=10 # 由于我们发现预测的最小值有负数，而真实情况下，pric

# save to csv: SaleID,price
sub_Weighted = np.expm1(sub_Weighted)
submission = pd.DataFrame({'SaleID': Test_data['SaleID'], 'price': sub_Weighted})
submission.to_csv('submission.csv', index=False)

# 查看预测值的分布
plt.hist(sub_Weighted, bins=50, color='blue')
plt.title('Prediction Distribution')
plt.show()

# ##查看预测值的统计进行
# plt.hist(Y_data)
# plt.show()
# plt.close()