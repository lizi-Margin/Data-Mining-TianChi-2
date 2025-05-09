import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_nc(data: pd.DataFrame):
    categorical_features = data.select_dtypes(include=['object', 'category', 'datetime', 'timedelta']).columns.tolist()
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return numeric_features, categorical_features

def drop(cols, train_data, test_data):
    train_data.drop(columns=cols, inplace=True)
    test_data.drop(columns=cols, inplace=True)
    return train_data, test_data

def fix_convert(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs):
    # regDate like 20040402 should be converted to 2004-04-02
    assert 'regDate' in train_data.columns, 'regDate not in train_data'
    assert 'regDate' in test_data.columns, 'regDate not in test_data'
    assert 'creatDate' in train_data.columns, 'creatDate not in train_data'
    assert 'creatDate' in test_data.columns, 'creatDate not in test_data'
    
    # 转换日期格式，保留NaN
    train_data['regDateRaw'] = train_data['regDate']
    test_data['regDateRaw'] = test_data['regDate']
    train_data['creatDateRaw'] = train_data['creatDate']
    test_data['creatDateRaw'] = test_data['creatDate']
    train_data['regDate'] = pd.to_datetime(train_data['regDate'], format='%Y%m%d', errors='coerce')
    test_data['regDate'] = pd.to_datetime(test_data['regDate'], format='%Y%m%d', errors='coerce')
    train_data['creatDate'] = pd.to_datetime(train_data['creatDate'], format='%Y%m%d', errors='coerce')
    test_data['creatDate'] = pd.to_datetime(test_data['creatDate'], format='%Y%m%d', errors='coerce')
    
    # 提取年份和月份，保留NaN
    train_data['regYear'] = train_data['regDate'].dt.year
    train_data['regMonth'] = train_data['regDate'].dt.month
    test_data['regYear'] = test_data['regDate'].dt.year
    test_data['regMonth'] = test_data['regDate'].dt.month
    
    # 提取月中日期和星期几，保留NaN
    train_data['regDay'] = train_data['regDate'].dt.day
    train_data['regWeekday'] = train_data['regDate'].dt.dayofweek
    test_data['regDay'] = test_data['regDate'].dt.day
    test_data['regWeekday'] = test_data['regDate'].dt.dayofweek
    
    # 把星期几转换为字符串，NaN映射为'Unknown'
    weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    train_data['regWeekday_cat'] = train_data['regWeekday'].map(weekday_map).fillna('Unknown')
    test_data['regWeekday_cat'] = test_data['regWeekday'].map(weekday_map).fillna('Unknown')
    
    # 把月中日期转换为字符串，NaN映射为'Unknown'
    def day_to_ordinal(day):
        if pd.isna(day):
            return 'Unknown'
        elif 0 <= day <= 10:
            return "1-10"
        elif 10 < day <= 20:
            return "11-20"
        elif 20 < day <= 31:
            return "21-31"
        else:
            return 'Unknown'
    
    train_data['regDay_cat'] = train_data['regDay'].apply(day_to_ordinal)
    test_data['regDay_cat'] = test_data['regDay'].apply(day_to_ordinal)
    
    # 把月份转换为字符串，NaN映射为'Unknown'
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    train_data['regMonth_cat'] = train_data['regMonth'].map(month_map).fillna('Unknown')
    test_data['regMonth_cat'] = test_data['regMonth'].map(month_map).fillna('Unknown')
    
    # 计算车龄，保留NaN
    train_data['carAge'] = (train_data['regDate'] - train_data['creatDate']).dt.days
    test_data['carAge'] = (test_data['regDate'] - test_data['creatDate']).dt.days

    # 计算regDate到现在的时间差，保留NaN
    train_data['timeSinceReg'] = (pd.Timestamp.now() - train_data['regDate']).dt.days
    test_data['timeSinceReg'] = (pd.Timestamp.now() - test_data['regDate']).dt.days

    # 计算creatDate到现在的时间差，保留NaN
    train_data['timeSinceCreat'] = (pd.Timestamp.now() - train_data['creatDate']).dt.days
    test_data['timeSinceCreat'] = (pd.Timestamp.now() - test_data['creatDate']).dt.days
    
    # 计算车龄的年份和月份，保留NaN
    train_data['carAgeYear'] = train_data['carAge'] // 365
    train_data['carAgeMonth'] = (train_data['carAge'] % 365) // 30
    test_data['carAgeYear'] = test_data['carAge'] // 365
    test_data['carAgeMonth'] = (test_data['carAge'] % 365) // 30

    # # if dtype is DateTime64, convert to string
    # train_data['regDate'] = train_data['regDate'].astype(int)
    # test_data['regDate'] = test_data['regDate'].astype(int)
    # train_data['creatDate'] = train_data['creatDate'].astype(int)
    # test_data['creatDate'] = test_data['creatDate'].astype(int)

    # 删除原始日期列
    train_data.drop(columns=['regDate', 'creatDate'], inplace=True)
    test_data.drop(columns=['regDate', 'creatDate'], inplace=True)
    

    # train_data, test_data = drop([
    #     'regYear',
    #     'regMonth',
    #     'regDay',
    #     'regWeekday',
    #     'regWeekday_cat',
    #     'regDay_cat',
    #     'regMonth_cat',
    #     'carAge',
    #     'carAgeYear',
    #     'carAgeMonth',
    #     'timeSinceReg',
    #     'timeSinceCreat'
    # ], train_data, test_data)
    
    return train_data, test_data

def 数据特征(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs):
    train_data, test_data = fix_convert(train_data, test_data, **kwargs)
    train_data.hist(bins=50,figsize=(20,15))
    plt.savefig('hist_train.png')
    plt.clf()
    test_data.hist(bins=50,figsize=(20,15))
    plt.savefig('hist_test.png')
    plt.clf()

    # 自动生成 categorical_features 和 numeric_features
    numeric_features, categorical_features = get_nc(train_data)

    print("Categorical Features:", categorical_features)
    print("Numeric Features:", numeric_features)
    print(train_data.columns)

    return train_data, test_data


def 数据分布(train_data, test_data, **kwargs):
    # 绘制标签的统计图，查看标签分布
    plt.hist(train_data['price'], bins=50, orientation='vertical', histtype='bar', color='red')
    plt.savefig('Price Distribution Raw.png')
    plt.clf()

    # 目标变量进行对数变换（使其更接近正态分布）
    train_data['price'] = np.log1p(train_data['price'])
    # 绘制对数变换后的价格分布直方图
    plt.hist(train_data['price'], orientation='vertical', histtype='bar', color='red')
    plt.title('Log-Transformed Price Distribution')
    plt.xlabel('Log(Price)')
    plt.ylabel('Frequency')
    plt.savefig('./Price Distribution Log-Transformed.png')
    plt.clf()

    # 使用seaborn绘制分布图（包含KDE曲线）
    sns.distplot(train_data['price'])
    plt.title('Price Distribution Log-Transformed with KDE')
    plt.savefig('Price Distribution Log-Transformed with KDE.png')
    plt.clf()

    # 计算并打印偏度和峰度
    print("Skewness: %f" % train_data['price'].skew())
    print("Kurtosis: %f" % train_data['price'].kurt())



    print(train_data['offerType'].value_counts())
    print(train_data['seller'].value_counts())
    # print(train_data['creatDate'].value_counts())

    # 统计numeric特征的取值范围和离散值数量
    numeric_features, categorical_features = get_nc(train_data)
    for feature in numeric_features:
        print(f"{feature}的取值范围: {train_data[feature].min()} - {train_data[feature].max()}")
        print(f"{feature}的离散值数量: {train_data[feature].nunique()}")
    

    # cat特征分布
    for cat_fea in categorical_features:
        print(cat_fea + "的特征分布如下:")
        print("{}特征有{}个不同的值".format(cat_fea, train_data[cat_fea].nunique()))
        print(train_data[cat_fea].value_counts())
    
    return train_data, test_data

def 数据质量(train_data, test_data, **kwargs):
    #缺失数据可视化
    missing = train_data.isnull().sum()
    missing= missing[missing>0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
    # 减小x轴字体
    plt.xticks(fontsize=6)
    plt.savefig('missing_train.png')
    plt.clf()


    #缺失数据可视化
    missing = test_data.isnull().sum()
    missing= missing[missing>0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
    # 减小x轴字体
    plt.xticks(fontsize=6)
    plt.savefig('missing_test.png')
    plt.clf()
    return train_data, test_data

def 数据清洗(train_data, test_data, **kwargs):
    numeric_features, categorical_features = get_nc(train_data)
    # 对numeric类型col中的nan进行填midean
    for col in numeric_features:
        if col!= 'price':
            train_data[col].fillna(train_data[col].median(), inplace=True)
            test_data[col].fillna(test_data[col].median(), inplace=True)
    
    

    return train_data, test_data



def 数据转换(train_data, test_data, **kwargs):
    """
    将object类型的数据变换成数值型等操作。
    """
    
    return train_data, test_data


def 特征处理(train_data, test_data, **kwargs):
    col_a, col_b = kwargs['col_a'], kwargs['col_b']
    numeric_features, categorical_features = get_nc(train_data)
    assert col_a in numeric_features and col_b in numeric_features, "col_a and col_b must be numeric features"
    numeric_features.remove(col_a)
    numeric_features.remove(col_b)

    # 对numeric特征进行标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
    test_data[numeric_features] = scaler.transform(test_data[numeric_features])

    # 对分类特征进行独热编码
    def onehot_encode(data, feature):
        onehot = pd.get_dummies(data[feature], prefix=feature)
        data = pd.concat([data, onehot], axis=1)
        data.drop(feature, axis=1, inplace=True)
        return data
    for cat_fea in categorical_features:
        train_data = onehot_encode(train_data, cat_fea)
        test_data = onehot_encode(test_data, cat_fea)

    numeric_features, categorical_features = get_nc(train_data)
    assert len(categorical_features) == 0, f"categorical features should be empty: {categorical_features}"
    
    return train_data, test_data

def 特征选择与构造(train_data, test_data, **kwargs):
    # correlation vector for price
    numeric_features, categorical_features = get_nc(train_data)
    # 计算相关系数矩阵
    correlation = train_data[numeric_features].corr()
    # visualize correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('Correlation Matrix.png')

    # 对与price的相关系数进行排序并输出
    corr_vector = correlation['price'].sort_values()
    # visualize correlation vector
    plt.figure(figsize=(12, 8))
    sns.barplot(x=corr_vector.index, y=corr_vector.values)
    plt.title('Correlation Vector')
    plt.xticks(rotation=90)


    if False:
        # columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
        # columns = Train_data.columns
        sns.pairplot(Train_data[numeric_features], height=2, diag_kind = None)  # Added pairplot visualization
        plt.savefig('pairplot.png')
        # plt.show()
        plt.clf()

        print('pairplot.png saved')


    X = train_data.drop(['price'], axis=1)
    y = train_data['price']
    # xgb特征重要性
    import xgboost as xgb

    model = xgb.XGBRegressor(n_jobs=kwargs['NJ'])
    model.fit(X, y)
    # plot feature importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model)
    plt.title('Feature Importance xgb')
    plt.savefig('Feature Importance xgb.png')
    plt.clf()

    # 随机森林特征重要性
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_jobs=kwargs['NJ'])
    model.fit(X, y)
    # plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x=model.feature_importances_, y=X.columns)
    plt.title('Feature Importance RandomForest')
    plt.savefig('Feature Importance RandomForest.png')
    plt.clf()

    # 线性回归特征重要性
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(n_jobs=kwargs['NJ'])
    model.fit(X, y)
    from copy import copy
    coef = abs(copy(model.coef_))
    # plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x=coef, y=X.columns)
    plt.title('Feature Importance LinearRegression')
    plt.savefig('Feature Importance LinearRegression.png')
    plt.clf()

    # # 逻辑回归特征重要性
    # from sklearn.linear_model import LogisticRegression
    # model = LogisticRegression(n_jobs=kwargs['NJ'])
    # model.fit(X, y)
    # # plot feature importance
    # plt.figure(figsize=(12, 8))
    # sns.barplot(x=model.coef_[0], y=X.columns)
    # plt.title('Feature Importance LogisticRegression')
    # plt.savefig('Feature Importance LogisticRegression.png')
    # plt.clf()

    return train_data, test_data