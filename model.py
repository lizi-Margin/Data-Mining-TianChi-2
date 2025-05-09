import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# regression models: xgb, rf, svr, lgbm, lr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgbm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# opt methods: bayesian opt
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_objective, plot_convergence
from skopt.utils import use_named_args
# traintest split
from sklearn.model_selection import train_test_split
# save model
import joblib
import os

N_ITR = 30
CV = 3

def get_best_xgb_model(train_data, **kwargs):
    col_a = kwargs['col_a']  # id
    col_b = kwargs['col_b']  # price

    # 划分训练集和测试集
    X_train, y_train = train_data.drop(columns=[col_a, col_b]), train_data[col_b]

    if kwargs['search']:
        # use bayesian opt to find the best xgb model
        # define the search space
        param_rf = {
            'n_estimators': Integer(40, 50),
            'max_depth': Integer(50, 60),
            # 'min_samples_split': Integer(2, 3),
            # 'criterion': Categorical(['absolute_error']),
        }
        # define the model
        model = xgb.XGBRegressor(n_jobs=1)
        # define the bayesian opt
        opt = BayesSearchCV(model, param_rf, n_iter=N_ITR, cv=CV, verbose=kwargs['verbose'], scoring='neg_mean_absolute_error', n_jobs=kwargs['NJ'], random_state=42)
        # fit the model
        opt.fit(X_train, y_train)
        model = opt.best_estimator_
        # print the best params
        print("Best params: ", opt.best_params_)
        # print the best score
        print("Best score: ", opt.best_score_)

        # save the best params and score to xgb_best_params.txt
        with open('xgb_best_params.txt', 'w') as f:
            f.write("Best params: " + str(opt.best_params_) + '\n')
            f.write("Best score: " + str(opt.best_score_) + '\n')
        joblib.dump(opt, 'xgb_model_opt.pkl')
        joblib.dump(model, 'xgb_model.pkl')
        print("Model saved")
    else:
        LOAD = False
        if not LOAD:
            model = xgb.XGBRegressor(
                n_jobs=kwargs['NJ'],
                n_estimators=150,
                max_depth=7,
            )

            model.fit(X_train, y_train)
            joblib.dump(model, 'xgb_model.pkl')
            print("Model saved")
        else:
            # load model from xgb_model_opt.pkl
            if os.path.exists('xgb_model_opt.pkl'):
                model = joblib.load('xgb_model_opt.pkl')
                model = model.best_estimator_
            else:
                model = joblib.load('xgb_model.pkl')

    return model


def get_best_rf_model(train_data, **kwargs):
    col_a = kwargs['col_a']  # id
    col_b = kwargs['col_b']  # price
    X_train, y_train = train_data.drop(columns=[col_a, col_b]), train_data[col_b]

    if kwargs['search']:
        # use bayesian opt to find the best rf model
        # define the search space
        param_rf = {
            'n_estimators': Integer(49, 50),
            'max_depth': Integer(59, 60),
            'min_samples_split': Integer(2, 3),
            'criterion': Categorical(['absolute_error']),
        }

        model = RandomForestRegressor(n_jobs=kwargs['NJ']//CV)
        
        opt = BayesSearchCV(model, param_rf, n_iter=N_ITR, cv=CV, verbose=kwargs['verbose'], scoring='neg_mean_absolute_error', n_jobs=kwargs['NJ'], random_state=42)

        opt.fit(X_train, y_train)
        model = opt.best_estimator_

        # print the best params
        print("Best params: ", opt.best_params_)
        
        # print the best score
        print("Best score: ", opt.best_score_)

        # save the best params and score to rf_best_params.txt
        with open('rf_best_params.txt', 'w') as f:
            f.write("Best params: " + str(opt.best_params_) + '\n')
            f.write("Best score: " + str(opt.best_score_) + '\n')


        # save the model
        joblib.dump(opt, 'rf_model_opt.pkl')
        joblib.dump(model, 'rf_model.pkl')
        print("Model saved")
    else:
        LOAD = False
        if not LOAD:
            model = RandomForestRegressor(
                n_jobs=kwargs['NJ']
            )
            model.fit(X_train, y_train)
            joblib.dump(model, 'rf_model.pkl')
            print("Model saved")
        else:
            # load model from rf_model_opt.pkl
            if os.path.exists('rf_model_opt.pkl'):
                model = joblib.load('rf_model_opt.pkl')
                model = model.best_estimator_
            else:
                model = joblib.load('rf_model.pkl')
    return model

def get_best_lr_model(train_data, **kwargs):
    col_a = kwargs['col_a']  # id
    col_b = kwargs['col_b']  # price
    X_train, y_train = train_data.drop(columns=[col_a, col_b]), train_data[col_b]
    # use bayesian opt to find the best lr model
    # define the search space

    if kwargs['search'] or True:
        # define the model
        # model = LinearRegression(n_jobs=kwargs['NJ']//CV)
        model = LinearRegression(n_jobs=1, copy_X=True, fit_intercept=True, positive=False)
        # define the bayesian opt
        # opt = BayesSearchCV(model, param_lr, n_iter=N_ITR, cv=CV, verbose=kwargs['verbose'], scoring='neg_mean_absolute_error', n_jobs=kwargs['NJ'], random_state=42)
        # fit the model
        model.fit(X_train, y_train)
        # model = opt.best_estimator_
        
        # print the best params
        # print("Best params: ", opt.best_params_)
        # print the best score
        # print("Best score: ", opt.best_score_)
        # save the best params and score to lr_best_params.txt
        # with open('lr_best_params.txt', 'w') as f:
            # f.write("Best params: " + str(opt.best_params_) + '\n')
            # f.write("Best score: " + str(opt.best_score_) + '\n')
        # save the model
        # joblib.dump(opt, 'lr_model_opt.pkl')
        joblib.dump(model, 'lr_model.pkl')
        print("Model saved")
    else:
        # model = LinearRegression(
        #     n_jobs=kwargs['NJ']
        # )
        # load model from lr_model_opt.pkl
        model = joblib.load('lr_model.pkl')
    return model




















def get_soft_voting_model(trained_models: list, train_data, **kwargs):
    """
    不训练，直接集成多个模型
    """
    # 定义集成模型
    class SoftVotingRegressor:
        def __init__(self, models):
            self.models = models
        def fit(self, X, y):
            assert False, "NO TRAINING"
        def predict(self, X):
            predictions = np.array([model.predict(X) for model in self.models])
            return np.mean(predictions, axis=0)
    model = SoftVotingRegressor(trained_models)
    return model