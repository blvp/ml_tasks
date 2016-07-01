from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster.k_means_ import KMeans
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.decomposition.pca import PCA, RandomizedPCA
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble.forest import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.svm.classes import SVR
from sklearn.tree.tree import DecisionTreeRegressor

from data import load_data, normalize_data, get_train_data, submit_model, load_data_new


def mape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)


def eval_error(preds, dmatrix):
    label = dmatrix.get_label()
    return 'mape_error', mape(label, preds)


def predict_per_cpu_full():
    data, target = load_data()
    data, target, labels = normalize_data(data, target)

    data = data[['C0', 'cpuFull']]
    data['target'] = target
    split_by_types = dict()

    cpu_groups = data.groupby('cpuFull')
    for name, group in cpu_groups:
        X_train, X_test, y_train, y_test = train_test_split(group['C0'].reshape(-1, 1), group['target'])
        split_by_types[str(name)] = {
            'train': {
                'data': X_train,
                'target': y_train
            },
            'test': {
                'data': X_test,
                'target': y_test
            }
        }

    # print split_by_types
    summ = 0.0
    for cpu, data_set in split_by_types.iteritems():
        plt.figure()
        # reg = SGDRegressor(loss='huber', n_iter=100, alpha=0.0)
        reg = RandomForestRegressor(n_estimators=5)
        reg.fit(data_set['train']['data'], data_set['train']['target'])
        test_data = data_set['test']['data']
        y_pred = reg.predict(test_data)
        print mape(data_set['test']['target'], y_pred), cpu
        plt.scatter(test_data, data_set['test']['target'], s=3, color='g', label='actual')
        plt.scatter(test_data, y_pred, s=3, color='r', label='predicted')
        plt.legend(loc='upper left')
        plt.ylabel('mul time')
        plt.title('Category: {}'.format(cpu))
        plt.savefig('imgs/{}.png'.format(cpu))


def xgb_800_without_mnk():
    data, target = load_data()
    data, target, labels = normalize_data(data, target)
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    reg = xgb.XGBRegressor(max_depth=12, learning_rate=0.007, n_estimators=800)
    reg.fit(X_train, y_train, eval_metric=eval_error, eval_set=[(X_train, y_train), (X_test, y_test)])
    bst = reg.booster()
    fscore = bst.get_fscore()
    submit_model(reg)
    print sorted(fscore.iteritems(), key=lambda b: b[1], reverse=True)
    train_err = reg.evals_result_['validation_0']['error']
    test_err = reg.evals_result_['validation_1']['error']
    ind = np.arange(len(train_err))
    plt.figure()
    plt.plot(ind, train_err, label='train')
    plt.plot(ind, test_err, label='test')
    plt.ylim([0.0, 0.2])
    plt.legend(loc='upper left')
    plt.show()


data, target, submit = load_data_new()

X_train, X_test, y_train, y_test = train_test_split(data, target)


class StackedRegression(object):
    def __init__(self, regressors, meta_regressor, verbose=False):
        self._regressors = regressors
        self._meta_regressor = meta_regressor
        self.verbose = verbose
        self.coef_ = None

    def fit(self, X, y):
        for name, reg in self._regressors:
            if self.verbose:
                print "training, ", name
            reg.fit(X, y)

        data = self._predict(X)
        if self.verbose:
            print "training predictions"

        self._meta_regressor.fit(data, y)
        self.coef_ = self._meta_regressor.coef_

    def _predict(self, X):
        data = dict()
        for name, reg in self._regressors:
            data[name] = reg.predict(X)
        return pd.DataFrame(data, dtype=np.float32)

    def predict(self, X):
        return self._meta_regressor.predict(self._predict(X))


# reg = xgb.XGBRegressor(max_depth=8, learning_rate=0.05, n_estimators=250)
# reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=eval_error)
# y_submit = reg.predict(submit)
# y_submit[y_submit < 1] = 1
# pd.Series(y_submit).to_csv('y_submit_xgb.csv', index=False)
#
#
# result = reg.evals_result_
# train_err = result['validation_0']['error']
# test_err = result['validation_1']['error']
# ind = range(1, len(train_err) + 1)
#
# plt.plot(ind, train_err, label='train')
# plt.plot(ind, test_err, label='test')
# plt.legend(loc='best')
# plt.show()
ys = None
pd.read_csv()
with open('y_submit_0_078.csv', mode='r') as f:
    ys = np.array([y for y in f.readline().split(",")])

print ys
