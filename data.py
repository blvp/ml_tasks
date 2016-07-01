import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn import linear_model
from sklearn.cluster.k_means_ import KMeans
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.coordinate_descent import Lasso
from sklearn.linear_model.ransac import RANSACRegressor
from sklearn.linear_model.ridge import Ridge
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.metrics.regression import mean_squared_error
from sklearn.metrics.scorer import r2_scorer, make_scorer
from sklearn.preprocessing.data import MinMaxScaler, StandardScaler
from sklearn.preprocessing.label import LabelEncoder
from sklearn.svm import SVR
from sklearn.tree.tree import DecisionTreeRegressor


def load_data():
    data = pd.read_csv('data/x_train.csv', na_values='None')
    target = pd.read_csv('data/y_train.csv')['time']
    return data, target


def plot_actual_data_distribution():
    data, target = load_data()
    data, target, labels = normalize_data(data, target)

    data = data[['C0', 'C1', 'cpuFull']]
    data['target'] = target
    split_by_types = dict()

    cpu_groups = data.groupby('cpuFull')

    for cpu, group in cpu_groups:
        plt.figure()
        plt.scatter(group['C0'], group['target'], s=4)
        plt.savefig('imgs/actual/C0_{}.png'.format(labels['cpuFull'].inverse_transform(cpu)))

    for cpu, group in cpu_groups:
        plt.figure()
        plt.scatter(group['C1'], group['target'], s=4)
        plt.savefig('imgs/actual/C1_{}.png'.format(labels['cpuFull'].inverse_transform(cpu)))


def normalize_data(data, target):
    data.replace({'None': np.nan}, inplace=True)
    types = pd.read_csv('data/datatypes.csv')
    for i, row in types.iterrows():
        data[row['feature']] = data[row['feature']].astype(row['type'])
    data['memFreq'].fillna(0, inplace=True)
    data['memtRFC'].fillna(0, inplace=True)

    os_le = LabelEncoder()
    cpu_full_le = LabelEncoder()
    cpu_arch_le = LabelEncoder()
    mem_type_le = LabelEncoder()
    data['cpuFull'] = cpu_full_le.fit_transform(data['cpuFull'])
    data['os'] = os_le.fit_transform(data['os'])
    data['cpuArch'] = cpu_arch_le.fit_transform(data['cpuArch'])
    data['memType'] = mem_type_le.fit_transform(data['memType'])
    # drop single value columns
    data = data.drop(['cacheL3IsShared', 'BMI', 'CLF_._Cache_Line_Flush', 'CMOV_._Conditionnal_Move_Inst.',
                      'CX8_._CMPXCHG8B', 'FXSR.FXSAVE.FXRSTOR', 'IA.64_Technology',
                      'MMX_Technology', 'SSE', 'SSE2', 'SSE4a', 'SSE5', 'TBM', 'X3DNow_Pro_Technology'], axis=1)

    data['C0'] = np.log(data['n'] * data['m'] * data['k'])
    data = data.drop(['m', 'n', 'k'], axis=1)
    return data, target, {
        'os': os_le,
        'cpuFull': cpu_full_le,
        'cpuArch': cpu_arch_le,
        'memType': mem_type_le,
    }


def submit_model(model):
    X_submit = pd.read_csv('data/x_test.csv')
    X_submit, _, _ = normalize_data(X_submit, None)
    y_submit = model.predict(X_submit)
    pd.DataFrame([y_submit, ]).to_csv('data/y_submit.csv', index=False, header=False)


def get_train_data():
    data, target = load_data()
    data, target, _ = normalize_data(data, target)
    return train_test_split(data, target)


def plot_split_by_arch():
    data, target = load_data()
    data, target, labels = normalize_data(data, target)
    data['target'] = target
    plt.figure()
    for arch, group in data.groupby('cpuArch'):
        plt.clf()
        plt.scatter(group['C0'], group['target'])
        plt.title('Category: {}'.format(labels['cpuArch'].inverse_transform(arch)))
        plt.savefig('imgs/arch/C0_{}.png'.format(labels['cpuArch'].inverse_transform(arch)))


def load_data_new():
    data, target = load_data()
    submit = pd.read_csv('data/x_test.csv', na_values='None')
    data['C0'] = data['m'] * data['n'] * data['k']
    # data['C1'] = ((data['m'] + data['k'] + data['n']) / 3.0) ** math.log(7, 2)
    submit['C0'] = submit['m'] * submit['n'] * submit['k']
    # submit['C1'] = ((submit['m'] + submit['k'] + submit['n']) / 3.0) ** math.log(7, 2)
    # for name, group in data.groupby('cpuFull'):
    #     group_data = group['C0'].reshape(-1, 1)
    #     gtrain, g_test, ytrain, ytest = train_test_split(group_data, target[group.index],
    #                                                      test_size=0.5)
    #     reg = RANSACRegressor(LinearRegression())
    #     reg.fit(gtrain, ytrain)
    #
    #     data.loc[group.index, 'C2'] = reg.predict(group_data)
    #     submit['C2'] = reg.predict(submit['C0'].reshape(-1, 1))

    numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
    col_drops = [c for c in numerical_columns if len(data[c].unique()) == 1]
    data.drop(col_drops, axis=1, inplace=True)
    submit.drop(col_drops, axis=1, inplace=True)

    numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
    non_binary_numerical_cats = [c for c in numerical_columns
                                 if len(data[c].unique()) != 2]
    numerical_data = data[non_binary_numerical_cats]
    drop_thr = 0.4
    drop_col = [c for c in non_binary_numerical_cats
                if float(len(numerical_data[c][numerical_data[c] == 0])) / data.shape[0] > drop_thr]
    data.drop(drop_col, axis=1, inplace=True)
    submit.drop(drop_col, axis=1, inplace=True)

    categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
    data_describe = data.describe(include=['object'])
    submit_desc = submit.describe(include=['object'])
    for col in categorical_columns:
        data[col] = data[col].fillna(data_describe[col]['top'])
        submit[col] = submit[col].fillna(submit_desc[col]['top'])

    numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
    numerical_count = data[numerical_columns].count(axis=0)
    # have missing values
    total = len(data)
    thr = 0.5
    numerical_count = numerical_count[numerical_count < total]
    for col in numerical_count.index:
        if float(numerical_count.get(col)) / total <= thr:
            data.drop([col, ], axis=1, inplace=True)
            submit.drop([col, ], axis=1, inplace=True)
        else:
            data[col] = data[col].fillna(data[col].median(), axis=0)
            submit[col] = submit[col].fillna(submit[col].median(), axis=0)

    data_nonbinary = pd.get_dummies(data[categorical_columns])
    submit_nonbinary = pd.get_dummies(submit[categorical_columns])

    numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
    numerical_binary = [c for c in numerical_columns
                        if len(data[c].unique()) == 2]
    numerical_nonbinary = [c for c in numerical_columns
                           if len(data[c].unique()) != 2]
    data_numerical = data[numerical_nonbinary]
    data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
    submit_numerical = submit[numerical_nonbinary]
    submit_numerical = (submit_numerical - submit_numerical.mean()) / submit_numerical.std()
    submit = pd.DataFrame(pd.concat((submit_nonbinary, submit_numerical, submit[numerical_binary]), axis=1),
                          dtype=float)
    data = pd.DataFrame(pd.concat((data_nonbinary, data_numerical, data[numerical_binary]), axis=1), dtype=float)
    return data, target, submit
#
#
# data, target = load_data()
# data['C0'] = data['n'] * data['k'] * data['m']
# for name, group in data.groupby('cpuFull'):
#     group_data = group['C0'].reshape(-1, 1)
#     gtrain, gtest, ytrain, ytest = train_test_split(group_data, target[group.index])
#     plt.figure()
#     # for title, reg in [('linear', LinearRegression()), ('l2', Ridge(alpha=0.5)), ('l1', Lasso(alpha=0.5))]:
#     # reg = DecisionTreeRegressor()
#     reg = RANSACRegressor(LinearRegression())
#     reg.fit(gtrain, ytrain)
#     err = mean_squared_error(ytest, reg.predict(gtest))
#     plt.plot(group_data, reg.predict(group_data), label='{}: mse {}'.format("svr", err))
#     plt.scatter(group['C0'], target[group.index], label='actual data', s=4, c='b')
#     plt.legend(loc='best')
#     plt.savefig('imgs/reg_lin_{}.png'.format(name))
#     print "saved fig"
