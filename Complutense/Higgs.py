import numpy as np
import xgboost as xgb
import pandas as pd

size = 100000 #Full dataset: 11000000
train_size = int(0.8*size)
test_size = int(0.2*size)

data = pd.read_csv('HIGGS.csv.gz', nrows=train_size+test_size, compression='gzip',
                   error_bad_lines=False, skiprows=1, delimiter=',', converters={32: lambda x:int(x=='s'.encode('utf-8')) })

label = data.iloc[:, 0]
dataSet = data.iloc[:, 1:]

xgmatTrain = xgb.DMatrix(dataSet.iloc[:train_size, :], label=label.iloc[:train_size], missing = -999.0)
xgmatTest = xgb.DMatrix(dataSet.iloc[train_size:train_size+test_size, :], label=label.iloc[train_size:train_size+test_size], missing = -999.0)

param = {}
param['objective'] = 'binary:logitraw'
param['eta'] = 0.01
param['max_depth'] = 9
param['eval_metric'] = 'auc'
param['silent'] = 0
param['alpha'] = 1
param['gamma'] = 1
param['sub_sample'] = 0.9

plst = list(param.items())+[('eval_metric', 'logloss')]

watchlist = [(xgmatTrain,'train'), (xgmatTest,'eval')]
num_round = 1500 #For better results: 3000
bst = xgb.train(plst, xgmatTrain, num_round, watchlist);
bst.save_model('higgs.model')
