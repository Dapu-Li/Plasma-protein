

import numpy as np
import pandas as pd
from tqdm import tqdm

from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/home/mrli/桌面/py/shuai/Data/'
outpath = '/home/mrli/桌面/py/shuai/Results/10Years/'
outputfile = outpath + 'ProRS_Data.csv'

pro_f_df = pd.read_csv(outpath + 'AccAUC_TotalGain.csv')
from param.TenYears import *
pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]

pro_df = pd.read_csv(dpath + 'pro_imputed.csv',usecols = ['ID'] + pro_f_lst)
pro_dict = pd.read_csv(dpath + 'ProCode.csv', usecols = ['Pro_code', 'Prode_finition'])
target_df = pd.read_csv(dpath + 'outcome.csv')
#age + sex + ethnic + wctohc + Pulserate + Diabetes + dtrinking + units + smoke + Towns
cov_f_lst = ['age', 'Pulserate', 'Diabetes', 'units', 'smoke', 'Towns', 'IPAQ', 'BMI', 'hyper', 'viral', 'wctohc']
# cov_f_lst = ['Towns', 'age','sex','ethnic','wctohc','Pulserate','Diabetes','dtrinking','units','smoke',
#             'IPAQ', 'BMI', 'hyper', 'hyperli', 'edu2', 'dietscore','fib4', 'apri', 'viral']
cov_df = pd.read_csv(dpath + 'nonpro_imputed_int_scaled.csv', usecols = ['ID'] + cov_f_lst)
reg_df = pd.read_csv(dpath + 'region.csv')

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, reg_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, cov_df, how = 'inner', on = ['ID'])
fold_id_lst = [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

mydf.loc[mydf.CIRtotal_interuse > 10*365, 'cirtotal'] = 0

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

AUC_cv = []
tmp_f = pro_f_lst

fold_id = 1
for fold_id in tqdm(fold_id_lst):
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
    y_train, y_test = mydf.iloc[train_idx].cirtotal, mydf.iloc[test_idx].cirtotal
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2024)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    y_pred_train = my_lgb.predict_proba(X_train)[:, 1]
    prors_train_df = pd.DataFrame({'ID': mydf.iloc[train_idx].ID.tolist(),
                                   'Region_code': mydf.iloc[train_idx].Region_code.tolist(),
                                   'cirtotal': y_train.tolist(),
                                   'CIRtotal_interuse': mydf.iloc[train_idx].CIRtotal_interuse.tolist(),
                                   'ProRS': y_pred_train.tolist()})
    y_pred_test = my_lgb.predict_proba(X_test)[:, 1]
    prors_test_df = pd.DataFrame({'ID': mydf.iloc[test_idx].ID.tolist(),
                                   'Region_code': mydf.iloc[test_idx].Region_code.tolist(),
                                   'cirtotal': y_test.tolist(),
                                   'CIRtotal_interuse': mydf.iloc[test_idx].CIRtotal_interuse.tolist(),
                                   'ProRS': y_pred_test.tolist()})
    prors_df = pd.concat([prors_test_df, prors_train_df], axis = 0)
    prors_df.to_csv(outpath + 'ProRS/Test_fold' + str(fold_id) + '.csv', index = False)


print('finished')

