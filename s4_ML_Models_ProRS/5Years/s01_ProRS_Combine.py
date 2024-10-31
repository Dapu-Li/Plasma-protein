

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
outpath = '/home/mrli/桌面/py/shuai/Results/5Years/'
outpath1 = outpath + 'ProRS/'
outputfile = outpath1 + 'ProRS_Results.csv'

basic_f_lst = ['age', 'Pulserate', 'Diabetes', 'units', 'smoke', 'Towns', 'IPAQ', 'BMI', 'hyper', 'viral', 'wctohc']
lab_f_lst = ['GGT', 'ALT', 'AST', 'Albumin', 'Platelet']
fib = ['fib4']
apri = ['apri']
cov_df = pd.read_csv(dpath + 'nonpro_imputed_int_scaled.csv', usecols = ['ID'] + basic_f_lst + lab_f_lst + fib + apri)
fold_id_lst = [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

AUC_cv1, AUC_cv2, AUC_cv3, AUC_cv4, AUC_cv5, AUC_cv6 = [], [], [], [], [], []
tmp_f1 = ['ProRS']
tmp_f2 = ['ProRS'] + basic_f_lst
tmp_f3 = ['ProRS'] + fib
tmp_f4 = ['ProRS'] + apri
tmp_f5 = ['ProRS'] + lab_f_lst
tmp_f6 = ['ProRS'] + basic_f_lst + lab_f_lst


for fold_id in tqdm(fold_id_lst):
    pro_df = pd.read_csv(outpath1 + 'Test_fold' + str(fold_id) + '.csv')
    mydf = pd.merge(pro_df, cov_df, how = 'inner', on = ['ID'])
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_train, y_test = mydf.iloc[train_idx].cirtotal, mydf.iloc[test_idx].cirtotal
    X_train1, X_test1 = mydf.iloc[train_idx][tmp_f1], mydf.iloc[test_idx][tmp_f1]
    X_train2, X_test2 = mydf.iloc[train_idx][tmp_f2], mydf.iloc[test_idx][tmp_f2]
    X_train3, X_test3 = mydf.iloc[train_idx][tmp_f3], mydf.iloc[test_idx][tmp_f3]
    X_train4, X_test4 = mydf.iloc[train_idx][tmp_f4], mydf.iloc[test_idx][tmp_f4]
    X_train5, X_test5 = mydf.iloc[train_idx][tmp_f5], mydf.iloc[test_idx][tmp_f5]
    X_train6, X_test6 = mydf.iloc[train_idx][tmp_f6], mydf.iloc[test_idx][tmp_f6]
    my_lgb1 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2024)
    my_lgb1.set_params(**my_params)
    my_lgb1.fit(X_train1, y_train)
    y_pred_prob1 = my_lgb1.predict_proba(X_test1)[:, 1]
    AUC_cv1.append(np.round(roc_auc_score(y_test, y_pred_prob1), 3))
    my_lgb2 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2024)
    my_lgb2.set_params(**my_params)
    my_lgb2.fit(X_train2, y_train)
    y_pred_prob2 = my_lgb2.predict_proba(X_test2)[:, 1]
    AUC_cv2.append(np.round(roc_auc_score(y_test, y_pred_prob2), 3))
    my_lgb3 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2024)
    my_lgb3.set_params(**my_params)
    my_lgb3.fit(X_train3, y_train)
    y_pred_prob3 = my_lgb3.predict_proba(X_test3)[:, 1]
    AUC_cv3.append(np.round(roc_auc_score(y_test, y_pred_prob3), 3))
    my_lgb4 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2024)
    my_lgb4.set_params(**my_params)
    my_lgb4.fit(X_train4, y_train)
    y_pred_prob4 = my_lgb4.predict_proba(X_test4)[:, 1]
    AUC_cv4.append(np.round(roc_auc_score(y_test, y_pred_prob4), 3))
    my_lgb5 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2024)
    my_lgb5.set_params(**my_params)
    my_lgb5.fit(X_train5, y_train)
    y_pred_prob5 = my_lgb5.predict_proba(X_test5)[:, 1]
    AUC_cv5.append(np.round(roc_auc_score(y_test, y_pred_prob5), 3))
    my_lgb6 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2024)
    my_lgb6.set_params(**my_params)
    my_lgb6.fit(X_train6, y_train)
    y_pred_prob6 = my_lgb6.predict_proba(X_test6)[:, 1]
    AUC_cv6.append(np.round(roc_auc_score(y_test, y_pred_prob6), 3))


tmp_out1 = ['ProRS'] + [np.round(np.mean(AUC_cv1), 3), np.round(np.std(AUC_cv1), 3)] + AUC_cv1
tmp_out2 = ['ProRS+Basic'] + [np.round(np.mean(AUC_cv2), 3), np.round(np.std(AUC_cv2), 3)] + AUC_cv2
tmp_out3 = ['ProRS+Fib'] + [np.round(np.mean(AUC_cv3), 3), np.round(np.std(AUC_cv3), 3)] + AUC_cv3
tmp_out4 = ['ProRS+Apri'] + [np.round(np.mean(AUC_cv4), 3), np.round(np.std(AUC_cv4), 3)] + AUC_cv4
tmp_out5 = ['ProRS+Lab'] + [np.round(np.mean(AUC_cv4), 3), np.round(np.std(AUC_cv4), 3)] + AUC_cv4
tmp_out6 = ['ProRS+Basic+Lab'] + [np.round(np.mean(AUC_cv4), 3), np.round(np.std(AUC_cv4), 3)] + AUC_cv4

AUC_df1 = pd.DataFrame(tmp_out1).T
AUC_df2 = pd.DataFrame(tmp_out2).T
AUC_df3 = pd.DataFrame(tmp_out3).T
AUC_df4 = pd.DataFrame(tmp_out4).T
AUC_df5 = pd.DataFrame(tmp_out5).T
AUC_df6 = pd.DataFrame(tmp_out6).T


AUC_df1.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]
AUC_df2.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]
AUC_df3.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]
AUC_df4.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]
AUC_df5.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]
AUC_df6.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]

myout_df = pd.concat([AUC_df1, AUC_df2], axis = 0)
myout_df = pd.concat([myout_df, AUC_df3], axis = 0)
myout_df = pd.concat([myout_df, AUC_df4], axis = 0)
myout_df = pd.concat([myout_df, AUC_df5], axis = 0)
myout_df = pd.concat([myout_df, AUC_df6], axis = 0)

myout_df.to_csv(outputfile, index = False)

print('finished')

