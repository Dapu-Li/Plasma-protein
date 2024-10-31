

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import shap
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/home/mrli/桌面/py/shuai/Data/'
outpath = '/home/mrli/桌面/py/shuai/Results/Over10Years/'
outpath1 = outpath + 'ProRS/'
outputfile = outpath1 + 's2_SeqSelector.csv'

basic_f_lst = ['age', 'sex', 'ethnic', 'wctohc', 'Pulserate', 'Diabetes', 'dtrinking', 'units', 'smoke', 'Towns']
lab_f_lst = ['GGT', 'ALT', 'AST', 'Albumin', 'Platelet']

cov_df = pd.read_csv(dpath + 'nonpro_imputed_int_scaled.csv', usecols = ['ID'] + basic_f_lst + lab_f_lst)
fold_id_lst = [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

f_rank_df = pd.read_csv(outpath1 + 's1_FeaImpRank.csv')
f_rank_df.sort_values(by = 'TotalGain_cv', ascending = False, inplace = True)
my_f_lst = f_rank_df.Pro_code.tolist()

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

y_test_full = np.zeros(shape = [1,1])
for fold_id in fold_id_lst:
    pro_df = pd.read_csv(outpath1 + 'Test_fold' + str(fold_id) + '.csv')
    mydf = pd.merge(pro_df, cov_df, how='inner', on=['ID'])

    rm_idx = mydf.index[(mydf.CIRtotal_interuse < 10 * 365) & (mydf.cirtotal == 1)]
    mydf.drop(rm_idx, axis=0, inplace=True)
    mydf.reset_index(inplace=True)

    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_test_full = np.concatenate([y_test_full, np.expand_dims(mydf.iloc[test_idx].cirtotal, -1)])

y_pred_full_prev = y_test_full
tmp_f, AUC_cv_lst= [], []

for f in my_f_lst:
    tmp_f.append(f)
    AUC_cv = []
    y_pred_full = np.zeros(shape = [1,1])
    for fold_id in fold_id_lst:
        pro_df = pd.read_csv(outpath1 + 'Test_fold' + str(fold_id) + '.csv')
        mydf = pd.merge(pro_df, cov_df, how='inner', on=['ID'])

        rm_idx = mydf.index[(mydf.CIRtotal_interuse < 10 * 365) & (mydf.cirtotal == 1)]
        mydf.drop(rm_idx, axis=0, inplace=True)
        mydf.reset_index(inplace=True)

        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
        y_train, y_test = mydf.iloc[train_idx].cirtotal, mydf.iloc[test_idx].cirtotal
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True,  n_jobs=4, verbosity=-1, seed=2024)
        my_lgb.set_params(**my_params)
        my_lgb.fit(X_train, y_train)
        y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
        AUC_cv.append(np.round(roc_auc_score(y_test, y_pred_prob), 3))
        y_pred_full = np.concatenate([y_pred_full, np.expand_dims(y_pred_prob, -1)])
    log10_p = delong_roc_test(y_test_full[:,0], y_pred_full_prev[:,0], y_pred_full[:,0])
    y_pred_full_prev = y_pred_full
    tmp_out = np.array([np.round(np.mean(AUC_cv), 3), np.round(np.std(AUC_cv), 3), 10**log10_p[0][0]] + AUC_cv)
    AUC_cv_lst.append(tmp_out)
    print((f, np.mean(AUC_cv), 10**log10_p[0][0]))

AUC_df = pd.DataFrame(AUC_cv_lst, columns = ['AUC_mean', 'AUC_std', 'p_delong'] + ['AUC_' + str(i) for i in range(10)])

AUC_df = pd.concat((pd.DataFrame({'Pro_code':tmp_f}), AUC_df), axis = 1)
AUC_df.to_csv(outputfile, index = False)

print('finished')



