import numpy as np
import pandas as pd
from tqdm import tqdm

from Utility.Training_Utilities import *
from Utility.DelongTest import *
from lightgbm import LGBMClassifier
import scipy.stats as stats
import warnings
from Utility.dl import delong_test
warnings.filterwarnings("ignore")

dpath = '/home/mrli/桌面/py/shuai/Data/'
outpath = '/home/mrli/桌面/py/shuai/Results/5Years/'

pro_df = pd.read_csv(dpath + 'pro_imputed.csv')
pro_dict = pd.read_csv(dpath + 'ProCode.csv', usecols = ['Pro_code', 'Prode_finition'])
target_df = pd.read_csv(dpath + 'outcome.csv')
reg_df = pd.read_csv(dpath + 'region.csv')

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, reg_df, how = 'inner', on = ['ID'])

mydf['cirtotal'].loc[mydf.CIRtotal_interuse > 5*365] = 0

fold_id_lst = [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

pro_f_df = pd.read_csv(outpath + 'ProImportance.csv')
pro_f_df.sort_values(by = 'TotalGain_cv', ascending=False, inplace = True)
pro_f_lst = pro_f_df.Pro_code.tolist()
outfile = outpath + 'AccAUC_TotalGain.csv'

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.8,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

y_test_full = np.zeros(shape = [1,1])
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_test_full = np.concatenate([y_test_full, np.expand_dims(mydf.iloc[test_idx].cirtotal, -1)])

y_pred_full_prev = y_test_full
tmp_f, AUC_cv_lst= [], []
f = pro_f_lst[0]
f = pro_f_lst[1]
for f in tqdm(pro_f_lst[:100]): # 【10】: number of features to be selected
    tmp_f.append(f)
    my_X = mydf[tmp_f]
    AUC_cv = []
    y_pred_full = np.zeros(shape = [1,1])
    for fold_id in fold_id_lst:
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
    #p = delong_test(y_pred_full[:,0],y_test_full[:,0], y_pred_full_prev[:,0],2 )

    y_pred_full_prev = y_pred_full
    tmp_out = np.array([np.round(np.mean(AUC_cv), 3), np.round(np.std(AUC_cv), 3), 10**log10_p[0][0]] + AUC_cv)
    AUC_cv_lst.append(tmp_out)
    print((f, np.mean(AUC_cv), 10**log10_p[0][0]))

AUC_df = pd.DataFrame(AUC_cv_lst, columns = ['AUC_mean', 'AUC_std', 'p_delong'] + ['AUC_' + str(i) for i in range(10)])

AUC_df = pd.concat((pd.DataFrame({'Pro_code':tmp_f}), AUC_df), axis = 1)
myout = pd.merge(AUC_df, pro_dict, how='left', on=['Pro_code'])
myout.to_csv(outfile, index = False)

print('finished')