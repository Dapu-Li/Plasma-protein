

import numpy as np
import pandas as pd
from tqdm import tqdm

from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore")


def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)):
        i+=1
    return i

def get_pred_probs(tmp_f, mydf, fold_id_lst, my_params, col_name):
    ID_lst, region_lst = [], []
    y_test_lst, y_pred_lst = [], []
    for fold_id in fold_id_lst:
        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
        y_train, y_test = mydf.iloc[train_idx].cirtotal, mydf.iloc[test_idx].cirtotal
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2024)
        my_lgb.set_params(**my_params)
        my_lgb.fit(X_train, y_train)
        y_pred_prob = my_lgb.predict_proba(X_test)[:, 1].tolist()
        y_pred_lst += y_pred_prob
        y_test_lst += mydf.cirtotal.iloc[test_idx].tolist()
        ID_lst += mydf.ID.iloc[test_idx].tolist()
        region_lst += mydf.Region_code.iloc[test_idx].tolist()
    myout_df = pd.DataFrame([ID_lst, region_lst, y_test_lst, y_pred_lst]).T
    myout_df.columns = ['ID', 'Region_code', 'cirtotal', 'y_pred_'+col_name]
    myout_df[['ID', 'Region_code']] = myout_df[['ID', 'Region_code']].astype('int')
    return myout_df

dpath = '/home/mrli/桌面/py/shuai/Data/'
outpath = '/home/mrli/桌面/py/shuai/Results/5Years/'
outpath1 = outpath + 'ROC/'
outputfile = outpath1 + 'pred_probs.csv'

target_df = pd.read_csv(dpath + 'outcome.csv')
pro_f_df = pd.read_csv(outpath + 'AccAUC_TotalGain.csv')
from param.FiveYears import *
pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]
pro_need_lst = ['GDF15', 'GGT1', 'TNFRSF10A', 'COL4A1', 'CEACAM1']
pro_f_lst1 = pro_f_lst + pro_need_lst
pro_df = pd.read_csv(dpath + 'pro_imputed.csv', usecols = ['ID'] + pro_f_lst1)

base_f_lst = ['age', 'Pulserate', 'Diabetes', 'units', 'smoke', 'Towns', 'IPAQ', 'BMI', 'hyper', 'viral', 'wctohc']
fib4 = ['fib4']
apri = ['apri']
lab = ['GGT', 'ALT', 'AST', 'Albumin', 'Platelet']
cov_df = pd.read_csv(dpath + 'nonpro_imputed_int_scaled.csv', usecols = ['ID'] + base_f_lst + fib4 + apri + lab)
reg_df = pd.read_csv(dpath + 'region.csv')

mydf = pd.merge(target_df, cov_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, pro_df, how = 'left', on = ['ID'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['ID'])

mydf.loc[mydf.CIRtotal_interuse > 5*365, 'cirtotal'] = 0

fold_id_lst = [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

my_params1 = {'n_estimators': 100,
             'max_depth': 3,
             'num_leaves': 7,
             'subsample': 1,
             'learning_rate': 0.01,
             'colsample_bytree': 1}

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.8,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

y_test_lst, region_lst = [], []
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_test_lst.append(mydf.cirtotal.iloc[test_idx].tolist())
    region_lst.append(mydf.Region_code.iloc[test_idx].tolist())

print(pro_f_lst)
pro_need_lst = ['GDF15', 'GGT1', 'TNFRSF10A', 'COL4A1', 'CEACAM1']

pro = 'GDF15'
for pro in tqdm(pro_need_lst):
    print(pro)
    tmp_f1 = [pro]
    tmp_f2 = fib4
    tmp_f3 = apri
    tmp_f4 = [pro] + base_f_lst
    tmp_f5 = [pro] + lab
    tmp_f6 = [pro] + base_f_lst + lab

    pred_df1 = get_pred_probs(tmp_f1, mydf, fold_id_lst, my_params, 'm1')
    pred_df2 = get_pred_probs(tmp_f2, mydf, fold_id_lst, my_params, 'm2')
    pred_df3 = get_pred_probs(tmp_f3, mydf, fold_id_lst, my_params, 'm3')
    pred_df4 = get_pred_probs(tmp_f4, mydf, fold_id_lst, my_params, 'm4')
    pred_df5 = get_pred_probs(tmp_f5, mydf, fold_id_lst, my_params, 'm5')
    pred_df6 = get_pred_probs(tmp_f6, mydf, fold_id_lst, my_params, 'm6')
    # pred_df7 = get_pred_probs(tmp_f7, mydf, fold_id_lst, my_params, 'm7')
    # pred_df8 = get_pred_probs(tmp_f8, mydf, fold_id_lst, my_params, 'm8')


    myout_df = pd.merge(pred_df1, pred_df2[['ID', 'y_pred_m2']], how = 'inner', on = ['ID'])
    myout_df = pd.merge(myout_df, pred_df3[['ID', 'y_pred_m3']], how = 'inner', on = ['ID'])
    myout_df = pd.merge(myout_df, pred_df4[['ID', 'y_pred_m4']], how = 'inner', on = ['ID'])
    myout_df = pd.merge(myout_df, pred_df5[['ID', 'y_pred_m5']], how = 'inner', on = ['ID'])
    myout_df = pd.merge(myout_df, pred_df6[['ID', 'y_pred_m6']], how = 'inner', on = ['ID'])
    # myout_df = pd.merge(myout_df, pred_df7[['ID', 'y_pred_m7']], how = 'inner', on = ['ID'])
    # myout_df = pd.merge(myout_df, pred_df8[['ID', 'y_pred_m8']], how = 'inner', on = ['ID'])


    myout_df.to_csv(outpath1 + 'pred_probs_' + pro + '.csv', index = False)
