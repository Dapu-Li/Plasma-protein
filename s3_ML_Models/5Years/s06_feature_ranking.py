

import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

# def get_top_pros(mydf):
#     p_lst = mydf.p_delong.tolist()
#     i = 0
#     while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)):
#         i+=1
#     return i

dpath = '/home/mrli/桌面/py/shuai/Data/'
outpath = '/home/mrli/桌面/py/shuai/Results/5Years/'
outputfile = outpath + 's6_FeaImpRank.csv'

pro_f_df = pd.read_csv(outpath + 'AccAUC_TotalGain.csv')

nb_top_pros = 11

pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]
pro_df = pd.read_csv(dpath + 'pro_imputed.csv',usecols = ['ID'] + pro_f_lst)
pro_dict = pd.read_csv(dpath + 'ProCode.csv', usecols = ['Pro_code', 'Prode_finition'])
target_df = pd.read_csv(dpath + 'outcome.csv')
cov_f_lst = ['age', 'Pulserate', 'Diabetes', 'units', 'smoke', 'Towns', 'IPAQ', 'BMI', 'hyper', 'viral', 'wctohc']

cov_df = pd.read_csv(dpath + 'nonpro_imputed_int_scaled.csv', usecols = ['ID'] + cov_f_lst)
reg_df = pd.read_csv(dpath + 'region.csv')


mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, reg_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, cov_df, how = 'inner', on = ['ID'])
fold_id_lst = [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

mydf.loc[mydf.CIRtotal_interuse > 5*365, 'cirtotal'] = 0

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

tmp_f_lst = cov_f_lst + pro_f_lst
tg_imp_cv = Counter()
tc_imp_cv = Counter()
shap_imp_cv = np.zeros(len(tmp_f_lst))

for fold_id in fold_id_lst:
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][tmp_f_lst], mydf.iloc[test_idx][tmp_f_lst]
    y_train, y_test = mydf.iloc[train_idx].cirtotal, mydf.iloc[test_idx].cirtotal
    my_lgb = LGBMClassifier(objective = 'binary', metric = 'auc', is_unbalance = True, verbosity = -1, seed = 2024)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
    totalcover_imp = my_lgb.booster_.feature_importance(importance_type='split')
    totalcover_imp = dict(zip(my_lgb.booster_.feature_name(), totalcover_imp.tolist()))
    tg_imp_cv += Counter(normal_imp(totalgain_imp))
    tc_imp_cv += Counter(normal_imp(totalcover_imp))
    explainer = shap.TreeExplainer(my_lgb)
    shap_values = explainer.shap_values(X_test)
    shap_values = np.abs(np.average(shap_values[0], axis=0))
    shap_imp_cv += shap_values / np.sum(shap_values)


shap_imp_df = pd.DataFrame({'Pro_code': tmp_f_lst,
                            'ShapValues_cv': shap_imp_cv/10})
shap_imp_df.sort_values(by = 'ShapValues_cv', ascending = False, inplace = True)

tg_imp_cv = normal_imp(tg_imp_cv)
tg_imp_df = pd.DataFrame({'Pro_code': list(tg_imp_cv.keys()),
                          'TotalGain_cv': list(tg_imp_cv.values())})

tc_imp_cv = normal_imp(tc_imp_cv)
tc_imp_df = pd.DataFrame({'Pro_code': list(tc_imp_cv.keys()),
                          'TotalCover_cv': list(tc_imp_cv.values())})

my_imp_df = pd.merge(left = shap_imp_df, right = tg_imp_df, how = 'left', on = ['Pro_code'])
my_imp_df = pd.merge(left = my_imp_df, right = tc_imp_df, how = 'left', on = ['Pro_code'])
my_imp_df['Ensemble'] = (my_imp_df['ShapValues_cv'] + my_imp_df['TotalGain_cv'] + my_imp_df['TotalCover_cv'])/3
my_imp_df.sort_values(by = 'Ensemble', ascending = False, inplace = True)

my_imp_df.to_csv(outputfile, index = False)

print('finished')