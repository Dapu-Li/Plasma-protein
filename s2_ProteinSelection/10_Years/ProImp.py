from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from collections import Counter
import shap
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore")
dpath = '/home/mrli/桌面/py/shuai/Data/'
outpath = '/home/mrli/桌面/py/shuai/Results/'

pro_df = pd.read_csv(dpath + 'pro_imputed.csv')
pro_dict = pd.read_csv(dpath + 'ProCode.csv', usecols = ['Pro_code', 'Prode_finition'])
target_df = pd.read_csv(dpath + 'outcome.csv')
reg_df = pd.read_csv(dpath + 'region.csv')

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, reg_df, how = 'inner', on = ['ID'])
mydf['cirtotal'].loc[mydf.CIRtotal_interuse > 10*365] = 0
mydf.cirtotal.value_counts()
pro_f_m1_df = pd.read_csv(outpath + 'Cox/M1_Cox_protein.csv')
pro_f_m2_df = pd.read_csv(outpath + 'Cox/M2_Cox_protein.csv')
pro_f_m1 = pro_f_m1_df.loc[pro_f_m1_df.p_val_bfi<0.05].Pro_code.tolist()
pro_f_m2 = pro_f_m2_df.loc[pro_f_m2_df.p_val_bfi<0.05].Pro_code.tolist()
pro_f_lst = [ele for ele in pro_f_m2 if ele in pro_f_m1]



col = mydf.columns.tolist()

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.8,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

tg_imp_cv = Counter()
tc_imp_cv = Counter()
shap_imp_cv = np.zeros(len(pro_f_lst))
fold_id_lst = [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
#LR = LogisticRegression(random_state=2024)
#fold_id = 1
for fold_id in tqdm(fold_id_lst):
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][pro_f_lst], mydf.iloc[test_idx][pro_f_lst]
    y_train, y_test = mydf.iloc[train_idx].cirtotal, mydf.iloc[test_idx].cirtotal
    my_lgb = LGBMClassifier(objective = 'binary', metric = 'auc', is_unbalance = True, verbosity = -1, seed = 2024)
    #my_lgb = LR
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
    shap_values = np.abs(np.average(shap_values, axis=0))
    shap_imp_cv += shap_values / np.sum(shap_values)


shap_imp_df = pd.DataFrame({'Pro_code': pro_f_lst,
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
my_imp_df = pd.merge(my_imp_df, pro_dict, how = 'left', on=['Pro_code'])

my_imp_df.to_csv(outpath + '10Years/ProImportance.csv', index = False)
#my_imp_df.to_csv(outpath + 'ProImportance_LR.csv', index = False)

print('finished')
