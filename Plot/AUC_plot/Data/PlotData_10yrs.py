

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

import time
start = time.time()

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


dpath = 'C:/Users/Administrator/Desktop/pro/shuai/Data/'
outpath = 'C:/Users/Administrator/Desktop/pro/shuai/Results/10Years/'
outpath1 = outpath + 'Delong/'
outputfile = outpath1 + 'pred_probs.csv'

target_df = pd.read_csv(dpath + 'outcome.csv')
pro_f_df = pd.read_csv(outpath + 'AccAUC_TotalGain.csv')
from param.TenYears import nb_top_pros
pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]

use_f_lst = ['GDF15', 'GGT1', 'TNFRSF10A', 'COL4A1', 'CEACAM1']

pro_f_lst1 = pro_f_lst + use_f_lst
pro_f_lst1 = list(set(pro_f_lst1))

pro_df = pd.read_csv(dpath + 'pro_imputed.csv', usecols = ['ID'] + pro_f_lst1)

base_f_lst = ['age', 'sex', 'ethnic', 'wctohc', 'Pulserate', 'Diabetes', 'dtrinking', 'units', 'smoke', 'Towns']
lab_f_lst = ['GGT', 'ALT', 'AST', 'Albumin', 'Platelet']
fib = ['fib4']
apri = ['apri']
cov_df = pd.read_csv(dpath + 'nonpro_imputed_int_scaled.csv', usecols = ['ID'] + base_f_lst + lab_f_lst + fib + apri)
reg_df = pd.read_csv(dpath + 'region.csv')

mydf = pd.merge(target_df, cov_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, pro_df, how = 'left', on = ['ID'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['ID'])

mydf.loc[mydf.CIRtotal_interuse > 10*365, 'cirtotal'] = 0

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
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

y_test_lst, region_lst = [], []
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_test_lst.append(mydf.cirtotal.iloc[test_idx].tolist())
    region_lst.append(mydf.Region_code.iloc[test_idx].tolist())

print(pro_f_lst)

tmp_f1 = ['GDF15']
tmp_f2 = ['GDF15'] + fib
tmp_f3 = ['GDF15'] + apri
tmp_f4 = ['GDF15'] + base_f_lst
tmp_f5 = ['GDF15'] + lab_f_lst
tmp_f6 = ['GDF15'] + base_f_lst + lab_f_lst

tmp_f7 = ['GGT1']
tmp_f8 = ['GGT1'] + fib
tmp_f9 = ['GGT1'] + apri
tmp_f10 = ['GGT1'] + base_f_lst
tmp_f11 = ['GGT1'] + lab_f_lst
tmp_f12 = ['GGT1'] + base_f_lst + lab_f_lst

tmp_f13 = ['TNFRSF10A']
tmp_f14 = ['TNFRSF10A'] + fib
tmp_f15 = ['TNFRSF10A'] + apri
tmp_f16 = ['TNFRSF10A'] + base_f_lst
tmp_f17 = ['TNFRSF10A'] + lab_f_lst
tmp_f18 = ['TNFRSF10A'] + base_f_lst + lab_f_lst

tmp_f19 = ['COL4A1']
tmp_f20 = ['COL4A1'] + fib
tmp_f21 = ['COL4A1'] + apri
tmp_f22 = ['COL4A1'] + base_f_lst
tmp_f23 = ['COL4A1'] + lab_f_lst
tmp_f24 = ['COL4A1'] + base_f_lst + lab_f_lst

tmp_f25 = ['CEACAM1']
tmp_f26 = ['CEACAM1'] + fib
tmp_f27 = ['CEACAM1'] + apri
tmp_f28 = ['CEACAM1'] + base_f_lst
tmp_f29 = ['CEACAM1'] + lab_f_lst
tmp_f30 = ['CEACAM1'] + base_f_lst + lab_f_lst

tmp_f31 = pro_f_lst
tmp_f32 = base_f_lst
tmp_f33 = lab_f_lst
tmp_f34 = pro_f_lst + base_f_lst
tmp_f35 = pro_f_lst + lab_f_lst
tmp_f36 = pro_f_lst + base_f_lst + lab_f_lst


pred_df1 = get_pred_probs(tmp_f1, mydf, fold_id_lst, my_params1, 'm1')
pred_df2 = get_pred_probs(tmp_f2, mydf, fold_id_lst, my_params, 'm2')
pred_df3 = get_pred_probs(tmp_f3, mydf, fold_id_lst, my_params, 'm3')
pred_df4 = get_pred_probs(tmp_f4, mydf, fold_id_lst, my_params, 'm4')
pred_df5 = get_pred_probs(tmp_f5, mydf, fold_id_lst, my_params, 'm5')
pred_df6 = get_pred_probs(tmp_f6, mydf, fold_id_lst, my_params, 'm6')
print(time.time() - start)

pred_df7 = get_pred_probs(tmp_f7, mydf, fold_id_lst, my_params1, 'm7')
pred_df8 = get_pred_probs(tmp_f8, mydf, fold_id_lst, my_params, 'm8')
pred_df9 = get_pred_probs(tmp_f9, mydf, fold_id_lst, my_params, 'm9')
pred_df10 = get_pred_probs(tmp_f10, mydf, fold_id_lst, my_params, 'm10')
pred_df11 = get_pred_probs(tmp_f11, mydf, fold_id_lst, my_params, 'm11')
pred_df12 = get_pred_probs(tmp_f12, mydf, fold_id_lst, my_params, 'm12')
print(time.time() - start)

pred_df13 = get_pred_probs(tmp_f13, mydf, fold_id_lst, my_params1, 'm13')
pred_df14 = get_pred_probs(tmp_f14, mydf, fold_id_lst, my_params, 'm14')
pred_df15 = get_pred_probs(tmp_f15, mydf, fold_id_lst, my_params, 'm15')
pred_df16 = get_pred_probs(tmp_f16, mydf, fold_id_lst, my_params, 'm16')
pred_df17 = get_pred_probs(tmp_f17, mydf, fold_id_lst, my_params, 'm17')
pred_df18 = get_pred_probs(tmp_f18, mydf, fold_id_lst, my_params, 'm18')
print(time.time() - start)

pred_df19 = get_pred_probs(tmp_f19, mydf, fold_id_lst, my_params1, 'm19')
pred_df20 = get_pred_probs(tmp_f20, mydf, fold_id_lst, my_params, 'm20')
pred_df21 = get_pred_probs(tmp_f21, mydf, fold_id_lst, my_params, 'm21')
pred_df22 = get_pred_probs(tmp_f22, mydf, fold_id_lst, my_params, 'm22')
pred_df23 = get_pred_probs(tmp_f23, mydf, fold_id_lst, my_params, 'm23')
pred_df24 = get_pred_probs(tmp_f24, mydf, fold_id_lst, my_params, 'm24')
print(time.time() - start)

pred_df25 = get_pred_probs(tmp_f25, mydf, fold_id_lst, my_params1, 'm25')
pred_df26 = get_pred_probs(tmp_f26, mydf, fold_id_lst, my_params, 'm26')
pred_df27 = get_pred_probs(tmp_f27, mydf, fold_id_lst, my_params, 'm27')
pred_df28 = get_pred_probs(tmp_f28, mydf, fold_id_lst, my_params, 'm28')
pred_df29 = get_pred_probs(tmp_f29, mydf, fold_id_lst, my_params, 'm29')
pred_df30 = get_pred_probs(tmp_f30, mydf, fold_id_lst, my_params, 'm30')
print(time.time() - start)

pred_df31 = get_pred_probs(tmp_f31, mydf, fold_id_lst, my_params, 'm31')
pred_df32 = get_pred_probs(tmp_f32, mydf, fold_id_lst, my_params, 'm32')
pred_df33 = get_pred_probs(tmp_f33, mydf, fold_id_lst, my_params, 'm33')
pred_df34 = get_pred_probs(tmp_f34, mydf, fold_id_lst, my_params, 'm34')
pred_df35 = get_pred_probs(tmp_f35, mydf, fold_id_lst, my_params, 'm35')
pred_df36 = get_pred_probs(tmp_f36, mydf, fold_id_lst, my_params, 'm36')
print(time.time() - start)



myout_df = pd.merge(pred_df1, pred_df2[['ID', 'y_pred_m2']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df3[['ID', 'y_pred_m3']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df4[['ID', 'y_pred_m4']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df5[['ID', 'y_pred_m5']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df6[['ID', 'y_pred_m6']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df7[['ID', 'y_pred_m7']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df8[['ID', 'y_pred_m8']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df9[['ID', 'y_pred_m9']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df10[['ID', 'y_pred_m10']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df11[['ID', 'y_pred_m11']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df12[['ID', 'y_pred_m12']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df13[['ID', 'y_pred_m13']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df14[['ID', 'y_pred_m14']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df15[['ID', 'y_pred_m15']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df16[['ID', 'y_pred_m16']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df17[['ID', 'y_pred_m17']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df18[['ID', 'y_pred_m18']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df19[['ID', 'y_pred_m19']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df20[['ID', 'y_pred_m20']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df21[['ID', 'y_pred_m21']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df22[['ID', 'y_pred_m22']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df23[['ID', 'y_pred_m23']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df24[['ID', 'y_pred_m24']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df25[['ID', 'y_pred_m25']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df26[['ID', 'y_pred_m26']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df27[['ID', 'y_pred_m27']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df28[['ID', 'y_pred_m28']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df29[['ID', 'y_pred_m29']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df30[['ID', 'y_pred_m30']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df31[['ID', 'y_pred_m31']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df32[['ID', 'y_pred_m32']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df33[['ID', 'y_pred_m33']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df34[['ID', 'y_pred_m34']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df35[['ID', 'y_pred_m35']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df36[['ID', 'y_pred_m36']], how = 'inner', on = ['ID'])
print(time.time() - start)
myout_df.to_csv(outputfile, index = False)



my_array = np.zeros((37, 37))

for i in tqdm(range(1,37)):
    for j in range(1,37):
        col1 = 'y_pred_m' + str(i)
        col2 = 'y_pred_m' + str(j)
        stat = delong_roc_test(myout_df['cirtotal'], myout_df[col1], myout_df[col2])
        my_array[i, j] = np.exp(stat)[0][0]

myout_df = pd.DataFrame(my_array)

cata = ['Catagory',
        'GDF15', 'GDF15+Fib4', 'GDF15+APRI', 'GDF15+Basic', 'GDF15+Lab', 'GDF15+Basic+Lab',
        'GGT1', 'GGT1+Fib4', 'GGT1+APRI', 'GGT1+Basic', 'GGT1+Lab', 'GGT1+Basic+Lab',
        'TNFRSF10A', 'TNFRSF10A+Fib4', 'TNFRSF10A+APRI', 'TNFRSF10A+Basic', 'TNFRSF10A+Lab', 'TNFRSF10A+Basic+Lab',
        'COL4A1', 'COL4A1+Fib4', 'COL4A1+APRI', 'COL4A1+Basic', 'COL4A1+Lab', 'COL4A1+Basic+Lab',
        'CEACAM1', 'CEACAM1+Fib4', 'CEACAM1+APRI', 'CEACAM1+Basic', 'CEACAM1+Lab', 'CEACAM1+Basic+Lab',
        'ProPANEL', 'Basic', 'Lab', 'ProPANEL+Basic', 'ProPANEL+Lab', 'ProPANEL+Basic+Lab']
myout_df.columns = cata
myout_df.Catagory = cata
myout_df = myout_df.drop(0, axis=0)

myout_df.to_csv(outpath1 + 'DelongTest.csv', index = False)