import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import interp
from tqdm import tqdm

from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.stats as stats
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

mpl.rcParams['font.family'] = 'Arial'

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
outpath = 'C:/Users/Administrator/Desktop/pro/shuai/Results/5Years/'
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

tmp_f1 = ['GDF15']
tmp_f2 = ['GGT1']
tmp_f3 = ['TNFRSF10A']
tmp_f4 = ['COL4A1']
tmp_f5 = ['CEACAM1']
tmp_f6 = pro_f_lst
tmp_f7 = pro_f_lst + base_f_lst
tmp_f8 = pro_f_lst + lab
tmp_f9 = pro_f_lst + base_f_lst + lab


pred_df1 = get_pred_probs(tmp_f1, mydf, fold_id_lst, my_params, 'm1')
pred_df2 = get_pred_probs(tmp_f2, mydf, fold_id_lst, my_params, 'm2')
pred_df3 = get_pred_probs(tmp_f3, mydf, fold_id_lst, my_params, 'm3')
pred_df4 = get_pred_probs(tmp_f4, mydf, fold_id_lst, my_params, 'm4')
pred_df5 = get_pred_probs(tmp_f5, mydf, fold_id_lst, my_params, 'm5')
pred_df6 = get_pred_probs(tmp_f6, mydf, fold_id_lst, my_params, 'm6')
pred_df7 = get_pred_probs(tmp_f7, mydf, fold_id_lst, my_params, 'm7')
pred_df8 = get_pred_probs(tmp_f8, mydf, fold_id_lst, my_params, 'm8')
pred_df9 = get_pred_probs(tmp_f9, mydf, fold_id_lst, my_params, 'm9')


myout_df = pd.merge(pred_df1, pred_df2[['ID', 'y_pred_m2']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df3[['ID', 'y_pred_m3']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df4[['ID', 'y_pred_m4']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df5[['ID', 'y_pred_m5']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df6[['ID', 'y_pred_m6']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df7[['ID', 'y_pred_m7']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df8[['ID', 'y_pred_m8']], how = 'inner', on = ['ID'])
myout_df = pd.merge(myout_df, pred_df9[['ID', 'y_pred_m9']], how = 'inner', on = ['ID'])


myout_df.to_csv(outpath1 + 'pred_probs_panel.csv', index = False)


def AUC_calculator(mydf, y_pred_col, fold_id_lst):
    auc_cv_lst = []
    for fold_id in fold_id_lst:
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col].iloc[test_idx]
        auc_cv_lst.append(roc_auc_score(y_true, y_pred))
    auc_mean = np.round(np.mean(auc_cv_lst), 3)
    auc_std = np.round(np.std(auc_cv_lst), 3)
    return auc_mean, auc_std

mydf = myout_df


mywidth = 3
titlename = '5-year incidents'
col1, col2, col3 = 'steelblue', 'deepskyblue', 'yellowgreen'
col4, col5, col6 = 'darkorange', 'lightcoral', 'darkred'
col7, col8, col9 = 'darkviolet', 'darkgreen', 'darkblue'

y_pred_col1, y_pred_col2, y_pred_col3 = 'y_pred_m1', 'y_pred_m2', 'y_pred_m3'
y_pred_col4, y_pred_col5, y_pred_col6 = 'y_pred_m4', 'y_pred_m5', 'y_pred_m6'
y_pred_col7, y_pred_col8, y_pred_col9 = 'y_pred_m7', 'y_pred_m8', 'y_pred_m9'

auc_mean_m1, auc_sd_m1 = AUC_calculator(mydf, y_pred_col1, fold_id_lst)
auc_mean_m2, auc_sd_m2 = AUC_calculator(mydf, y_pred_col2, fold_id_lst)
auc_mean_m3, auc_sd_m3 = AUC_calculator(mydf, y_pred_col3, fold_id_lst)
auc_mean_m4, auc_sd_m4 = AUC_calculator(mydf, y_pred_col4, fold_id_lst)
auc_mean_m5, auc_sd_m5 = AUC_calculator(mydf, y_pred_col5, fold_id_lst)
auc_mean_m6, auc_sd_m6 = AUC_calculator(mydf, y_pred_col6, fold_id_lst)
auc_mean_m7, auc_sd_m7 = AUC_calculator(mydf, y_pred_col7, fold_id_lst)
auc_mean_m8, auc_sd_m8 = AUC_calculator(mydf, y_pred_col8, fold_id_lst)
auc_mean_m9, auc_sd_m9 = AUC_calculator(mydf, y_pred_col9, fold_id_lst)


legend1 = 'Plasma GDF15 only (AUC =' + str(auc_mean_m1) + ')'
legend2 = 'Plasma GGT1 only (AUC =' + str(auc_mean_m2) +')'
legend3 = 'Plasma TNFRSF10A only (AUC =' + str(auc_mean_m3) +')'
legend4 = 'Plasma COL4A1 only (AUC =' + str(auc_mean_m4) +')'
legend5 = 'Plasma CEACAM1 only (AUC =' + str(auc_mean_m5) +')'
legend6 = 'Protein panel (AUC =' + str(auc_mean_m6) +')'
legend7 = 'Protein panel + demographic (AUC =' + str(auc_mean_m7) +')'
legend8 = 'Protein panel + laboratory (AUC =' + str(auc_mean_m8) +')'
legend9 = 'Protein panel + demographic + laboratory (AUC =' + str(auc_mean_m9) +')'




fig, ax = plt.subplots(figsize = (13, 13))

tprs = []
base_fpr = np.linspace(0, 1, 101)

for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col1].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col1, linewidth = mywidth, label = legend1)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col1, alpha = 0.1)


tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col2].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col2, linewidth = mywidth, label = legend2)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col2, alpha = 0.1)



tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col3].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col3, linewidth = mywidth, label = legend3)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col3, alpha = 0.1)



tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col4].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col4, linewidth = mywidth, label = legend4)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col4, alpha = 0.1)



tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col5].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col5, linewidth = mywidth, label = legend5)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col5, alpha = 0.1)




tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col6].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col6, linewidth = mywidth, label = legend6)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col6, alpha = 0.1)

tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col7].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col7, linewidth = mywidth, label = legend7)

tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col8].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col8, linewidth = mywidth, label = legend8)

tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].cirtotal, mydf[y_pred_col9].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col9, linewidth = mywidth, label = legend9)

plt.legend(loc=4, fontsize=20, labelspacing=1.5, facecolor='gainsboro')

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.ylabel('True positive rate', fontsize=30, labelpad=25)
plt.xlabel('False positive rate', fontsize=30, labelpad=25)
plt.title(titlename, fontsize=30)

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=25)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=25)
ax.tick_params(axis='both', which='both', length=10)
plt.grid(which='minor', alpha=0.2, linestyle=':')
plt.grid(which='major', alpha=0.5, linestyle='--')
plt.tight_layout()
ax = plt.gca()  # 获取当前的Axes对象
ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                       facecolor='none', edgecolor='black', linewidth=mywidth))
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')


plt.savefig(outpath1 + 'ROC_panel.png')
plt.savefig(outpath1 + 'ROC_panel.png')