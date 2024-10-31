import seaborn as sns
import matplotlib.pyplot as plt
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
outpath = '/home/mrli/桌面/py/shuai/Results/SHAP/'
outfile = outpath + 'shap.csv'
outfile1 = outpath + 'shap1.csv'

cox1 = pd.read_csv(outpath + '../Cox/M1_Cox_protein.csv')
cox2 = pd.read_csv(outpath + '../Cox/M2_Cox_protein.csv')
pro1 = cox1.loc[cox1.p_val_bfi < 0.05].Pro_code.tolist()
pro2 = cox2.loc[cox2.p_val_bfi < 0.05].Pro_code.tolist()
pro_f_lst = [ele for ele in pro1 if ele in pro2]
pro_df = pd.read_csv(dpath + 'pro_imputed.csv', usecols=['ID'] + pro_f_lst)
target_df = pd.read_csv(dpath + 'outcome.csv')
reg_df = pd.read_csv(dpath + 'region.csv')
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, reg_df, how = 'inner', on = ['ID'])

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.8,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

tg_imp_cv = Counter()
tc_imp_cv = Counter()
shap_imp_cv = np.zeros(len(pro_f_lst))
fold_id_lst = [i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

fold_id = 1
shap_df = pd.DataFrame()
pro_shap_list = []
for i in pro_f_lst:
    name = i + '_shap'
    pro_shap_list += [name]


for fold_id in tqdm(fold_id_lst):
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][pro_f_lst], mydf.iloc[test_idx][pro_f_lst]
    y_train, y_test = mydf.iloc[train_idx].cirtotal, mydf.iloc[test_idx].cirtotal
    my_lgb = LGBMClassifier(objective = 'binary', metric = 'auc', is_unbalance = True, verbosity = -1, seed = 2024)
    #my_lgb = LR
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)

    # 计算SHAP值
    explainer = shap.TreeExplainer(my_lgb)
    shap_values = explainer.shap_values(X_test)

    # 将SHAP值和对应的蛋白质值添加到DataFrame中
    # 这里假设shap_values的形状是(len(X_test), len(pro_f_lst))
    temp_df = X_test.copy()
    shap_df_fold = pd.DataFrame(shap_values, columns=pro_shap_list, index=X_test.index)
    temp_df = pd.concat([temp_df, shap_df_fold], axis=1)
    temp_df['fold_id'] = fold_id
    shap_df = pd.concat([shap_df, temp_df], axis=0)

shap_df_transformed = pd.DataFrame(columns=['shap_value', 'Pro_values', 'Pro_code'])


shap_df_transformed = pd.DataFrame(columns=['shap_value', 'Pro_values', 'Pro_code'])

for protein in pro_f_lst:
    temp_df = shap_df[['fold_id', protein, protein + '_shap']].copy()
    temp_df.rename(columns={protein: 'Pro_values', protein + '_shap': 'shap_value'}, inplace=True)
    temp_df['Pro_code'] = protein

    shap_df_transformed = pd.concat([shap_df_transformed, temp_df], axis=0)

# 重置索引
shap_df_transformed.reset_index(drop=True, inplace=True)
shap_df_transformed.to_csv(outfile1, index=False)