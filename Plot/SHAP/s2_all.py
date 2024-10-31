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
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

dpath = 'C:/Users/Administrator/Desktop/pro/shuai/Data/'
outpath = 'C:/Users/Administrator/Desktop/pro/shuai/Results/SHAP/'
outfile = outpath + 'shap_all.pdf'

pro_f_df = pd.read_csv(outpath + '../ALL/AccAUC_TotalGain.csv')
from param.ALL import nb_top_pros
pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]
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




X = mydf[pro_f_lst]
y = mydf.cirtotal

my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, verbosity=-1, seed=2024)
my_lgb.set_params(**my_params)
my_lgb.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(my_lgb)
shap_values = explainer.shap_values(X)


# 绘制SHAP图
shap.summary_plot(shap_values, X, plot_type='dot', show=False)

plt.gcf().set_size_inches(25, 10)
plt.savefig(outfile)