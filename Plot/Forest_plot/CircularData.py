

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
import warnings
import re
import shap
pd.options.mode.chained_assignment = None  # default='warn'

outpath = '/home/mrli/桌面/py/shuai/Results/Cox/'

pro_f_m1_df = pd.read_csv(outpath + 'M1_Cox_protein.csv')
pro_f_m2_df = pd.read_csv(outpath + 'M2_Cox_protein.csv')
pro_f_m1 = pro_f_m1_df.loc[pro_f_m1_df.p_val_bfi < 0.05].Pro_code.tolist()
pro_f_m2 = pro_f_m2_df.loc[pro_f_m2_df.p_val_bfi < 0.05].Pro_code.tolist()
pro_f_lst = [ele for ele in pro_f_m2 if ele in pro_f_m1]
selected_df = pro_f_m2_df[pro_f_m2_df.Pro_code.isin(pro_f_lst)]
selected_df['Target'] = 'cirtotal'

selected_df.to_csv(outpath + 'CircularData.csv', index = False)