

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
pd.options.mode.chained_assignment = None  # default='warn'

outpath = '/home/mrli/桌面/py/shuai/Results/10Years/'

s1_df = pd.read_csv(outpath + 's1_1Pro.csv')
s1_df['Category'] = 'SinglePro'

s20_df = pd.read_csv(outpath + 's20_BasicCov.csv')
s20_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s20_df], axis = 1)
s20_df['Category'] = 'BasicCovar'

s21_df = pd.read_csv(outpath + 's21_BasicCov.csv')
s21_df['Category'] = 'SinglePro+BasicCovar'

s22_df = pd.read_csv(outpath + 's22_FibCov.csv')
s22_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s22_df], axis = 1)
s22_df['Category'] = 'FibCovar'

s23_df = pd.read_csv(outpath + 's23_FibCov.csv')
s23_df['Category'] = 'SinglePro+FibCovar'

s24_df = pd.read_csv(outpath + 's24_ApriCov.csv')
s24_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s24_df], axis = 1)
s24_df['Category'] = 'ApriCovar'

s25_df = pd.read_csv(outpath + 's25_ApriCov.csv')
s25_df['Category'] = 'SinglePro+ApriCovar'

s26_df = pd.read_csv(outpath + 's26_Fib_BasicCov.csv')
s26_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s26_df], axis = 1)
s26_df['Category'] = 'FibCovar+BasicCovar'

s27_df = pd.read_csv(outpath + 's27_Fib_BasicCov.csv')
s27_df['Category'] = 'SinglePro+FibCovar+BasicCovar'

s28_df = pd.read_csv(outpath + 's28_Apri_BasicCov.csv')
s28_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s28_df], axis = 1)
s28_df['Category'] = 'ApriCovar+BasicCovar'

s29_df = pd.read_csv(outpath + 's29_Apri_BasicCov.csv')
s29_df['Category'] = 'SinglePro+ApriCovar+BasicCovar'


s36_df = pd.read_csv(outpath + 's36_GGTCov.csv')
s36_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s36_df], axis = 1)
s36_df['Category'] = 'GGTCovar'

s37_df = pd.read_csv(outpath + 's37_GGTCov.csv')
s37_df['Category'] = 'SinglePro+GGTCovar'

s38_df = pd.read_csv(outpath + 's38_LabCov.csv')
s38_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s38_df], axis = 1)
s38_df['Category'] = 'LabCovar'

s39_df = pd.read_csv(outpath + 's39_LabCov.csv')
s39_df['Category'] = 'SinglePro+LabCovar'

s40_df = pd.read_csv(outpath + 's40_Lab_BasicCov.csv')
s40_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s40_df], axis = 1)
s40_df['Category'] = 'LabCovar+BasicCovar'

s41_df = pd.read_csv(outpath + 's41_Lab_BasicCov.csv')
s41_df['Category'] = 'SinglePro+LabCovar+BasicCovar'

####
s30_df = pd.read_csv(outpath + 's30_ProPANEL.csv')
s30_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s30_df], axis = 1)
s30_df['Category'] = 'ProPANEL'

s31_df = pd.read_csv(outpath + 's31_ProPANEL_BasicCov.csv')
s31_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s31_df], axis = 1)
s31_df['Category'] = 'ProPANEL+BasicCovar'

s32_df = pd.read_csv(outpath + 's32_ProPANEL_FibCov.csv')
s32_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s32_df], axis = 1)
s32_df['Category'] = 'ProPANEL+FibCovar'

s33_df = pd.read_csv(outpath + 's33_ProPANEL_ApriCov.csv')
s33_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s33_df], axis = 1)
s33_df['Category'] = 'ProPANEL+ApriCovar'

s34_df = pd.read_csv(outpath + 's34_ProPANEL_Fib_BasicCov.csv')
s34_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s34_df], axis = 1)
s34_df['Category'] = 'ProPANEL+FibCovar+BasicCovar'

s35_df = pd.read_csv(outpath + 's35_ProPANEL_Apri_BasicCov.csv')
s35_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s35_df], axis = 1)
s35_df['Category'] = 'ProPANEL+ApriCovar+BasicCovar'

s42_df = pd.read_csv(outpath + 's42_ProPANEL_LabCov.csv')
s42_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s42_df], axis = 1)
s42_df['Category'] = 'ProPANEL+LabCovar'

s43_df = pd.read_csv(outpath + 's43_ProPANEL_Lab_BasicCov.csv')
s43_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s43_df], axis = 1)
s43_df['Category'] = 'ProPANEL+LabCovar+BasicCovar'



mydf = pd.concat([s1_df, s20_df, s21_df, s22_df, s23_df, s24_df, s25_df, s26_df, s27_df, s28_df, s29_df,
                  s36_df, s37_df, s38_df, s39_df, s40_df, s41_df,
                  s30_df, s31_df, s32_df, s33_df, s34_df, s35_df, s42_df, s43_df], axis = 0)

mydf.to_csv(outpath + 'CombineResults.csv', index = False)

