

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines import KaplanMeierFitter as kmf
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

dpath = '/home/mrli/桌面/py/shuai/Data/'
outpath = '/home/mrli/桌面/py/shuai/Results/KM/'
outfile = outpath + 'KM_count.csv'
tgt_outcome = 'cirtotal'

pro_f_df = pd.read_csv(outpath + 'KM_info2.csv')
pro_f_lst = pro_f_df.Pro_code.tolist()
cut_f_lst = pro_f_df.opt_ct.tolist()
riskdir_f_lst = pro_f_df.HR.tolist()
pval_f_lst = pro_f_df.bfn.tolist()
hrci_f_lst = pro_f_df.HR_95CI.tolist()

pro_df = pd.read_csv(dpath + 'pro_imputed.csv', usecols=['ID'] + pro_f_lst)
target_df = pd.read_csv(dpath + 'outcome.csv')
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['ID'])
cut_lst = [0, 2.5*365, 5*365, 7.5*365, 10*365, 12.5*365, 15*365, 17.5*365]

def get_risk_info(mydf, pro_f, pro_f_cut, cut_lst):
    h_risk_df = mydf.loc[mydf[pro_f] > pro_f_cut]
    h_risk_df.reset_index(inplace = True, drop = True)
    l_risk_df = mydf.loc[mydf[pro_f] <= pro_f_cut]
    l_risk_df.reset_index(inplace = True, drop = True)
    high_risk_lst, low_risk_lst = [], []
    for cut in cut_lst:
        h_tmpdf1 = h_risk_df.loc[h_risk_df.CIRtotal_interuse > cut]
        h_tmpdf2 = h_risk_df.loc[h_risk_df.CIRtotal_interuse <= cut]
        nb_at_risk_h = len(h_tmpdf1)
        nb_events_h = h_tmpdf2.cirtotal.sum()
        high_risk_lst.append(str(nb_at_risk_h) + ' (' + str(nb_events_h) + ')')
        l_tmpdf1 = l_risk_df.loc[l_risk_df.CIRtotal_interuse > cut]
        l_tmpdf2 = l_risk_df.loc[l_risk_df.CIRtotal_interuse <= cut]
        nb_at_risk_l = len(l_tmpdf1)
        nb_events_l = l_tmpdf2.cirtotal.sum()
        low_risk_lst.append(str(nb_at_risk_l) + ' (' + str(nb_events_l) + ')')
    return (high_risk_lst, low_risk_lst)


myout_df = pd.DataFrame()

for i in range(len(pro_f_lst)):
    pro_f, pro_f_cut = pro_f_lst[i], cut_f_lst[i]
    high_risk_lst, low_risk_lst = get_risk_info(mydf, pro_f, pro_f_cut, cut_lst=cut_lst)
    tmpdf = pd.DataFrame([low_risk_lst, high_risk_lst]).T
    tmpdf.columns = ['lrisk_' + pro_f + '_' + tgt_outcome, 'hrisk_' + pro_f + '_' + tgt_outcome]
    myout_df = pd.concat([myout_df, tmpdf.T], axis = 0)

myout_df.to_csv(outfile)
