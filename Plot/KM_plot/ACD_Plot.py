

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines import KaplanMeierFitter as kmf
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

dpath = 'C:/Users/Administrator/Desktop/pro/shuai//Data/'
outpath = 'C:/Users/Administrator/Desktop/pro/shuai/Results/KM/'
outfile = outpath + 'KM.png'

pro_f_df = pd.read_csv(outpath + 'KM_info2.csv')
pro_f_df.bfn = pro_f_df.bfn.astype(str)

pro_f_df['P'] = pro_f_df['bfn'].astype(str).str[:4]
last_two_digits = pro_f_df['bfn'].astype(str).str[-3:]
pro_f_df['P'] = pro_f_df['P'] + ' x 10^' + last_two_digits

pro_f_lst = pro_f_df.Pro_code.tolist()
cut_f_lst = pro_f_df.opt_ct.tolist()
riskdir_f_lst = pro_f_df.HR.tolist()
pval_f_lst = pro_f_df.P.tolist()
hrci_f_lst = pro_f_df.HR_95CI.tolist()

pro_df = pd.read_csv(dpath + 'pro_imputed.csv', usecols=['ID'] + pro_f_lst)
target_df = pd.read_csv(dpath + 'outcome.csv')
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['ID'])
# cut_lst = [0, 2.5*365, 5*365, 7.5*365, 10*365, 12.5*365, 15*365, 17.5*365]

mydf.CIRtotal_interuse = mydf.CIRtotal_interuse/365

for i in range(len(pro_f_lst)):
    pro_f, f_cut, risk_dir = pro_f_lst[i], np.round(cut_f_lst[i], 2), riskdir_f_lst[i]
    f_hrci, f_pval = hrci_f_lst[i], pval_f_lst[i]
    plotdf = mydf[['ID', 'cirtotal', 'CIRtotal_interuse'] + [pro_f]]
    plotdf.rename(columns={pro_f: 'target_pro'}, inplace=True)
    rm_idx = plotdf.index[plotdf.target_pro.isnull() == True]
    plotdf = plotdf.drop(rm_idx, axis=0)
    plotdf.reset_index(inplace=True)
    if risk_dir > 1:
        high_risk = (plotdf.target_pro > f_cut)
        prop = np.round(high_risk.sum()/len(plotdf)*100,2)
        high_risk_label = 'High risk group (>' + str(f_cut) + ', ' + str(prop) + '%)'
        low_risk_label = 'Rest control'
    elif risk_dir < 1:
        high_risk = (plotdf.target_pro < f_cut)
        prop = np.round(high_risk.sum() / len(plotdf) * 100, 2)
        high_risk_label = 'High risk group (<' + str(f_cut) + ', ' + str(prop) + '%)'
        low_risk_label = 'Rest control'
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(durations=plotdf.CIRtotal_interuse[~high_risk], event_observed=plotdf.cirtotal[~high_risk], label='Low')
    kmf.plot_survival_function(ax=ax, color='#abc3f0', linewidth=3)
    kmf.fit(durations=plotdf.CIRtotal_interuse[high_risk], event_observed=plotdf.cirtotal[high_risk], label='High')
    kmf.plot_survival_function(ax=ax, color='red', linewidth=3)
    ax.set_title(pro_f, weight='normal', fontsize=20)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Survival probability', weight='normal', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('Follow-up time (years)', weight='normal', fontsize=14)
    ax.plot([], [], ' ', label="HR = " + f_hrci)
    ax.plot([], [], ' ', label="p value = " + f_pval)
    ax.legend(loc='lower left', fontsize='large', labelspacing=1, frameon=False, facecolor='none')
    #fontsize='x-large'
    plt.subplots_adjust(left=0.2, bottom=0.2)
    fig.tight_layout()
    plt.savefig(outfile + pro_f + '.pdf', bbox_inches='tight',pad_inches = 0.05)

