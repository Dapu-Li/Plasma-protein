


import glob
import os
import numpy as np
import pandas as pd
import re
from joblib import Parallel, delayed
import multiprocessing
import time

t = time.time()




dpath = 'C:/Users/Administrator/Desktop/pro/shuai/Data/'
outpath = 'C:/Users/Administrator/Desktop/pro/shuai/Results/control/'
outfile = outpath + 'S1_case_control_eid_df.csv'

pro_f_df = pd.read_csv(outpath + '../ALL/AccAUC_TotalGain.csv')

pro_f_lst = pro_f_df.Pro_code.tolist()[:20]
pro_df = pd.read_csv(dpath + 'pro_imputed.csv',usecols = ['ID'] + pro_f_lst)
pro_dict = pd.read_csv(dpath + 'ProCode.csv', usecols = ['Pro_code', 'Prode_finition'])
target_df = pd.read_csv(dpath + 'outcome2.csv')
cov_f_lst = ['age', 'sex', 'ethnic', 'wctohc', 'Pulserate', 'Diabetes', 'dtrinking', 'units', 'smoke', 'Towns', 'BMI', 'viral']

cov_df = pd.read_csv(dpath + 'nonpro_imputed_int.csv', usecols = ['ID'] + cov_f_lst)
reg_df = pd.read_csv(dpath + 'region.csv')

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['ID'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['ID'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['ID'])

case_df = mydf.loc[mydf['cirtotal'] == 1].copy()  # Create a copy of the slice
case_df.reset_index(inplace=True)
case_df.drop(['index'], axis=1, inplace=True)

control_df = mydf.loc[mydf['cirtotal'] == 0].copy()  # Create a copy of the slice
control_df.reset_index(inplace=True)
control_df.drop(['index'], axis=1, inplace=True)

col = case_df.columns.tolist()
print(col)

def get_match_subject(case_id, case_df, control_df):
    tmp_case = case_df[case_df['ID'] == case_id]
    tmp_case_flyrs = float(tmp_case['CIRtotal_interuse'].iloc[0])
    tmp_case_adyrs = float(tmp_case['CIRtotal_interuse'].iloc[0])
    tmp_case_age = float(tmp_case['age'].iloc[0])
    tmp_case_gender = int(tmp_case['sex'].iloc[0])
    tmp_case_ethnicity = int(tmp_case['ethnic'].iloc[0])
    tmp_case_viral = int(tmp_case['viral'].iloc[0])
    tmp_case_education = int(tmp_case['Diabetes'].iloc[0])
    tmp_case_dk = int(tmp_case['dtrinking'].iloc[0])
    tmp_case_TDI = int(tmp_case['Towns'].iloc[0])
    tmp_case_bmi = float(tmp_case['BMI'].iloc[0])
    tmp_control_f = control_df.loc[control_df['CIRtotal_interuse'] >= tmp_case_flyrs]
    tmp_control_fa = tmp_control_f.loc[((pd.isna(tmp_control_f['BL2Death_yrs'])) | (tmp_control_f['BL2Death_yrs'] >= tmp_case_adyrs))]
    tmp_control_faa = tmp_control_fa.loc[((tmp_control_fa['age'] >= tmp_case_age - 2) & (tmp_control_fa['age'] <= tmp_case_age + 2))]
    tmp_control_faag = tmp_control_faa.loc[tmp_control_faa['sex'] == tmp_case_gender]
    #tmp_control_faage = tmp_control_faag.loc[tmp_control_faag['ethnic'] == tmp_case_ethnicity]
    tmp_control_faagev = tmp_control_faag.loc[tmp_control_faag['viral'] == tmp_case_viral]
    #tmp_control_faagee = tmp_control_faagev.loc[((tmp_control_faagev['Diabetes'] == tmp_case_education))]
    tmp_control_faageetb = tmp_control_faagev.loc[((tmp_control_faagev['dtrinking'] == tmp_case_dk))]
    tmp_control_faageet = tmp_control_faageetb.loc[((tmp_control_faageetb['Towns'] >= tmp_case_TDI - 0.5) & (tmp_control_faageetb['Towns'] <= tmp_case_TDI + 0.5))]
    tmp_control_faageetbmi = tmp_control_faageet.loc[((tmp_control_faageet['BMI'] >= tmp_case_bmi - 1) & (tmp_control_faageet['BMI'] <= tmp_case_bmi + 1))]
    if len(tmp_control_faageetbmi) < 10:
        tmp_control_faageetbmi = tmp_control_faageet
    if len(tmp_control_faageet) < 30:
        tmp_control_faageetbmi = tmp_control_faageetb
    # if len(tmp_control_faageetb) < 30:
    #     tmp_control_faageetbmi = tmp_control_faa
    # if len(tmp_control_faagee) < 30:
    #     tmp_control_faageetbmi = tmp_control_faagev
    return tmp_control_faageetbmi['ID'].tolist()


def get_match_population(case_df, control_df):
    results = []
    case_eid = case_df['ID'].tolist()
    i = 0
    for ID in case_eid:
        control_eids = get_match_subject(ID, case_df, control_df)
        results.append([len(control_eids)] + [ID] + control_eids)
        i+=1
        if i % 250 == 0:
            print(i)
    return results
# control_eids = get_match_subject(1031222, case_df, control_df)
# results = []
# results.append([len(control_eids)] + [1031222] + control_eids)

case_control_eid = get_match_population(case_df, control_df)
case_control_eid_df = pd.DataFrame(case_control_eid)

case_control_eid_df.columns = ['nb_available_controls', 'case_ids'] + \
                              [str(ele) for ele in range(1, case_control_eid_df.shape[1]-1)]

case_control_eid_df.sort_values(by=['nb_available_controls'], ascending= True, inplace= True)
rm_idx = case_control_eid_df.index[case_control_eid_df['nb_available_controls'] < 10]
case_control_eid_df.drop(rm_idx, axis = 0, inplace = True)
case_control_eid_df.reset_index(inplace = True)
case_control_eid_df.drop(['index'], axis = 1, inplace = True)

case_control_eid_df.to_csv(outfile, index = False)

print(time.time() - t)

print('finished')


