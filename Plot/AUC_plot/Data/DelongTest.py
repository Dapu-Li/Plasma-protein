

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
import pandas as pd
from Utility.DelongTest import delong_roc_test
import warnings
warnings.filterwarnings('ignore')

# if __name__ == '__main__':
#
#     dpath = 'C:/Users/Administrator/Desktop/pro/shuai/Results/5Years/Delong/'
#
#     mydf = pd.read_csv(dpath + 'pred_probs.csv')
#
#     my_array = np.zeros((14, 14))
#
#     i = 1
#     j = 1
#     for i in range(1,14):
#         for j in range(1,14):
#             col1 = 'y_pred_m' + str(i)
#             col2 = 'y_pred_m' + str(j)
#             stat = delong_roc_test(mydf['cirtotal'], mydf[col1], mydf[col2])
#             my_array[i, j] = np.exp(stat)[0][0]
#
#     myout_df = pd.DataFrame(my_array)
#     myout_df.to_csv(dpath + 'DelongTest.csv', index = False)
