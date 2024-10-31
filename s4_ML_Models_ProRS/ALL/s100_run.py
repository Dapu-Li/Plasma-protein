import os
import sys
import warnings
warnings.filterwarnings("ignore")
import time
sys.path.extend(['/home/mrli/桌面/py/shuai/s4_ML_Models_ProRS/ALL'])

path = os.path.dirname('/home/mrli/桌面/py/shuai/s4_ML_Models_ProRS/ALL/')
files = os.listdir(path)
files = [file for file in files if file.endswith('.py')]

current_file = os.path.basename('s100_run.py')
if current_file in files:
    files.remove(current_file)

files = sorted(files)


for file in files:
    start = time.time()
    print(f'Running {file}')
    os.system(f'python {file}')
    print(f'Finished {file} in {time.time()-start} seconds')
