import os
import sys
import warnings
warnings.filterwarnings("ignore")
import time

current_path = 'C:/Users/Administrator/Desktop/pro/shuai/Plot/AUC_plot/Plot/'

sys.path.extend([current_path])

path = os.path.dirname(current_path)
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
