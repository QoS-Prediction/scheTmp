import os
from typing import Literal


MAIN_DIR = "/data2/IWQOS/VMpred4mig"
DATA_DIR = os.path.join(MAIN_DIR,"data")
raw_data_save_path = os.path.join(DATA_DIR,"rawdata.csv")
raw_data_save_path2 = os.path.join(DATA_DIR,"rawdata2.csv")
PROCESS_DATA_DIR = os.path.join(DATA_DIR,"dataprocessed")
PROCESS_DATA_DIR2 = os.path.join(DATA_DIR,"dataprocessed2")
PROCESS_DATA_DIR3 = os.path.join(DATA_DIR,"dataprocessed3")
PROCESS_DATA_DIR4 = os.path.join(DATA_DIR,"dataprocessed4")
APPLIST = ['memcache','redis','mysql','keydb','nginx']
APPS = ["memcache", "mysql", "keydb", "redis", "nginx"]
App = Literal["memcache", "mysql", "keydb", "redis", "nginx"]

MODEL_SAVE_PATH = '/data2/Alioth/performance_estimation/model/Alioth'
MODEL_PATH = os.path.join(MAIN_DIR,"baselines/model/Tracealpha_transdcn.pt")

# PM指标/agg指标 维度
comb_dim_begin = 31
comb_dim_end = 154
# 31 vm 124agg
vm_dim_num = 31
agg_dim_num = 124
x_dim_num = vm_dim_num + agg_dim_num
deepset_dim_num = 31

MAXRATIO = 3
maskratio = 0.5

RANDOM_SEED = 2024