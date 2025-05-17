# VM 队列实现
import schedule_lib._conf as sch_conf
from schedule_lib.class_vm import VM
import random
import pandas as pd
App = sch_conf.App

def rand_app() -> App:
    return random.choice(sch_conf.APPS)

class Input:  # 包含 vmqueue 以及 user_list
    def __init__(self, data_file: str, VM_close=True):
        self.VMqueue: list[VM] = None  # 要求按照时间排序
        self.VM_len = None
        self.VM_close = VM_close  # 虚机是否会关机。若 VM_close=False, 则每个VM的运行时间将长过最后VM的开启时间
        self.data_file = data_file

    def sample_or_keep(self,group,maxVMArriveSameTime = 5):
        if len(group) > maxVMArriveSameTime:
            return group.sample(5)
        else:
            return group

    def init_vmqueue_from_huawei(self, start=0, end=13660):  
        # 从 Huawei-East 数据集初始化 VM 队列，选择 [start, end) 部分的数据
        n_rows = end - start
        df = pd.read_csv(self.data_file, skiprows=start+1, nrows=n_rows, header=None)

        header = pd.read_csv(self.data_file, nrows=0).columns.tolist()
        df.columns = header

        print(df.shape)
        print(df.columns)
        
        # 对 DataFrame 按 'at' 列进行分组，并应用函数
        result_df = df.groupby('at').apply(self.sample_or_keep).reset_index(drop=True)
        print(result_df.shape)
        # sys.exit(0)

        if self.VM_close:
            self.VMqueue = [VM(s['vmid'], s['cpu'], s['mem'], rand_app(), s['at'], s['lt']) for _, s in df.iterrows()]
        else:
            self.VMqueue = [VM(s['vmid'], s['cpu'], s['mem'], rand_app(), s['at'], 3_000_000) for _, s in df.iterrows()]
        self.VM_len = len(self.VMqueue)

    def get_start_t(self):  # 返回第一个 VM 的开始时间，相当于调度的初始时间
        return self.VMqueue[0].start_time
