import Conf
from schedule_lib.class_input import Input
from schedule_lib.class_cluster import Cluster
from schedule_lib.class_scheduler import *
from reschedule_algs.PEAS import PEAS
from reschedule_algs.RIAL import RIAL
from reschedule_algs.simple_resch_alg import FFD, WFD
import heapq
import random
import numpy as np
import os
import json
import warnings
import argparse
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
huawei_file = f"{Conf.MAIN_DIR}/data/Huawei-East-1-4u8g-lt.csv"
Time = int

EXCEPTAPP = 'total'

class Simulation:
    def __init__(self, env_no, max_vm2pm_ratio, PM_number, PM_cpu, PM_mem, OC_ratio, 
                 VM_index_start, VM_index_end, resch_method, is_mask, mask_params, VM_close=True):
        self.cluster = Cluster()
        self.cluster.initialize(PM_number, PM_cpu, PM_mem, OC_ratio)
        self.input = Input(huawei_file, VM_close)
        self.input.init_vmqueue_from_huawei(VM_index_start, VM_index_end)  # 共 index_end - index_start 个 VM
        self.online_scheduler = WorstFit()
        self.is_mask = is_mask
        self.mask_params = mask_params
        self.heuristic = resch_method(self.cluster, self.is_mask, self.mask_params)
        self.PMNUM = PM_number
        
        self.info = {
            'Random_seed': Conf.RANDOM_SEED, 'VM_to_PM_ratio': max_vm2pm_ratio,
            'PM_num': PM_number, 'PM_cpu': PM_cpu, 'PM_mem': PM_mem, 'OC_ratio': OC_ratio,
            'VM_index_start': VM_index_start, 'VM_index_end': VM_index_end, 'VM_close': VM_close,
            'DetectMethod': self.heuristic.name, 'is_mask': self.is_mask
        }
        self.max_vm2pm_ratio = max_vm2pm_ratio
        self.VM_close = VM_close        
        self.t = 0 
        self.total_VM_num = 0  # 共创建了多少 VM

    def start_reschedule(self) -> bool:
        ''' 判断是否开始进行二次调度 '''
        total_running_VM_num = sum(len(vms) for vms in self.cluster.running_VM_info.values())
        if total_running_VM_num >= self.cluster.PM_number:
            return True
        else:
            return False 

    def reschedule(self) -> tuple[list[dict], list[dict], dict[str, dict[str, int|float]], list[dict], list[dict]]:
        ''' 
        实行二次调度, 并返回
        调度数据, 调度前各 PM 的干扰指标,
        调度前与后的指标 ( result[before|after][deg_QoS_sum|deg_num|cpu_utilz|mem_utilz] ),
        调度前集群状态，调度后集群状态
        '''
        # 记录二次调度前指标
        result = {}
        deg_num, deg_QoS_sum = self.cluster.get_PMs_degradation()
        cpu_utilz, mem_utilz = self.cluster.get_average_utilz()
        result['before'] = {
            'deg_QoS_sum': deg_QoS_sum,
            'deg_vm_num': deg_num, 'deg_pm_num': self.cluster.get_deg_PM_num(),
            'cpu_utilz': cpu_utilz, 'mem_utilz': mem_utilz,
        }
        # 记录调度前主机状态
        cluster_before = [
            {
                'pmid': pm.id, 'apps': [vm.app for vm in pm.running_VMs], 
                'QoSs': (vms_QoS := [vm.get_QoS() for vm in pm.running_VMs]), 
                'QoS_sum': sum(vms_QoS)
            }
            for pm in self.cluster.PMlist
        ]
        # 记录二次调度
        reschedule_data: list[dict] = []
        PM_metrics_dict_data = [{}]

        # 二次调度
        self.heuristic.reschedule()

        # 记录二次调度后指标
        deg_num, deg_QoS_sum = self.cluster.get_PMs_degradation()
        cpu_utilz, mem_utilz = self.cluster.get_average_utilz()
        result['after'] = {
            'deg_QoS_sum': deg_QoS_sum,
            'deg_vm_num': deg_num, 'deg_pm_num': self.cluster.get_deg_PM_num(),
            'cpu_utilz': cpu_utilz, 'mem_utilz': mem_utilz,
        }
        cluster_after = [
            {
                'pmid': pm.id, 'apps': [vm.app for vm in pm.running_VMs], 
                'QoSs': (vms_QoS := [vm.get_QoS() for vm in pm.running_VMs]), 
                'QoS_sum': sum(vms_QoS)
            }
            for pm in self.cluster.PMlist
        ]
        return reschedule_data, PM_metrics_dict_data, result, cluster_before, cluster_after
    

    def start(self):
        self.t = self.input.get_start_t()
        vm_index = 0  # VM 队列中的第几个 VM

        later_t: list[int] = []   # 存储下一个 VM 的开始时间，以及运行 VM 的结束时间。最小堆，每次pop必定pop最小的值
        all_VM_created = False  # 是否所有 VM 已创建

        last_resche_time = -1
        # second sche after vm placement 
        # until reach the highest VM2PMRatio
        # resche 迁移一步

        while (vm_index < self.input.VM_len):        
            # 安装 vm, (一次调度)
            vm = self.input.VMqueue[vm_index]
            available_PMs = self.cluster.get_available_PMs(vm)
            if len(available_PMs) == 0:
                raise ValueError("No available PM for new VM!")
                               
            pm = self.online_scheduler.schedule(available_PMs, vm)
            self.cluster.run_VM(pm, vm)
            self.total_VM_num += 1 
            
            # 超过极限超分比直接返回
            currentratio = sum(len(vms) for vms in self.cluster.running_VM_info.values()) / self.cluster.PM_number
            if currentratio > self.max_vm2pm_ratio or vm_index > 1000:
                print("reach max vm2pm ratio")
                return
            
            # 二次调度
            if self.start_reschedule() and self.t != last_resche_time:
                reschedule_data, PM_inter_metrics, result, clus_bef, clus_aft = self.reschedule()
                self.save_result(reschedule_data, PM_inter_metrics, result, clus_bef, clus_aft, \
                                 savebyVMindex = self.t, currentratio = currentratio)
                last_resche_time = self.t

            if vm_index % 10 == 0:
                print(f"sche {vm_index} ratio{currentratio:.2f}")  

            # 记录未来用于跳转时间（要么是新 VM 创建，要么是 VM 删除），并增加index
            heapq.heappush(later_t, self.t + vm.running_time)  # 将此 VM 结束时间存储

            vm_index += 1
            
            if vm_index == len(self.input.VMqueue):
                all_VM_created = True
            
            if not all_VM_created:
                heapq.heappush(later_t, self.input.VMqueue[vm_index].start_time)  # 将下一个 VM 开始时间存储 

            # 跳到下一个安装 VM 的时间，并删除中间结束的vm
            while True:
                if not all_VM_created:  # 如果还有 VM 未创建
                    self.t = heapq.heappop(later_t)  # 跳到下一时刻
                    if self.VM_close:
                        self.cluster.finish_VM(self.t)
                    if self.t == self.input.VMqueue[vm_index].start_time:  # 若现在是下一个 VM 的安装时间，
                        break  # 则结束循环，跳到安装 VM 的代码
                else:  # 若所有 VM 已创建
                    if self.VM_close:
                        while len(later_t) > 0:  # 且还存在未删除 VM
                            self.t = heapq.heappop(later_t) # 跳到下一时刻
                            self.cluster.finish_VM(self.t)  # 删除所有结束时间为 t 的 VM  
                    break
    
    def _save_result(self, data, file_name, log_dir):
        def flatten_item(item):
            flattened_item = {}
            for key, value in item.items():
                if isinstance(value, list) and all(isinstance(i, str) for i in value):   # 如果是list[str]，用"|"连接
                    flattened_item[key] = "|".join(value)
                elif isinstance(value, list) and all(isinstance(i, float) for i in value):
                    flattened_item[key] = ", ".join(f"{val:.2f}" for val in value)
                elif isinstance(value, np.ndarray):   # 如果是ndarray，转成字符串并格式化为小数
                    flattened_item[key] = ", ".join(f"{val:.2f}" for val in value.tolist())
                else:   # 其他类型保持不变
                    flattened_item[key] = value
            return flattened_item
    
        flattened_data = [flatten_item(item) for item in data]
        file_path = os.path.join(log_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(flattened_data, f, indent=4)

    def save_result(self, reschedule_data, pm_inter_metrics, result, cluster_before, cluster_after,savebyVMindex,currentratio = "None"):
        log_dir = f"{Conf.MAIN_DIR}/log/reschedule_dynamic/PM{self.PMNUM}/Ratio{self.max_vm2pm_ratio}/" + \
                  f"except{EXCEPTAPP}/{self.heuristic.name}/{savebyVMindex}"
        os.makedirs(log_dir, exist_ok=True)

        self._save_result(reschedule_data, "reschedule.json", log_dir)
        self._save_result(pm_inter_metrics, "pm_inter_metrics.json", log_dir) 
        self._save_result(cluster_before, "cluster_before.json", log_dir)
        self._save_result(cluster_after, "cluster_after.json", log_dir)
        
        result_path = os.path.join(log_dir, f"info_result.json")
        with open(result_path, "w") as f:
            json.dump({"info": self.info, "result": result,"current ratio": currentratio,"time":self.t}, f, indent=4)

 

def initialize_seed(seed=Conf.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def inputargs():
    ''' 解析命令行参数，如 python simulation_dynamic.py --vm_ratio 1 ... --mask_ratio 0.5 '''
    ''' 方便并行化 '''
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1'):
            return True
        elif v.lower() in ('false', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser(description="simulation case")
    parser.add_argument('--vm_ratio', type=float)
    parser.add_argument('--case', type=int, help='0-9')
    parser.add_argument('--PMnum', type=int)
    parser.add_argument('--resch', type=str, choices=['WFD', 'FFD', 'PAVMM', 'PEAS', 'RIAL'])
    parser.add_argument('--is_mask', type=str2bool)
    parser.add_argument('--is_mask_oracle', type=str2bool, default=None)
    parser.add_argument('--is_mask_dyn', type=str2bool, default=None)
    parser.add_argument('--max_dyn_mask_ratio', type=float, default=None)
    parser.add_argument('--g_alpha', type=float, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = inputargs()
    case = args.case
    pm_num = args.PMnum
    max_vm2pm_ratio = args.vm_ratio
    resch = args.resch
    is_mask = args.is_mask
    if is_mask:
        is_mask_oracle = args.is_mask_oracle
        if is_mask_oracle:
            mask_ratio = args.mask_ratio
            mask_params = {'is_oracle': is_mask_oracle, 'mask_ratio': mask_ratio}
        else:
            is_mask_dynamic = args.is_mask_dyn
            if is_mask_dynamic:
                max_dyn_ratio = args.max_dyn_mask_ratio
                g_alpha = args.g_alpha
                mask_params = {'is_oracle': is_mask_oracle, 'is_dynamic': is_mask_dynamic, \
                               'max_dyn_ratio': max_dyn_ratio, 'g_alpha': g_alpha}
            else:
                mask_ratio = args.mask_ratio
                mask_params = {'is_oracle': is_mask_oracle, 'is_dynamic': is_mask_dynamic, 'mask_ratio': mask_ratio}
    else:
        mask_params = None

    # case = 1
    # pm_num = 10
    # resch = 'RIAL'   # 'WFD', 'FFD', 'PEAS', 'RIAL'
    # is_mask = True   # True, False
    # max_vm2pm_ratio = 3
    # mask_params = {'is_oracle': False, 'is_dynamic': False, 'max_dyn_ratio': 0.51, 'mask_ratio': 0.5}
    # # mask_params = {'is_oracle': True, 'mask_ratio': 0.5}
    
    resch_dict = {'WFD': WFD, 'FFD': FFD, 'PEAS': PEAS, 'RIAL': RIAL}  # 二此调度方案
    resch_method = resch_dict[resch]

    max_vm2pm_ratio += case / 1000
    Simulation_Dict = {
        'max_vm2pm_ratio': max_vm2pm_ratio, 'PM_number': pm_num, 'PM_cpu': 4, 'PM_mem': 80, 'OC_ratio': 5,
        'VM_index_start': case * 6000, 'VM_index_end': case * 6000 + 30*pm_num,
        'resch_method': resch_method, 'is_mask': is_mask, 'mask_params': mask_params,
    }
    initialize_seed()
    simulation = Simulation(0, **Simulation_Dict)
    simulation.start()
