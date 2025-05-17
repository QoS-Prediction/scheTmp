import Conf
from schedule_lib.class_input import Input
from schedule_lib.class_cluster import Cluster
from schedule_lib.class_scheduler import *
from detect2sche import QoSDetect_Alioth
from detect2sche import QoSDetect_Monitorless
from detect2sche import QoSDetect_Practical
from detect2sche import QoSDetect_XGB
from detect2sche import QoSDetect_CART
from sklearn.compose import ColumnTransformer
import heapq
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import json
import pickle
from datetime import datetime
from itertools import product
import concurrent.futures
import traceback
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
huawei_file = f"{Conf.MAIN_DIR}/data/Huawei-East-1-4u8g-lt.csv"
Time = int

DetectMethod = ["Alioth","Monitorless","Practical","XGB","CART","4DIMS","onlineWithoutoffline","new method"]
DetectMethodID = 4
VM2PMRatio = 1.5
EXCEPTAPP = 'total'
rescheMethodID = 0


random.seed(Conf.RANDOM_SEED)
np.random.seed(Conf.RANDOM_SEED)

with open(f'{Conf.MODEL_SAVE_PATH}/preprocessor.pkl', 'rb') as f:
    preprocessor: ColumnTransformer = pickle.load(f)

@dataclass(order=True)  # 可比较
class deg_order:
    deg_num: int
    deg_score: float  # 1 - QoS
    
    def to_tuple(self) -> tuple[int, float]:
        return (self.deg_num, self.deg_score)

    def __add__(self, other):
        if not isinstance(other, deg_order):  raise TypeError
        return deg_order(self.deg_num + other.deg_num, self.deg_score + other.deg_score)

    def __sub__(self, other):
        if not isinstance(other, deg_order):  raise TypeError
        return deg_order(self.deg_num - other.deg_num, self.deg_score - other.deg_score)
    
    def __str__(self):
        return f"DEG({self.deg_num}, {self.deg_score:.3f})"


class Simulation:
    def __init__(self, env_no, VM_to_PM_ratio, PM_number, PM_cpu, PM_mem, OC_ratio, 
                 VM_index_start, VM_index_end, VM_close=True):
        self.cluster = Cluster()
        self.cluster.initialize(PM_number, PM_cpu, PM_mem, OC_ratio)
        self.input = Input(huawei_file, VM_close)
        self.input.init_vmqueue_from_huawei(VM_index_start, VM_index_end)  # 共 index_end - index_start 个 VM
        self.online_scheduler = WorstFit()
        
        self.info = {
            'Random_seed': Conf.RANDOM_SEED, 'VM_to_PM_ratio': VM_to_PM_ratio,
            'PM_num': PM_number, 'PM_cpu': PM_cpu, 'PM_mem': PM_mem, 'OC_ratio': OC_ratio,
            'VM_index_start': VM_index_start, 'VM_index_end': VM_index_end, 'VM_close': VM_close,
            'DetectMethod': DetectMethod[DetectMethodID],
        }
        self.VM_to_PM_ratio = VM_to_PM_ratio
        self.VM_close = VM_close        
        self.t = 0 
        self.total_VM_num = 0  # 共创建了多少 VM
    
    def scale_interfer_metrics(self, df: pd.DataFrame) -> np.ndarray:
        ''' 使用 preprocessor 归一化数据, 并输出四维的干扰指标. 主机的话输出的是干扰指标对虚机之和 '''
        total_cols = preprocessor.feature_names_in_.tolist()
        missing_cols = [col for col in total_cols if col not in df.columns]
        missing_data = pd.DataFrame({col: [0.0] * len(df) for col in missing_cols})
        df = pd.concat([df, missing_data], axis=1)
        df = df[total_cols]
        
        data = preprocessor.transform(df)
        
        a = np.sum(data[:,2])                      # perf_cache_misses
        b = np.sum((data[:,12] + data[:,15]) / 2)  # lvirt_domain_block_stats_read/write_bytes_total
        c = np.sum(data[:,20])                     # lvirt_domain_info_memory_usage_bytes
        d = np.sum((data[:,23] + data[:,26]) / 2)  # lvirt_domain_interface_stats_receive/transmit_bytes_total
        return np.array([a, b, c, d])

    def start_reschedule(self) -> bool:
        total_running_VM_num = sum(len(vms) for vms in self.cluster.running_VM_info.values())
        if total_running_VM_num >= self.VM_to_PM_ratio * self.cluster.PM_number:
            return True
        else:
            return False 
    
    # 二次调度与一次调度自适应，如果还有空的主机由于worstfit，认为一次调度会直接放，无需二次调度
    def start_reschedule2(self) -> bool:
        for pm in self.cluster.PMlist:
            if len(pm.running_VMs) == 0:
                return False 
        return True

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

        '''
        二次调度
        监测每个主机，选择主机 QoS 最低的，选择 QoS VM 最低的，然后迁移。
        选择迁移主机，一是选择 QoS 最好的，二是选择 interfer metrics 错峰的。
        '''
        # 1. 使用 QoS 检测得到每个 PM 的 QoS 总和以及各 VM 的 QoS
        pm_QoS: dict[PM, dict[str, dict[int, float]]] = {}  # pm_QoS[pm][nginx][i] 是 pm 里第 i 个 app=nginx 的 vm 的 QoS 检测值
        pm_QoS_sum: dict[PM, deg_order] = {}  # 记录调度前各 PM 的 QoS 和
        for pm in self.cluster.PMlist:
            if DetectMethod[DetectMethodID] == "Alioth":
                pm_QoS[pm] = QoSDetect_Alioth([vm.app for vm in pm.running_VMs], pm.colocate_ind,EXCEPTAPP)[0]
            elif DetectMethod[DetectMethodID] == "Monitorless":
                pm_QoS[pm] = QoSDetect_Monitorless([vm.app for vm in pm.running_VMs], pm.colocate_ind,EXCEPTAPP)[0]
            elif DetectMethod[DetectMethodID] == "Practical":
                pm_QoS[pm] = QoSDetect_Practical([vm.app for vm in pm.running_VMs], pm.colocate_ind,EXCEPTAPP)[0]
            elif DetectMethod[DetectMethodID] == "XGB":
                pm_QoS[pm] = QoSDetect_XGB([vm.app for vm in pm.running_VMs], pm.colocate_ind,EXCEPTAPP)[0]
            elif DetectMethod[DetectMethodID] == "CART":
                pm_QoS[pm] = QoSDetect_CART([vm.app for vm in pm.running_VMs], pm.colocate_ind,EXCEPTAPP)[0]
            else:
                pm_QoS[pm] = QoSDetect_Alioth([vm.app for vm in pm.running_VMs], pm.colocate_ind,EXCEPTAPP)[0]

            deg_num = sum(val < Conf.QOS_THRESHOLD for val_dict in pm_QoS[pm].values() for val in val_dict.values())
            deg_score_sum = sum(1 - val for val_dict in pm_QoS[pm].values() for val in val_dict.values())
            pm_QoS_sum[pm] = deg_order(deg_num, deg_score_sum)
        pm_QoS_sum = dict(sorted(pm_QoS_sum.items(), key=lambda x: x[1], reverse=True))  # deg 越大越严重, 排越前面

        # 2. 预先计算所有 PM 的干扰指数
        PM_metrics_dict: dict[PM, np.ndarray] = {}
        for pm in pm_QoS_sum: 
            pm_metrics = pd.DataFrame.from_dict(pm.get_metrics(), orient='index')
            pm_metrics.reset_index(drop=True, inplace=True)
            PM_metrics_dict[pm] = self.scale_interfer_metrics(pm_metrics)
        PM_metrics_dict_data = [
            {
                'pmid': pm.id,
                'apps': [vm.app for vm in pm.running_VMs],
                'metrics': val
            }
            for pm, val in PM_metrics_dict.items()
        ]
        

        # 3. 针对劣化最严重的 20% 主机进行二次调度 ！！！！modify！！！！！
        if DetectMethodID <5:
            cnt = 0
            for pm in pm_QoS_sum.keys():
                # ！！！！！！！modify！！！！！！！！！
                if cnt > int(0.05 * len(pm_QoS_sum)):  break  # 只处理前 5% QoS 最低的 PM
                # if cnt > 1: break

                cnt += 1
                re_data = {}  # 存储调度数据
                
                # 3.1 找到主机里劣化最严重的 app 对应的 vm
                app_min_qos_dict: dict[str, float] = {}
                for app, apps_qos in pm_QoS[pm].items():
                    app_min_qos_dict[app] = min(qos for qos in apps_qos)
                min_qos_app = min(app_min_qos_dict, key=app_min_qos_dict.get)
                vm = [vm for vm in pm.running_VMs if vm.app == min_qos_app][0]
                re_data['vm_app'] = vm.app
                
                # 3.2 迁移出去
                re_data['deg_pm'] = sorted([vm.app for vm in pm.running_VMs])
                re_data['deg_pm_id'] = pm.id
                pm.complete_run_VM(vm)
                
                # 3.3 选择迁移目标（主机）
                
                # A. 选择劣化最轻微的主机
                # target_pm = pm_QoS_sum[-1][0] 
                # pm_QoS_sum.pop()  # 被迁移的主机后续就不考虑了
                
                # B. 选择 对应干扰维度最小的 PM (对应干扰维度为 VM 的最大值)
                VM_metrics = pd.DataFrame([vm.current_metrics])
                scaled_VM_metrics = self.scale_interfer_metrics(VM_metrics)
                max_inter_ind = np.argmax(scaled_VM_metrics)
                re_data['vm_inter'] = scaled_VM_metrics

                filtered_dict = {pm: PM_metrics_dict[pm] for pm in PM_metrics_dict if pm_QoS_sum[pm].deg_num == 0}
                if len(filtered_dict)<=1:
                    target_pm = min(set(PM_metrics_dict.keys()) & set(self.cluster.get_available_PMs(vm)), \
                                    key = lambda pm: PM_metrics_dict[pm][max_inter_ind])
                else:
                    target_pm = min(set(filtered_dict.keys()) & set(self.cluster.get_available_PMs(vm)), \
                                    key = lambda pm: filtered_dict[pm][max_inter_ind])
                
                # 3.4 迁移去新主机
                re_data['dest_pm_inter'] = PM_metrics_dict[target_pm]
                re_data['dest_pm_id'] = target_pm.id
                re_data['dest_pm'] = sorted([vm.app for vm in target_pm.running_VMs])
                target_pm.run_VM(vm)
                del PM_metrics_dict[target_pm]  # 迁移后后续不考虑此主机，因为干扰指标已更新
                reschedule_data.append(re_data)


        # 3*. 根据四个维度指标及阈值判断迁移虚机
        else:

            # 3.1  对每个维度根据阈值筛选待迁移虚机
            metric_threshold = [0.8,0.6,1,0.75]
            filtered_pms: dict[PM, int] = {}

            if not DetectMethod[DetectMethodID] == "onlineWithoutoffline":
                for pm, metrics in PM_metrics_dict.items():
                    for index in range(len(metrics)):
                        if metrics[index] > metric_threshold[index]:
                            filtered_pms[pm] = index
                            continue
                     
            
                
            # 3.2 对指标超过某个阈值的维度的主机上迁移  该“劣化”维度上占比最高的虚机,并迁移
            for pm,dimindex in filtered_pms.items():
                re_data = {}  # 存储调度数据
                # 可能被选为目标迁移主机，若被选中则不考虑了
                if pm not in PM_metrics_dict.keys():
                    continue

                migvm = max(pm.running_VMs, key=lambda vm: self.scale_interfer_metrics(pd.DataFrame([vm.current_metrics]))[dimindex])
                re_data['vm_app'] = migvm.app
            
                #  迁移出去
                re_data['deg_pm'] = sorted([vm.app for vm in pm.running_VMs])
                re_data['deg_pm_id'] = pm.id
                pm.complete_run_VM(migvm)


                # B. 选择 对应干扰维度最小的 PM (对应干扰维度为 VM 的最大值)
                max_inter_ind = dimindex
                re_data['vm_inter'] = self.scale_interfer_metrics(pd.DataFrame([migvm.current_metrics]))
                
                filtered_dict = {pm: PM_metrics_dict[pm] for pm in PM_metrics_dict if pm_QoS_sum[pm].deg_num == 0}
                if len(filtered_dict)<=1:
                    target_pm = min(set(PM_metrics_dict.keys()) & set(self.cluster.get_available_PMs(migvm)), \
                                    key = lambda pm: PM_metrics_dict[pm][max_inter_ind])
                else:
                    target_pm = min(set(filtered_dict.keys()) & set(self.cluster.get_available_PMs(migvm)), \
                                    key = lambda pm: filtered_dict[pm][max_inter_ind])

                
                # 3.4 迁移去新主机
                re_data['dest_pm_inter'] = PM_metrics_dict[target_pm]
                re_data['dest_pm_id'] = target_pm.id
                re_data['dest_pm'] = sorted([vm.app for vm in target_pm.running_VMs])
                target_pm.run_VM(migvm)
                del PM_metrics_dict[target_pm]  # 迁移后后续不考虑此主机，因为干扰指标已更新
                reschedule_data.append(re_data)



                        
        # 4. 记录二次调度后指标
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
        # untill reach the highest VM2PMRatio
        # resche 迁移一步


        while True:        
  
            # 安装 vm,(一次调度)
            vm = self.input.VMqueue[vm_index]
            available_PMs = self.cluster.get_available_PMs(vm)
            if len(available_PMs) == 0:
                raise ValueError("No available PM for new VM!")
                               
            pm = self.online_scheduler.schedule(available_PMs, vm)
            self.cluster.run_VM(pm, vm)
            self.total_VM_num += 1 


            # 二次调度
            currentratio = total_running_VM_num = sum(len(vms) for vms in self.cluster.running_VM_info.values())/self.cluster.PM_number
            if self.start_reschedule2() and self.t!=last_resche_time:
                # for i in range(0,20):
                #     reschedule_data, PM_inter_metrics, result, clus_bef, clus_aft = self.reschedule()
                #     # 将二次调度结果同步更新成为新的状态
                #     self.save_result(reschedule_data, PM_inter_metrics, result, clus_bef, clus_aft,savebyVMindex = i,currentratio= i)
                # return
                
                # 
                reschedule_data, PM_inter_metrics, result, clus_bef, clus_aft = self.reschedule()
                self.save_result(reschedule_data, PM_inter_metrics, result, clus_bef, clus_aft,savebyVMindex = self.t,currentratio= currentratio)
                last_resche_time = self.t
            # 超过极限超分比直接返回
            if currentratio>VM2PMRatio:
                print("reach max vm2om ratio")
                return

            if vm_index % 10 == 0:
                print(f"sche {vm_index} ratio{currentratio}")  

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


    def flatten_item(self, item):
        flattened_item = {}
        
        for key, value in item.items():
            if isinstance(value, list) and all(isinstance(i, str) for i in value):
                # 如果是list[str]，用"|"连接
                flattened_item[key] = "|".join(value)
            elif isinstance(value, list) and all(isinstance(i, float) for i in value):
                flattened_item[key] = ", ".join(f"{val:.2f}" for val in value)
            elif isinstance(value, np.ndarray):
                # 如果是ndarray，转成字符串并格式化为小数
                flattened_item[key] = ", ".join(f"{val:.2f}" for val in value.tolist())
            else:
                # 其他类型保持不变
                flattened_item[key] = value
        
        return flattened_item
    
    def _save_result(self, data, file_name, log_dir):
        flattened_data = [self.flatten_item(item) for item in data]
        file_path = os.path.join(log_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(flattened_data, f, indent=4)

    def save_result(self, reschedule_data, pm_inter_metrics, result, cluster_before, cluster_after,savebyVMindex,currentratio = "None"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # log_dir = f"{Conf.MAIN_DIR}/log/reschedule_dynamic_log/Ratio{VM2PMRatio}/except{EXCEPTAPP}/{DetectMethod[DetectMethodID]}/{timestamp}"
        log_dir = f"{Conf.MAIN_DIR}/log/reschedule_dynamic3_log/Ratio{VM2PMRatio}/except{EXCEPTAPP}/{DetectMethod[DetectMethodID]}/{savebyVMindex}"
        os.makedirs(log_dir, exist_ok=True)

        self._save_result(reschedule_data, "reschedule.json", log_dir)
        self._save_result(pm_inter_metrics, "pm_inter_metrics.json", log_dir) 
        self._save_result(cluster_before, "cluster_before.json", log_dir)
        self._save_result(cluster_after, "cluster_after.json", log_dir)
        
        result_path = os.path.join(log_dir, f"info_result.json")
        with open(result_path, "w") as f:
            json.dump({"info": self.info, "result": result,"current ratio": currentratio,"time":self.t}, f, indent=4)


def _simulation_main(input_dict: dict):  # 使用单输入是方便并行计算
    ''' input_dict.keys() = ['sim_env_no', 'start_ind'] '''

    if not all([(key in input_dict) for key in ('sim_env_no', 'start_ind')]):
        raise ValueError('_simulation_main函数输入不对')
    sim_env_no = input_dict['sim_env_no']
    start_ind = input_dict['start_ind']

    Simulation_Dicts = [
        {
            'VM_to_PM_ratio': VM2PMRatio, 'PM_number': 100, 'PM_cpu': 4, 'PM_mem': 80, 'OC_ratio': 5,
            'VM_index_start': start_ind, 'VM_index_end': start_ind + 1000,
        },
    ]

    simulation = Simulation(sim_env_no, **Simulation_Dicts[sim_env_no])
    simulation.start()  

    # 测试
    # simulation = Simulation(
    #     env_no=0, Sch_no=2, PM_number=400, PM_cpu=4, PM_mem=80, OC_ratio=5, 
    #     User_num=100, VM_index_start=0, VM_index_end=40000,
    # )     
    # simulation = Simulation(Sch_no=1, PM_number=20, PM_cpu=4, PM_mem=80, User_num=50, VM_index_start=0, VM_index_end=80, VM_close=False) 
    # simulation.start()   

def initialize_seed(seed=Conf.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def aggResult(Method):
    log_dir = f"{Conf.MAIN_DIR}/log/reschedule_dynamic3_log/Ratio{VM2PMRatio}/except{EXCEPTAPP}/{Method}"
    import os
    import json


    
    counter = 0
    before_agg = {'deg_QoS_sum': 0, 'deg_vm_num':0,'deg_pm_num':0,'cpu_utilz':0}
    after_agg = {'deg_QoS_sum': 0, 'deg_vm_num':0,'deg_pm_num':0,'cpu_utilz':0}
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(log_dir):
        for filename in files:
            if filename == "info_result.json":
                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    print(data["result"])
                    data = data['result']
                    counter += 1
                    before_agg['deg_QoS_sum'] += data['before']['deg_QoS_sum']
                    before_agg['deg_vm_num'] += data['before']['deg_vm_num']
                    before_agg['deg_pm_num'] += data['before']['deg_pm_num']
                    before_agg['cpu_utilz'] += data['before']['cpu_utilz']

                    after_agg['deg_QoS_sum'] += data['after']['deg_QoS_sum']
                    after_agg['deg_vm_num'] += data['after']['deg_vm_num']
                    after_agg['deg_pm_num'] += data['after']['deg_pm_num']
                    after_agg['cpu_utilz'] += data['after']['cpu_utilz']

    result_agg = {}
    result_agg['method'] = Method
    result_agg['before'] = before_agg
    result_agg['after'] = after_agg
    outputdir = f"/data2/Alioth2Journal/log/reschedule_dynamic3_log/Ratio{VM2PMRatio}/except{EXCEPTAPP}/resche_compare"
    os.makedirs(os.path.join(outputdir), exist_ok=True)
    with open(os.path.join(outputdir,f'{Method}.json'), "w") as f:
        json.dump(result_agg, f, indent=4)
    print(f"agg {counter} reseults for {Method}")
    return 0

# 1.8  1000
# 1.6  2000
# 1.7  3000

# 1.70x x000

if __name__ == "__main__":

    casenum = float(0.004)

    VM2PMRatio = 3.000+casenum
    # idxlist = [0,3,2,4,1]
    DetectMethodIDlist = [4,3,2,1]
    
    # rescheMethodIDlist = [0,1]
    EXCEPTAPPList = ["total"]#,'nginx','mysql',"memcache",'redis','keydb'
    for tmpapp in EXCEPTAPPList:
        EXCEPTAPP = tmpapp
        for idx in DetectMethodIDlist:
            DetectMethodID = idx
            # rescheMethodID = idx
            initialize_seed()
            _simulation_main( {'sim_env_no': 0, 'start_ind': int(casenum*1000)*1000})
    from plot import plotfig
    plotfig(VM2PMRatio,EXCEPTAPP)



    # VM2PMRatio = 1.6
    # DetectMethodID = 4
    # _simulation_main( {'sim_env_no': 0, 'start_ind': 1000})


    # EXCEPTAPPList = ["total","memcache",'redis','nginx','mysql','keydb']
    # while(VM2PMRatio<4):
    #     for tmpapp in EXCEPTAPPList:
    #         EXCEPTAPP = tmpapp
    #         for startidx in range(0,10):
    #             for idx in range(0,5):
    #                 print(f"Test Group{startidx} Method{idx}")
    #                 initialize_seed()
    #                 DetectMethodID = idx
    #                 _simulation_main( {'sim_env_no': 0, 'start_ind': 1000*startidx} )
    #         for idx in range(0,5):
    #             aggResult(DetectMethod[idx])
    #     VM2PMRatio += 0.2


    # aggResult(DetectMethod[0])
    
    # sim_env_range = range(2) # range(4)
    # start_ind_range = [ind for ind in list(range(0, 5001, 500)) if ind not in {2000, 5000}]
    # params_comb = [
    #     {'sim_env_no': a, 'start_ind': b} 
    #     for a, b in product(sim_env_range, start_ind_range)
    # ]
    
    # total_tasks = len(params_comb)
    # with concurrent.futures.ProcessPoolExecutor(initializer=initialize_seed, max_workers=1) as executor:
    #     futures = {executor.submit(_simulation_main, params): params for params in params_comb}
    #     completed_tasks = 0
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             params = futures[future]
    #             print(f"params {params}: exception {e}")
    #             traceback.print_exc()
    #         finally:
    #             completed_tasks += 1
    #             print(f"{completed_tasks} / {total_tasks}")

    # cProfile.run("_simulation_main( {'sim_env_no': 3, 'sch_no': 0, 'start_ind': 0})", 
    #               '/opt/data/ctpcode/online_scheduling/__other/prof/PP_FirstFit_env3.prof')
    # cProfile 会减慢一倍




# 4to1
# 迁出交给一次调度
# oracle放置


# 多个 连续 一次     
# 单个 中间 多次


# 中间 多次     *10次实验平均
# 连续 一次       1次实验

# 中间 连续多次  多次实验平均
