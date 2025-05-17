import Conf
from schedule_lib.class_vm import VM
from schedule_lib.class_pm import PM
from schedule_lib.class_cluster import Cluster
from datapre.pred2sche import tracepred
from reschedule_algs.tracepred_oracle import tracepred_oracle
from sklearn.compose import ColumnTransformer
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm

with open(f'{Conf.MODEL_SAVE_PATH}/preprocessor.pkl', 'rb') as f:
    preprocessor: ColumnTransformer = pickle.load(f)
    
def scale_interfer_metrics(df: pd.DataFrame) -> np.ndarray:
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


class RIAL:
    def __init__(self, cluster: Cluster, is_mask: bool, mask_params: dict):
        self.cluster = cluster
        self.is_mask = is_mask
        if self.is_mask:
            self.is_mask_oracle = mask_params['is_oracle']
            if self.is_mask_oracle:
                self.mask_ratio = mask_params['mask_ratio']
                self.name = f'RIAL_mask_oracle_{self.mask_ratio}'    
            else:
                self.is_mask_dynamic = mask_params['is_dynamic']
                if self.is_mask_dynamic:
                    self.max_dyn_mask_ratio = mask_params['max_dyn_ratio']
                    self.g_alpha = mask_params['g_alpha']
                    self.name = f'RIAL_dyn_mask_{self.max_dyn_mask_ratio}_{self.g_alpha}'
                else:
                    self.mask_ratio = mask_params['mask_ratio']
                    self.name = f'RIAL_mask_{self.mask_ratio}'
        else:
            self.name = 'RIAL'

    def get_pm_mask_by_tracepred(self, vm: VM, pm_apps_list: list) -> np.ndarray:
        if self.is_mask_oracle:
            pm_mask = tracepred_oracle([vm.app], pm_apps_list, self.mask_ratio)
        else:
            if self.is_mask_dynamic:
                pm_mask = tracepred([vm.app], [vm.current_colocate + [vm.app]], pm_apps_list, \
                                    dynamic=True, dynamicmaxratio=self.max_dyn_mask_ratio, g_alpha=self.g_alpha)
            else:
                pm_mask = tracepred([vm.app], [vm.current_colocate + [vm.app]], pm_apps_list, \
                                    dynamic=False, maskratio=self.mask_ratio)
        return pm_mask

    def is_pm_overload(self, pm: PM) -> tuple[bool, Optional[VM]]:
        ''' 第一个 bool 表示是否 overload, 第二个表示待迁移 vm '''
        if len(pm.running_VMs) == 0:
            return False, None
        
        vm_inter_metrics = {
            vm: scale_interfer_metrics(pd.DataFrame([vm.current_metrics]))
            for vm in pm.running_VMs
        }
        overload_threshold = [0.5, 0.4, 1, 0.5] # 每个维度对应的阈值
        # 将所有 VM 的指标按行堆叠，得到 shape=(N, 4) 的数组（N 个 VM）
        metrics_array = np.array(list(vm_inter_metrics.values()))  # shape (N, 4)

        overload_flags = []
        ideal_metrics = np.zeros(4)
        for d in range(4):
            total_metric = np.sum(metrics_array[:, d])
            is_overload = (total_metric > overload_threshold[d])
            overload_flags.append(is_overload)
            if is_overload:
                ideal_metrics[d] = np.max(metrics_array[:, d])
            else:
                ideal_metrics[d] = np.min(metrics_array[:, d])

        # 如果没有任何维度 overload，则不需要迁移
        if not any(overload_flags):
            return False, None

        # 对于每个虚机计算与 ideal_metrics 的距离
        vm_distances = {}
        for vm, metrics in vm_inter_metrics.items():
            diff = metrics - ideal_metrics
            # 计算欧氏距离：先对每个维度求平方，再求和，最后开根号
            distance = np.sqrt(np.sum(diff ** 2))
            vm_distances[vm] = distance

        # 选择距离最小的虚机
        selected_vm = min(vm_distances, key=vm_distances.get)
        return True, selected_vm
    
    def choose_migrate_pm(self, underload_pms: list[PM]) -> PM:
        def get_pm_inter_metrics(pm: PM) -> np.ndarray:
            if len(pm.running_VMs) == 0:
                return np.array([0, 0, 0, 0])
            vm_inter_metrics = [scale_interfer_metrics(pd.DataFrame([vm.current_metrics])) for vm in pm.running_VMs]
            return sum(vm_inter_metrics)
            
        pm_inter_metrics = {pm: get_pm_inter_metrics(pm) for pm in underload_pms}
        metrics_array = np.array(list(pm_inter_metrics.values()))  # shape (N, 4)
        ideal_metrics = np.min(metrics_array, axis=0)

        # 对于每个虚机计算与 ideal_metrics 的距离
        pm_distances = {}
        for pm, metrics in pm_inter_metrics.items():
            diff = metrics - ideal_metrics
            distance = np.sqrt(np.sum(diff ** 2))
            pm_distances[pm] = distance

        selected_pm = min(pm_distances, key=pm_distances.get)
        return selected_pm
        
    def reschedule(self) -> None:
        # 分开劣化与未劣化主机
        overloaded_pms_with_migrate_vm: dict[PM, VM] = {}
        underload_pms: list[PM] = []
        for pm in self.cluster.PMlist:
            is_overload, selected_vm = self.is_pm_overload(pm)
            if is_overload:
                overloaded_pms_with_migrate_vm[pm] = selected_vm
            else:
                underload_pms.append(pm)
        
        # 迁移虚机
        # for host_pm, vm in tqdm(overloaded_pms_with_migrate_vm.items()):
        for host_pm, vm in overloaded_pms_with_migrate_vm.items():
            if len(underload_pms) == 0:
                break
            
            # 选择候选待迁移主机
            if self.is_mask:
                available_pms = [pm for pm in underload_pms if len(pm.running_VMs) <= 2]
                if len(available_pms) > 0:
                    pm_apps_list = []
                    for pm in available_pms:
                        pm_apps_list.append([vm.app for vm in pm.running_VMs])
                    pm_mask = self.get_pm_mask_by_tracepred(vm, pm_apps_list)
                    pm_mask = pm_mask.flatten().astype(bool)
                    available_pms = [pm for pm, mask in zip(available_pms, pm_mask) if mask]
                else:
                    available_pms = underload_pms
            else:
                available_pms = underload_pms
                
            available_pms = [pm for pm in available_pms if not pm.check_capacity_exceed(vm)]
            if len(available_pms) == 0:
                continue
                
            dest_pm = self.choose_migrate_pm(available_pms)
            host_pm.complete_run_VM(vm)
            dest_pm.run_VM(vm)
           
            is_overload, _ = self.is_pm_overload(dest_pm)
            if is_overload:
                underload_pms.remove(dest_pm)      
    