import Conf
from schedule_lib.class_vm import VM
from schedule_lib.class_pm import PM
from schedule_lib.class_cluster import Cluster
from datapre.pred2sche import tracepred
from reschedule_algs.tracepred_oracle import tracepred_oracle
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import pickle
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


class FFD:
    ''' 虚机按照四维干扰指标的最大值进行排序，然后从大到小依次选择主机放置。主机放满才到下一个主机 '''
    def __init__(self, cluster: Cluster, is_mask: bool, mask_params: dict):
        self.cluster = cluster
        self.is_mask = is_mask
        if self.is_mask:
            self.is_mask_oracle = mask_params['is_oracle']
            if self.is_mask_oracle:
                self.mask_ratio = mask_params['mask_ratio']
                self.name = f'FFD_mask_oracle_{self.mask_ratio}'    
            else:
                self.is_mask_dynamic = mask_params['is_dynamic']
                if self.is_mask_dynamic:
                    self.max_dyn_mask_ratio = mask_params['max_dyn_ratio']
                    self.g_alpha = mask_params['g_alpha']
                    self.name = f'FFD_dyn_mask_{self.max_dyn_mask_ratio}_{self.g_alpha}'
                else:
                    self.mask_ratio = mask_params['mask_ratio']
                    self.name = f'FFD_mask_{self.mask_ratio}'
        else:
            self.name = 'FFD' 

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

    def get_available_pms(self, vm: VM, pms: list[PM], allocation: dict[PM, list[VM]]) -> list[PM]:
        potential_pms = [pm for pm in pms if len(allocation[pm]) < 3]
        vm_apps_in_pms = [[vm.app for vm in allocation[pm]] for pm in potential_pms]
        mask_array = self.get_pm_mask_by_tracepred(vm, vm_apps_in_pms)
        indices = np.where(mask_array[0] == 1)[0]
        selected_pms = [potential_pms[i] for i in indices]
        return selected_pms
    
    def _run(self, vms: list[VM], pms: list[PM]) -> dict[PM, list[VM]]:
        # 计算每个VM的干扰指标
        vm_inter = {
            vm: np.max(scale_interfer_metrics(pd.DataFrame([vm.current_metrics])))
            for vm in vms
        }
        # 按干扰指标从小到大排序，干扰越小的在前
        vm_sorted = sorted(vms, key=lambda vm: vm_inter[vm])
        allocation = {pm: [] for pm in pms}

        # for vm in tqdm(reversed(vm_sorted), total=len(vm_sorted)):
        for vm in reversed(vm_sorted):
            if self.is_mask:
                available_pms = self.get_available_pms(vm, pms, allocation)
            else:
                available_pms = [pm for pm in pms if len(allocation[pm]) < 3]
                
            if available_pms:
                allocation[available_pms[0]].append(vm)
            else:
                raise Exception(f"No suitable PM!")
            
        return allocation
    
    def reschedule(self):
        vms = [vm for vm_set in self.cluster.running_VM_info.values() for vm in vm_set]
        pms = self.cluster.PMlist
        reallocation = self._run(vms, pms)
        try:
            self.cluster.reallocate(reallocation)
        except Exception as e:
            print('allocation:')
            for pm in pms:
                print(f'pm {pm.id}: {[vm.app for vm in reallocation[pm]]}')
            exit()


class WFD:
    ''' 虚机按照四维干扰指标的最大值进行排序。依次选择干扰指数和最小的主机放置。 '''
    def __init__(self, cluster: Cluster, is_mask: bool, mask_params: dict):
        self.cluster = cluster
        self.is_mask = is_mask
        if self.is_mask:
            self.is_mask_oracle = mask_params['is_oracle']
            if self.is_mask_oracle:
                self.mask_ratio = mask_params['mask_ratio']
                self.name = f'WFD_mask_oracle_{self.mask_ratio}'    
            else:
                self.is_mask_dynamic = mask_params['is_dynamic']
                if self.is_mask_dynamic:
                    self.max_dyn_mask_ratio = mask_params['max_dyn_ratio']
                    self.name = f'WFD_dyn_mask_{self.max_dyn_mask_ratio}'
                else:
                    self.mask_ratio = mask_params['mask_ratio']
                    self.name = f'WFD_mask_{self.mask_ratio}'
        else:
            self.name = 'WFD'

    def get_pm_mask_by_tracepred(self, vm: VM, pm_apps_list: list) -> np.ndarray:
        if self.is_mask_oracle:
            pm_mask = tracepred_oracle([vm.app], pm_apps_list, self.mask_ratio)
        else:
            if self.is_mask_dynamic:
                pm_mask = tracepred([vm.app], [vm.current_colocate + [vm.app]], pm_apps_list, \
                                    dynamic=True, dynamicmaxratio=self.max_dyn_mask_ratio)
            else:
                pm_mask = tracepred([vm.app], [vm.current_colocate + [vm.app]], pm_apps_list, \
                                    dynamic=False, maskratio=self.mask_ratio)
        return pm_mask
    
    def get_inter_sum(self, pm: PM, allocation: dict[PM, list[VM]], vm_inter: dict[VM, float]):
        return sum(vm_inter[vm] for vm in allocation[pm])

    def get_available_pms(self, vm: VM, pms: list[PM], allocation: dict[PM, list[VM]]) -> list[PM]:
        potential_pms = [pm for pm in pms if len(allocation[pm]) < 3]
        vm_apps_in_pms = [[vm.app for vm in allocation[pm]] for pm in potential_pms]
        mask_array = self.get_pm_mask_by_tracepred(vm, vm_apps_in_pms)
        indices = np.where(mask_array[0] == 1)[0]
        selected_pms = [potential_pms[i] for i in indices]
        return selected_pms
    
    def _run(self, vms: list[VM], pms: list[PM]) -> dict[PM, list[VM]]:
        vm_inter = {vm: np.max(scale_interfer_metrics(pd.DataFrame([vm.current_metrics]))) 
                    for vm in vms}
        vm_sorted = sorted(vms, key = lambda vm: vm_inter[vm])
        allocation = {pm: [] for pm in pms}

        #for vm in tqdm(reversed(vm_sorted), total=len(vm_sorted)):
        for vm in reversed(vm_sorted):
            if self.is_mask:
                available_pms = self.get_available_pms(vm, pms, allocation)
            else:
                available_pms = [pm for pm in pms if len(allocation[pm]) < 3]
                
            sorted_pms = sorted(available_pms, key = lambda pm: self.get_inter_sum(pm, allocation, vm_inter))   
            target_pm = sorted_pms[0]
            allocation[target_pm].append(vm)
        
        return allocation
    
    def reschedule(self):
        vms = [vm for vm_set in self.cluster.running_VM_info.values() for vm in vm_set]
        pms = self.cluster.PMlist
        reallocation = self._run(vms, pms)
        self.cluster.reallocate(reallocation)    
