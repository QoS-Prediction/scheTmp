import Conf
from schedule_lib.class_vm import VM
from schedule_lib.class_pm import PM
from schedule_lib.class_cluster import Cluster
from datapre.pred2sche import tracepred
from reschedule_algs.tracepred_oracle import tracepred_oracle
from collections import deque
from scipy.stats import iqr
import numpy as np
import random
from tqdm import tqdm

class PEAS:
    def __init__(self, cluster: Cluster, is_mask: bool, mask_params: dict):
        self.cluster = cluster
        self.is_mask = is_mask
        if self.is_mask:
            self.is_mask_oracle = mask_params['is_oracle']
            if self.is_mask_oracle:
                self.mask_ratio = mask_params['mask_ratio']
                self.name = f'PEAS_mask_oracle_{self.mask_ratio}'    
            else:
                self.is_mask_dynamic = mask_params['is_dynamic']
                if self.is_mask_dynamic:
                    self.max_dyn_mask_ratio = mask_params['max_dyn_ratio']
                    self.g_alpha = mask_params['g_alpha']
                    self.name = f'PEAS_dyn_mask_{self.max_dyn_mask_ratio}_{self.g_alpha}'
                else:
                    self.mask_ratio = mask_params['mask_ratio']
                    self.name = f'PEAS_mask_{self.mask_ratio}'
        else:
            self.name = 'PEAS'
        self.UP_THR = 0.9
        self.pms_utilz_record = {pm: deque(maxlen=3) for pm in self.cluster.PMlist}
    
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
    
    def PEAP(self, vm: VM): 
        # 按照PM利用率排
        available_PMs = []
        if not self.is_mask:
            available_PMs = [pm for pm in self.cluster.PMlist if len(pm.running_VMs) < 5]
        else:
            potential_PMs = [pm for pm in self.cluster.PMlist if len(pm.running_VMs) < 3]
            if len(potential_PMs) == 0:
                raise ValueError('all PM has more or equal than 3 running VMs!')
            
            pm_apps_list = []
            for pm in potential_PMs:
                pm_apps_list.append([vm.app for vm in pm.running_VMs])
            pm_mask = self.get_pm_mask_by_tracepred(vm, pm_apps_list)
            pm_mask = pm_mask.flatten().astype(bool)
            available_PMs = [pm for pm, mask in zip(potential_PMs, pm_mask) if mask]

        sorted_pms = sorted(available_PMs, key = lambda pm: self.get_pm_cpu_utilz(pm))
        target_pm = sorted_pms[0]
        target_pm.run_VM(vm)
            
    def get_pm_cpu_utilz(self, pm: PM) -> float:
        return sum(vm.current_metrics['lvirt_domain_info_cpu_usage'] for vm in pm.running_VMs) / 100
         
    def load_mov_avg(self, pm: PM) -> float:
        load_deque = self.pms_utilz_record[pm]
        if len(load_deque) == 3:
            return load_deque[-1] * 0.67 + load_deque[-2] * 0.24 + load_deque[-3] * 0.09
        elif len(load_deque) == 2:
            return load_deque[-1] * 0.73 + load_deque[-2] * 0.27
        elif len(load_deque) == 1:
            return load_deque[-1]
        else:
            raise ValueError(f'load_deque of PM{pm.id} has length {len(load_deque)}, incorrect!')
    
    def PEACR(self) -> None:
        pms = self.cluster.PMlist
        pending_vms: list[VM] = []
        # LW_THR = self.UP_THR - iqr([self.get_pm_cpu_utilz(pm) for pm in pms])
        
        for pm in pms:
            rctload = self.load_mov_avg(pm)
            
            if rctload > self.UP_THR:
                while True:
                    pending_vm = random.choice(list(pm.running_VMs))
                    pending_vms.append(pending_vm)
                    pm.complete_run_VM(pending_vm)
                    if self.get_pm_cpu_utilz(pm) < self.UP_THR:
                        break
        
            # if rctload < LW_THR:
            #     shutdown_vms = pm.shut_down()
            #     pending_vms.extend(shutdown_vms)
        
        # for vm in tqdm(pending_vms):
        for vm in pending_vms:
            self.PEAP(vm)

    def renew_pm_utilz_record(self) -> None:
        for pm in self.cluster.PMlist:
            self.pms_utilz_record[pm].append(
                sum(vm.current_metrics['lvirt_domain_info_cpu_usage'] for vm in pm.running_VMs) / 100
            )
    
    def reschedule(self) -> None:
        self.renew_pm_utilz_record()
        self.PEACR()
