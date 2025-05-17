from schedule_lib.class_vm import VM
from schedule_lib.class_pm import PM
from schedule_lib.class_cluster import Cluster
import random
import copy
Flavor = tuple[int, int]
# 默认schedule时全程只使用一种scheduler

class OnScheduler:
    def schedule(self, available_PMs: list[PM], vm: VM) -> PM:  # 选择 PM
        # available_PMs 一定是 cluster 里能运行 VM 的所有 PM，无论 pm.type
        return
    
    def reallocate(self, allocation: dict[int, dict[str, dict | int]]) -> dict[int, dict[str, dict | int]]:
        ''' allocation[PMid] = {vm_dict: {VMid: {app: app, cpu: cpu, mem: mem}}, pm_cpu: PM_CPU, pm_mem: PM_MEM} '''
        pass

    def PM_close_interrupt(self, pm: PM) -> None:
        return
    
    def VM_close_interrupt(self, vm: VM, pm: PM) -> None:
        return


class FirstFit(OnScheduler):
    def __init__(self):
        self.name = 'FirstFit'
        self.consider_pred_err = False

    def schedule(self, available_PMs: list[PM], vm: VM) -> PM:
        return available_PMs[0]


class BestFit(OnScheduler):  # 剩余资源最小，使用资源最大的 PM
    def __init__(self):
        self.name = 'BestFit'
        self.consider_pred_err = False

    # def get_remain_capacity(self, pm: PM) -> int:
    #     return (5 * pm.CPUCapacity) + pm.MEMCapacity - (5 * pm.CPUCapacityUsage) - pm.MEMCapacityUsage

    def get_remain_capacity(self, pm: PM) -> int:
        return (5 * pm.CPUCapacity) + pm.MEMCapacity - (pm.CPUCapacityUsage) - pm.MEMCapacityUsage
    
    def schedule(self, available_PMs: list[PM], vm: VM) -> PM:
        best_current_pm = available_PMs[0]
        for pm in available_PMs:
            if self.get_remain_capacity(pm) < self.get_remain_capacity(best_current_pm):
                best_current_pm = pm
        return best_current_pm


class WorstFit(OnScheduler):  # 剩余资源最大，使用资源最小的 PM
    def __init__(self):
        self.name = 'WorstFit'
        self.consider_pred_err = False

    # def get_remain_capacity(self, pm: PM) -> int:
    #     return (5 * pm.CPUCapacity) + pm.MEMCapacity - (5 * pm.CPUCapacityUsage) - pm.MEMCapacityUsage

    def get_remain_capacity(self, pm: PM) -> int:
        return (5 * pm.CPUCapacity) + pm.MEMCapacity - (pm.CPUCapacityUsage) - pm.MEMCapacityUsage
    
    def schedule(self, available_PMs: list[PM], vm: VM) -> PM:
        best_current_pm = available_PMs[0]
        for pm in available_PMs:
            if self.get_remain_capacity(pm) > self.get_remain_capacity(best_current_pm):
                best_current_pm = pm
        return best_current_pm


class RandomFit(OnScheduler): 
    def __init__(self) -> None:
        self.name = 'RandomFit'
        self.consider_pred_err = False

    def schedule(self, available_PMs: list[PM], vm: VM) -> PM:
        return random.choice(available_PMs)


class Move2Front(OnScheduler):
    def __init__(self, cluster: Cluster):
        self.name = 'Move2Front'
        self.consider_pred_err = False
        self.cluster = cluster
        self.ordered_PMs: list[PM] = copy.deepcopy(cluster.PMlist)
    
    def schedule(self, available_PMs: list[PM], vm: VM) -> PM:
        for i, pm in enumerate(self.ordered_PMs):
            if pm in available_PMs:
                del self.ordered_PMs[i]
                self.ordered_PMs.append(pm)
                return pm



# 二次调度
class AliothSch():
    def __init__(self, cluster: Cluster):
        self.cluster = cluster
    
    def reschedule(self) -> None:
        pass
