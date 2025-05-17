from __future__ import annotations
from typing import Iterable
from schedule_lib._get_data import get_real_metrics
from Conf import App
Time = int

# VM 无 open, close 属性. 由是否在 pm 上决定是否开关
class VM:
    def __init__(self, id, CPU, MEM, app, start_time, running_time):
        self.id = int(id)
        self.CPU = CPU
        self.MEM = MEM
        self.app = app
        self.start_time = int(start_time)
        self.running_time = int(running_time)
        self.end_time = int(start_time + running_time)
        self.current_metrics: dict[str, float] = None 
        self.current_colocate: list[App] = []  # 不包含自己本身

    def renew_metrics(self, colocate_VMs: Iterable[VM], colocate_ind: int, app_ind = 0) -> dict[str, float]:
        ''' 更新并返回本 vm 与其他 colocate_VMs 合并后的 metrics '''
        self.current_colocate = [vm.app for vm in colocate_VMs]

        metrics = get_real_metrics(self.app, [vm.app for vm in colocate_VMs], colocate_ind, app_ind)
        if 'perf_branch_misses' not in metrics:
            raise ValueError(f"get_real_metrics({self.app}, {[vm.app for vm in colocate_VMs]}) don't have metrics!")
        self.current_metrics = metrics
        return metrics
    
    def get_QoS(self) -> float:
        return self.current_metrics['QoS']
    
    def get_resource_usage(self) -> tuple[float, float]:  # cpu 利用率，mem 利用量
        cpu_utilz = self.current_metrics["lvirt_domain_vcpu_time"]
        mem_utilz = self.current_metrics["lvirt_domain_info_memory_usage_bytes"]
        return cpu_utilz, mem_utilz
    
    def get_interfer_metrics(self) -> dict[str, float]:
        result = {
            'perf_cache_misses': self.current_metrics['perf_cache_misses'],
            'lvirt_domain_block_stats_rw_bytes_total': \
               (self.current_metrics['lvirt_domain_block_stats_read_bytes_total'] + \
                self.current_metrics['lvirt_domain_block_stats_write_bytes_total']) / 2,
            'lvirt_domain_info_memory_usage_bytes': \
                self.current_metrics['lvirt_domain_info_memory_usage_bytes'],
            'lvirt_domain_interface_stats_rt_bytes_total': \
               (self.current_metrics['lvirt_domain_block_stats_read_bytes_total'] + \
                self.current_metrics['lvirt_domain_block_stats_write_bytes_total']) / 2,
        }
        return result
         
    def __str__(self):
        return f"VM {self.id} (cpu {self.CPU}, mem {self.MEM})"

    def __hash__(self):  # 使 VM 可哈希，让 VM 能够变成 set 的元素
        return hash(self.id)
    
    def __eq__(self, other):  # 使 VM 可比较，让 VM 能够变成 set 的元素
        if isinstance(other, VM):
            return (self.id == other.id)
        return False

