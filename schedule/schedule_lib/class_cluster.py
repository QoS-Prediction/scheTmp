from schedule_lib.class_vm import VM
from schedule_lib.class_pm import PM
import logging
vmid = type

class Cluster:
    def __init__(self):
        self.PMlist: list[PM] = []
        self.running_VM_info: dict[PM, set[VM]] = {}  # running_VM_info[pm] 与 pm.running_VMs 引用同一对象, 会一起修改
        self.PMid_to_PM: dict[int, PM] = {}
        self.VMid_to_VM: dict[int, VM] = {}
    
    def initialize(self, PM_number: int, cpu: int, mem: int, OC_ratio: list[float] | float):
        ''' OC_ratio CPU overcommit ratio 超分比 '''
        self.PM_number = PM_number
        self.cpu = cpu
        self.mem = mem
        self.OC_ratio = OC_ratio

        for i in range(PM_number):
            if isinstance(OC_ratio, (float, int)):
                pm = PM(i, cpu, mem, OC_ratio)
            elif isinstance(OC_ratio, list):
                pm = PM(i, cpu, mem, OC_ratio[i % len(OC_ratio)])
            else:
                raise TypeError(f"Cluster initialization variable OC_ratio ({OC_ratio}) type must be int, float, or list[int | float] !")
            self.PMlist.append(pm)
            self.running_VM_info[pm] = pm.running_VMs  # 这两个共享同一对象，会一起修改！
        
        self.PMid_to_PM = {pm.id: pm for pm in self.PMlist}

    def run_VM(self, pm: PM, vm: VM):
        self.VMid_to_VM[vm.id] = vm
        pm.run_VM(vm)

    def reallocate(self, allocation: dict[PM, list[VM]]):
        for pm, vms in allocation.items():
            pm.reallocate_VMs(vms)
            self.running_VM_info[pm] = pm.running_VMs  # 应该可以删除
    
    def get_allocation(self) -> dict[int, dict[str, dict | int]]:
        ''' 
        allocation[PMid] = {vm_dict: {VMid: {app: app, 
                                             cpu: cpu, 
                                             mem: mem},
                                     ...}, 
                            pm_cpu: PM_CPU_Constraint, 
                            pm_mem: PM_MEM_Constraint} 
        '''
        allocation = {}
        for pm in self.PMlist:
            allocation[pm.id] = {
                'vm_dict': {vm.id: {'app': vm.app, 'cpu': vm.CPU, 'mem': vm.MEM} 
                            for vm in pm.running_VMs},
                'pm_cpu': pm.CPUCapacity * pm.OC_ratio,
                'pm_mem': pm.MEMCapacity,
            } 
        return allocation

    def get_close_PMs(self) -> list[PM]:
        return [pm for pm in self.PMlist if not pm.open]

    def get_open_PMs(self) -> list[PM]:
        return [pm for pm in self.PMlist if pm.open]

    def get_available_PMs(self, vm: VM) -> list[PM]:  
        return [pm for pm in self.PMlist if not pm.check_capacity_exceed(vm)]
    
    def get_PMs_degradation(self) -> tuple[int, float]:
        ''' 返回真实总劣化虚机数，真实总劣化分数(由 1-QoS 计算) '''
        open_PMs = self.get_open_PMs()
        PMs_deg_num = [pm.get_deg_num() for pm in open_PMs]
        real_deg_VM_num = sum(PMs_deg_num)
        PMs_deg_QoS = sum(pm.get_deg_QoS() for pm in open_PMs)
        return real_deg_VM_num, PMs_deg_QoS
        
    def finish_VM(self, t: int) -> bool: 
        ''' 删除所有结束时间为 t 的 VM, 返回是否有 VM 被删除 '''
        is_vm_finished = False
        for pm in self.PMlist:
            for vm in self.running_VM_info[pm].copy():
                if (vm.end_time == t):
                    pm.complete_run_VM(vm)
                    is_vm_finished = True
        return is_vm_finished

    def get_average_utilz(self):
        ''' 返回cpu利用率平均值, mem利用率平均值 '''
        open_PMs = self.get_open_PMs()
        open_PM_num = len(open_PMs)
        total_cpu_utilz = 0
        total_mem_utilz = 0
        for pm in open_PMs:
            cpu_utilz, mem_utilz = pm.get_resource_usage()
            total_cpu_utilz += cpu_utilz
            total_mem_utilz += mem_utilz
        cpu_avg_utilz = total_cpu_utilz / open_PM_num 
        return cpu_avg_utilz, total_mem_utilz / open_PM_num
    
    def get_interfer_metrics(self) -> dict[PM, dict[VM, dict[str, float]]]:
        return {pm: pm.get_interfer_metrics() for pm in self.PMlist}
    
    def get_deg_PM_num(self) -> int:
        ''' 返回劣化 PM 数 (只要有一个 VM 劣化, 该 PM 就算劣化) '''
        return len([1 for pm in self.PMlist if (pm.get_deg_num() >= 1)])

    def add_PM(self, pm: PM):
        self.PMlist.append(pm)
        self.running_VM_info[pm] = pm.running_VMs
