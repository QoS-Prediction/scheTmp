from schedule_lib.class_vm import VM
from schedule_lib._get_data import get_real_deg_num, get_QoS_sum
import schedule_lib._conf as sch_conf
import random
from collections import defaultdict
Metric_agg = str

def rand_colocate_ind(max_num = sch_conf.COLOCATE_DATA_NUM) -> int:
    ''' 随机选择一个整数 (0,1,2,3) 表示共置数据的 index '''
    return random.randint(0, max_num-1)

class PM:
    def __init__(self, id, CPU, MEM, OC_ratio):
        ''' OC_ratio overcommit_ratio CPU 超分比 '''
        self.id = int(id)
        self.CPUCapacity = int(CPU)
        self.MEMCapacity = int(MEM)
        self.OC_ratio = OC_ratio  # CPU 超分比
        self.CPUCapacityUsage = 0  # 容量利用率量（由VM容量约束）
        self.MEMCapacityUsage = 0
        self.running_VMs: set[VM] = set()
        self.colocate_ind = 0
        self.open = False

    def check_capacity_exceed(self, vm: VM) -> bool:  
        # 约束为 cpu 超分比 <= OC_ratio, mem 小于容量
        if (self.CPUCapacityUsage + vm.CPU > self.OC_ratio * self.CPUCapacity) or (
            self.MEMCapacityUsage + vm.MEM > self.MEMCapacity
        ):
            return True
        return False

    def run_VM(self, vm: VM):  # 运行 VM
        if not self.check_capacity_exceed(vm):
            self.CPUCapacityUsage += vm.CPU
            self.MEMCapacityUsage += vm.MEM
            if vm in self.running_VMs:
                raise Exception(f"VM {vm.id} already running in PM {self.id}")
            self.running_VMs.add(vm)
            self.open = True
            self.colocate_ind = rand_colocate_ind()
            self.renew_VM_metrics(self.colocate_ind)
        else:
            raise Exception(
                "Capacity constraint violate! CPU {} MEM {} vmCPU {} vmMEM {}".format(
                    self.CPUCapacityUsage, self.MEMCapacityUsage, vm.CPU, vm.MEM
                )
            )

    def complete_run_VM(self, vm: VM):  # 结束 VM
        if vm not in self.running_VMs:
            raise Exception(f"VM {vm.id} not running in PM {self.id}")
        self.running_VMs.remove(vm)
        self.CPUCapacityUsage -= vm.CPU
        self.MEMCapacityUsage -= vm.MEM
        if len(self.running_VMs) == 0: 
            self.open = False
        else:
            self.colocate_ind = rand_colocate_ind()
            self.renew_VM_metrics(self.colocate_ind)
    
    def shut_down(self) -> list[VM]:
        ''' 关机, 并输出所有运行中的虚机 '''
        vms = list(self.running_VMs)
        for vm in vms:
            self.complete_run_VM(vm)
        return vms
        
    def reallocate_VMs(self, VMs: list[VM]):
        ''' PM 重排 VMs '''
        self.running_VMs = set(VMs)
        self.CPUCapacityUsage = sum([vm.CPU for vm in self.running_VMs])
        self.MEMCapacityUsage = sum([vm.MEM for vm in self.running_VMs])
        if (self.CPUCapacityUsage > self.OC_ratio * self.CPUCapacity) or (self.MEMCapacityUsage > self.MEMCapacity):
            raise ValueError(f"When reallocate VMs, capacity exceed!")

        if len(self.running_VMs) == 0: 
            self.open = False
        else:
            self.open = True

        self.colocate_ind = rand_colocate_ind()
        self.renew_VM_metrics(self.colocate_ind)

    def renew_VM_metrics(self, colocate_ind):
        ''' 更新主机所在的 VM metrics '''
        # 为每个同样 app 的 VM 生成不同的 app_ind
        app_groups = defaultdict(list)
        for vm in self.running_VMs:
            app_groups[vm.app].append(vm)

        vm_app_ind: dict[VM, int] = {}
        for app, vms in app_groups.items():
            for app_ind, vm in enumerate(vms):
                vm_app_ind[vm] = app_ind
        
        # 更新
        for vm in self.running_VMs:
            vm.renew_metrics(self.running_VMs - set([vm]), colocate_ind, vm_app_ind[vm])

    def get_deg_num(self) -> int:
        ''' 返回真实的劣化虚机数 '''
        if len(self.running_VMs) >= 5:
            return len(self.running_VMs)
        if len(self.running_VMs) == 0:
            return 0
        return get_real_deg_num([vm.app for vm in self.running_VMs], self.colocate_ind)
    
    def get_deg_QoS(self) -> float:
        ''' 返回真实的劣化分数总和，由 deg = 1 - QoS 计算 '''
        if len(self.running_VMs) >= 5:
            return len(self.running_VMs) * 1.0
        if len(self.running_VMs) == 0:
            return 0.0
        QoS_sum = get_QoS_sum([vm.app for vm in self.running_VMs], self.colocate_ind)
        deg_QoS = len(self.running_VMs) - QoS_sum
        return deg_QoS

    def get_resource_usage(self) -> tuple[float, float]:
        CPUUsage_lvirt = 0
        MEMUsage_lvirt = 0
        for vm in self.running_VMs:
            cpu_utilization_rate, mem_utilize_usage = vm.get_resource_usage()
            CPUUsage_lvirt += (cpu_utilization_rate * vm.CPU) / self.CPUCapacity
            MEMUsage_lvirt += mem_utilize_usage / (2 ** 30)   # 换算成 GB
        return CPUUsage_lvirt, MEMUsage_lvirt
    
    def get_metrics(self) -> dict[VM, dict[str, float]]:
        return {vm: vm.current_metrics for vm in self.running_VMs}
    
    def get_interfer_metrics(self) -> dict[VM, dict[str, float]]:
        return {vm: vm.get_interfer_metrics() for vm in self.running_VMs}
    
    def __eq__(self, other):
        if isinstance(other, PM):
            if self.id == other.id:
                return True
        return False
    
    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"Host {self.id} (CPU {self.CPUCapacityUsage}, MEM {self.MEMCapacityUsage}, running {len(self.running_VMs)} VMs)"

    def __repr__(self):
        return self.__str__()
