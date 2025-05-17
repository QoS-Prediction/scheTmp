from schedule_lib.class_vm import VM
from schedule_lib.class_pm import PM
from schedule_lib.class_cluster import Cluster
from datapre.pred2sche import tracepred
import random
import copy

class PAVMM_aco:
    def __init__(self, cluster: Cluster, is_mask: bool, num_ants=5, num_iterations=20, alpha=1, rho=0.1, q0=0.9) -> None:
        self.cluster = cluster
        self.is_mask = is_mask
        if self.is_mask:
            self.name = 'PAVMM_mask'
        else:
            self.name = 'PAVMM'
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.rho = rho
        self.q0 = q0

    def calculate_pm_performance(self, pm_utilization: float) -> float:
        """
        根据性能模型计算主机的性能
        """
        if pm_utilization <= 97.3:
            return -0.00227 * pm_utilization + 0.62288
        else:
            return -0.00128 * pm_utilization + 0.53754

    def calculate_total_performance(self, allocation: dict[PM, list[VM]], vm_cpu_utilz: dict[VM, float]) -> float:
        """
        计算所有虚拟机的总性能
        """
        total_performance = 0
        for pm, vm_list in allocation.items():
            pm_utilization = sum(vm_cpu_utilz[vm] for vm in vm_list)
            total_performance += self.calculate_pm_performance(pm_utilization) * len(vm_list)
        return total_performance
    
    def get_avail_PMs(self, pms: list[PM], vm: VM, allocation: dict[PM, list[VM]]) -> list[PM]:
        potential_PMs = [pm for pm in pms if len(allocation[pm]) < 3]
        if len(potential_PMs) == 0:
            raise ValueError('all pms has more or equal than 3 running VMs!')
        
        pm_apps_list = []
        for pm in potential_PMs:
            pm_apps_list.append([vm.app for vm in allocation[pm]])
        pm_mask = tracepred([vm.app], [vm.current_colocate + [vm.app]], pm_apps_list)
        pm_mask = pm_mask.flatten().astype(bool)
        available_PMs = [pm for pm, mask in zip(potential_PMs, pm_mask) if mask]
        return available_PMs

    def _run(self, vms: list[VM], pms: list[PM], vm_cpu_utilz: dict[VM, float], org_allocation: dict[PM, list[VM]]) -> dict[PM, list[VM]]:
        # 初始化信息素矩阵
        pheromone = {
            vm: {
                pm: self.calculate_total_performance(org_allocation, vm_cpu_utilz)  # 避免除以零
                for pm in pms
            }
            for vm in vms
        }
        original_pheromone = copy.deepcopy(pheromone)
        best_allocation = None
        best_performance = float('-inf')

        # 外部集合存储最优解及其首次迭代次数
        external_set = {"solution": None, "resided_iterations": 0}

        for iteration in range(self.num_iterations):
            solutions: list[dict[PM, list[VM]]] = []

            for ant in range(self.num_ants):
                allocation: dict[PM, list[VM]] = {pm: [] for pm in pms}

                for vm in vms:
                    if self.is_mask:
                        avail_pms = self.get_avail_PMs(pms, vm, allocation)
                    else:
                        avail_pms = [pm for pm in pms if len(allocation[pm]) < 5] 
                        
                    if random.random() < self.q0:
                        # 按照信息素选择最优主机
                        pm = max(avail_pms, key=lambda x: pheromone[vm][x])
                    else:
                        # 按等概率随机选择主机
                        pm = random.choice(avail_pms)

                    allocation[pm].append(vm)

                    # 更新局部信息素
                    pheromone[vm][pm] *= (1 - self.rho)
                    pheromone[vm][pm] += self.rho * original_pheromone[vm][pm]

                solutions.append(allocation)

            # 更新全局最优解
            current_best_solution = None
            current_best_performance = float('-inf')

            for solution in solutions:
                performance = self.calculate_total_performance(solution, vm_cpu_utilz)
                if performance > current_best_performance:
                    current_best_performance = performance
                    current_best_solution = solution

            # 如果新的解优于全局最优解，则更新全局最优解
            if current_best_performance > best_performance:
                best_performance = current_best_performance
                best_allocation = current_best_solution

                # 更新外部集合
                if external_set["solution"] == current_best_solution:
                    # 当前解已经在外部集合中，增加存续迭代次数
                    external_set["resided_iterations"] += 1
                else:
                    # 当前解是新的最优解，替换外部集合并重置存续迭代次数
                    external_set["solution"] = current_best_solution
                    external_set["resided_iterations"] = 1
            
            # 信息素全局挥发
            for vm in vms:
                for pm in pms:
                    pheromone[vm][pm] *= (1 - self.rho)

            # 增强全局最优解的信息素
            gamma = self.num_ants / (iteration+1 - external_set["resided_iterations"] + 1)
            for pm, vm_list in best_allocation.items():
                for vm in vm_list:
                    pheromone[vm][pm] += self.rho * gamma * best_performance

        return best_allocation

    def reschedule(self) -> None:
        vms = [vm for vm_set in self.cluster.running_VM_info.values() for vm in vm_set]
        pms = self.cluster.PMlist
        vm_cpu_utilz = {vm: vm.current_metrics['lvirt_domain_info_cpu_usage'] for vm in vms}
        org_allocation = {pm: list(pm.running_VMs) for pm in pms}
        reallocation = self._run(vms, pms, vm_cpu_utilz, org_allocation)

        self.cluster.reallocate(reallocation)
        