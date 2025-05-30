# 正常二次调度的实现，请看other/simulation_dynamic3.py

1. 怎么从集群中获取到VM和主机对应的聚合数据（欣阳）接口写过？
2. 聚合数据传给千惠，得到预测QoS结果
3. 欣阳对比所有主机的预测结果，进行调度主机判断

vscode 如何创建文件夹考虑rc中的umask

class FEDGE:
    def __init__(self, cluster: Cluster):
        self.cluster = cluster
    
    def _run(self, vm:detected_vm, pms: list[PM], vm_cpu_utilz: dict[VM, float], org_allocation: dict[PM, list[VM]]) -> dict[PM, list[VM]]:
        
        for pm in pms:
            max_qos_pm
            fedge(vm,pm,dist[pm])     