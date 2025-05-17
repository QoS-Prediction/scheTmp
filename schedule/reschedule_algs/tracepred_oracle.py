from datapre.datapre_allsample import QoS_Oracle
from Conf import App
import numpy as np

def tracepred_oracle(VMAPPlist: list[App], PMCOMBlist: list[list[App]], mask: float) -> np.ndarray:
    results = np.zeros((len(VMAPPlist), len(PMCOMBlist)))

    for i, app in enumerate(VMAPPlist):
        for j, pm_apps in enumerate(PMCOMBlist):
             _, app_qos_dict = QoS_Oracle(pm_apps + [app]) 
             results[i, j] = app_qos_dict[app]

    masked_results = np.ones_like(results, dtype=int)  # 默认全为 1
    num_mask = np.ceil(results.shape[1] * mask).astype(int)
    if num_mask == results.shape[1]:
        num_mask -= 1

    # 对每一行进行处理
    for i in range(results.shape[0]):
        sorted_indices = np.argsort(results[i])  # 按值排序，返回索引
        mask_indices = sorted_indices[:num_mask]  # 选取最低的 num_mask 个索引
        masked_results[i, mask_indices] = 0  # 设为 0
    
    return masked_results
