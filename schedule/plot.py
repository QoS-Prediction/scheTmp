# 计算结果
import Conf
import os
import json
import pandas as pd

DetectMethod = ['PEAS', 'FFD']  # ["PEAS", "FFD", "RIAL"]
dyn_mask_w = [0.8]
g_alphas = [0.7]
mask_w = [0.5]
extended_methods = []
for method in DetectMethod:
    extended_methods.append(method)  # 原方法
    extended_methods.extend([f"{method}_dyn_mask_{w}_{g_a}" for w in dyn_mask_w for g_a in g_alphas])  # 动态掩码
    #extended_methods.extend([f"{method}_mask_{w}" for w in mask_w])
    #extended_methods.extend([f"{method}_mask_oracle_{w}" for w in mask_w])# 普通掩码

DetectMethod = extended_methods
Labeldict = DetectMethod

def _plotfig(VM2PMRatio,EXCEPTAPP,metric_name,lenPM,index_max = 220,xName = "time"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    data_all_dict = {}
    # idxlist = [0,1]
    for idx in range(len(DetectMethod)):
        Method = DetectMethod[idx]
        data_dict = {}
        log_dir = f"{Conf.MAIN_DIR}/log_main/reschedule_dynamic/PM{lenPM}/Ratio{VM2PMRatio}/except{EXCEPTAPP}/{Method}"
        for root, dirs, files in os.walk(log_dir):
            for filename in files:
                index = int(os.path.basename(root))
                if filename == "info_result.json":
                    # if index >index_max: continue
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        ratio = data[xName]
                        data_dict[ratio] = data['result']['after'][metric_name]

        ratios = sorted(data_dict.keys())
        metrics = [data_dict[ratio] for ratio in ratios]
        
        if xName == "time" and len(ratios)>0:
            min_ratio = ratios[0]
            ratios = [ratio - min_ratio for ratio in ratios]
        print(len(metrics))
        data_all_dict[Method] = data_dict

        ax.plot(ratios, metrics, label=Labeldict[idx])
    plt.legend()
    plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
    plt.savefig(f"{Conf.MAIN_DIR}/log_main/reschedule_dynamic/PM{lenPM}/Ratio{VM2PMRatio}/except{EXCEPTAPP}/all_{metric_name}.png",bbox_inches = 'tight')
    # plt.savefig(f"{Conf.MAIN_DIR}/log/reschedule_dynamic3_log/Ratio{VM2PMRatio}/except{EXCEPTAPP}/all_{metric_name}.pdf",format = 'pdf',bbox_inches = 'tight')
    return

def plotfig(VM2PMRatio,EXCEPTAPP,lenPM,index_max = 300,xName = "time"):
    metric_names = ['deg_QoS_sum',"deg_vm_num","deg_pm_num",'cpu_utilz']
    for metric_name in metric_names:
        _plotfig(VM2PMRatio,EXCEPTAPP,metric_name,lenPM,index_max,xName)

def _cal_avg(EXCEPTAPP,metric_name,lenPM,xName = 'time', cases=range(10)):
    cal_dict = {}

    for idx in range(len(DetectMethod)):
        Method = DetectMethod[idx]
        count = 0
        t_len = 0
        for case in cases:
            ratio = round(float(3+0.001*case),3)
            print(ratio)
            log_dir = f"{Conf.MAIN_DIR}/log_main/reschedule_dynamic/PM{lenPM}/Ratio{ratio}/except{EXCEPTAPP}/{Method}"
            data_dict = {}
            for root, dirs, files in os.walk(log_dir):
                for filename in files:
                    index = int(os.path.basename(root))
                    if filename == "info_result.json":
                        # if index >index_max: continue
                        filepath = os.path.join(root, filename)
                        with open(filepath, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                            ratio = data[xName]
                            data_dict[ratio] = data['result']['after'][metric_name]

            ratios = sorted(data_dict.keys())
            metrics = [data_dict[ratio] for ratio in ratios]
            
            if xName == "time" and len(ratios)>0:
                min_ratio = ratios[0]
                ratios = [ratio - min_ratio for ratio in ratios]

            for len_index in range(len(ratios)-1):
                period = ratios[len_index+1] - ratios[len_index]
                count = count + data_dict[ratios[len_index]+min_ratio]*period
                t_len += period

            # print(Method)
            # print(metric_name)
            print(f"case{case}")
            # print(count/t_len)
            print(t_len)
        if not t_len == 0:
            cal_dict[Method] = count/t_len
    return cal_dict,t_len
            
def cal_avg(EXCEPTAPP,lenPM,cases):
    metric_names = ['deg_QoS_sum',"deg_vm_num",'cpu_utilz',"deg_pm_num"]
    metric_avg_dict = {}
    for metric_name in metric_names:
        metric_avg_dict[metric_name],t_len = _cal_avg(EXCEPTAPP,metric_name,lenPM,'time',cases)
        print(f"time_len{t_len}")
    print(metric_avg_dict)
    return metric_avg_dict
    


if __name__ == "__main__":
    VM2PMRatio = 3.0
    EXCEPTAPP = "total"
    PMnum = 10

    # plotfig(VM2PMRatio,EXCEPTAPP,PMnum)

    result_dict = cal_avg(EXCEPTAPP,PMnum,cases=range(10))  # [0,1,4,5,6]
    result_df = pd.DataFrame(result_dict)
    print(result_df)
    
    # for each in PMnumlist:
    #     PMnum = each
    #     result_dict = cal_avg(EXCEPTAPP,PMnum,case_begin= 0,case_end=9)
    #     result_df = pd.DataFrame(result_dict)
    #     print(result_df)
    #     result_df.to_csv(f"metricdata{PMnum}.csv",index = True)
        # plotfig(VM2PMRatio,EXCEPTAPP,xName = "current ratio")