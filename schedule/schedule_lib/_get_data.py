import Conf
import schedule_lib._conf as sch_conf
import os
import pickle
main_dir = Conf.MAIN_DIR
App = sch_conf.App
Metric = str

# 读取数据
if not os.path.exists(f"{main_dir}/data/MetricsData.pickle"):
    raise ValueError(f"{main_dir}/data/MetricsData.pickle doesn't exists! Please run Utils_data.py first!")
else:
    with open(f"{main_dir}/data/MetricsData.pickle", "rb") as f:
        apps_metric_dict = pickle.load(f)
# apps_metric_dict[memcacheredis][i][index] 是这个共置下第index个VM在第i个子情况下的指标
# apps_metric_dict[memcacheredis][i]['agg'] 是这个共置下所有虚机在第i个子情况下的聚合指标
# 每个共置情况共有4种子情况

# 一堆函数
def get_real_metrics(app: App, applist: list[App], colocate_ind: int, app_ind: int) -> dict[Metric, float]:
    ''' 返回 app 在与 applist 共置时的 metrics '''
    applist.append(app)
    applist_merge = "".join(sorted(applist))

    if applist_merge in apps_metric_dict:
        app_ind_count = 0
        for data in apps_metric_dict[applist_merge][colocate_ind].values():
            if 'app' not in data: continue
            if data['app'] == app:
                if app_ind_count == app_ind:
                    return data
                app_ind_count += 1
        raise ValueError(f"applist: {applist_merge}, colocate_ind: {colocate_ind}, app_ind: {app_ind} don't have data!")
    else:
        raise ValueError(f"applist: {applist_merge} not in apps_metric_dict!")
    
def get_real_deg_num(applist: list[App], colocate_ind: int) -> int:
    ''' 返回 applist 共置时有多少个劣化虚机 '''
    applist_merge = "".join(sorted(applist))
    if applist_merge in apps_metric_dict:
        deg_num = 0
        for data in apps_metric_dict[applist_merge][colocate_ind].values():
            if 'QoS' not in data: continue
            if data['QoS'] < sch_conf.QOS_THRESHOLD:
                deg_num += 1
        return deg_num
    else:
        raise ValueError(f"applist: {applist_merge} not in apps_metric_dict!")
    
def get_QoS_sum(applist: list[App], colocate_ind: int) -> float:
    ''' 返回 applist 共置时所有虚机的 QoS 总和 '''
    applist_merge = "".join(sorted(applist))
    if applist_merge in apps_metric_dict:
        result = sum(data['QoS'] for data in apps_metric_dict[applist_merge][colocate_ind].values() if 'QoS' in data)
        return result
    else:
        raise ValueError(f"applist: {applist_merge} not in apps_metric_dict!")
