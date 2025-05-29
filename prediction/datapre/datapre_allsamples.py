# notification：严谨抽样version
# 如不对PM、VM抽样请把测试集抽样率设为0
# dataprocessed2 对VM抽样
# dataprocessed3 对VM PM 抽样
# dataprocessed4 对PM抽样
import sys
sys.path.append("..")
sys.path.append(".")
import datapre.Conf
import pickle
import os
import pandas as pd
import numpy as np
from functools import lru_cache

from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations_with_replacement
from itertools import product
import itertools
import pickle
from tqdm import tqdm


# 输入 comb 
# 输出该comb聚合的数据   
# 网络中间的监督值
def get_comb_aggdata(applist,timestamp = 0):
    comb_dim_begin = Conf.comb_dim_begin
    comb_dim_end = Conf.comb_dim_end
    raw_data_df = get_allapp_df_all()
    return raw_data_df[(raw_data_df['comb'] == ",".join(sorted(applist))) & (raw_data_df['index'] == 0)].iloc[timestamp:timestamp+1, comb_dim_begin:Conf.comb_dim_end+1]

def get_comb_alldata(applist,timestamp = 0):
    raw_data_df = get_allapp_df_all()
    return raw_data_df[(raw_data_df['comb'] == ",".join(sorted(applist))) & (raw_data_df['timestamp'] == timestamp)]



# 输入 comb app 
# 输出 该comb下该app数据  
# 用于调度预测的时候的输入
def get_comb_appdata(applist,app):
    raw_data_df = get_allapp_df_all()
    return raw_data_df[(raw_data_df['app'] == app) & (raw_data_df['comb'] == ",".join(sorted(applist)))]


# 输入 app
# 输出 包含该app的所有数据的df
def get_app_df_all(app):
    raw_data_df = get_allapp_df_all()
    return raw_data_df[raw_data_df['app'] == app]


# 所有数据的df,维度为 31 + 124 + 1 + 3
# 1 为QoS,  3 为三个索引 一个是 comb 一个是 app  另一个是index(主机上的虚机id可以区分两个相同应用虚机)
@lru_cache
def get_allapp_df_all(regenerate = False,timestamplist = [0,1,2,3],maxratio = Conf.MAXRATIO,savedata = True):
    raw_data_save_path = Conf.raw_data_save_path
    if os.path.exists(raw_data_save_path) and not regenerate:
        print(f"The path '{raw_data_save_path}' exists! Loading data from that path!")
        return pd.read_csv(raw_data_save_path)
    with open(os.path.join(Conf.DATA_DIR,"MetricsData.pickle"), "rb") as f:
        apps_metric_dict = pickle.load(f)
    raw_data_df = pd.DataFrame()
    for key,timestamp_dict in apps_metric_dict.items():
        for timestamp,apps_dict in timestamp_dict.items():
            if timestamp not in timestamplist:
                continue
            applist = [dict['app'] for index,dict in apps_dict.items() if index != 'agg']
            if len(applist) > maxratio:
                continue
            print(applist)
            for index, dict in apps_dict.items():
                if index == 'agg':
                    continue
                df = pd.DataFrame([dict])
                aggdata = apps_dict['agg']
                aggdata_df = pd.DataFrame([aggdata])
                aggdata_df['QoS'] = df['QoS']
                aggdata_df['comb'] = ",".join(sorted(applist))
                aggdata_df['app'] = df['app']
                aggdata_df['index'] = index
                aggdata_df['timestamp'] = timestamp
                df = df.drop(columns = ['QoS','app'])
                raw_data = pd.concat([df,aggdata_df],axis = 1)
                raw_data_df = pd.concat([raw_data_df,raw_data],ignore_index = True)
    print(raw_data_df.shape)
    if savedata and maxratio == Conf.MAXRATIO:
        raw_data_df.to_csv(raw_data_save_path, index=False)
    else:
        raw_data_df.to_csv(Conf.raw_data_save_path2, index=False)
    return raw_data_df



def add2dict(dict,app,QoS):
    if app not in dict.keys():
        dict[app]={0:QoS}
        return dict
    else:
        index = len(dict[app])
        dict[app][index] = QoS
        return dict

def QoS_Oracle(comb,timestamp = 0):
    raw_data_save_path = Conf.raw_data_save_path2
    if os.path.exists(raw_data_save_path):
        # print(f"The path '{raw_data_save_path}' exists! Loading data from that path!")
        raw_df = pd.read_csv(raw_data_save_path)
    else:
        raw_df = get_allapp_df_all(regenerate = True,maxratio = 5,savedata = True)
    comb_df = raw_df[(raw_df['comb']==",".join(sorted(comb))) & (raw_df['timestamp'] == timestamp)]
    WhiteQoS_dict = {}
    avg_qos_values = {}
    for _, row in comb_df.iterrows():
        WhiteQoS_dict = add2dict(WhiteQoS_dict,row['app'],row['QoS'])
    for app, metrics in WhiteQoS_dict.items():
        average_value = sum(metrics.values()) / len(metrics)
        avg_qos_values[app] = average_value
    return WhiteQoS_dict,avg_qos_values




def get_train_test_vm(sample_ratio = 0.2):
    raw_df = get_allapp_df_all()
    if sample_ratio == 0:
        return raw_df,raw_df
    np.random.seed(3)
    raw_df = get_allapp_df_all()
    # grouped = raw_df.groupby(['app', 'comb','index'])
    grouped = raw_df.groupby(['app', 'comb'])
    # 输出有多少组
    num_groups = len(grouped)
    print(f"Number of groups: {num_groups}")
    group_names = list(grouped.groups.keys())

    # 随机选择 20% 的组作为测试集
    num_groups = len(group_names)
    num_test_groups = int(sample_ratio * num_groups)
    test_group_indices = np.random.choice(num_groups, num_test_groups, replace=False)
    test_group_names = [group_names[i] for i in test_group_indices]

    # 将测试集和训练集分开
    test_df = raw_df[raw_df.set_index(['app', 'comb']).index.isin(test_group_names)].reset_index(drop=True)
    train_df = raw_df[~raw_df.set_index(['app', 'comb']).index.isin(test_group_names)].reset_index(drop=True)

    # 输出结果
    # print("Test VM DataFrame all:")
    # print(test_df.shape)
    # print("\nTrain VM DataFrame all:")
    # print(train_df.shape)
    return train_df,test_df

def get_train_test_vm_expapp(expapp):
    raw_df = get_allapp_df_all()
    # 将测试集和训练集分开
    test_df = raw_df[raw_df['app']==expapp].reset_index(drop=True)
    train_df = raw_df[raw_df['app']!=expapp].reset_index(drop=True)
    # 输出结果
    print("Test VM DataFrame all:")
    print(test_df.shape)
    print("\nTrain VM DataFrame all:")
    print(train_df.shape)
    return train_df,test_df


def get_vm_ext_data(VMAPPlist,VMCOMBlist,VMtimestamplist = [0,1,2,3]):
    raw_data_save_path = Conf.raw_data_save_path2
    if os.path.exists(raw_data_save_path):
        # print(f"The path '{raw_data_save_path}' exists! Loading data from that path!")
        raw_df = pd.read_csv(raw_data_save_path)
    else:
        raw_df = get_allapp_df_all(regenerate = True,maxratio = 5,savedata = True)

    test_df = pd.DataFrame()
    assert len(VMAPPlist) == len(VMCOMBlist)
    for vmindex in range(len(VMAPPlist)):
        vmapp = VMAPPlist[vmindex]
        vmcomb = VMCOMBlist[vmindex]
        for timestamp in VMtimestamplist:
            # print(list(vmcomb))
            tmpdf = raw_df[(raw_df['comb'] == ",".join(sorted(list(vmcomb)))) & (raw_df['app'] == vmapp) & (raw_df['timestamp'] == timestamp) ]
            # print(tmpdf.shape)
            if len(tmpdf) >= 2:
                tmpdf = tmpdf.iloc[:1]
            test_df = pd.concat([test_df,tmpdf],ignore_index=True)
    # print(test_df)
    return test_df

def get_combpm_vms_agg_data(comblist,PMtimestamplist = [0,1,2,3]):
    baseline_agg_df = pd.DataFrame()
    deepsetvms_list = []
    for comb in comblist:
        for timestamp in PMtimestamplist:
            tmpdf = get_comb_alldata(comb,timestamp).iloc[:,:Conf.deepset_dim_num]
            agg_df = get_comb_aggdata(comb,timestamp).copy()
            
            agg_df['comb'] = ",".join(sorted(list(comb)))
            # print(agg_df)

            baseline_agg_df = pd.concat([baseline_agg_df,agg_df],ignore_index = True)
            reshape_tmp = tmpdf.to_numpy()[np.newaxis,:,:]
            # print(reshape_tmp.shape)
            deepsetvms_list.extend(reshape_tmp.tolist())
    return deepsetvms_list,baseline_agg_df

def get_train_test_pm(maxratio = Conf.MAXRATIO,sample_ratio = 0.2,PMtimestamplist = [0,1,2,3]):
    raw_df = get_allapp_df_all()
    comblist = []
    for num in range(1,maxratio):
        for comb in combinations_with_replacement(Conf.APPLIST,num):
            comblist.append(comb)

    if sample_ratio == 0:
        testcomblist = comblist
        traincomblist = comblist
    else:
        # 随机选择 20% 的组合作为测试集
        num_combs = len(comblist)
        num_test_combs = int(sample_ratio * num_combs)
        test_comb_indices = np.random.choice(num_combs, num_test_combs, replace=False)
        testcomblist = [comblist[i] for i in test_comb_indices]
        # print(len(testcomblist))
        # 将剩余的组合作为训练集
        traincomblist = [comblist[i] for i in range(num_combs) if i not in test_comb_indices]
        # print(len(traincomblist))

    deepsetvms_df_train,deepsetagg_df_train = get_combpm_vms_agg_data(traincomblist)
    print(len(deepsetvms_df_train),deepsetagg_df_train.shape)
    
    deepsetvms_df_test, deepsetagg_df_test  = get_combpm_vms_agg_data(testcomblist)
    print(len(deepsetvms_df_test),deepsetagg_df_test.shape)

    return deepsetvms_df_train,deepsetagg_df_train,deepsetvms_df_test, deepsetagg_df_test

def get_vm2pm(extvm_df,deepsetvms_list,deepsetagg_df):
    extVMfeature_df = pd.DataFrame()
    deepset_list = []
    y_list = []
    baseline_agg_df = pd.DataFrame()

    for _, rowVM in tqdm(extvm_df.iterrows()):
        vmapp = rowVM['app']
        for PMindex in range(len(deepsetvms_list)):
            deepsetvms = [deepsetvms_list[PMindex]]
            deepsetagg = deepsetagg_df.iloc[[PMindex]]
            combapp = str(deepsetagg_df.iloc[PMindex]["comb"]).split(",")
            applist = list(combapp)
            deepsetagg = deepsetagg.drop(columns = ["comb"])
            applist.append(vmapp)
            _,avgqosdict = QoS_Oracle(applist)
            QoS_vm = avgqosdict[vmapp]

            extVMfeature_df = pd.concat([extVMfeature_df,rowVM[:Conf.x_dim_num].to_frame().T],ignore_index = True)
            # print(np.array(deepsetvms).shape)
            deepset_list.extend(deepsetvms)
            baseline_agg_df = pd.concat([baseline_agg_df,deepsetagg],ignore_index = True)
            y_list.append(QoS_vm)

    return extVMfeature_df,deepset_list,baseline_agg_df,y_list




def normalpad_traindata(extdata,data,dataagg,Path):
    # 将数据展平到第 2 维度
    flattened_data = np.concatenate([np.array(sublist) for sublist in data], axis=0)
    # print(flattened_data)

    # 在第 2 维度上进行归一化
    # normalized_flattened_data = normalize(flattened_data, axis=0)
    ext_scaler = MinMaxScaler()
    deepsetx_scaler = MinMaxScaler()
    deepsetagg_scaler = MinMaxScaler()
    normalized_ext = ext_scaler.fit_transform(extdata.to_numpy())
    normalized_flattened_data = deepsetx_scaler.fit_transform(flattened_data)
    normalized_agg = deepsetagg_scaler.fit_transform(dataagg)

    # print(normalized_flattened_data)

    # 将数据恢复到原来的形状
    normalized_data = []
    start = 0
    for sublist in data:
        length = len(sublist)
        normalized_data.append(normalized_flattened_data[start:start+length].tolist())
        start += length

    # print(normalized_data)

    # 找到第一维度的最大长度
    max_length = max(len(sublist) for sublist in normalized_data)

    # 对每个子列表进行填充
    padded_normalized_data = [sublist + [[0] * len(sublist[0])] * (max_length - len(sublist)) for sublist in normalized_data]
    # print(padded_normalized_data)


    with open(os.path.join(Path,'deepsetx_scaler.pkl'),'wb') as file:
        pickle.dump(deepsetx_scaler,file)

    with open(os.path.join(Path,'deepsetagg_scaler.pkl'),'wb') as file:
        pickle.dump(deepsetagg_scaler,file)
    with open(os.path.join(Path,'ext_scaler.pkl'),'wb') as file:
        pickle.dump(ext_scaler,file)
    return np.array(normalized_ext),np.array(padded_normalized_data),np.array(normalized_agg)

def normalpad_testdata(extdata,data,dataagg,Path = os.path.join(Conf.DATA_DIR,f"dataprocessed2")):
    # 将数据展平到第 2 维度
    flattened_data = np.concatenate([np.array(sublist) for sublist in data], axis=0)
    # print(flattened_data.shape)

    # 在第 2 维度上进行归一化
    # normalized_flattened_data = normalize(flattened_data, axis=0)


    with open(os.path.join(Path,'deepsetx_scaler.pkl'),'rb') as file:
        deepsetx_scaler = pickle.load(file)

    with open(os.path.join(Path,'deepsetagg_scaler.pkl'),'rb') as file:
        deepsetagg_scaler = pickle.load(file)
    with open(os.path.join(Path,'ext_scaler.pkl'),'rb') as file:
        ext_scaler = pickle.load(file)
    normalized_flattened_data = deepsetx_scaler.transform(flattened_data)
    normalized_agg = deepsetagg_scaler.transform(dataagg)
    normalized_ext = ext_scaler.transform(extdata.to_numpy())

    # print(normalized_flattened_data)

    # 将数据恢复到原来的形状
    normalized_data = []
    start = 0
    for sublist in data:
        length = len(sublist)
        normalized_data.append(normalized_flattened_data[start:start+length].tolist())
        start += length

    # print(normalized_data)

    # 找到第一维度的最大长度
    max_length = max(len(sublist) for sublist in normalized_data)

    # 对每个子列表进行填充
    padded_normalized_data = [sublist + [[0] * len(sublist[0])] * (max_length - len(sublist)) for sublist in normalized_data]
    # print(padded_normalized_data)

    return np.array(normalized_ext),np.array(padded_normalized_data),np.array(normalized_agg)


def myreshape(data,num = 560):
    n = int(data.shape[0] / num)
    reshaped_array = data.reshape((num, n) + data.shape[1:])
    # print(reshaped_array.shape)
    return reshaped_array

def datapre_main(savepath,expapp = None):
    if expapp == None:
        extvm_df_train,extvm_df_test = get_train_test_vm()
    else:
        extvm_df_train,extvm_df_test = get_train_test_vm_expapp(expapp)
    deepsetvms_df_train,deepsetagg_df_train,deepsetvms_df_test, deepsetagg_df_test = get_train_test_pm()
    lendict = {}
    lendict["vm_train"] = len(extvm_df_train)
    lendict["vm_test"]  = len(extvm_df_test)
    lendict["pm_train"] = len(deepsetvms_df_train)
    lendict["pm_test"]  = len(deepsetvms_df_test)
    
    extVMfeature_df_train,deepset_list_train,baseline_agg_df_train,y_list_train = get_vm2pm(extvm_df_train,deepsetvms_df_train,deepsetagg_df_train)
    extVMfeature_df_test,deepset_list_test,baseline_agg_df_test,y_list_test = get_vm2pm(extvm_df_test,deepsetvms_df_test,deepsetagg_df_test)


    norm_extVMfeature_train,norm_deepsetx_train,norm_deepsetagg_train = normalpad_traindata(extVMfeature_df_train,deepset_list_train,baseline_agg_df_train,Path=savepath)
    print(norm_extVMfeature_train.shape,norm_deepsetx_train.shape,norm_deepsetagg_train.shape)
    norm_extVMfeature_test, norm_deepsetx_test ,norm_deepsetagg_test  = normalpad_testdata(extVMfeature_df_test ,deepset_list_test ,baseline_agg_df_test,Path=savepath)
    print(norm_extVMfeature_test.shape,norm_deepsetx_test.shape,norm_deepsetagg_test.shape)
    y_list_train = np.array(y_list_train)
    y_list_test = np.array(y_list_test)

    # print(vmlen_train,vmlen_test,pmlen_train,pmlen_test)
    print(np.array(y_list_train).shape,np.array(y_list_test).shape)

    return lendict,norm_extVMfeature_train,norm_deepsetx_train,norm_deepsetagg_train,y_list_train,norm_extVMfeature_test, norm_deepsetx_test ,norm_deepsetagg_test,y_list_test

    
if __name__ == "__main__":
    expapp = "nginx"
    Path = os.path.join(Conf.DATA_DIR,f"dataprocessed{expapp}")
    if not os.path.exists(Path):
        os.makedirs(Path)
    lendict,norm_extVMfeature_train,norm_deepsetx_train,norm_deepsetagg_train,y_list_train,norm_extVMfeature_test, norm_deepsetx_test ,norm_deepsetagg_test,y_list_test = datapre_main(Path,expapp)
    
    with open(os.path.join(Path,'norm_extVMfeature_train.pkl'),'wb') as f:
        pickle.dump(myreshape(norm_extVMfeature_train,lendict["vm_train"]),f)
    with open(os.path.join(Path,'norm_deepsetx_train.pkl'),'wb') as f:
        pickle.dump(myreshape(norm_deepsetx_train,lendict["vm_train"]),f)    
    with open(os.path.join(Path,'norm_deepsetagg_train.pkl'),'wb') as f:
        pickle.dump(myreshape(norm_deepsetagg_train,lendict["vm_train"]),f)
    with open(os.path.join(Path,'y_list_train.pkl'),'wb') as f:
        pickle.dump(myreshape(y_list_train,lendict["vm_train"]),f)

    with open(os.path.join(Path,'norm_extVMfeature_test.pkl'),'wb') as f:
        pickle.dump(myreshape(norm_extVMfeature_test,lendict["vm_test"]),f)
    with open(os.path.join(Path,'norm_deepsetx_test.pkl'),'wb') as f:
        pickle.dump(myreshape(norm_deepsetx_test,lendict["vm_test"]),f)    
    with open(os.path.join(Path,'norm_deepsetagg_test.pkl'),'wb') as f:
        pickle.dump(myreshape(norm_deepsetagg_test,lendict["vm_test"]),f)
    with open(os.path.join(Path,'y_list_test.pkl'),'wb') as f:
        pickle.dump(myreshape(y_list_test,lendict["vm_test"]),f)



    
