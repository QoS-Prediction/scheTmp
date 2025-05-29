# import datapre_allsample as datapre
import datapre.datapre_allsamples as datapre
from itertools import combinations_with_replacement
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("../baselines")
import datapre.Conf
import numpy as np
# from baselines.trace import Trace
from baselines.traceSetTransDCN import Trace
import os
import torch
import torch.nn as nn

def change_shape(data):
    shape = data.shape
    if len(shape) < 2:
        raise ValueError("输入张量的维度必须至少为 2")
    data = data.view(-1, *shape[2:])
    return data

def return_shape(data, batch_shape):
    shape = data.shape
    data = data.view(*batch_shape, *shape[1:])
    return data

# def dynamic_mask_ratio(arr):
#     arr_transformed = (arr + 1) / 2
#     mean_value = np.mean(arr_transformed)
#     # 计算最大值
#     max_value = np.max(arr_transformed)
#     # 计算平均值与最大值的比值
#     ratio = mean_value / max_value
#     # 0-1   1 波动小(比较多的)  0波动大(比较少)
#     with open('mask_ratio_RIAL.txt', 'a') as f:
#         f.write(f"{1-ratio}\n")
#     return 1-ratio

def gini_coefficient(x):
    # 确保数据为非负且为一维数组
    x = np.array(x)
    x[x < 0] = 0
    
    # 排序
    sorted_x = np.sort(x)
    n = len(x)
    # 分子部分
    numerator = np.sum((2 * np.arange(1, n+1) - n - 1) * sorted_x)
    denominator = n * np.sum(sorted_x)
    return numerator / denominator if denominator != 0 else 0

def dynamic_mask_ratio(arr, g_alpha):
    # from inequality.gini import Gini
    # g = Gini(arr).g

    g = gini_coefficient(arr)
    g0 = 0.22
    ratio = g_alpha * g /g0
    
    # with open('mask_ratio.txt', 'a') as f:
    #     f.write(f"g_alpha: {g_alpha}, ratio: {ratio}\n")
    
    return ratio


# INPUT VM
# APPLIST COMBLIST/APPLIST

# INPUT PM
# COMBLIST/APPLIST

def tracepred(VMAPPlist, VMCOMBlist, PMCOMBlist, modelpath=Conf.MODEL_PATH, \
              maskratio=0.5, dynamic=True, dynamicmaxratio=0.8, dynamicminratio=0.4, g_alpha=0.8) -> np.ndarray:
    assert len(PMCOMBlist) != 0
    assert len(VMAPPlist) != 0
    assert len(VMAPPlist) == len(VMCOMBlist)
    # 记录空列表的位置
    emptylist_indices = [index for index, sublist in enumerate(PMCOMBlist) if not sublist]
    # 删除空列表
    PMCOMBlist = [sublist for sublist in PMCOMBlist if sublist]

    if not PMCOMBlist == []:
        extvm_df_test = datapre.get_vm_ext_data(VMAPPlist,VMCOMBlist,[0])
        # print(extvm_df_test.shape)
        # print(extvm_df_test)
        deepsetvms_df_test, deepsetagg_df_test  = datapre.get_combpm_vms_agg_data(PMCOMBlist,[0])
        # print(len(deepsetvms_df_test),deepsetagg_df_test.shape)
        extVMfeature_df_test,deepset_list_test,baseline_agg_df_test,y_list_test = datapre.get_vm2pm(extvm_df_test,deepsetvms_df_test,deepsetagg_df_test)
        norm_extVMfeature_test, norm_deepsetx_test ,norm_deepsetagg_test  = datapre.normalpad_testdata(extVMfeature_df_test ,deepset_list_test ,baseline_agg_df_test )
        
        y_list_test = np.array(y_list_test)
        # print(norm_extVMfeature_test.shape,norm_deepsetx_test.shape,norm_deepsetagg_test.shape,y_list_test.shape)
        norm_extVMfeature_test = datapre.myreshape(norm_extVMfeature_test,len(VMAPPlist))
        norm_deepsetx_test = datapre.myreshape(norm_deepsetx_test,len(VMAPPlist))
        norm_deepsetagg_test = datapre.myreshape(norm_deepsetagg_test,len(VMAPPlist))
        y_list_test = datapre.myreshape(y_list_test,len(VMAPPlist))
        # print(norm_extVMfeature_test.shape,norm_deepsetx_test.shape,norm_deepsetagg_test.shape,y_list_test.shape)

        val_data = { 
        'VMfeature': torch.tensor(norm_extVMfeature_test, dtype=torch.float32),
        'deepsetx': torch.tensor(norm_deepsetx_test, dtype=torch.float32),
        'deepsetagg': torch.tensor(norm_deepsetagg_test, dtype=torch.float32),
        'target': torch.tensor(y_list_test, dtype=torch.float32)
        }

        # load model and predict
        loaded_model = Trace()
        # loaded_model.load_state_dict(torch.load(modelpath,weights_only=True))
        loaded_model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
        loaded_model.eval()

        batch_shape = val_data['target'].shape
        with torch.no_grad():
            for key, data in val_data.items():
                if key == 'target':  continue
                val_data[key] = change_shape(data)
            VMfeature = val_data['VMfeature']
            deepsetx = val_data['deepsetx']
            deepsetagg = val_data['deepsetagg']
            target = val_data['target']

            output = loaded_model(VMfeature, deepsetx, deepsetagg)
            with open('trace_data_settrans.txt', 'a') as f:
                f.write(f'VMAPPlist: {VMAPPlist}\n')
                f.write(f'VMCOMBlist: {VMCOMBlist}\n')
                f.write(f'PMlist: {PMCOMBlist}\n')
                f.write(f'Output:\n')
                f.write(str(output))
                f.write('\n\n')
            
            output = return_shape(output, target.shape)
            if output.dim() == 3 and output.size(-1) == 1:
                output = output.squeeze(-1)

            for key, data in val_data.items():
                if key == 'target':  continue
                val_data[key] = return_shape(data, batch_shape)
            assert output.shape == target.shape
            # print(output.shape)
            # print(target.shape)

            # mse_loss = nn.MSELoss()
            # # 计算 MSE 损失
            # loss = mse_loss(output, target)
            # # 打印损失值
            # print("MSE Loss:", loss.item())

            mask_np = output.numpy().copy()
            # 对每一行进行处理
            for i in range(mask_np.shape[0]):
                row = mask_np[i]

                # 找到最小的 maskratio 的值的索引
                if dynamic:
                    maskratio = max(dynamicminratio, min(dynamic_mask_ratio(row, g_alpha), dynamicmaxratio))
                    
                
                threshold_index = int((len(row)+len(emptylist_indices)) * maskratio)
                if threshold_index >= len(row) and len(emptylist_indices) == 0:
                    threshold_index = len(row) - 1

                # print(threshold_index)
                sorted_indices = np.argsort(row)
                min_indices = sorted_indices[:threshold_index]
                # print(min_indices)
                # 创建一个全为 1 的数组
                binary_row = np.ones_like(row)
                # 将最小的 maskratio 的值设置为 0
                binary_row[min_indices] = 0
                # 更新原数组
                mask_np[i] = binary_row
    else:
        lenvm = len(VMAPPlist)
        lenpm = len(emptylist_indices)
        return np.ones((lenvm, lenpm))

    # print(mask_np)
    # print(output.numpy())
    # print(target.numpy())
    if len(emptylist_indices)>0:
        for row in range(mask_np.shape[0]):
            for indice in emptylist_indices:
                mask_np = np.insert(mask_np, indice, 1, axis=1)

    return mask_np


def _tracepred(VMAPPlist, VMCOMBlist, PMCOMBlist, modelpath=Conf.MODEL_PATH, \
              maskratio=0.5, dynamic=True, dynamicmaxratio=0.8, dynamicminratio=0.4) -> np.ndarray:
    assert len(PMCOMBlist) != 0
    assert len(VMAPPlist) != 0
    assert len(VMAPPlist) == len(VMCOMBlist)
    # 记录空列表的位置
    emptylist_indices = [index for index, sublist in enumerate(PMCOMBlist) if not sublist]
    # 删除空列表
    PMCOMBlist = [sublist for sublist in PMCOMBlist if sublist]

    extvm_df_test = datapre.get_vm_ext_data(VMAPPlist,VMCOMBlist,[0])
    # print(extvm_df_test.shape)
    # print(extvm_df_test)
    deepsetvms_df_test, deepsetagg_df_test  = datapre.get_combpm_vms_agg_data(PMCOMBlist,[0])
    # print(len(deepsetvms_df_test),deepsetagg_df_test.shape)
    extVMfeature_df_test,deepset_list_test,baseline_agg_df_test,y_list_test = datapre.get_vm2pm(extvm_df_test,deepsetvms_df_test,deepsetagg_df_test)
    norm_extVMfeature_test, norm_deepsetx_test ,norm_deepsetagg_test  = datapre.normalpad_testdata(extVMfeature_df_test ,deepset_list_test ,baseline_agg_df_test )
    
    y_list_test = np.array(y_list_test)
    # print(norm_extVMfeature_test.shape,norm_deepsetx_test.shape,norm_deepsetagg_test.shape,y_list_test.shape)
    norm_extVMfeature_test = datapre.myreshape(norm_extVMfeature_test,len(VMAPPlist))
    norm_deepsetx_test = datapre.myreshape(norm_deepsetx_test,len(VMAPPlist))
    norm_deepsetagg_test = datapre.myreshape(norm_deepsetagg_test,len(VMAPPlist))
    y_list_test = datapre.myreshape(y_list_test,len(VMAPPlist))
    # print(norm_extVMfeature_test.shape,norm_deepsetx_test.shape,norm_deepsetagg_test.shape,y_list_test.shape)

    val_data = { 
    'VMfeature': torch.tensor(norm_extVMfeature_test, dtype=torch.float32),
    'deepsetx': torch.tensor(norm_deepsetx_test, dtype=torch.float32),
    'deepsetagg': torch.tensor(norm_deepsetagg_test, dtype=torch.float32),
    'target': torch.tensor(y_list_test, dtype=torch.float32)
    }

    # load model and predict
    loaded_model = Trace()
    # loaded_model.load_state_dict(torch.load(modelpath,weights_only=True))
    loaded_model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    loaded_model.eval()

    batch_shape = val_data['target'].shape
    with torch.no_grad():
        for key, data in val_data.items():
            if key == 'target':  continue
            val_data[key] = change_shape(data)
        VMfeature = val_data['VMfeature']
        deepsetx = val_data['deepsetx']
        deepsetagg = val_data['deepsetagg']
        target = val_data['target']

        output = loaded_model(VMfeature, deepsetx, deepsetagg)
        
        output = return_shape(output, target.shape)
        if output.dim() == 3 and output.size(-1) == 1:
            output = output.squeeze(-1)

    return output

if __name__ == "__main__":
    VMAPPlist = ["memcache","redis","nginx","keydb","mysql"]
    VMCOMBlist = [
        ["memcache","redis","nginx"],
        ["redis","keydb"],
        ["nginx","memcache","mysql"],
        ["keydb","nginx","mysql"],
        ["mysql","keydb"]
        ]
    PMCOMBlist = []
    for num in range(1,Conf.MAXRATIO):
        for comb in combinations_with_replacement(Conf.APPLIST,num):
            PMCOMBlist.append(comb)
    # PMCOMBlist.append([])
    print(tracepred(VMAPPlist,VMCOMBlist,PMCOMBlist))