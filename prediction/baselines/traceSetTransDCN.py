import Conf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def np_to_tensor(input_array):
    """
    将输入转换为 PyTorch Tensor，并统一为 torch.float32 类型。
    如果输入是 ndarray，则转换为 Tensor 并设为 float32。
    如果输入已经是 Tensor，则直接转换为 float32。
    """
    if isinstance(input_array, np.ndarray):
        return torch.tensor(input_array, dtype=torch.float32)
    elif isinstance(input_array, torch.Tensor):
        return input_array.float()  # 确保转换为 float32
    else:
        raise TypeError("输入必须是 numpy.ndarray 或 torch.Tensor")


class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0, x):
        # x0: shape (batch_size, d), 原始输入
        # x: shape (batch_size, d), 当前输入
        # 对每个样本计算 x * w 的点积，结果形状为 (batch_size, 1)
        dot = torch.sum(x * self.weight, dim=1, keepdim=True)  # (batch_size, 1)
        # x0 * dot 会自动广播到 (batch_size, d)
        out = x0 * dot + self.bias + x  # 保证 x0 * dot: (batch_size, d), bias: (d,), x: (batch_size, d)
        return out

class MLP(nn.Module):
    def __init__(self, output_dim=64,num_cross_layers = 2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(Conf.x_dim_num, 64)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, output_dim)

        self.cross_layers = nn.ModuleList([CrossLayer(Conf.x_dim_num) for _ in range(num_cross_layers)])
        self.output_layer = nn.Linear(Conf.x_dim_num+output_dim, output_dim)
        # self.quadratic1 = nn.Linear(Conf.x_dim_num, 64)
        # self.quadratic2 = nn.Linear(64, output_dim)

    def forward(self, VMfeature: torch.Tensor):  # Shape: (b, 155)
        x0 = VMfeature
        x_cross = VMfeature

        x = F.relu(self.fc1(VMfeature))
        x = self.dropout(x)
        x = self.fc2(x)

        for layer in self.cross_layers:
            # print("!!!!!!!!!!!!begin!!!!!!!!!!!")
            # print(x_cross.shape,x0.shape)
            x_cross = layer(x0,x_cross)
            
        # 合并交叉网络和深度网络的输出
        combined_output = torch.cat([x,x_cross], dim=-1)
        # 输出层
        x = F.relu(self.output_layer(combined_output))
        return x


class DeepSetModel(nn.Module):
    def __init__(self, output_dim=64):
        super(DeepSetModel, self).__init__()
        self.fc1 = nn.Linear(Conf.vm_dim_num, 128) 
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 128)   
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128 + Conf.agg_dim_num, 64)  
        self.dropout3 = nn.Dropout(0.25)    
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, deepsetx: torch.Tensor, deepsetagg: torch.Tensor):  # deepsetx (b, 2, 31),  deepsetagg (b, 124)
        non_zero_mask = (deepsetx.abs().sum(dim=2) != 0)  # (b, 2)  # 计算非全零掩码

        x = torch.tanh(self.fc1(deepsetx))  
        x = self.dropout1(x)
        x = torch.tanh(self.fc2(x)) 
        x = self.dropout2(x)
        # 对非零项求和    
        non_zero_mask_expanded = non_zero_mask.unsqueeze(-1).expand_as(x)   
        x = x * non_zero_mask_expanded
        x = torch.sum(x, dim=1)  # (b, 128)

        x = F.relu(self.fc3( torch.cat((x, deepsetagg), dim=1) ))  # 拼接后 (b, 128 + agg_dim_num)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


from baselines.modules import SAB, PMA
class SmallSetTransformer(nn.Module):
    def __init__(self, dim_input,output_dim,num_cross_layers=2):
        super(SmallSetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=dim_input, dim_out=128, num_heads=4),
            SAB(dim_in=128, dim_out=128, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=128, num_heads=4, num_seeds=1),
            nn.Linear(in_features=128, out_features=128),
        )
        self.fc3 = nn.Linear(128 + Conf.agg_dim_num, 64)  
        self.dropout3 = nn.Dropout(0.25)    
        self.fc4 = nn.Linear(64, output_dim)

        self.cross_layers = nn.ModuleList([CrossLayer(128 + Conf.agg_dim_num) for _ in range(num_cross_layers)])
        self.output_layer = nn.Linear(128 + Conf.agg_dim_num+output_dim, output_dim)

    def forward(self, x ,setagg):
        # x: shape (batch, set_size, dim_input)
        x = self.enc(x)      # 输出形状: (batch, set_size, 64)
        x = self.dec(x)      # 输出形状: (batch, 1, 128)
        x = x.squeeze(1)  # 输出形状: (batch, 128)


        x_concat =  torch.cat((x, setagg), dim=1)
        x = F.relu(self.fc3(x_concat))  # 拼接后 (b, 128 + agg_dim_num)
        x = self.dropout3(x)
        x = self.fc4(x)

        x_cross = x_concat
        x0 = x_concat
        for layer in self.cross_layers:
            # print("!!!!!!!!!!!!begin!!!!!!!!!!!")
            # print(x_cross.shape,x0.shape)
            x_cross = layer(x0, x_cross)
        # 合并交叉网络和深度网络的输出
        combined_output = torch.cat([x, x_cross], dim=-1)
        # 输出层
        x = F.relu(self.output_layer(combined_output))
        return x


class Trace(nn.Module):
    def __init__(self, output_dim=64):
        super(Trace, self).__init__()
        self.mlp = MLP(output_dim)
        self.settrans = SmallSetTransformer(Conf.vm_dim_num,output_dim)

    
    def forward(self, VMfeature, deepsetx, deepsetagg):  # shape: (batch size, 特征 dim)
        VMfeature = np_to_tensor(VMfeature)
        deepsetx = np_to_tensor(deepsetx)
        deepsetagg = np_to_tensor(deepsetagg)
        y_mlp = self.mlp(VMfeature)
        y_deepset = self.settrans(deepsetx, deepsetagg)
        cos_similarity = F.cosine_similarity(y_mlp, y_deepset, dim=1, eps=1e-8).unsqueeze(1)
        return cos_similarity

    def forward_fast(self, VMfeature, deepsetx, deepsetagg):  # VMfeature batch size 与 deepset 不同
        VMfeature = np_to_tensor(VMfeature)
        deepsetx = np_to_tensor(deepsetx)
        deepsetagg = np_to_tensor(deepsetagg)
        y_mlp = self.mlp(VMfeature)
        y_deepset = self.settrans(deepsetx, deepsetagg)

        similarity_matrix = torch.mm(y_mlp, y_deepset.T)  # (batch_mlp, batch_deepset)
        y_mlp_norm = torch.norm(y_mlp, dim=1, keepdim=True)  # (batch_mlp, 1)
        y_deepset_norm = torch.norm(y_deepset, dim=1, keepdim=True)  # (batch_deepset, 1)
        cosine_similarity_matrix = similarity_matrix / (y_mlp_norm * y_deepset_norm.T + 1e-8)  # (batch_mlp, batch_deepset)

        return cosine_similarity_matrix