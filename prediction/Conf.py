alpha = 1.0
dataset = "4"
dirname = f"../IWQOS/VMpred4mig/data/dataprocessed{dataset}"
resultSavePath = f"./{dataset}results"
# alpha * NDCG + (1-alpha)*mse

logdir = f"./log/log{dataset}"

SEED = "0"

# PM指标/agg指标 维度
comb_dim_begin = 31
comb_dim_end = 154
# 31 vm 124agg
vm_dim_num = 31
agg_dim_num = 124
x_dim_num = vm_dim_num + agg_dim_num
deepset_dim_num = 31

MAXRATIO = 3
maskratio = 0.5