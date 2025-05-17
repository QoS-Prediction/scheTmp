# %%
import os
import argparse
import logging
import pickle
import numpy as np
np.random.seed(0)
import torch
# # set torch random seed so that the result is reproducible
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mymodel import *
from tqdm import tqdm
from utils import save_finalresult
import Conf
from utils import approxNDCGLoss
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
# %%
# setting up logger
logfile = "./log/FEDGE.log"
logger = logging.getLogger("FEDGE logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile, "w")
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_train_data(dirname):
    ''' return dict(VMfeature, deepsetx, deepsetagg, y_list) '''
    with open(f'{dirname}/norm_extVMfeature_train.pkl', 'rb') as f:
        VMfeature_train = pickle.load(f)
    with open(f'{dirname}/norm_deepsetx_train.pkl', 'rb') as f:
        deepsetx_train = pickle.load(f)
    with open(f'{dirname}/norm_deepsetagg_train.pkl', 'rb') as f:
        deepsetagg_train = pickle.load(f)
    with open(f'{dirname}/y_list_train.pkl', 'rb') as f:
        y_list_train = pickle.load(f)
    train_data = { 
        'VMfeature': torch.tensor(VMfeature_train, dtype=torch.float32),
        'deepsetx': torch.tensor(deepsetx_train, dtype=torch.float32),
        'deepsetagg': torch.tensor(deepsetagg_train, dtype=torch.float32),
        'target': torch.tensor(y_list_train, dtype=torch.float32)
    }
    return train_data

def get_test_data(dirname):
    ''' return dict(VMfeature, deepsetx, deepsetagg, y_list) '''
    with open(f'{dirname}/norm_extVMfeature_test.pkl', 'rb') as f:
        VMfeature_test = pickle.load(f)
    with open(f'{dirname}/norm_deepsetx_test.pkl', 'rb') as f:
        deepsetx_test = pickle.load(f)
    with open(f'{dirname}/norm_deepsetagg_test.pkl', 'rb') as f:
        deepsetagg_test = pickle.load(f)
    with open(f'{dirname}/y_list_test.pkl', 'rb') as f:
        y_list_test = pickle.load(f)
    test_data = { 
        'VMfeature': torch.tensor(VMfeature_test, dtype=torch.float32),
        'deepsetx': torch.tensor(deepsetx_test, dtype=torch.float32),
        'deepsetagg': torch.tensor(deepsetagg_test, dtype=torch.float32),
        'target': torch.tensor(y_list_test, dtype=torch.float32)
    }
    return test_data

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# %%
batch_size = 256
test_batch_size = 4096
epoches = 400
ae_lr = 0.001
dis_lr = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

dirname = Conf.dirname
train_data_raw = get_train_data(dirname)
app_dim = train_data_raw["VMfeature"].shape[2] # 原始特征数
noapp_dim = train_data_raw["deepsetagg"].shape[2] # 聚合特征数
pm_dim_train = train_data_raw["VMfeature"].shape[1] #待迁移pm数量
train_data_x1 = train_data_raw["VMfeature"]
train_data_x2 = train_data_raw["deepsetagg"] # [448, 80, 124]
train_data_y = train_data_raw["target"]
# ([448, 80, 155]) 448种迁移前VM环境，80种迁移主机环境，31(监测指标)+124(聚合指标)
test_data_raw = get_test_data(dirname)
pm_dim_test = test_data_raw["VMfeature"].shape[1]
test_data_x1 = test_data_raw["VMfeature"]
test_data_x2 = test_data_raw["deepsetagg"]
test_data_y = test_data_raw["target"]
# ([112, 80, 155])
batch_size = batch_size // pm_dim_train
test_batch_size = test_batch_size // pm_dim_test # Why?

class fedgeDataset(Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y
    def __len__(self):
        return self.x1.shape[0]
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

fedgeTrainDS = fedgeDataset(train_data_x1, train_data_x2, train_data_y)
fedgeTrainDS, fedgeValDS = torch.utils.data.random_split(fedgeTrainDS, [0.9, 0.1])
fedgeTestDS = fedgeDataset(test_data_x1, test_data_x2, test_data_y)
fedgeTrainDataLoader = DataLoader(fedgeTrainDS, batch_size, shuffle=True, drop_last=False) # batch size here?
fedgeValDataLoader = DataLoader(fedgeValDS, batch_size, shuffle=True, drop_last=False)
fedgeTestDataLoader = DataLoader(fedgeTestDS, test_batch_size, drop_last=False)
# %%
lambda_0 = 0.5
# lambda_1 = 1
lambda_2 = 0.1


fedge = FEDGE(app_dim, noapp_dim).to(device)

Q_optimizer = optim.Adam(fedge.Q.parameters(), lr=ae_lr)
P_optimizer = optim.Adam(fedge.P.parameters(), lr=ae_lr)
R_optimizer = optim.Adam(fedge.R.parameters(), lr=ae_lr)

D_optimizer = optim.Adam(fedge.D.parameters(), lr=dis_lr)

# MMD_loss = MMD_multidis_loss(7)
# %%
def train(model):
    # def train(Q, P, R, D, train_loaders, MMD_loss, Q_optimizer, P_optimizer, R_optimizer, D_optimizer):
    # train
    model.train()
    for batch, (app_data, noapp_data, labels) in tqdm(enumerate(fedgeTrainDataLoader), desc="Train fedge"):

        app_data = app_data.view(-1, app_dim).to(device)
        noapp_data = noapp_data.view(-1, noapp_dim).to(device)
        labels = labels.to(device)
        labelsFlat = labels.view(-1, 1).to(device)
        # reconstruction and adversarial minimization
        code = model.Q(app_data)
        re_data = model.P(code)
        D_fake = model.D(code)

        R_input = torch.cat([code, noapp_data], dim=1)
        output = model.R(R_input).view(-1, pm_dim_train)

        # L_MMD = MMD_loss(code, domain_labels)
        L_re = F.mse_loss(app_data, re_data)
        # L_re = F.mse_loss(nostress_data, re_data)
        L_qos = F.mse_loss(output, labels)

        fake_ones = torch.ones(D_fake.shape).to(device)
        # adversarial loss
        # make the generated code close to 1 to cheat discriminator
        L_adv_min = F.mse_loss(D_fake, fake_ones)

        total_loss = L_qos + lambda_0 * L_re + lambda_2 * L_adv_min
        logger.info("Batch {}, L_qos: {:.3f}, L_re: {:.3f}, L_adv_min: {:.3f}".format(batch, L_qos.item(), L_re.item(), L_adv_min.item()))
        model.Q.zero_grad()
        model.P.zero_grad()
        model.R.zero_grad()
        total_loss.backward()
        R_optimizer.step()
        P_optimizer.step()
        Q_optimizer.step()

        # adversarial
        code = code.detach()
        real_prior = torch.tensor(np.random.laplace(0, np.sqrt(2)/2, code.shape)).float().to(device)
        # different prior
        # real_prior = torch.tensor(np.random.normal(0, 0.1, code.shape)).float().to(device)
        # real_prior = torch.tensor(np.random.uniform(-0.1, 0.1, code.shape)).float().to(device)

        real_labels = torch.ones(real_prior.shape[0])
        fake_labels = torch.zeros(code.shape[0])
        datas = torch.cat([code, real_prior], dim=0).to(device)
        labels = torch.cat([real_labels, fake_labels], dim=0).unsqueeze(1).to(device)

        model.D.zero_grad()
        D_output = model.D(datas)
        # In the equation we want to maximize the GAN loss, 
        # so we can multiply a -1 to minimize the loss instead. 
        # The minimum of the loss is -1
        L_adv_max = -F.mse_loss(D_output, labels)
        logger.info("\tL_adv: {}".format(L_adv_max.item()))
        # writer.add_scalars("Loss", {"L_qos": L_qos.item(), "L_re": L_re.item(), "L_MMD": L_MMD.item(), "L_adv_min": L_adv_min.item(), "L_adv_max": L_adv_max.item()}, itr)
        # writer.add_scalars("Loss", {"L_qos": L_qos.item(), "L_re": L_re.item(), "L_adv_min": L_adv_min.item(), "L_adv_max": L_adv_max.item()}, itr)
        L_adv_max.backward()
        D_optimizer.step()

# %%
def val(model, criterion):
    model.eval()
    total_loss = 0.
    total_count = 0
    for app, noapp, label in tqdm(fedgeValDataLoader, desc="Val fedge"):
        app = app.view(-1, app_dim).to(device)
        noapp = noapp.view(-1, noapp_dim).to(device)
        labels = label.to(device)
        labelsFlat = labels.view(-1, 1).to(device)

        code = model.Q(app)
        R_input = torch.cat([code, noapp], dim=1)
        outputs = model.R(R_input).view(-1, pm_dim_train)

        loss = criterion(outputs, labels)
        total_loss += loss.item() * app.shape[0]
        total_count += app.shape[0]
    total_loss /= total_count
    return total_loss

def test(model, criterion):
    model.eval()
    total_loss = 0.
    total_count = 0
    for app, noapp, label in tqdm(fedgeTestDataLoader, desc="Test fedge"):
        app = app.view(-1, app_dim).to(device)
        noapp = noapp.view(-1, noapp_dim).to(device)
        labels = label.to(device)
        labelsFlat = labels.view(-1, 1).to(device)

        code = model.Q(app)
        R_input = torch.cat([code, noapp], dim=1)
        outputs = model.R(R_input).view(-1, pm_dim_test)

        loss = criterion(outputs, labels)
        total_loss += loss.item() * app.shape[0]
        total_count += app.shape[0]
    total_loss /= total_count
    return total_loss

# %%
model_path = "./model/fedge.pt"
mae_criterion = nn.L1Loss(reduction='mean')
mse_criterion = nn.MSELoss(reduction='mean')
es = EarlyStopping(patience=10, path=model_path, trace_func=logger.info, delta=-0.0001)

for epoch in range(epoches):
    print(f"Epoch: {epoch} / {epoches}")
    train(fedge)
    val_loss_ndcg = val(fedge, approxNDCGLoss)
    val_loss_mse = val(fedge, mse_criterion)
    es(val_loss_mse, fedge)
    if es.early_stop:
        logger.warning("Early stop! Val NDCG loss: {}".format(es.val_loss_min))
        break
    else:
        logger.info("\tVal NDCG loss: {}".format(val_loss_ndcg))
        logger.info("\tVal MSE loss: {}".format(val_loss_mse))

fedge.load_state_dict(torch.load(model_path)) # load训练好的参数
test_mae_loss = test(fedge, mae_criterion)
test_mse_loss = test(fedge, mse_criterion)
test_NDCG = -test(fedge, approxNDCGLoss)
logger.info(f"Test MAE loss: {test_mae_loss}, Test MSE loss: {test_mse_loss}, Test NDCG: {test_NDCG}")

# %%
save_finalresult("FEDGE", test_mse_loss, Conf.resultSavePath)