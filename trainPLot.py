
import os
import torch
from matplotlib import pyplot as plt

from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from dataloader import load_data


Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=30)
parser.add_argument("--tune-epochs",default=10)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=16)
parser.add_argument("--temperatureH", default=0.5)#0.5,1.0
parser.add_argument("--temperatureL", default=1)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 根据数据集设置种子和对比训练的 epochs
if args.dataset == "MNIST-USPS":
    args.con_epochs = 10
    seed = 10
if args.dataset == "NoisyMNIST":
    args.con_epochs = 10#50
    seed = 10
elif args.dataset == "BDGP":
    args.con_epochs = 20
    seed = 30
elif args.dataset == "CCV":
    args.con_epochs = 50
    seed = 100
    args.tune_epochs = 200
elif args.dataset == "Fashion":
    args.con_epochs = 50
    seed = 10
elif args.dataset == "Caltech-2V":
    args.con_epochs = 100
    seed = 200
    args.tune_epochs = 40
elif args.dataset == "Caltech-3V":
    args.con_epochs = 100
    seed = 30
elif args.dataset == "Caltech-4V":
    args.con_epochs = 100
    seed = 100
elif args.dataset == "Caltech-5V":
    args.con_epochs = 100
    seed = 1000000
elif args.dataset == "Cifar10":
    args.con_epochs = 10
    seed = 10
elif args.dataset == "Cifar100":
    args.con_epochs = 20
    seed = 10
elif args.dataset == "Prokaryotic":
    args.con_epochs = 20
    seed = 10000
elif args.dataset == "Synthetic3d":
    args.con_epochs = 100
    seed = 100
elif args.dataset == "Hdigit":
    args.con_epochs = 100
    seed = 10
elif args.dataset == "Caltech101_20":
    args.con_epochs = 100
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

# 加载数据集
dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)

# 计算视图权重
def compute_view_value(rs, H, view):
    N = H.shape[0]
    w = []
    global_sim = torch.matmul(H, H.t())

    # 替换 NaN 值
    global_sim = torch.nan_to_num(global_sim, nan=0.0)

    for v in range(view):
        view_sim = torch.matmul(rs[v], rs[v].t())
        related_sim = torch.matmul(rs[v], H.t())

        # 替换 NaN 值
        view_sim = torch.nan_to_num(view_sim, nan=0.0)
        related_sim = torch.nan_to_num(related_sim, nan=0.0)

        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N * N)
        w.append(torch.exp(-w_v))
    w = torch.stack(w)
    w = w / torch.sum(w)
    return w.squeeze()

# 预训练函数
def pretrain(epoch):
    tot_loss = 0
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss = criterion(xs[v], xrs[v])
            loss = torch.nan_to_num(loss, nan=0.0)  # 替换 NaN 值
            loss_list.append(loss)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
def tune_train(epoch):
    tot_loss = 0
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, rs, H = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1,view):
                loss_list.append(criterion(rs[v],rs[w]))
            loss_list.append(mes(xs[v],xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
# 对比训练函数
def contrastive_train(epoch):
    tot_loss = 0
    mse = torch.nn.MSELoss()

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, rs, H = model(xs)

        # 替换 H 中的 NaN 值
        H = torch.nan_to_num(H, nan=0.0)

        loss_list = []
        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        for v in range(view):
            for wid in range(v+1,view):
                loss_list.append(criterion(rs[v],rs[wid]))
            contrastive_loss_value = criterion(rs[v], H, w[v])
            contrastive_loss_value = torch.nan_to_num(contrastive_loss_value, nan=0.0)  # 替换 NaN 值
            loss_list.append(contrastive_loss_value)
            mse_loss_value = mse(xs[v], xrs[v])
            mse_loss_value = torch.nan_to_num(mse_loss_value, nan=0.0)  # 替换 NaN 值
            loss_list.append(mse_loss_value)

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

# 主训练循环
accs = []
nmis = []
purs = []

# 主训练循环
if not os.path.exists('./models'):
    os.makedirs('./models')
if not os.path.exists('./loss_curves'):
    os.makedirs('./loss_curves')
if not os.path.exists('./metric_curves'):
    os.makedirs('./metric_curves')

T = 1
for i in range(T):
    print("ROUND: {}".format(i + 1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterionh = ContrastiveLoss(args.batch_size, args.temperatureH, device)
    criterion = ContrastiveLoss(args.batch_size, args.temperatureL, device)
    best_acc, best_nmi, best_pur = 0, 0, 0
    epoch = 1

    # 预训练阶段
    while epoch <= args.pre_epochs:
        pretrain(epoch)
        epoch += 1

    # 微调阶段
    while epoch <= args.pre_epochs + args.tune_epochs:
        tune_train(epoch)
        if epoch % 1 == 0:
            acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
            accs.append(acc)
            nmis.append(nmi)
            purs.append(pur)
        epoch += 1

    # 对比训练阶段
    while epoch <= args.pre_epochs + args.tune_epochs + args.con_epochs:
        contrastive_train(epoch)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
        accs.append(acc)
        nmis.append(nmi)
        purs.append(pur)

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')

        epoch += 1

    print('The best clustering performance: ACC = {:.4f} NMI = {:.4f} PUR = {:.4f}'.format(best_acc, best_nmi, best_pur))

