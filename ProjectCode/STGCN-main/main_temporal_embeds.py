import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import matplotlib.pyplot as plt

from script import dataloader, utility, earlystopping
from model import models

#import nni

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='metr-la', choices=['metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=2)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0005, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2, help='epochs, default as 100')
    # parser.add_argument('--opt', type=str, default='rmsprop', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):  # 2
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    return args, device, blocks

def data_preparate(args, device):    
    adj, n_vertex = dataloader.load_adj(args.dataset)       # n_vertex = 207
    gso = utility.calc_gso(adj, args.gso_type)              # 207 x 207
    if args.graph_conv_type == 'cheb_graph_conv':           
        gso = utility.calc_chebynet_gso(gso)                
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]  # 34271
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    val_and_test_rate = 0.15

    len_val = int(math.floor(data_col * val_and_test_rate))         # 5140      0.15
    len_test = int(math.floor(data_col * val_and_test_rate))        # 5140      0.15
    len_train = int(data_col - len_val - len_test)                  # 23991     0.7

    train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform_embeds(train, args.n_his, args.n_pred, device)    # torch.Size([23976, 1, 12, 207]), torch.Size([23976, 207])
    x_val, y_val = dataloader.data_transform_embeds(val, args.n_his, args.n_pred, device)          # torch.Size([5125, 1, 12, 207]), torch.Size([5125, 207])
    x_test, y_test = dataloader.data_transform_embeds(test, args.n_his, args.n_pred, device)       # torch.Size([5125, 1, 12, 207]), torch.Size([5125, 207])

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter

def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler

def train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter):
    train_loss_list = []
    val_loss_list = []
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            # x: (N,1,T,nodes)     torch.Size([32, 1, 12, 207])  
            # y: (N,nodes)     torch.Size([32, 207])
            y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        train_loss_list.append(l_sum / n)
        val_loss_list.append(val_loss)
        
        if es.step(val_loss):
            print('Early stopping.')
            break
    return train_loss_list, val_loss_list


def plot_curves(train_loss_history, val_loss_history):

    epochs = list(range(1, len(train_loss_history) + 1))
    fig, ax = plt.subplots(figsize=(16, 8)) 

    plt.plot(epochs, train_loss_history, marker='o', linestyle='-', colo11r='green', label='Train')
    plt.plot(epochs, val_loss_history, marker='o', linestyle='-', color='red', label='Validation')

    plt.title("Loss (Mean Squared Error)")         
    plt.xlabel("Epoch")
    plt.xticks(np.arange(1, len(epochs) + 1, step=1))
    plt.legend(loc='upper right')
    # plt.legend(["Train", "Validation"])
    # if args.middle_layer == True:
    #     plt.savefig(str(args.stblock_num) + " " + str(args.Ks)  +'.png')
    # else:
    plt.savefig('./figure/Default.png')

@torch.no_grad()
def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad() 
def test(zscore, loss, model, test_iter, args):
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f} ')
    return test_MSE,test_MAE, test_RMSE, test_WMAPE




def plot_losses(train_losses, val_losses, lr, Ks, weight_decay_rate, n_his, opt, batch_size):
    
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(16, 8)) 

    plt.plot(epochs, train_losses, marker='o', linestyle='-', color='green', label='Train')
    plt.plot(epochs, val_losses, marker='o', linestyle='-', color='red', label='Validation')

    plt.xticks(np.arange(1, len(epochs) + 1, step=1), rotation=90)

    plt.title('Loss(Mean Squared Error)')

    plt.xlabel('Epoch')

    plt.legend(loc='upper right')
    
    fname = './figure/hyper/Embeds_ + ' + str(val_loss_history[-1]) + " lr: " + str(lr) + " Ks: " + str(Ks) + " weight_decay_rate: " + str(weight_decay_rate) + " n_his: " + str(n_his) + "opt: " + opt + 'batch_size: ' + str(batch_size)


    # plt.savefig('./figure/STGCN Learning Curves-Embeds + ' +  + '.png')
    plt.savefig(fname + '.png')
    


    # plt.show()


# def plot_curves(train_loss_history, valid_loss_history):
#     """
#     Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
#     training accuracy and validation accuracy are plot in the same figure too.
#     :param train_loss_history: training loss history of epochs
#     :param valid_loss_history: validation loss history of epochs
#     :return: None, save two figures in the current directory
#     """

#     # print(train_acc_history)
#     # print(valid_acc_history)

#     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))

#     x1 = list(range(len(train_loss_history)))
#     axes.plot(x1,train_loss_history, label="train",)
#     axes.plot(x1,valid_loss_history, label="valid")
#     # axes[0].plot(x1,train_loss_history, "--bo", label="train",)
#     # axes[0].plot(x1,valid_loss_history, "--gD",label="valid")
#     axes.set_title(str("Learning_Curve: Loss, Train Loss: {train_loss: .4f}".format(train_loss=train_loss_history[-1])) + str(" Val Loss:  {valid_loss: .4f}".format(valid_loss=valid_loss_history[-1])))
#     axes.set(xlabel="Epochs")
#     axes.legend(loc="upper right")


#     # x2 = list(range(len(train_acc_history)))
#     # axes[1].plot(x2,train_acc_history, label="train",)
#     # axes[1].plot(x2,valid_acc_history, label="valid")

#     # # axes[1].plot(x2,train_acc_history, "--bo", label="train")
#     # # axes[1].plot(x2,valid_acc_history, "--gD", label="valid")
#     # axes[1].set_title(str("Learning_Curve: Accuracy, Train Acc: {train_acc: .4f}".format(train_acc=train_acc_history[-1])) + str(" Val Acc:  {valid_acc: .4f}".format(valid_acc=valid_acc_history[-1])))
#     # axes[1].set(xlabel="Epochs")
#     # axes[1].legend(loc="lower right")
    

#     epochs = len(train_loss_history)
#     filename = "./figure/Epoch: " + str(epochs) + ", Learning Curve" + ".png"
    
#     fig.savefig(filename)




if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    args, device, blocks = get_parameters()
    for batch_size in [32,64]:
        args.batch_size = batch_size
        for lr in [0.001, 0.0015, 0.002, 0.03, 0.01]:   
            args.lr = lr
            for Ks in [2,3]:
                args.Ks = Ks
                for weight_decay_rate in [0.0005, 0.00075, 0.001]:
                    args.weight_decay_rate = weight_decay_rate
                    for n_his in [15,9,18,21,24]:
                        args.his = n_his
                        n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)    # n_vertex = 207, 
                        for opt in ['adam','rmsprop', 'adamw']:
                            args.opt = opt
                            # print(args)
                            loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)
                            train_loss_history, val_loss_history = train(loss, args, optimizer, scheduler, es, model, train_iter, val_iter)
                            val_loss_history = [float(x) for x in val_loss_history]
                            test_MSE,test_MAE, test_RMSE, test_WMAPE = test(zscore, loss, model, test_iter, args)
                            # print(train_loss_history)
                            # print(val_loss_history)
            
                            fname = './figure/hyper/Embeds_ + ' + str(val_loss_history[-1]) + " lr: " + str(lr) + " Ks: " + str(Ks) + " weight_decay_rate: " + str(weight_decay_rate) + " n_his: " + str(n_his) + "opt: " + opt + 'batch_size: ' + str(batch_size)


                            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                            
                            
                            c = []
                            c.append(str('Epoch: 100 | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.format(lr, train_loss_history[-1], val_loss_history[-1],gpu_mem_alloc)) + '\n')
                            
                            c.append(str(f'Dataset {args.dataset:s} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f} '))
                            
                            with open (fname+'.txt' , "w") as f:
                                f.write(str(c))
                            plot_losses(train_loss_history, val_loss_history, lr, Ks, weight_decay_rate, n_his, opt,batch_size)