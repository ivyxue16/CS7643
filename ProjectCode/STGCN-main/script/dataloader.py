import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    
    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228

    return adj, n_vertex

def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]                        # (23991, 207)
    val = vel[len_train: len_train + len_val]       # (5140, 207)
    test = vel[len_train + len_val:]                # (5140, 207)
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)              # 23991
    num = len_record - n_his - n_pred   # 23991 - 12 - 3
    
    x = np.zeros([num, 1, n_his, n_vertex])   # (23976, 1, 12, 207)
    y = np.zeros([num, n_vertex])             # (23976, 207)
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)    # (1,12,207)
        y[i] = data[tail + n_pred - 1]                                  # (207,)

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)