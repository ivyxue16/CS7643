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


def data_transform_embeds(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)              # 23991
    num = len_record - n_his - n_pred   # 23991 - 12 - 3
    
    x = np.zeros([num, 1, n_his, n_vertex])   # (23976, 1, 12, 207)
    y = np.zeros([num, n_vertex])             # (23976, 207)

    data_after_embeds = embeds(data, time_of_hour=True, time_of_day = True, time_of_week = True)
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data_after_embeds[head: tail].reshape(1, n_his, n_vertex)    # (1,12,207)
        y[i] = data[tail + n_pred - 1]                                  # (207,)

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)



def embeds(data, time_of_hour=True, time_of_day = True, time_of_week = True):
    ts_index = np.arange(data.shape[0])

    if time_of_hour:
        time_of_hour_size = 12          # 12
        ts_hour = ts_index % time_of_hour_size
        ts_hour_sin = np.sin(2* np.pi * ts_hour / ts_hour.max())
        ts_hour_cos = np.cos(2* np.pi * ts_hour / ts_hour.max())

    if time_of_day:
        time_of_day_size = 12 * 24      # 288
        ts_day = ts_index % time_of_day_size
        ts_day_sin = np.sin(2* np.pi * ts_day / ts_day.max())
        ts_day_cos = np.cos(2* np.pi * ts_day / ts_day.max())

    if time_of_week:
        time_of_week_size = 12* 24 * 7  # 2016
        ts_week = ts_index % time_of_week_size
        ts_week_sin = np.sin(2* np.pi * ts_week / ts_week.max())
        ts_week_cos = np.cos(2* np.pi * ts_week / ts_week.max())


    if time_of_hour and time_of_day and time_of_week:
        temporal_embeds = np.vstack([ts_hour_sin, ts_hour_cos,ts_day_sin,ts_day_cos,ts_week_sin,ts_week_cos]).T

    elif time_of_hour and (not time_of_day) and (not time_of_week):
        temporal_embeds = np.vstack([ts_hour_sin, ts_hour_cos]).T
    
    elif time_of_day and (not time_of_hour) and (not time_of_week):
        temporal_embeds = np.vstack([ts_day_sin,ts_day_cos]).T

    elif time_of_week and (not time_of_hour) and (not time_of_day):
        temporal_embeds = np.vstack([ts_week_sin,ts_week_cos]).T

    temporal_embeds_sum = temporal_embeds.sum(axis=1)
    data_after_embeds = np.repeat([temporal_embeds_sum], data.shape[1], axis=0).T + data


    return data_after_embeds