import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch


class DataLoader(object):

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()


class StandardScaler():

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        mask = (data == 0)
        data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    adj = sp.coo_matrix(adj)            # (207,207)
    rowsum = np.array(adj.sum(1))       # (207,1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.   # (207,)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # (207,207)
    # https://github.com/tkipf/gcn/issues/142
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()    # (207,207)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    try:
        _, _, adj_mx = load_pickle(pkl_filename)
    except:
        adj_mx = load_pickle(pkl_filename)
    adj = [sym_adj(adj_mx), sym_adj(np.transpose(adj_mx))]  # (207,207)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']: # 7:1:2
        # train: (23974, 12, 207, 2) 7
        # val: (3425, 12, 207, 2) 1
        # test: (6850, 12, 207, 2) 2
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz')) 
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    torch.Tensor(data['x_train'])

    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_hadj(pkl_filename,top_k):
    try:
        _, _, adj_mx = load_pickle(pkl_filename)
    except:
        adj_mx = load_pickle(pkl_filename)

    hadj = adj_mx                                     # (207,207)

    top = top_k

    hadj = hadj - np.identity(hadj.shape[0])          # (207,207)
    hadj = torch.from_numpy(hadj.astype(np.float32))  # (207,207)
    # 参数K，当graph转化为hypergraph时，因为边的数量太多，会导致计算量过大，
    # 所以，只取每个节点的top-K条边作为hypergraph的数据
    _, idx= torch.topk(hadj, top, dim=0)                # (4,207)
    _, idy = torch.topk(hadj, top, dim=1)             # (207,4)

    base_mx_lie = torch.zeros([hadj.shape[0], hadj.shape[1]])     # (207,207)
    for i in range(hadj.shape[0]):
        base_mx_lie[idx[:, i], i] = hadj[idx[:, i], i]
    base_mx_hang = torch.zeros([hadj.shape[0], hadj.shape[1]])    # (207,207)
    for j in range(hadj.shape[0]):
        base_mx_hang[j, idy[j, :]] = hadj[j, idy[j, :]]

    base_mx = torch.where(base_mx_lie != 0, base_mx_lie, base_mx_hang)   # (207,207)

    hadj = base_mx + torch.eye(hadj.shape[0])           # (207,207)
    hadj = hadj.numpy()

    # node_nums: 207
    n = hadj.shape[0] 
    # edge_nums: 1083
    l = int((len(np.nonzero(hadj)[0])))
    H = np.zeros((l, n))                              # (1083, 207)
    H_a = np.zeros((l, n))                            # (1083, 207)
    H_b = np.zeros((l, n))                            # (1083, 207)
    #  the static features of the edges
    lwjl = np.zeros((l,1))                            # (1083, 1)
    a=0

    for i in range(hadj.shape[0]):  # 207
        for j in range(hadj.shape[1]):  # 207
            if(hadj[i][j]!=0.0):
                H[a, i] = 1.0
                H[a, j] = 1.0
                H_a[a, i] = 1.0
                H_b[a, j] = 1.0
                if(i == j):
                    lwjl[a, 0] = 1.0
                else:
                    lwjl[a, 0] = adj_mx[i,j]
                a = a + 1

    lwjl = 1.0-lwjl                                 # (1083, 1)

    W = np.ones(n)                                  # (207,)

    DV = np.sum(H * W, axis=1)                      # (1083, )
    DE = np.sum(H, axis=0)                          # (207,)
    DE_= np.power(DE, -1)                           # (207,)
    DE_[np.isinf(DE_)] = 0.
    invDE = np.mat(np.diag(DE_))                    # (207,207)
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))       # (1083, 1083)
    W = np.mat(np.diag(W))                          # (207,)
    H = np.mat(H)                                   # (1083, 207)
    HT = H.T                                        # (207, 1083)

    HT = sp.coo_matrix(HT)                          # (207, 1083)
    rowsum = np.array(HT.sum(1)).flatten()          # (207,)
    d_inv = np.power(rowsum, -1).flatten()          
    d_inv[np.isinf(d_inv)] = 0.                     # (207,207)
    d_mat = sp.diags(d_inv)                         # (207,207)
    H_T_new = d_mat.dot(HT).astype(np.float32).todense() # (207, 1083)

    G0 = DV2 * H                                    # (1083, 207)
    G1 = invDE * HT * DV2                           # (207, 1083)

    # ========================================================
    n = adj_mx.shape[0]            # 207                 
    l = int((len(np.nonzero(adj_mx)[0])))  # 1083         
    H_all = np.zeros((l, n))                        # (1072, 207)
    edge_1 = np.array([])
    edge_2 = np.array([])
    a=0

    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):
            if(adj_mx[i][j]!=0.0):
                H_all[a, i] = 1.0
                H_all[a, j] = 1.0
                edge_1 = np.hstack((edge_1, np.array([i])))  # (1722, 1722)
                edge_2 = np.hstack((edge_2, np.array([j])))  # (1722, 1722)
                a = a + 1

    W_all = np.ones(n)                              # (207,)
    DV_all = np.sum(H_all * W_all, axis=1)          # (1722, )
    DE_all = np.sum(H_all, axis=0)                  # (207,)

    DE__all=np.power(DE_all, -1)
    DE__all[np.isinf(DE__all)] = 0.
    invDE_all = np.mat(np.diag(DE__all))
    DV2_all = np.mat(np.diag(np.power(DV_all, -0.5)))
    W_all = np.mat(np.diag(W_all))
    H_all = np.mat(H_all)
    HT_all = H_all.T

    HT_all = sp.coo_matrix(HT_all)
    rowsum_all = np.array(HT_all.sum(1)).flatten()
    d_inv_all = np.power(rowsum_all, -1).flatten()
    d_inv_all[np.isinf(d_inv_all)] = 0.
    d_mat_all = sp.diags(d_inv_all)
    H_T_new_all = d_mat_all.dot(HT_all).astype(np.float32).todense()

    G0_all = DV2_all * H_all
    G1_all = invDE_all * HT_all * DV2_all

    coo_hadj = adj_mx - np.identity(n)
    coo_hadj = sp.coo_matrix(coo_hadj)
    coo_hadj = coo_hadj.tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((edge_1, edge_2)).astype(np.int64))

    G0 = G0.astype(np.float32)
    G1 = G1.astype(np.float32)
    H = H.astype(np.float32)
    HT = H.T.astype(np.float32)
    H_T_new = torch.from_numpy(H_T_new.astype(np.float32))
    H_a = torch.from_numpy(H_a.astype(np.float32))
    H_b = torch.from_numpy(H_b.astype(np.float32))
    lwjl = torch.from_numpy(lwjl.astype(np.float32))

    G0_all = G0_all.astype(np.float32)
    G1_all = G1_all.astype(np.float32)

    return H_a, H_b, HT, lwjl, G0,G1, indices,G0_all,G1_all


def feature_node_to_edge(feature_node,H_a,H_b,operation="concat"):
    feature_edge_a = torch.einsum('ncvl,wv->ncwl', (feature_node, H_a))
    feature_edge_b = torch.einsum('ncvl,wv->ncwl', (feature_node, H_b))
    if operation == "concat":
        feature_edge = torch.cat([feature_edge_a, feature_edge_b], dim=1)
    elif  operation == "sum":
        feature_edge = feature_edge_a + feature_edge_b
    elif operation == "subtract":
        feature_edge = feature_edge_a - feature_edge_b
    return feature_edge


def fusion_edge_node(x, x_h, H_T_new):
    x_h_new = torch.einsum('ncvl,wv->ncwl', (x_h, H_T_new))
    x = torch.cat([x, x_h_new], dim=1)
    return x
