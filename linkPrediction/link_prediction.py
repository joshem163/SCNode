import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import time
import multiprocessing
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid

import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T

path1 = '....' # path to data files
name = 'Cornell' # dataset name


# Assuming the GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BatchNormModel(nn.Module):
    def __init__(self, num_features):
        super(BatchNormModel, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # Apply batch normalization
        return self.batch_norm(x)

    def reset_parameters(self):
        # Reset the parameters of the batch normalization layer
        self.batch_norm.reset_running_stats()
        self.batch_norm.reset_parameters()


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.bn1 = nn.BatchNorm1d(in_channels)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        #x = self.bn1(x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.elu(x)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, pos_train_edge, neg_train_edge, optimizer, batch_size, x):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        h = model(x)

        edge = pos_train_edge[perm].t().to(device)
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = neg_train_edge[perm].t().to(device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, pos_valid_edge, neg_valid_edge, y_valid, pos_test_edge, neg_test_edge, y_test, batch_size, x):
    model.eval()
    h = model(x)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t().to(device)
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t().to(device)
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t().to(device)
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t().to(device)
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}

    # validation AUC
    valid_preds = torch.cat([pos_valid_pred, neg_valid_pred], dim=0)
    valid_preds_np = valid_preds.cpu().numpy()

    auc_v = roc_auc_score(y_valid.cpu(), valid_preds_np)

    # test AUC
    test_preds = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_preds_np = test_preds.cpu().numpy()
    auc_t = roc_auc_score(y_test.cpu(), test_preds_np)

    results['valid'] = auc_v
    results['test'] = auc_t

    return results


def train_and_evaluate(input_dim, hidden_dim, lr, num_epoch, batch_size,
                      pos_valid_edge, neg_valid_edge, y_valid, pos_test_edge, neg_test_edge, y_test,
                      x, pos_train_edge, neg_train_edge):

    print(input_dim, hidden_dim, lr, num_epoch, batch_size)

    model = BatchNormModel(input_dim).to(device)
    predictor = LinkPredictor(input_dim, hidden_dim, 1, 3).to(device)

    Y = []
    for k in range(10):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr)
        res = []

        for epoch in range(1, 1 + num_epoch):
            loss = train(model, predictor, pos_train_edge[k], neg_train_edge[k], optimizer, batch_size, x.to(device))
            results = test(model, predictor, pos_valid_edge[k], neg_valid_edge[k], y_valid[k], pos_test_edge[k], neg_test_edge[k], y_test[k], batch_size, x.to(device))
            if epoch % 20 == 0:
                print(epoch, results)
            res.append(results)

        Y.append([item['test'] for sublist in [res] for item in sublist])
    Y = np.array(Y)
    
    return np.mean(Y[:,-1]), np.std(Y[:,-1])


def prediction_hits(pos_train_edge, neg_train_edge, pos_valid_edge, neg_valid_edge, y_valid, pos_test_edge,
                    neg_test_edge, y_test, x, input_dim):


    results = train_and_evaluate(input_dim, hidden_dim = 16, lr = 0.001, num_epoch = 100, batch_size = 128,
                                    pos_valid_edge, neg_valid_edge, y_valid, pos_test_edge, neg_test_edge, y_test,
                                    x, pos_train_edge, neg_train_edge)

    return results




pos_train_edge, neg_train_edge, pos_valid_edge, neg_valid_edge, pos_test_edge, neg_test_edge = [], [], [], [], [], []
y_valid, y_te = [], []

for i in range(10):
    with open(path + name + '_database' + str(i) + '.csv', 'r') as f: # read the pre-computed edge sets
        data = f.readlines()[1:]
    d_data = np.array([list(map(float, edge.strip().split('\t')[1:])) for edge in data])
    r, c = np.shape(d_data)

    pos_edges = np.array([line for line in d_data if line[c - 1] == 1])
    neg_edges = np.array([line for line in d_data if line[c - 1] == 0])

    yp = pos_edges[:, c - 1]
    yn = neg_edges[:, c - 1]

    train_pos = np.array(pd.read_csv(path + 'train_pos_index' + str(i) + '.csv'))[:, 1:] # training edges index
    valid_pos = np.array(pd.read_csv(path + 'valid_pos_index' + str(i) + '.csv'))[:, 1:] # validation edges index
    test_pos = np.array(pd.read_csv(path + 'test_pos_index' + str(i) + '.csv'))[:, 1:]  # test edges index

    # separating into their respective set the labels
    y_train_pos = np.squeeze(yp[sorted(train_pos)])
    y_val_pos = np.squeeze(yp[sorted(valid_pos)])
    y_test_pos = np.squeeze(yp[sorted(test_pos)])
    y_train_neg = np.squeeze(yn[sorted(train_pos)])
    y_val_neg = np.squeeze(yn[sorted(valid_pos)])
    y_test_neg = np.squeeze(yn[sorted(test_pos)])

    y_train = np.concatenate((y_train_pos, y_train_neg))
    y_val = np.concatenate((y_val_pos, y_val_neg))
    y_test = np.concatenate((y_test_pos, y_test_neg))

    Ep = pos_edges[:, :2]
    En = neg_edges[:, :2]

    # separating into their respective set the edges
    E_train_pos = np.squeeze(Ep[sorted(train_pos)])
    E_val_pos = np.squeeze(Ep[sorted(valid_pos)])
    E_test_pos = np.squeeze(Ep[sorted(test_pos)])
    E_train_neg = np.squeeze(En[sorted(train_pos)])
    E_val_neg = np.squeeze(En[sorted(valid_pos)])
    E_test_neg = np.squeeze(En[sorted(test_pos)])

    # conversion to torch
    pos_train_edge_i = torch.tensor(E_train_pos, dtype=torch.long).to(device)
    neg_train_edge_i = torch.tensor(E_train_neg, dtype=torch.long).to(device)
    pos_valid_edge_i = torch.tensor(E_val_pos, dtype=torch.long).to(device)
    neg_valid_edge_i = torch.tensor(E_val_neg, dtype=torch.long).to(device)
    pos_test_edge_i = torch.tensor(E_test_pos, dtype=torch.long).to(device)
    neg_test_edge_i = torch.tensor(E_test_neg, dtype=torch.long).to(device)

    pos_train_edge.append(pos_train_edge_i)
    neg_train_edge.append(neg_train_edge_i)
    pos_valid_edge.append(pos_valid_edge_i)
    neg_valid_edge.append(neg_valid_edge_i)
    pos_test_edge.append(pos_test_edge_i)
    neg_test_edge.append(neg_test_edge_i)

    y_valid_i = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test_i = torch.tensor(y_test, dtype=torch.long).to(device)

    y_valid.append(y_valid_i)
    y_te.append(y_test_i)

# import scn features
with open( path + 'SCN_Feature_'+name+'.csv', 'r') as f1:
    data_x = f1.readlines()[1:]

data_split = [line.strip().split(',') for line in data_x]
data_array = np.array(data_split, dtype=float)
R, C = data_array.shape
x = data_array[:, 1:C-1]

N = np.shape(x)[1] # input dimension
x = torch.tensor(x, dtype=torch.float).to(device)

auc = prediction_hits(pos_train_edge, neg_train_edge, pos_valid_edge, neg_valid_edge, y_valid, pos_test_edge,
                      neg_test_edge, y_te, x, N)

print(auc)
