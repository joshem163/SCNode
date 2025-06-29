import argparse
import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

from logger import Logger
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    #print(len(out))
    #print(data.y.squeeze(1)[train_idx])
    loss = F.nll_loss(out, data.y.squeeze()[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def ACC(Prediction, Label):
    correct = Prediction.view(-1).eq(Label).sum().item()
    total=len(Label)
    return correct / total

@torch.no_grad()
def test(model, data, train_idx,valid_idx,test_idx):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_pred=y_pred.view(-1)
    train_acc=ACC(data.y[train_idx],y_pred[train_idx])
    valid_acc=ACC(data.y[valid_idx],y_pred[valid_idx])
    test_acc =ACC(data.y[test_idx],y_pred[test_idx])
    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='cora')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = Planetoid(root='/tmp/cora', name='Cora',split='geom-gcn', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    data = dataset[0]
    print(data)
    #data.adj_t = data.adj_t.to_symmetric()

    # train_idx = np.where(data.train_mask)[0]
    # valid_idx = np.where(data.val_mask)[0]
    # test_idx = np.where(data.test_mask)[0]

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        idx_train=[data.train_mask[i][run] for i in range(len(data.y))]
        train_idx = np.where(idx_train)[0]
        idx_val=[data.val_mask[i][run] for i in range(len(data.y))]
        valid_idx = np.where(idx_val)[0]
        idx_test=[data.test_mask[i][run] for i in range(len(data.y))]
        test_idx = np.where(idx_test)[0]

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, train_idx,valid_idx,test_idx)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                # print(f'Run: {run + 1:02d}, '
                #       f'Epoch: {epoch:02d}, '
                #       f'Loss: {loss:.4f}, '
                #       f'Train: {100 * train_acc:.2f}%, '
                #       f'Valid: {100 * valid_acc:.2f}% '
                #       f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()

