import argparse
import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from ClassContrast import *
from model import LinkX
from torch_geometric.datasets import Planetoid,WebKB, WikipediaNetwork
from torch_geometric.nn import LINKX


from logger import Logger
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    #print(len(out))
    #print(data.y.squeeze(1)[train_idx])
    loss = F.cross_entropy(out, data.y.squeeze()[train_idx])
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
    parser = argparse.ArgumentParser(description='experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='cora')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.dataset_name == 'cora':
        dataset = Planetoid(root='/tmp/cora', name='Cora', split='geom-gcn',
                            transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    elif args.dataset_name == 'citeseer':
        dataset = Planetoid(root='/tmp/citeSeer', name='citeSeer', split='geom-gcn',
                            transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    elif args.dataset_name == 'pubmed':
        dataset = Planetoid(root='/tmp/pubmed', name='pubmed', split='geom-gcn',
                            transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    # elif args.dataset_name == 'cornell':
    #     dataset = WebKB(root='/tmp/cornell', name='cornell',
    #                     transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    # elif args.dataset_name == 'wisconsin':
    #     dataset = WebKB(root='/tmp/wisconsin', name='wisconsin',
    #                     transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    # elif args.dataset_name == 'chameleon':
    #     dataset = WikipediaNetwork(root='/tmp/chameleon', name='chameleon', geom_gcn_preprocess=True,
    #                                transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    # elif args.dataset_name == 'squirrel':
    #     dataset = WikipediaNetwork(root='/tmp/squirrel', name='squirrel', geom_gcn_preprocess=True,
    #                                transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    elif args.dataset_name == 'texas':
        dataset = WebKB(root='/tmp/texas', name='texas',
                        transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    else:
        raise NotImplementedError

    #dataset = Planetoid(root='/tmp/cora', name='Cora',split='geom-gcn', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
    cc_feature = CC_feature(args.dataset_name)
    data = dataset[0]
    data.x=cc_feature
    print(data)

    model = LINKX(len(data.y),data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,1,1,
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
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-3)
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

