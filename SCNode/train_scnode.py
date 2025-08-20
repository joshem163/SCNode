import argparse
import sys
import itertools
from sklearn.metrics import accuracy_score, roc_auc_score
from logger_search import *
from dataloader import *
from modules import *
from model import *
from data_utils import load_fixed_splits

def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze()[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def ACC(prediction, label):
    correct = prediction.eq(label).sum().item()
    total = len(label)
    return correct / total

@torch.no_grad()
def test(model, data, train_idx, valid_idx, test_idx, metric='accuracy'):
    model.eval()
    out = model(data.x)
    y_true = data.y.squeeze()

    if metric == 'accuracy':
        y_pred = out.argmax(dim=-1)
        train_score = ACC(y_pred[train_idx], y_true[train_idx])
        valid_score = ACC(y_pred[valid_idx], y_true[valid_idx])
        test_score = ACC(y_pred[test_idx], y_true[test_idx])
    elif metric == 'roc_auc':
        probs = F.softmax(out, dim=-1)[:, 1]
        train_score = roc_auc_score(y_true[train_idx].cpu(), probs[train_idx].cpu())
        valid_score = roc_auc_score(y_true[valid_idx].cpu(), probs[valid_idx].cpu())
        test_score = roc_auc_score(y_true[test_idx].cpu(), probs[test_idx].cpu())
    else:
        raise ValueError("Unsupported metric: choose 'accuracy' or 'roc_auc'")
    return train_score, valid_score, test_score


def main():
    parser = argparse.ArgumentParser(description='MLP Experiment with Grid Search')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='cornell')
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Define grid search space
    lr_list = [0.001, 0.005, 0.01]
    dropout_list = [0.3, 0.5, 0.7]
    hidden_list = [64, 128, 256]
    wd_list = [0, 1e-4, 5e-4]


    # Cartesian product of all hyperparams
    grid = list(itertools.product(lr_list, dropout_list, hidden_list, wd_list))

    print(f"Total configs to try: {len(grid)}")

    best_valid, best_test, best_cfg = -1, -1, None

    for (lr, dropout, hidden, wd) in grid:
        print(f"\n=== Config: lr={lr}, dropout={dropout}, hidden={hidden}, weight_decay={wd} ===")
        logger = Logger(args.runs, args)

        for run in range(args.runs):
            # ----- load dataset and splits -----
            if args.dataset_name in ['chameleon', 'squirrel']:
                data = load_Sq_Cha_filterred(args.dataset_name)
                split_idx_lst = load_fixed_splits('data', args.dataset_name, name=args.dataset_name)
                out_dim = 5
                train_idx = split_idx_lst[run]['train']
                valid_idx = split_idx_lst[run]['valid']
                test_idx = np.array(split_idx_lst[run]['test'])
            else:
                dataset = load_data(args.dataset_name)
                data = dataset[0]
                train_mask=[data.train_mask[i][run] for i in range(len(data.train_mask))]
                val_mask= [data.val_mask[i][run] for i in range(len(data.val_mask))]
                test_mask = [data.test_mask[i][run] for i in range(len(data.test_mask))]
                train_idx = np.where(train_mask)[0]
                valid_idx = np.where(val_mask)[0]
                test_idx = np.where(test_mask)[0]
                out_dim=dataset.num_classes

            # ----- feature engineering -----
            f = spatial_embeddings(data, test_idx)
            if args.dataset_name in ['pubmed','roman-empire','amazon-ratings','questions','tolokers']:
                f1,f2,f3 = Contextual_embeddings_eucledian(data,test_idx)
                concatenated_list = np.concatenate((f1,f2,f3,f), axis=1)
            else:
                f1, f2,f3,f4 = Contextual_embeddings(data, args.dataset_name,test_idx)
                concatenated_list = np.concatenate((f1,f2,f3,f4,f), axis=1)
            data.x = torch.tensor(concatenated_list, dtype=torch.float).to(device)
            data.y = data.y.to(device)

            train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
            valid_idx = torch.tensor(valid_idx, dtype=torch.long, device=device)
            test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

            # ----- model + optimizer -----
            model = MLP(len(data.x[0]), hidden, out_dim, args.num_layers, dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            # ----- training loop -----
            for epoch in range(1, 1 + args.epochs):
                loss = train(model, data, train_idx, optimizer)
                metric = 'roc_auc' if args.dataset_name in ['minesweeper','questions','tolokers'] else 'accuracy'
                result = test(model, data, train_idx, valid_idx, test_idx, metric=metric)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run+1:02d}, Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, Train: {100*train_acc:.2f}%, '
                          f'Valid: {100*valid_acc:.2f}%, Test: {100*test_acc:.2f}%')

            logger.print_statistics(run)

        (valid_mean, valid_std), (test_mean, test_std) = logger.print_statistics()
        if test_mean > best_test:
            best_valid, best_test, best_std, best_cfg = valid_mean, test_mean,test_std, (lr, dropout, hidden, wd)

    print("\n=== Best Config ===")
    print(f"lr={best_cfg[0]}, dropout={best_cfg[1]}, hidden={best_cfg[2]}, weight_decay={best_cfg[3]}")
    print(f"Best Valid = {valid_mean:.4f} ± {valid_std:.4f}")
    print(f"Best Test  = {best_test:.4f} ± {test_std:.4f}")


if __name__ == "__main__":
    main()
