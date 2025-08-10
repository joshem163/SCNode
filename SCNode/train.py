import argparse
import sys
from logger import *
from dataloader import *
from modules import *
from model import *
#sys.argv = [sys.argv[0]]

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
def test(model, data, train_idx, valid_idx, test_idx):
    model.eval()
    out = model(data.x)
    y_pred = out.argmax(dim=-1)
    train_acc = ACC(y_pred[train_idx], data.y[train_idx])
    valid_acc = ACC(y_pred[valid_idx], data.y[valid_idx])
    test_acc = ACC(y_pred[test_idx], data.y[test_idx])
    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='MLP Experiment')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='pubmed')  # cora,pubmed,cornell
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()

    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        dataset = load_data(args.dataset_name)
        data = dataset[0]
        train_mask=[data.train_mask[i][run] for i in range(len(data.train_mask))]
        val_mask= [data.val_mask[i][run] for i in range(len(data.val_mask))]
        test_mask = [data.test_mask[i][run] for i in range(len(data.test_mask))]
        train_idx = np.where(train_mask)[0]
        valid_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        f = spatial_embeddings(data,test_idx)

        if args.dataset_name == 'pubmed':
            f1,f2 = Contextual_embeddings_eucledian(data)
            concatenated_list = np.concatenate((f1,f2, f), axis=1)
        else:
            f1, f2 = Contextual_embeddings(data, args.dataset_name)

            concatenated_list = np.concatenate((f1, f2, f), axis=1)
        data.x = torch.tensor(concatenated_list, dtype=torch.float)

        # model.reset_parameters()
        model = MLP(len(data.x[0]), args.hidden_channels, dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, train_idx, valid_idx, test_idx)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
