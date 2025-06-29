from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
import warnings
#warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric.data.dataset")

warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric")

def load_data(dataset_Name):
    if dataset_Name=='cora':
        data_loaded = Planetoid(root='/tmp/cora', name='Cora', split='geom-gcn')
    elif dataset_Name=='citeseer':
        data_loaded = Planetoid(root='/tmp/citeseer', name='citeseer', split='geom-gcn')
    elif dataset_Name=='pubmed':
        data_loaded = Planetoid(root='/tmp/pubmed', name='pubmed', split='geom-gcn')
    elif dataset_Name=='texas':
        data_loaded = WebKB(root='/tmp/texas', name='texas')
    elif dataset_Name=='cornell':
        data_loaded = WebKB(root='/tmp/cornell', name='cornell')
    elif dataset_Name=='wisconsin':
        data_loaded = WebKB(root='/tmp/wisconsin', name='wisconsin')
    elif dataset_Name=='chameleon':
        data_loaded = WikipediaNetwork(root='/tmp/chameleon', name='chameleon')
    elif dataset_Name=='squirrel':
        data_loaded = WikipediaNetwork(root='/tmp/squirrel', name='squirrel')
    else:
        raise NotImplementedError
    return data_loaded