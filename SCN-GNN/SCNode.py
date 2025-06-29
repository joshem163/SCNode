
import numpy as np
import pandas as pd
import torch

from torch_geometric.datasets import Planetoid, WebKB

def spatial_two(Node_class, Edge_indices, label, n):
    F_vec = []
    for i in range(n):
        # print("\rProcessing file {} ({}%)".format(i, 100*i//(n-1)), end='', flush=True)
        node_F = []
        list_out = []
        list_In = []
        S_nbd_out = []
        S_nbd_in = []
        for edge in Edge_indices:
            src, dst = edge
            if src == i:
                list_out.append(label[dst])
                for edge_2 in Edge_indices:
                    src_2, dst_2 = edge_2
                    if src_2 == dst and src_2 != dst_2:
                        S_nbd_out.append(label[dst_2])

        # print(list_out)
        # print(list_In)
        for d in Node_class:
            count = 0
            count_in = 0

            for node in list_out:
                if Node_class[node] == d:
                    count += 1
            node_F.append(count)

        for d in Node_class:
            count_S_out = 0
            count_S_in = 0
            for node in S_nbd_out:
                if Node_class[node] == d:
                    count_S_out += 1
            node_F.append(count_S_out)

        F_vec.append(node_F)
    return F_vec
def Similarity(array1, array2):
    intersection = np.sum(np.logical_and(array1, array2))
    return intersection

def Domain_Fe(DataFram, basis, sel_basis, feature_names):
    Fec = []
    SFec = []

    for i in range(len(DataFram)):
        vec = []
        Svec = []

        # Extract the features for the current node
        f = DataFram.loc[i, feature_names].values.flatten().tolist()

        # Compute similarities for basis
        for b in basis:
            vec.append(Similarity(f, b))

        # Compute similarities for sel_basis
        for sb in sel_basis:
            Svec.append(Similarity(f, sb))

        # Clear the feature list and append results
        f.clear()
        Fec.append(vec)
        SFec.append(Svec)

    return Fec, SFec
def SCN_feature(dataset_name):
    if dataset_name=='cora':
        dataset = Planetoid(root='/tmp/cora', name='Cora', split='geom-gcn')
    elif dataset_name=='Citeseer':
        dataset = Planetoid(root='/tmp/citeseer', name='citeseer', split='geom-gcn')
    elif dataset_name=='texas':
        dataset = WebKB(root='/tmp/texas', name='texas')
    else:
        raise NotImplementedError
    data = dataset[0]
    Number_nodes = len(data.y)
    label = data.y.numpy()
    Edge_idx = data.edge_index.numpy()
    Node = range(Number_nodes)
    Edgelist = []
    for i in range(len(Edge_idx[1])):
        Edgelist.append((Edge_idx[0][i], Edge_idx[1][i]))
    # print(Edgelist)
    Node_class = range(max(label) + 1)
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Data.head()
    label = data.y.numpy()
    Number_nodes = len(data.y)
    fe_len = len(data.x[0])
    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    basis = [[max(df[i]) for i in range(len(df.columns))] for df in data_by_class.values()]
    sel_basis = [[int(list(df[i].to_numpy()).count(1) >= int(len(df[i].index) * 0.1))
                  for i in range(len(df.columns))]
                 for df in data_by_class.values()]

    F_vec = spatial_two(Node_class, Edgelist,label, Number_nodes)
    feature_names = [ii for ii in range(fe_len)]
    Fec, SFec = Domain_Fe(Data, basis, sel_basis, feature_names)
    # Concatenate the lists along axis 0 (vertically)
    concatenated_list = np.concatenate((Fec, SFec, F_vec,), axis=1)

    # Convert to a NumPy array (tensor)
    tensor_scn = torch.tensor(concatenated_list).float()
    return tensor_scn



