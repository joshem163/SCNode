import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm
import pandas as pd

def Average(lst):
    # average function
    avg = np.average(lst)
    return (avg)

def spatial_embeddings(data,test_index):
    Node_class = list(range(max(data.y) + 2))
    n = len(data.y)
    label = data.y.clone()
    Edge_idx = data.edge_index.numpy()
    Node = range(n)
    Edge_indices = []
    for i in range(len(Edge_idx[1])):
        Edge_indices.append((Edge_idx[0][i], Edge_idx[1][i]))
    #test_index = np.where(data.test_mask)[0]
    test_class = max(data.y) + 1
    for idx_test in test_index:
        label[idx_test] = test_class

    F_vec = []
    for i in tqdm(range(n), desc="Processing spatial features"):
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


def Contextual_embeddings(data, dataset_name):
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Data.head()
    label = data.y.numpy()
    if dataset_name == 'squirrel':
        Ir = 0.01
    else:
        Ir = 0.1

    Number_nodes = len(data.y)
    fe_len = len(data.x[0])

    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    basis = [[max(df[i]) for i in range(len(df.columns))] for df in data_by_class.values()]
    sel_basis = [[int(list(df[i].to_numpy()).count(1) >= int(len(df[i].index) * Ir))
                  for i in range(len(df.columns))]
                 for df in data_by_class.values()]
    Fec = []
    SFec = []
    fe_len = len(data.x[0])
    feature_names = [ii for ii in range(fe_len)]

    for i in tqdm(range(len(Domain_Fec)), desc="Processing contextual feature"):
        vec = []
        Svec = []

        # Extract the features for the current node
        f = Domain_Fec.loc[i, feature_names].values.flatten().tolist()

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


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def ContextualPubmed(data):
    # Scale data before applying PCA
    DataAttribute = pd.DataFrame(data.x.numpy())
    scaling = StandardScaler()

    # Use fit and transform method
    scaling.fit(DataAttribute)
    Scaled_data = scaling.transform(DataAttribute)

    # Set the n_components=3
    m = 100
    principal = PCA(n_components=m)
    principal.fit(Scaled_data)
    x = principal.transform(Scaled_data)
    return x

def Contextual_embeddings_eucledian(data):
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Number_nodes = len(data.y)
    fe_len = len(data.x[0])
    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    sel_basis = [[Average(list(df[i].to_numpy())) for i in range(len(df.columns))] for df in data_by_class.values()]
    feature_names = [ii for ii in range(fe_len)]
    Fec = []
    for i in tqdm(range(Number_nodes),desc='processing contextual features'):
        vec = []
        f = Data.loc[i, feature_names].values.flatten().tolist()
        for j in range(max(catagories) + 1):
            vec.append(np.linalg.norm(np.array(f) - np.array(sel_basis[j])))
        f.clear()
        Fec.append(vec)
    cont_fe=ContextualPubmed(data)
    return Fec,cont_fe
