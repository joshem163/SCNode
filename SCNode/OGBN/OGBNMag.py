import numpy as np
import pandas as pd
import ogb
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
#H:\\PhD Research\Data\PROTEINS\PROTEINS_node_attributes.txt
# Download and process data at './dataset/ogbg_molhiv/'
dataset = NodePropPredDataset(name = "ogbn-mag")
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0] # pyg graph object
D=graph[0]
Class=graph[1]
Cla=Class['paper']
Node_Class=Cla[:,0]
df1 = pd.DataFrame(Node_Class,columns=["Class"])
df1.head()
feature_names = ["w_{}".format(ii) for ii in range(128)]
#print(feature_names)
Node_Fec=list(D['node_feat_dict']['paper'])
df2 = pd.DataFrame(Node_Fec,columns=feature_names)
df2.head()
Data=pd.concat([df1, df2],axis=1)
Data.head()
Node_Year=D['node_year']['paper']
Node_Year_list=list(Node_Year[:,0])

E=D['edge_index_dict'][('paper', 'cites', 'paper')]
EdgeL=E[0]
EdgeR=E[1]

Node_Year_list.index(2019)
list_size = len(Node_Year_list)
NIndex_19 = []
# declare for loop
for itr in range(list_size):

    # check the condition
    if (Node_Year_list[itr] == 2019):
        NIndex_19.append(itr)

        # print the indices
#print(len(NIndex_19))
N_class_train=Node_Class
for d in NIndex_19:
    N_class_train[d]=350


def CountX(lst, x):
    return (lst.count(x))


Node_class = list(range(349))
F_vec = []
n = 736389
for i in range(n):
    #print("Processing file {} ({}%)".format(i, 100 * i // n))
    node_F = []
    list_out = []
    list_In = []
    S_nbd_out=[]
    S_nbd_in=[]
    indx = np.where(EdgeL == i)
    indxIn = np.where(EdgeR == i)
    # print(indx)
    for j in indx[0]:
        list_out.append(N_class_train[EdgeR[j]])
        idx_sec = np.where(EdgeL == EdgeR[j])
        for l in idx_sec[0]:
            S_nbd_out.append(N_class_train[EdgeR[l]])
    # indx[0].clear()
    for k in indxIn[0]:
        list_In.append(N_class_train[EdgeL[k]])
        idx_sec_in = np.where(EdgeR == EdgeL[k])
        for m in idx_sec_in[0]:
            S_nbd_in.append(N_class_train[EdgeL[m]])
    # indxIn[0].clear()

    for d in Node_class:
        node_F.append(CountX(list_out, d))
        node_F.append(CountX(list_In, d))
        node_F.append(CountX(S_nbd_out, d))
        node_F.append(CountX(S_nbd_in, d))
    F_vec.append(node_F)

k = len(F_vec[0])
Feture = []
for i in range(k):
    Feture.append("{}".format(i))
data = pd.DataFrame(F_vec, columns =Feture)
data.insert(loc=k,column='Class',value=Node_Class)
data.head()

result = pd.concat([data, Data], axis=1)

compression_opts = dict(method='zip',
                        archive_name='Feature_2nbd_mag.csv')
result.to_csv('Feature_2nbd_mag.zip', index=True,
          compression=compression_opts)
feature = []
for i in range(k):
    feature.append("{}".format(i))
for i in range(128):
    feature.append("w_{}".format(i))


X=result[feature] # Features
y=result['Class']  # Labels
X_train=X.iloc[list(train_idx['paper'])+list(valid_idx['paper'])]
X_test=X.iloc[test_idx['paper']]
y_train=y.iloc[list(train_idx['paper'])+list(valid_idx['paper'])]
y_test=y.iloc[test_idx['paper']]
from xgboost import XGBClassifier

bst = XGBClassifier(n_estimators=100, max_depth=9, learning_rate=0.1)
bst.fit(X_train,y_train)
y_pred=bst.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")
