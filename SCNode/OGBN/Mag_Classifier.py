import pandas as pd
import numpy as np

data = pd.read_csv('Feature_2nbd_mag.csv')

data.head(10)
import ogb
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import DataLoader
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
#df1.head()
feature_names = ["w_{}".format(ii) for ii in range(128)]
#print(feature_names)
Node_Fec=list(D['node_feat_dict']['paper'])
df2 = pd.DataFrame(Node_Fec,columns=feature_names)
df2.head()
result=pd.concat([data, df2],axis=1)
result.head()
feature=['0']
for i in range(1,1392):
    feature.append("{}".format(i))
for i in range(128):
    feature.append("w_{}".format(i))

X=result[feature] # Features
y=df1['Class']  # Labels
X_train=X.iloc[list(train_idx['paper'])+list(valid_idx['paper'])]
X_test=X.iloc[test_idx['paper']]
y_train=y.iloc[list(train_idx['paper'])+list(valid_idx['paper'])]
y_test=y.iloc[test_idx['paper']]

# Create train and test features as a numpy array.
X_train = X_train[feature].to_numpy()
X_test = X_test[feature].to_numpy()

from xgboost import XGBClassifier
bst = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1)
bst.fit(X_train,y_train)
y_pred=bst.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")
