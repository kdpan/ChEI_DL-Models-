import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GCNNet(torch.nn.Module):
    def __init__(self,graph_output_dim=128,graph_features_dim=78,dropout=0.2):
        super(GCNNet,self).__init__()
        self.conv1 = GCNConv(graph_features_dim, graph_features_dim)
        self.conv2 = GCNConv(graph_features_dim, graph_features_dim*2)
        self.conv3 = GCNConv(graph_features_dim*2, graph_features_dim * 4)
        self.fc_g1 = torch.nn.Linear(graph_features_dim*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, graph_output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.classification_head = nn.Linear(graph_output_dim,2)

    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        feature = self.dropout(x)
        logits = self.classification_head(feature)
        return logits,feature

if __name__ == '__main__':
    from dataset import GraphDataset
    from torch_geometric.loader import DataLoader
    import pickle
    with open('../data/smile2graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)

    train_dataset = GraphDataset('../data/bbb.csv.train',smile2graph)
    loader = DataLoader(dataset=train_dataset,batch_size=8,shuffle=True)
    
    model = GCNNet()

    for graph,smiles in loader:
        logits,feat = model(graph)
        print(feat.shape)
        # print(logits)