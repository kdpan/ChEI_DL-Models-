from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import pickle
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader

class GraphDataset(Dataset):
    def __init__(self,dataset_file,smile2graph):
        self.smile2graph = smile2graph
        self.dataset = pd.read_csv(dataset_file)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,index):
        
        try:
            smile = self.dataset.iloc[index]['smiles']
            label = self.dataset.iloc[index]['BBBclass']
            label = int(label)
            c_size, features, edge_index = self.smile2graph[smile]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.LongTensor([label]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        except:
            smile = self.dataset.iloc[0]['smiles']
            label = self.dataset.iloc[0]['BBBclass']
            label = int(label)
            c_size, features, edge_index = self.smile2graph[smile]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.LongTensor([label]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
        return GCNData,smile



if __name__ == '__main__':

    with open('../data/smile2graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)

    train_dataset = GraphDataset('../data/bbb.csv.train',smile2graph)
    loader = DataLoader(dataset=train_dataset,batch_size=8,shuffle=True)
    
    for graph,smile in loader:
        print(graph.y)
        print(smile)
        break
