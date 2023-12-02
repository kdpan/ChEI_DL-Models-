from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import pickle
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader

class LMsGraphDataset(Dataset):
    def __init__(self,dataset_file,smile2graph,seq2emb):
        self.smile2graph = smile2graph
        self.seq2emb = seq2emb
        self.dataset = pd.read_csv(dataset_file)
        self.max_seq_len = 1024

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,index):
        try:
            smile = self.dataset.iloc[index]['compound_iso_smiles']
            seq = self.dataset.iloc[index]['target_sequence']
            label = self.dataset.iloc[index]['affinity']
            label = int(label)

            c_size, features, edge_index = self.smile2graph[smile]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.Tensor([label]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            
            seq_msg = torch.load(self.seq2emb[seq])
            seq_embed = seq_msg[0]
            seq_mask = seq_msg[1]
        except:
            smile = self.dataset.iloc[0]['compound_iso_smiles']
            seq = self.dataset.iloc[0]['target_sequence']
            label = self.dataset.iloc[0]['affinity']
            label = int(label)

            c_size, features, edge_index = self.smile2graph[smile]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.Tensor([label]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            
            seq_msg = torch.load(self.seq2emb[seq])
            seq_embed = seq_msg[0]
            seq_mask = seq_msg[1]

        return GCNData,seq_embed,seq_mask

class LMsGraphPredictionDataset(Dataset):
    def __init__(self,dataset_file,smile2graph):
        self.smile2graph = smile2graph
        self.dataset = pd.read_csv(dataset_file)
        self.max_seq_len = 1024

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,index):
        try:
            smile = self.dataset.iloc[index]['smi']
            prot_id = self.dataset.iloc[index]['prot_id']
            mol_id = self.dataset.iloc[index]['mol_id']
            entry = mol_id+"|"+prot_id 
            c_size, features, edge_index = self.smile2graph[smile]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    entry=entry)
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            
            seq_msg = torch.load(f'data/seq_embs/prot_t5_xl_uniref50/{prot_id}.pt')
            seq_embed = seq_msg[0]
            seq_mask = seq_msg[1]
        except:
            smile = self.dataset.iloc[0]['smi']
            prot_id = self.dataset.iloc[0]['prot_id']
            mol_id = self.dataset.iloc[0]['mol_id']
            entry = mol_id+"|"+prot_id 

            c_size, features, edge_index = self.smile2graph[smile]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    entry=entry)
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            
            seq_msg = torch.load(f'data/seq_embs/prot_t5_xl_uniref50/{prot_id}.pt')
            seq_embed = seq_msg[0]
            seq_mask = seq_msg[1]

        return GCNData,seq_embed,seq_mask


if __name__ == '__main__':
    with open('data/seq2path_prot_t5_xl_uniref50.pickle', 'rb') as handle:
        seq2path = pickle.load(handle)
    # with open('data/train_smile_graph.pickle', 'rb') as f:
    with open('screen_database/tcmsp.graph','rb') as f:
        smile2graph = pickle.load(f)

    # train_dataset = LMsGraphDataset('data/che_train.csv',smile2graph,seq2path)
    prediction_dataset = LMsGraphPredictionDataset('prediction/data/tcmsp-ache.csv',smile2graph)
    loader = DataLoader(dataset=prediction_dataset,batch_size=8,shuffle=True)
    
    for graph,seq_embed,seq_mask in loader:
        print(graph)
        # print(graph.entry)
        print(seq_embed.shape)
        break
    # graph,seq_embed,seq_mask = train_dataset.__getitem__(2)
    # print(graph.x.shape)
    # print(seq)    