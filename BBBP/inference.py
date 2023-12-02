import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
from dataset import GraphDataset
from model import GCNNet
from tqdm import tqdm

# valid_csv='../data/refined_from_tarbular.valid'
infer_csv='screen_database/tcmsp.csv'
saved_results = 'tcmsp-bbb-prediction.csv'
with open('screen_database/tcmsp.graph', 'rb') as f:
        smile2graph = pickle.load(f)

infer_dataset = GraphDataset(infer_csv,smile2graph)
loader = DataLoader(dataset=infer_dataset,batch_size=64,shuffle=False)

model = GCNNet()
model.load_state_dict(torch.load("graphbbb.pt"))


results = pd.DataFrame(columns=['smi','BBBP score'])
results.to_csv(saved_results,mode='w',index=None)
n = 0
for graph,smiles in tqdm(loader):
    model.eval()
    with torch.no_grad():
        logits,rep = model(graph)
        probs = F.softmax(logits, dim=-1)
        score = probs[:,1].cpu().numpy()
        for i,smi in enumerate(smiles):
            s = score[i]
            results.loc[n] = [smi,s]
            n+=1
            results.to_csv(saved_results,mode='a',index=None,header=None)
            results = pd.DataFrame(columns=['smi','BBBP score'])
            




