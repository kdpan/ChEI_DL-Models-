import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score

from config import LMsGraphModelConfig
from model import LMsGraphModel
from dataset import LMsGraphDataset, LMsGraphPredictionDataset
from utils import *
from tqdm import tqdm
def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

TEST_BATCH_SIZE =128
CUDA = 'cuda:1'
screen_dataset_name = 'L6030-ache.csv'
screen_cp_graph = 'screen_database/L6030.graph'
screen_dataset = os.path.join('prediction/data',screen_dataset_name)
pretrain = 'model_GCN_che.model'
saved_path = os.path.join('prediction/result',screen_dataset_name)


torch.cuda.empty_cache()
with open(screen_cp_graph, 'rb') as f:
        smile2graph = pickle.load(f)
cuda_name = CUDA

dataset = LMsGraphPredictionDataset(screen_dataset,smile2graph)
loader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
config = LMsGraphModelConfig()
graphNet = config['graphNet']
model = LMsGraphModel(config).to(device)
state_dict = torch.load(pretrain)
model.load_state_dict(state_dict)

res = pd.DataFrame(columns=['Entry','Score'])
res.to_csv(saved_path,mode='w',index=None)
n=0
model.eval()
with torch.no_grad():
    for graph,seq_embed,seq_mask in tqdm(loader):
        graph = graph.to(device)
        seq_embed = seq_embed.to(device)
        seq_mask = seq_mask.to(device)
        
        output = model(graph,seq_embed)
        m = nn.Sigmoid()
        score = torch.squeeze(m(output), 1).cpu().numpy()
        for i,entry in enumerate(graph.entry):
            res.loc[n] = [entry,score[i]]
            res.to_csv(saved_path,mode='a',index=None,header=None)
            res = pd.DataFrame(columns=['Entry','Score'])
            n+=1
sort_res = pd.read_csv(saved_path).sort_values(by='Score',ascending=False)
sort_res.to_csv(saved_path,index=None)


        
