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

# train_csv='../data/bbb.csv.train'
# valid_csv='../data/bbb.csv.valid'
train_csv='../data/refined_from_tarbular.train'
valid_csv='../data/refined_from_tarbular.valid'

with open('../data/smile2graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)

train_dataset = GraphDataset(train_csv,smile2graph)
valid_dataset = GraphDataset(valid_csv,smile2graph)
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=64,shuffle=True)
    
model = GCNNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(50):
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])
    for graph,smiles in train_loader:
        model.train()
        logits,rep = model(graph)
        loss = loss_fn(logits,graph.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)

    feat_data = {}
    for graph,smiles in valid_loader:

        model.eval()
        with torch.no_grad():
            logits,rep = model(graph)
            loss = loss_fn(logits,graph.y)
            
            lprobs = F.log_softmax(logits, dim=-1)
            y_pred_ = torch.argmax(lprobs,dim=-1).cpu().numpy()
            y_score_ = torch.max(lprobs, dim=-1).values.cpu().numpy()
            if graph.y.shape[0] == y_pred_.shape[0]:
                y_true = np.hstack((y_true,graph.y.cpu().numpy()))
                y_pred = np.hstack((y_pred,y_pred_))
                y_score = np.hstack((y_score,y_score_))
            
            # if epoch==48:
            features = rep.cpu().numpy()
            for i,smi in enumerate(smiles):
                y_true_smi = graph.y.cpu().numpy()[i]
                y_pred_smi = y_pred_[i]
                feat = features[i]
                feat_data[smi] = [y_true_smi,y_pred_smi,feat]
    
        

    valid_acc = accuracy_score(y_true,y_pred)
    roc_auc = roc_auc_score(y_true,y_score)
    print(valid_acc,roc_auc)

    if best_acc < valid_acc:
        best_acc = valid_acc
        with open("feature_refined_from_graphbbb.pickle","wb") as f:
            pickle.dump(feat_data,f)
        torch.save(model.state_dict(), 'refined_from_graphbbb.pt')