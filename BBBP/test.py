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
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
import matplotlib.pyplot as plt
from dataset import GraphDataset
from model import GCNNet
from tqdm import tqdm

def predicting(model, loader):
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])
    # feat_data = {}
    for graph,smiles in tqdm(loader):
        model.eval()
        with torch.no_grad():
            logits,rep = model(graph)

            probs = F.softmax(logits, dim=-1)
            y_pred_ = torch.argmax(probs,dim=-1).cpu().numpy()
            y_score_ = probs[:,1].cpu().numpy()
            if graph.y.shape[0] == y_pred_.shape[0]:
                y_true = np.hstack((y_true,graph.y.cpu().numpy()))
                y_pred = np.hstack((y_pred,y_pred_))
                y_score = np.hstack((y_score,y_score_))
            
            # # if epoch==48:
            # features = rep.cpu().numpy()
            # for i,smi in enumerate(smiles):
            #     y_true_smi = graph.y.cpu().numpy()[i]
            #     y_pred_smi = y_pred_[i]
            #     feat = features[i]
            #     feat_data[smi] = [y_true_smi,y_pred_smi,feat]

    return y_true,y_score

def metrics(y_label,y_pred,print_metrics=None):
    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    prec, recall, _ = precision_recall_curve(y_label, y_pred)
    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    if print_metrics is not None:
        y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
        cm1 = confusion_matrix(y_label, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        precision1 = precision_score(y_label, y_pred_s) 
        print(f'auroc:{auroc},auprc:{auprc},f1:{np.max(f1[5:])},acc{accuracy},sensitiviy{sensitivity},specificity{specificity},precision{precision1},thred:{thred_optim}')
    return prec, recall, auprc

valid_csv='data/bbb.csv.valid'
train_csv = 'data/bbb.csv.train'

with open('data/smile2graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)

valid_dataset = GraphDataset(valid_csv,smile2graph)
valid_loader = DataLoader(dataset=valid_dataset,batch_size=64,shuffle=False)

train_dataset = GraphDataset(train_csv,smile2graph)
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=False)
    
model = GCNNet()
model_random = GCNNet()
state_dict = torch.load('graphbbb.pt')
model.load_state_dict(state_dict)

T,P = predicting(model,valid_loader)
p,r,a = metrics(T,P,1)

T_t,P_t = predicting(model,train_loader)
p_t,r_t,a_t = metrics(T_t,P_t,1)

T_r,P_r = predicting(model_random,valid_loader)
p_r,r_r,a_r = metrics(T_r,P_r,1)

fig,ax = plt.subplots()
plt.plot(r, p,label='Test (AUC:{:.2f})'.format(a))
plt.plot(r_t, p_t,label='Train (AUC:{:.2f})'.format(a_t))
plt.plot(r_r[2:], p_r[2:],label='Random (AUC:{:.2f})'.format(a_r))


ax.set_xlabel('Recall',fontsize=12)
ax.set_ylabel('Precision',fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
# ax.legend(bbox_to_anchor=(0.8,0.8),fontsize=12)
ax.legend()
plt.title('BBBP Model Precision Recall Curve')
plt.savefig('prec_recall_curve.png',dpi=300)



# best_acc = 0

# y_true = np.array([])
# y_pred = np.array([])
# y_score = np.array([])


# feat_data = {}
# for graph,smiles in valid_loader:
#     model.eval()
#     with torch.no_grad():
#         logits,rep = model(graph)
#         loss = loss_fn(logits,graph.y)
        
#         lprobs = F.log_softmax(logits, dim=-1)
#         y_pred_ = torch.argmax(lprobs,dim=-1).cpu().numpy()
#         y_score_ = torch.max(lprobs, dim=-1).values.cpu().numpy()
#         if graph.y.shape[0] == y_pred_.shape[0]:
#             y_true = np.hstack((y_true,graph.y.cpu().numpy()))
#             y_pred = np.hstack((y_pred,y_pred_))
#             y_score = np.hstack((y_score,y_score_))
        
#         # if epoch==48:
#         features = rep.cpu().numpy()
#         for i,smi in enumerate(smiles):
#             y_true_smi = graph.y.cpu().numpy()[i]
#             y_pred_smi = y_pred_[i]
#             feat = features[i]
#             feat_data[smi] = [y_true_smi,y_pred_smi,feat]

        

# valid_acc = accuracy_score(y_true,y_pred)
# roc_auc = roc_auc_score(y_true,y_score)
# print(valid_acc,roc_auc)
