import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch import nn


# https://github.com/H-Ambrose/GNNs_on_node-level_tasks/blob/master/GATmodel.ipynb
class model(torch.nn.Module):
    def __init__(self, nfeat, hdim=8, dropout=0.6):
        super(model, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(nfeat, hdim, heads=8, dropout=dropout)
        self.conv2 = GATConv(hdim * 8, hdim, dropout=dropout)
        self.fc3 = nn.Linear(hdim, 1)
        self.dropout = 0.6
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        emb = x
        x = self.fc3(x)
        return {'emb':emb, 'score':x}


