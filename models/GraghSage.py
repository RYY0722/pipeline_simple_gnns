import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch import nn

# https://github.com/H-Ambrose/GNNs_on_node-level_tasks
class model(nn.Module):
    def __init__(self, nfeat, hdim, dropout):
        super(model, self).__init__()
        self.conv1 = SAGEConv(nfeat, hdim)
        self.conv2 = SAGEConv(hdim, hdim)
        self.fc3 = nn.Linear(hdim, 1)
        # self.dropout = dropout

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        emb = x
        x = self.fc3(x)
        return {'score':x,'emb':emb}

