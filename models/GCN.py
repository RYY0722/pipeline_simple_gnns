import torch.nn as nn
import torch.nn.functional as F
import torch
from models.layers import GraphConvolution

class model(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(model, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        emb = x
        x = self.fc3(x)

        return {'score':x,'emb':emb}