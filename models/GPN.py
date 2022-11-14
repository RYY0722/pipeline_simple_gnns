import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution

import torch
 
class GPN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5):
        super(GPN_Encoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout ,training=self.training)
        x = self.gc2(x, adj)

        return x


class GPN_Valuator(nn.Module):
    """
    For the sake of model efficiency, the current implementation is a little bit different from the original paper.
    Note that you can still try different architectures for building the valuator network.

    """
    def __init__(self, nfeat, nhid, dropout):
        super(GPN_Valuator, self).__init__()
        
        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc3(x)

        return x

class model(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(model, self).__init__()
        self.encoder = GPN_Encoder(nfeat, nhid, dropout)
        self.valuator = GPN_Valuator(nfeat, nhid, dropout)
    def forward(self, x, adj):
        emb = self.encoder(x, adj)
        score = self.valuator(x, adj)
        return {'score':score,'emb':emb}
       