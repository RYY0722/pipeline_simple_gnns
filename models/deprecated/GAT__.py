import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphAttentionLayer, SpGraphAttentionLayer


class model(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(model, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, 1, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


# class model(nn.Module):
#     def __init__(self, nfeat, nhid, dropout, alpha=0.2, nheads=8):
#         """Sparse version of GAT."""
#         super(model, self).__init__()
#         self.dropout = dropout

#         self.attentions = [SpGraphAttentionLayer(nfeat, 
#                                                  nhid, 
#                                                  dropout=dropout, 
#                                                  alpha=alpha, 
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = SpGraphAttentionLayer(nhid * nheads, 
#                                              1, 
#                                              dropout=dropout, 
#                                              alpha=alpha, 
#                                              concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)
