import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU

class model(torch.nn.Module):
    def __init__(self, nfeat, hdim=32, dropout=0.5):
        super(model, self).__init__()
        self.dropout = dropout
        num_features = nfeat

        nn1 = Sequential(Linear(num_features, hdim), ReLU(), Linear(hdim, hdim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hdim)
        nn2 = Sequential(Linear(hdim, hdim), ReLU(), Linear(hdim, hdim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hdim)
        
        self.fc1 = Linear(hdim, hdim)
        self.fc2 = Linear(hdim, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        emb = x
        x = self.fc2(x)
        return {'emb':emb, 'score':x}