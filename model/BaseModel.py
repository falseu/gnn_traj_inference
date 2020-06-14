import torch
from torch_geometric.nn import GINConv, SAGPooling, GCNConv, SAGEConv, TopKPooling   # noqa
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from model.mylayer import GraphConvolution

class BaseModel(torch.nn.Module):
    def __init__(self, in_channels, device):
        super(BaseModel, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 16),
            nn.ReLU(),
            nn.Linear(in_channels, 16)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # self.conv1 = GCNConv(in_channels, 32)
        # self.conv2 = GCNConv(32, 32)

        self.conv1 = GraphConvolution(in_channels, 2*in_channels)
        self.conv2 = GraphConvolution(2*in_channels, 2*in_channels)

        self.lin1 = nn.Linear(2*in_channels, 2*in_channels)
        self.lin2 = nn.Linear(2*in_channels, 1)

        self.device = device

    def forward(self, data):
        x = data.x.float()
        adj = torch.FloatTensor(data.adj).view(x.shape[0], x.shape[0])
        # adj np.inf denote disconnected
        # adj = adj + 1s
        adj = F.sigmoid(adj)
        # adj = torch.where(torch.isnan(adj), torch.zeros_like(adj), adj)
        adj[torch.isnan(adj)] = 0
        adj = adj * 4

        adj = adj.to(self.device)

        x = self.conv1(x, adj, self.device)
        x = self.conv2(x, adj, self.device)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

class BaseModelClassification(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BaseModelClassification, self).__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, in_channels)

        self.lin1 = nn.Linear(in_channels, in_channels)
        self.lin2 = nn.Linear(in_channels, num_classes)
    
    def forward(self, data, *input, **kwargs):
        x, edge_index = data.x.float(), data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
