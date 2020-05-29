import torch
from torch_geometric.nn import GINConv, SAGPooling, GCNConv, SAGEConv, TopKPooling   # noqa
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np

class BaseModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(BaseModel, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(in_channels, 64)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # self.conv1 = GINConv(self.mlp1)
        # self.conv2 = GINConv(self.mlp2)
        # self.conv3 = GINConv(self.mlp3)

        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        # self.conv3 = GCNConv(64, 64)

        self.lin1 = nn.Linear(64, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        
        x = self.conv1(x, edge_index)
        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = self.conv2(x, edge_index)
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # x = self.conv2(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
