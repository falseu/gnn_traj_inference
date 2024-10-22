import torch
from torch_geometric.nn import GINConv, SAGPooling, GCNConv, SAGEConv, TopKPooling   # noqa
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from model.mylayer import GraphConvolution

class BaseModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(BaseModel, self).__init__()
        # self.mlp1 = nn.Sequential(
        #     nn.Linear(in_channels, 16),
        #     nn.ReLU(),
        #     nn.Linear(in_channels, 16)
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16)
        # )

        # self.mlp3 = nn.Sequential(
        #     nn.Linear(16, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16)
        # )

        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, 32)

        self.lin1 = nn.Linear(32, 32)
        self.lin2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
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