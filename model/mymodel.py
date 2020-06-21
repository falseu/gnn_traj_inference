import torch
from torch_geometric.nn import GINConv, SAGPooling, GCNConv, SAGEConv, TopKPooling   # noqa
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from model.mylayer import GraphConvolution
from model.GraphSAGE import GraphConvolutionSage
from scipy.linalg import block_diag

class MyModel(torch.nn.Module):
    def __init__(self, in_channels, device):
        super(MyModel, self).__init__()
        self.device = device

        # self.conv1 = GraphConvolution(in_channels, 2*in_channels)
        # self.conv2 = GraphConvolution(2*in_channels, 2*in_channels)
        # self.conv3 = GraphConvolution(2*in_channels, 2*in_channels)

        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, in_channels)
        , nn.Linear(in_channels, in_channels)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(in_channels, in_channels)
        , nn.Linear(in_channels, in_channels)))

        self.lin1 = nn.Linear(in_channels, in_channels)
        # self.lin2 = nn.Linear(2*in_channels, in_channels)
        self.lin3 = nn.Linear(in_channels, 1)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.float()

        adj = data.adj

        
        # adj = torch.FloatTensor(data.adj).view(x.shape[0], x.shape[0])
        # # adj np.inf denote disconnected
        # # adj = adj + 1s
        # adj = F.sigmoid(adj)
        # # adj = torch.where(torch.isnan(adj), torch.zeros_like(adj), adj)
        # adj[torch.isnan(adj)] = 0
        # adj = adj * 4

        # adj = adj.to(self.device)

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # x = self.conv3(x, adj, self.device)
        x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, device):
        super(GraphSAGEModel, self).__init__()
        self.device = device

        self.conv1 = GraphConvolutionSage(in_channels, 32)
        self.conv2 = GraphConvolutionSage(32, 32)

        self.lin1 = nn.Linear(32, 32)
        self.lin2 = nn.Linear(32, 1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x = data.x.float()

        if len(data.adj) == 1:
            adj = torch.FloatTensor(data.adj).view(x.shape[0], x.shape[0])
        else:
            adj = torch.FloatTensor(block_diag(*[i[0] for i in data.adj]))

        # adj np.inf denote disconnected
        # adj = adj + 1s
        adj = F.sigmoid(adj)
        # adj = torch.where(torch.isnan(adj), torch.zeros_like(adj), adj)
        adj[torch.isnan(adj)] = 0
        # adj = adj * 4

        adj = adj.to(self.device)

        x = F.relu(self.conv1(x, adj, self.device))
        x = F.relu(self.conv2(x, adj, self.device))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x