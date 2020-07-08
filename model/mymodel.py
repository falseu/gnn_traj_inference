import torch
from torch_geometric.nn import GINConv, SAGPooling, GCNConv, SAGEConv, TopKPooling   # noqa
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from model.mylayer import GraphConvolution
from model.GraphSAGE import GraphConvolutionSage
from scipy.linalg import block_diag
from model.DiffusionNN import GraphDiffusion

class MyModel(torch.nn.Module):
    def __init__(self, in_channels, device):
        super(MyModel, self).__init__()
        self.device = device

        self.conv1 = GraphConvolution(in_channels, 2*in_channels)
        self.conv2 = GraphConvolution(2*in_channels, 2*in_channels)
        self.conv3 = GraphConvolution(2*in_channels, 2*in_channels)

        self.lin1 = nn.Linear(2*in_channels, 2*in_channels)
        self.lin2 = nn.Linear(2*in_channels, in_channels)
        self.lin3 = nn.Linear(in_channels, 1)

    def forward(self, data):
        x = data.x.float()

        adj = data.adj

        
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
        x = self.conv3(x, adj, self.device)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, device):
        super(GraphSAGEModel, self).__init__()
        self.device = device

        self.conv1 = GraphConvolutionSage(in_channels, 256)
        self.conv2 = GraphConvolutionSage(256, 64)

        # self.lin1 = nn.Linear(400, 64)
        # self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        # self.lin1.reset_parameters()
        # self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, v = data.x.float(), data.v.float()
        if len(data.adj) == 1:
            adj = torch.FloatTensor(data.adj).view(x.shape[0], x.shape[0])
        else:
            adj = torch.FloatTensor(block_diag(*[i[0] for i in data.adj]))
    
        # adj np.inf denote disconnected
        adj = F.sigmoid(adj)

        adj[torch.isnan(adj)] = 0
        adj[adj <= 0.7] = 0
        adj = adj * 4

        adj = adj.to(self.device)

        x = F.relu(self.conv1(x, adj, self.device))
        node_embed = F.relu(self.conv2(x, adj, self.device))
        # x = F.relu(self.lin1(x))
        # x = F.relu(self.lin2(torch.cat((x, v), dim=1)))
        # x = F.relu(self.lin2(x))
        x = F.sigmoid(self.lin3(node_embed))
        return x, node_embed, adj


class FCModel(torch.nn.Module):
    def __init__(self, in_channels, device, hidden1=128, hidden2=64, dropout1=0.5, dropout2=0.2):
        super(FCModel, self).__init__()
        self.device = device

        self.lin1 = nn.Linear(in_channels, hidden1)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lin3 = nn.Linear(hidden2, 1)

        self.p1 = dropout1
        self.p2 = dropout2
    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x = data.x.float()

        x = F.relu(F.dropout(self.lin1(x), p=self.p1))
        x = F.relu(F.dropout(self.lin2(x), p=self.p2))
        x = F.sigmoid(self.lin3(x))
        return x, None, None

class veloModel(torch.nn.Module):
    def __init__(self, in_channels, device):
        super(veloModel, self).__init__()
        self.device = device

        self.conv1 = GraphConvolutionSage(in_channels, 32)
        self.conv2 = GraphConvolutionSage(32, 8)

        self.lin1 = nn.Linear(16, 1)
        # self.lin2 = nn.Linear(16, 1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin1.reset_parameters()
        # self.lin2.reset_parameters()

    def forward(self, data):
        x, v = data.x.float(), data.v.float()
        if len(data.adj) == 1:
            adj = torch.FloatTensor(data.adj).view(x.shape[0], x.shape[0])
        else:
            adj = torch.FloatTensor(block_diag(*[i[0] for i in data.adj]))
    
        # coded in the dataset generation process
        adj = F.sigmoid(adj)

        adj[torch.isnan(adj)] = 0
        adj = adj * 4

        adj = adj.to(self.device)

        x = F.relu(self.conv1(x, adj, self.device))
        x = F.relu(self.conv2(x, adj, self.device))
        # v = F.relu(self.conv1(v, adj, self.device))
        # v = F.relu(self.conv2(v, adj, self.device))
        # x = F.relu(self.lin1(torch.cat((x, v), dim=1)))
        node_embed = torch.cat((x, v), dim=1)
        x = F.sigmoid(self.lin1(node_embed))
        # x = F.sigmoid(self.lin2(x))
        return x, node_embed, adj

class DiffusionModel(torch.nn.Module):
    def __init__(self, in_features, device, hidden1 = 128, hidden2 = 64, max_diffusion = 10, include_reversed = False, dropout1 = 0.5, dropout2 = 0.2):
        super(DiffusionModel, self).__init__()
        self.device = device

        self.conv1 = GraphDiffusion(in_features = in_features, out_features = hidden1, max_diffusion = max_diffusion, include_reversed = include_reversed)
        self.conv2 = GraphDiffusion(in_features = hidden1, out_features = hidden2, max_diffusion = max_diffusion, include_reversed = include_reversed)

        # self.norm1 = nn.BatchNorm1d(self.h1)
        # self.norm2 = nn.BatchNorm1d(self.h2)

        self.lin = nn.Linear(hidden2, 1)

        self.p1 = dropout1
        self.p2 = dropout2

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x = data.x.float()
        if len(data.adj) == 1:
            adj = torch.FloatTensor(data.adj).view(x.shape[0], x.shape[0])
        else:
            adj = torch.FloatTensor(block_diag(*[i[0] for i in data.adj]))
    
        # coded in the dataset generation process, adj with self loop adj[i,i] = 0
        adj = F.sigmoid(adj)
        adj[torch.isnan(adj)] = 0
        adj = adj.to(self.device)

        x = self.conv1(x, adj, self.device).squeeze()
        x = F.relu(F.dropout(x, p=self.p1))
        x = self.conv2(x, adj, self.device).squeeze()
        node_embed = F.relu(F.dropout(x, p=self.p2))
        output = F.sigmoid(self.lin(node_embed))

        return output, node_embed, adj