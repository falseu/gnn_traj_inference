import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphConvolutionSage(Module):
    """
    GraphSAGE
    """

    def __init__(self, in_features, out_features, dropout=0.):
        super(GraphConvolutionSage, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        # method 1
        self.lin = nn.Linear(in_features = in_features, out_features = out_features, bias= True)
        # method 2
        self.weight_neigh = Parameter(torch.FloatTensor(out_features, out_features))
        self.weight_self = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_support = Parameter(torch.FloatTensor(in_features, out_features))
        # with dimension (1, out_features), with broadcast -> (N, Dout)
        self.bias_support = Parameter(torch.FloatTensor(1, out_features))

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_neigh)
        torch.nn.init.xavier_uniform_(self.weight_self)
        torch.nn.init.xavier_uniform_(self.weight_support)
        # initialization requires two dimension
        torch.nn.init.xavier_uniform_(self.bias_support)
        self.lin.reset_parameters()
        

    def forward(self, input, adj, device):
        # first dropout some inputs
        input = F.dropout(input, self.dropout, self.training)

        # Message: two ways
        # method1, matrix multiplication, input of dimension N * Din, support of dimension N * Dout, message passing
        support = F.sigmoid(torch.mm(input, self.weight_support) + self.bias_support)
        
        # method2, use linear layer
        # support = F.sigmoid(self.lin(input))

        # make diagonal position 0
        # with torch.no_grad():
        #     ind = np.diag_indices(adj.shape[0])
        #     adj[ind[0], ind[1]] = torch.zeros(adj.shape[0]).to(device)

        # Aggregation:
        # addition here, could try element-wise max
        output = torch.mm(adj, support)

        # Update: 
        # output of dimension N * Dout
        output = F.relu(torch.mm(output, self.weight_neigh) + torch.mm(input, self.weight_self))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'