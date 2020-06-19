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

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolutionSage, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.lin = nn.Linear(in_features = in_features, out_features = out_features, bias= True)
        self.weight_neigh = Parameter(torch.FloatTensor(out_features, out_features))
        self.weight_self = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_support = Parameter(torch.FloatTensor(in_features, out_features))
        # self.lin_neigh = nn.Linear(out_features, out_features, bias=False)
        # self.lin_self = nn.Linear(in_features, out_features, False)
        #self.lin_support = nn.Linear(in_features, out_features, False)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_neigh)
        torch.nn.init.xavier_uniform_(self.weight_self)
        torch.nn.init.xavier_uniform_(self.weight_support)
        

    def forward(self, input, adj, device):
        # first dropout some inputs
        # input = F.dropout(input, self.dropout, self.training)

        # Message:
        # matrix multiplication, input of dimension N * Din, support of dimension N * Dout, message passing
        support = F.sigmoid(torch.mm(input, self.weight_support))
        # make diagonal position 0
        # with torch.no_grad():
        #     ind = np.diag_indices(adj.shape[0])
        #     adj[ind[0], ind[1]] = torch.zeros(adj.shape[0]).to(device)

        # Aggregation:
        # addition here, could try element-wise max
        output = torch.mm(adj, support)

        # Update: 
        # output of dimension N * Dout
        output = self.act(torch.mm(output, self.weight_neigh) + torch.mm(input, self.weight_self))
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'