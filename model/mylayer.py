import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj, device):
        # first dropout some inputs
        # input = F.dropout(input, self.dropout, self.training)
        
        # matrix multiplication, input of dimension N * Din, support of dimension N * Dout, message passing
        #support = torch.mm(torch.cat((input, torch.ones(input.shape[0], 1).to(device)), dim=1), self.weight)
        support = torch.mm(input, self.weight)
        # matrix multiplication, averaging the support of neighboring nodes, aggregation step
        # adj np.inf denote disconnected
        # adj = adj + 1
        # ones = torch.zeros_like(adj)
        # adj = torch.where(adj == float('inf'), ones, adj)
        output = torch.mm(adj, support)
        # update step, one layer of non-linearity, here rather than degree normalization, use relu.
        output = self.act(output)
        # output of dimension N * Dout
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'