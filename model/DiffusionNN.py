import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphDiffusion(Module):
    """
    GraphDiffusion
    """

    def __init__(self, in_features, out_features, max_diffusion, include_reversed = False):
        super(GraphDiffusion, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_diffusion = max_diffusion
        self.include_reversed = include_reversed
        if self.include_reversed:
            self.weight_support = Parameter(torch.FloatTensor((self.max_diffusion * 2 + 1) * self.in_features, self.out_features))
        else:
            self.weight_support = Parameter(torch.FloatTensor((self.max_diffusion + 1) * self.in_features, self.out_features))

        # with dimension (1, out_features), with broadcast -> (N, Dout)
        self.bias_support = Parameter(torch.FloatTensor(1, self.out_features))
        

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_support)
        torch.nn.init.xavier_uniform_(self.bias_support)
        
    def calculate_transition(self, adj):
        # note that the adj is larger than 1
        d = torch.sum(adj, dim=1)
        adj = torch.div(adj,d.reshape(-1,1))
        return adj

    def forward(self, input, adj, device):
        if len(input.shape) == 3:
            batch_size = input.shape[0]
            num_nodes = input.shape[1]
            input_size = input.shape[2]
        else:
            batch_size = 1
            num_nodes = input.shape[0]
            input_size = input.shape[1]
            input = input.reshape(batch_size, num_nodes, input_size)
        
        # make state0 (num_nodes, input_size * batch_size)
        state0 = input.permute(1,2,0)
        state0 = state0.reshape(num_nodes,-1)
        states = state0.reshape(1, num_nodes, -1)

        trans = self.calculate_transition(adj)
        
        if self.include_reversed:
            reversed_trans = self.calculate_transition(adj.T)
            transitions = [trans,reversed_trans]
        else:
            transitions = [trans]

        for idx, trans in enumerate(transitions):
            # state1 (num_nodes, input_size * batch_size)   
            state1 = torch.mm(trans, state0)
            states = torch.cat((states, state1.reshape(1, num_nodes, -1)),dim = 0)

            for k in range(2, self.max_diffusion + 1):
                # state2 = 2 * torch.mm(trans, state1) - state0
                state2 = torch.mm(trans, state1)
                # concatenate again (k, num_nodes, input_size * batch_size)
                states = torch.cat((states, state2.reshape(1, num_nodes, -1)),dim = 0)

                # watch out for the order
                state0 = state1
                state1 = state2
        
        K = self.max_diffusion * len(transitions) + 1
        # (batch_size, num_nodes, input_size * K)
        states = states.reshape(K, num_nodes, input_size, batch_size)
        states = states.permute(3, 1, 2, 0)
        states = states.reshape(batch_size, num_nodes, K * input_size)

        # (batch_size, num_nodes, output_size)
        output = torch.bmm(states, self.weight_support.reshape(batch_size,-1, self.out_features)) + self.bias_support.view(1,1,-1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'