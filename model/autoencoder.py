import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.mylayer import GraphConvolution
from model.GraphSAGE import GraphConvolutionSage
from scipy.linalg import block_diag
from model.DiffusionNN import GraphDiffusion

class AutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, latent_channels):
        super(AutoEncoder, self).__init__()

        # encoder
        self.lin1 = nn.Linear(in_channels, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, latent_channels)

        # decoder
        self.lin4 = nn.Linear(latent_channels, 64)
        self.lin5 = nn.Linear(64, 128)
        self.lin6 = nn.Linear(128, in_channels)

    def forward(self, x):
        # encode
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        # embedding
        x = self.lin3(x)
        embeddings = x

        # decode
        x = F.relu(self.lin4(x))
        before_reconstruct = F.relu(self.lin5(x))
        reconstruct = self.lin6(before_reconstruct)

        return reconstruct, before_reconstruct, embeddings