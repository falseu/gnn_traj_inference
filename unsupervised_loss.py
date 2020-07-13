import torch
import numpy as np
import random
import torch.nn.functional as F

class UnsupervisedLoss(object):
    def __init__(self, adj, train_nodes, device, Q, hop_pos=2, hop_neg=10):
        super(UnsupervisedLoss, self).__init__()
        self.Q = Q
        self.adj = adj
        self.train_nodes = train_nodes
        self.device = device
        self.node_pos_neighbor = torch.zeros_like(adj)
        self.node_neg_neighbor = torch.zeros_like(adj)
        self.hop_pos = hop_pos
        self.hop_neg = hop_neg

    def get_pos_neg_neighborhood(self, nodes):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.hop_neg):
                current = set()
                for outer in frontier:
                    index = set(np.where(self.adj[outer,:] != 0.0)[0])
                    # print("neg sampling:", index)
                    current |= index
                frontier = current - neighbors
                # print("current: ", current)
                neighbors |= current
                if i == self.hop_pos:
                    pos_neighbors = neighbors.copy()
            far_nodes = set(self.train_nodes) - neighbors
            
            self.node_pos_neighbor[node][list(pos_neighbors)] = 1
            self.node_neg_neighbor[node][list(far_nodes)] = 1

    # normal version of loss
    def get_loss_sage(self, embeddings, nodes, pos_size, neg_size):

        total = 0.0
        for node in nodes:
            pos = self.node_pos_neighbor[node]
            neg = self.node_neg_neighbor[node]

            pos = torch.where(pos == 1)[0]
            neg = torch.where(neg == 1)[0]

            pos_indices = torch.randperm(pos.shape[0])[:pos_size]
            neg_indices = torch.randperm(neg.shape[0])[:neg_size]

            pos_score = F.cosine_similarity(embeddings[torch.full((pos_indices.shape[0]), node),:], embeddings[pos_indices,:])
            neg_score = F.cosine_similarity(embeddings[torch.full((neg_indices.shape[0]), node),:], embeddings[neg_indices,:])

            pos_score = torch.log(torch.sigmoid(pos_score))
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)

            total += torch.mean(- pos_score - neg_score)

        total /= nodes.shape[0]
        return total
