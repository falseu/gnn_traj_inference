import torch
import numpy as np
import random

class UnsupervisedLoss(object):
    def __init__(self, adj, train_nodes, device):
        super(UnsupervisedLoss, self).__init__()
        self.Q = 10
        self.N_WALKS = 6
        self.WALK_LEN = 1
        self.N_WALK_LEN = 5
        self.MARGIN = 3
        # eliminate self node
        self.adj = adj - adj[0,0] * torch.diag(torch.ones((1,adj.shape[0])))
        self.train_nodes = train_nodes
        self.device = device

        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []


    def get_positive_nodes(self, nodes):
        """\
            Conduct graph random walk with set length self.WALK_LEN and num of walk N_WALK
            
            nodes:
                The starting nodes of the random walk, training nodes
        """
        return self._run_random_walks(nodes)
    
    def _run_random_walks(self, nodes):
        """\
            Conduct graph random walk with set length self.WALK_LEN and num of walk N_WALK
            
            nodes:
                The starting nodes of the random walk, training nodes
        """
        for node in nodes:
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    neighs = np.where(self.adj[curr_node,:] != 0)[0]
                    next_node = random.choice(neighs)
                    # self co-occurrences are useless
                    if next_node != node and next_node in self.train_nodes:
                        self.positive_pairs.append((node,next_node))
                        cur_pairs.append((node,next_node))
                    curr_node = next_node
            # return all neighbor pairs
            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs

    def extend_nodes(self, nodes, num_neg=6):
        """\
            Conduct positive sampling and negative sampling
            
            nodes:
                The starting nodes of the random walk, training nodes
        """
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        # dictionary, with focus node as key
        self.node_negtive_pairs = {}

        self.target_nodes = nodes
        # run random walk and obtain the neighborhood of each node
        self.get_positive_nodes(nodes)

        self.get_negtive_nodes(nodes, num_neg)
        # list of tuple [(node, neigh_i) for node in nodes for neigh_i in neigh(node)]
        print("positive pairs: ", self.positive_pairs)
        print("neg pairs: ", self.negtive_pairs)

        self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
        # print(set(self.unique_nodes_batch))
        assert set(self.target_nodes) < set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    # negative sampling nodes
    def get_negtive_nodes(self, nodes, num_neg):
        """\
            Conduct negative sampling
            
            nodes:
                The starting nodes of the random walk, training nodes
            num_neg:
                Sampling parameters
        """
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    index = set(np.where(self.adj[int(outer),:] != 0)[0])
                    current |= index
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train_nodes) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    # normal version of loss
    def get_loss_sage(self, embeddings, nodes):
        # assert for debug purpose. If true, nothing happens, or the Error is raised
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            # dictionary
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs,:], embeddings[neighb_indexs,:])
            neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            #print(neg_score)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs,:], embeddings[neighb_indexs,:])
            pos_score = torch.log(torch.sigmoid(pos_score))
            #print(pos_score)

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
                
        loss = torch.mean(torch.cat(nodes_score, 0))
        
        return loss