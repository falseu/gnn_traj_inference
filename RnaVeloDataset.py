import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
import anndata
import scvelo as scv
import scanpy as sc

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import kneighbors_graph
from utils import process_adata


def technological_noise(count_mt, capture_rate = 0.2):
    
    X = count_mt.astype('int')
    libsize_cell = [np.sum(X[cell,:]) for cell in range(X.shape[0])]

    gene_indices = [[0 for gene in range(libsize_cell[cell])] for cell in range(X.shape[0])]
    sampled_genes = []
    
    for cell_id, gene_idx in enumerate(gene_indices):
        subsample = np.random.uniform(0.0, 1.0, size = len(gene_indices)) > (1-capture_rate)
        sampled_genes.append(subsample)
        idx = 0
        for gene_id, gene_num in enumerate(X[cell_id,:]):
            count = np.sum(subsample[idx:(idx + int(gene_num))])
            X[cell_id, gene_id] = count
            
    return X

def calculate_adj(conn, x, v):
    adj = np.full_like(conn, np.nan)
    for i in range(conn.shape[0]):
        # self loop
        adj[i][i] = 0

        indices = conn[i,:].nonzero()[0]
        for k in indices:
            diff = x[i, :] - x[k, :] # 1,d
            distance = np.linalg.norm(diff, ord=2) #1
            # penalty = np.dot(diff, velo_matrix[k, :, None]) / np.linalg.norm(velo_matrix[k,:], ord=2) / distance
            penalty = np.dot(diff, v[k, :, None]) / np.linalg.norm(v[k,:], ord=2) / distance
            penalty = 0 if np.isnan(penalty) else penalty
            adj[i][k] = penalty
    return adj

class RnaVeloDataset(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(RnaVeloDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['Dyngen.dataset']

    def download(self):
        pass

    def process(self):
    
        num_cells = [300 + 10 * i for i in range(11)]
        root = 'data/dyngen/'

        data_list = []

        for nc in num_cells:
            path = root + 'bifurcating_1_1_1_' + str(nc)
            print(path)

            df = pd.read_csv(path + "_unspliced.csv")
            df = df.drop(df.columns[[0]], axis=1)
            X_unspliced = df.to_numpy()

            df = pd.read_csv(path + "_spliced.csv")
            df = df.drop(df.columns[[0]], axis=1)
            X_spliced = df.to_numpy()

            df = pd.read_csv(path + "_cell_info.csv")
            df = df.drop(df.columns[[0]], axis=1)
            X_obs = df

            adata = anndata.AnnData(X = csr_matrix(X_spliced),
                            obs = X_obs,
                            layers = dict(
                                unspliced = csr_matrix(X_unspliced),
                                spliced = csr_matrix(X_spliced)
                            ))

            data = process_adata(adata)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)