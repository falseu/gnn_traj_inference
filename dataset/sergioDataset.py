import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch.nn.functional as F

import pandas as pd
import anndata
import scvelo as scv
import scanpy as sc

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from backbone import nearest_neighbor
from utils import technological_noise, calculate_adj, process_adata, process_adata_novelo

class SergioBifur(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(SergioBifur, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SergioBifur.dataset']

    def download(self):
        pass

    def process(self):
        root = "data/sergio/"
        data_list = []

        # print(file_name)
        # print(root + file_name)
        # X_unspliced = pd.read_csv(root + file_name + "unspliced_counts.txt", sep="\t",header=None).to_numpy().T
        # X_spliced = pd.read_csv(root + file_name + "spliced_counts.txt", sep = "\t",header=None).to_numpy().T
        # true_velo = pd.read_csv(root + file_name + "true_velo.txt", sep="\t",header=None).to_numpy().T
        # true_time = pd.read_csv(root + file_name + "true_time.txt", lineterminator="\n",header=None).to_numpy().squeeze()
        # X_obs = pd.DataFrame(data=true_time, index = ['cell_' + str(x) for x in range(X_unspliced.shape[0])], columns = ['sim_time'])

        adata = adata = anndata.read_h5ad(root+"sergio_bifur_4.h5ad")

        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=0, n_top_genes=300, log=True)
        if adata.n_vars > 299:
            adata = adata[:,:299]
        elif adata.n_vars < 299:
            raise ValueError("Feature number", adata.n_vars)

        print(adata.X.shape)

        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode='stochastic') 

        adata2 = adata.copy()
        # dpt
        adata.uns['iroot'] = 0
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        #velo-dpt
        scv.tl.velocity_graph(adata2)
        scv.tl.velocity_pseudotime(adata2)
        y_dpt = adata.obs['dpt_pseudotime'].to_numpy()
        y_vdpt = adata2.obs['velocity_pseudotime'].to_numpy()

        X_spliced = adata.X

        pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
        X_pca = pipeline.fit_transform(X_spliced)

        conn, G = nearest_neighbor(X_pca, k=10, sigma=3)

        # X_spliced is original, X_pca is after pca
        x = X_spliced.copy()
        x = StandardScaler().fit_transform(x)
        x = torch.FloatTensor(x.copy())

        edge_index = np.array(np.nonzero(conn))
        edge_index = torch.LongTensor(edge_index)

        adj = conn.copy()

        data = Data(x=x, edge_index=edge_index, adj=adj, y_dpt = y_dpt, y_vdpt = y_vdpt)
        print(x.shape)
        # for capture_rate in [0.2, 0.4]:
        #     noisy_X_unspliced = technological_noise(X_unspliced, capture_rate=capture_rate)
        #     noisy_X_spliced = technological_noise(X_spliced, capture_rate=capture_rate)

        #     adata = anndata.AnnData(X = csr_matrix(X_spliced),
        #                 obs = X_obs,
        #                 layers = dict(
        #                     unspliced = csr_matrix(noisy_X_unspliced),
        #                     spliced = csr_matrix(noisy_X_spliced),
        #                     true_velo = true_velo
        #                 ))

        #     data = process_adata(adata, noise=capture_rate)
        #     data_list.append(data)
        
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
