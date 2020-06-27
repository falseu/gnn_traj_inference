import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
import anndata
import scvelo as scv
import scanpy

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import kneighbors_graph



class RnaVeloDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(RnaVeloDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['RnaVelo.dataset']

    def download(self):
        pass

    def process(self):
        # backbones = ["bifurcating", "binary_tree", "cycle", "linear", "trifurcating"]
        backbones = ["bifurcating", "linear", "binary_tree", "trifurcating"]
        seed = [1]
        trans_rate = [3, 5, 10]
        num_cells = [100, 150, 200, 250, 300]

        # seed = [6]
        # trans_rate = [1]
        # split_rate = [1]
        # num_cells=[3000]

        root = 'data/5_backbone/'
        
        combined = [(bb, tr, nc) for bb in backbones for tr in trans_rate for nc in num_cells]
        data_list = []

        for item in combined:
            bb, tr, nc = item
            path = root + bb + "_" + str(tr) + "_1_1_" + str(nc)
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

            scv.pp.filter_genes_dispersion(adata, n_top_genes = 72)
            adata = adata[:,:70]
            scv.pp.normalize_per_cell(adata)
            scv.pp.log1p(adata)

            # compute velocity
            scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
            scv.tl.velocity(adata)
            velo_matrix = adata.layers["velocity"].copy()
            X_spliced = adata.X.toarray()
            pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
            X_pca = pipeline.fit_transform(X_spliced)

            X_pre = X_spliced + velo_matrix
            X_pca_pre = pipeline.transform(X_pre)
            velo_pca = X_pca_pre - X_pca

            directed_conn = kneighbors_graph(X_pca, n_neighbors=5, mode='connectivity', include_self=False).toarray()
            conn = directed_conn + directed_conn.T
            conn[conn.nonzero()[0],conn.nonzero()[1]] = 1
            
            # X_spliced is original, X_pca is after pca
            x = X_spliced.copy()
            x = StandardScaler().fit_transform(x)
            x = torch.FloatTensor(x)

            # X_pca_pre is after pca
            v = StandardScaler().fit_transform(X_pre)
            v = torch.FloatTensor(v)

            # Simulation time label
            y = X_obs['sim_time'].to_numpy().reshape((-1, 1))
            scaler = MinMaxScaler((0, 1))
            scaler.fit(y)
            y = torch.FloatTensor(scaler.transform(y).reshape(-1, 1))

            # Graph type label
            # y = torch.LongTensor(np.where(np.array(backbones) == bb)[0])

            edge_index = np.array(np.where(conn == 1))
            edge_index = torch.LongTensor(edge_index)

            adj = np.full_like(conn, np.nan)
            for i in range(conn.shape[0]):
                adj[i][i] = 0

                indices = conn[i,:].nonzero()[0]
                for k in indices:
                    diff = X_pca[i, :] - X_pca[k, :] # 1,d
                    distance = np.linalg.norm(diff, ord=2) #1
                    penalty = np.dot(diff, velo_pca[k, :, None]) / np.linalg.norm(velo_pca[k,:], ord=2) / distance
                    penalty = 0 if np.isnan(penalty) else penalty
                    adj[i][k] = penalty
                
            # adj = torch.FloatTensor(adj)
            
            data = Data(x=x, edge_index=edge_index, y=y, adj=adj, v=v)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)