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

        root = '5_backbone/'
        
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

            # dimension reduction
            # X_concat = np.concatenate((X_spliced,X_unspliced),axis=1)
            # pca = PCA(n_components=10, svd_solver='arpack')
            # pipeline = Pipeline([('normalization', Normalizer()), ('pca', PCA(n_components=10, svd_solver='arpack'))])
            # X_pca_ori = pipeline.fit_transform(X_spliced)

            scv.pp.filter_genes_dispersion(adata, n_top_genes = 72)
            adata = adata[:,:70]
            scv.pp.normalize_per_cell(adata)
            scv.pp.log1p(adata)

            # compute velocity
            # scv.pp.filter_and_normalize(adata, n_top_genes=50)
            scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
            # scv.tl.recover_dynamics(adata)
            scv.tl.velocity(adata)
            velo_matrix = adata.layers["velocity"].copy()
            # genes_subset = ~np.isnan(velo_matrix).any(axis=0)
            # adata._inplace_subset_var(genes_subset)
            # velo_matrix = adata.layers["velocity"].copy()
            X_spliced = adata.X.toarray()
            X_pca_ori = X_spliced
            print(X_pca_ori.shape)

            # X_pre = X_spliced + velo_matrix/np.linalg.norm(velo_matrix,axis=1)[:,None]*3

            # X_pca_pre = pipeline.transform(X_pre)
            # velo_pca = X_pca_pre - X_pca_ori

            directed_conn = kneighbors_graph(X_pca_ori, n_neighbors=5, mode='connectivity', include_self=False).toarray()
            conn = directed_conn + directed_conn.T
            conn[conn.nonzero()[0],conn.nonzero()[1]] = 1
            
            x = torch.FloatTensor(X_pca_ori.copy())

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
                # indices = conn[i,:].nonzero()[0]
                # diff = X_pca_ori[indices,:] - X_pca_ori[i,:]
                # distance = np.linalg.norm(diff, axis=1, ord=2)[:,None]
                # penalty = np.matmul(diff, velo_pca[i,:, None])/\
                # (np.linalg.norm(velo_pca[i,:], ord=2) * distance)
                # penalty = np.nan_to_num(penalty, 0)
                # adj[i][indices] = penalty.squeeze()

                # self loop
                adj[i][i] = 0
                
                # weights = np.full((conn.shape[1]), np.inf)
                # weights[indices] = penalty.squeeze()
                # # Self loop
                # weights[i] = 1
                # adj = np.stack((adj, weights))

                indices = conn[i,:].nonzero()[0]
                for k in indices:
                    diff = X_pca_ori[i, :] - X_pca_ori[k, :] # 1,d
                    distance = np.linalg.norm(diff, ord=2) #1
                    penalty = np.dot(diff, velo_matrix[k, :, None]) / np.linalg.norm(velo_matrix[k,:], ord=2) / distance
                    penalty = 0 if np.isnan(penalty) else penalty
                    adj[i][k] = penalty
                
            # adj = torch.FloatTensor(adj)
            
            data = Data(x=x, edge_index=edge_index, y=y, adj=adj)
            data.adata = adata
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)