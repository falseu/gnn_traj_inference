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



class SymsimDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(SymsimDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SymsimVelo.dataset']

    def download(self):
        pass

    def process(self):
        start = 25
        end = 75
        root = 'data/symsim/'
        count = 2 * np.arange(start,end) + 1
        files = ['branch'+str(x) + '/' for x in count]
        data_list = []
        

        for file_name in files:
            X_unspliced = pd.read_csv(root + file_name + "unspliced_counts.txt", sep="\t",header=None).to_numpy().T
            X_spliced = pd.read_csv(root + file_name + "spliced_counts.txt", sep = "\t",header=None).to_numpy().T
            true_velo = pd.read_csv(root + file_name + "true_velo.txt", sep="\t",header=None).to_numpy().T
            true_time = pd.read_csv(root + file_name + "true_time.txt", lineterminator="\n",header=None).to_numpy().squeeze()
            X_obs = pd.DataFrame(data=true_time, index = ['cell_' + str(x) for x in range(X_unspliced.shape[0])], columns = ['sim_time'])


            adata = anndata.AnnData(X = csr_matrix(X_spliced),
                            obs = X_obs,
                            layers = dict(
                                unspliced = csr_matrix(X_unspliced),
                                spliced = csr_matrix(X_spliced),
                                true_velo = true_velo
                            ))

            scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=1000)
            # adata = adata[:,:70]
            # scv.pp.normalize_per_cell(adata)
            # scv.pp.log1p(adata)

            # compute velocity
            scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
            scv.tl.velocity(adata, mode='stochastic')            
            velo_matrix = adata.layers["velocity"].copy()

            X_spliced = adata.X.toarray()

            pipeline = Pipeline([('pca', PCA(n_components=80, svd_solver='arpack'))])
            X_pca = pipeline.fit_transform(X_spliced)

            # X_pre = X_spliced + velo_matrix/np.linalg.norm(velo_matrix,axis=1)[:,None]*3
            X_pre = X_spliced + velo_matrix

            X_pca_pre = pipeline.transform(X_pre)
            velo_pca = X_pca_pre - X_pca

            directed_conn = kneighbors_graph(X_pca, n_neighbors=10, mode='connectivity', include_self=False).toarray()
            conn = directed_conn + directed_conn.T
            conn[conn.nonzero()[0],conn.nonzero()[1]] = 1

            # X_spliced is original, X_pca is after pca
            x = X_spliced.copy()
            x = StandardScaler().fit_transform(x)
            x = torch.FloatTensor(x)
            # x = torch.FloatTensor(X_pca.copy())

            # X_pca_pre is after pca
            v = torch.FloatTensor(X_pre.copy())
            # v = torch.FloatTensor(X_pca_pre.copy())

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
                # self loop
                adj[i][i] = 0

                indices = conn[i,:].nonzero()[0]
                for k in indices:
                    diff = X_pca[i, :] - X_pca[k, :] # 1,d
                    distance = np.linalg.norm(diff, ord=2) #1
                    # penalty = np.dot(diff, velo_matrix[k, :, None]) / np.linalg.norm(velo_matrix[k,:], ord=2) / distance
                    penalty = np.dot(diff, velo_pca[k, :, None]) / np.linalg.norm(velo_pca[k,:], ord=2) / distance
                    penalty = 0 if np.isnan(penalty) else penalty
                    adj[i][k] = penalty
                
            
            data = Data(x=x, edge_index=edge_index, y=y, adj=adj, v=v)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)