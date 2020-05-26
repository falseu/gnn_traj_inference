import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
import anndata
import networkx as nx
import scvelo as scv

from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
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
        backbones = ["bifurcating"]
        trans_rate = [1, 5, 10]
        split_rate = [10, 1, 0.1]
        num_cells = [1000, 1500, 2000]
        root = 'result_files/'

        combined = [(bb, tr, sr, nc) for bb in backbones for tr in trans_rate for sr in split_rate for nc in num_cells]
        data_list = []

        for item in combined:
            bb, tr, sr, nc = item
            path = root + bb + "_" + str(tr) + "_" + str(sr) + "_" + str(nc)
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

            # compute velocity
            scv.tl.velocity_graph(adata)

            # dimension reduction
            X_concat = np.concatenate((X_spliced,X_unspliced),axis=1)
            pca = PCA(n_components=30, svd_solver='arpack')
            pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=30, svd_solver='arpack'))])
            features_pca = pipeline.fit_transform(X_concat)

            # predict gene expression data
            velo_matrix = adata.layers["velocity"].copy()
            X_pre = X_spliced + velo_matrix/np.linalg.norm(velo_matrix,axis=1)[:,None]*3

            X_pca_ori = pca.fit_transform(X_spliced)
            X_pca_pre = pca.transform(X_pre)
            velo_pca = X_pca_pre - X_pca_ori

            directed_conn = kneighbors_graph(X_pca_ori, n_neighbors=35, mode='connectivity', include_self=False).toarray()
            conn = directed_conn + directed_conn.T
            conn[conn.nonzero()[0],conn.nonzero()[1]] = 1

            x = torch.FloatTensor(X_pca_ori.copy())
            y = torch.FloatTensor(X_obs['sim_time'])

            edge_index = np.array(np.where(conn == 1))
            edge_index = torch.LongTensor(edge_index)

            edge_attr = np.array([])
            for i in range(conn.shape[0]):
                indices = conn[i,:].nonzero()[0]
                diff = X_pca_ori[indices,:] - X_pca_ori[i,:]
                distance = np.linalg.norm(diff, axis=1, ord=2)[:,None]
                penalty = np.matmul(diff, velo_pca[i,:, None])/\
                (np.linalg.norm(velo_pca[i,:], ord=2) * distance)
                penalty = np.nan_to_num(penalty, 0)
                weights = (np.exp(penalty)/np.sum(np.exp(penalty))).squeeze()
                edge_attr = np.append(edge_attr, weights)
            edge_attr = torch.FloatTensor(edge_attr)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            print(data)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)