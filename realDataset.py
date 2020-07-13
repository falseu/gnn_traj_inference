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

class forebrainDataset(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(forebrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['forebrainDataset.dataset']

    def download(self):
        pass

    def process(self):
        root = 'data/real_dataset/linear/'

        data_list = []
        
        path = root

        df = pd.read_csv(path + "unspliced.csv")
        df = df.drop(df.columns[[0]], axis=1)
        X_unspliced = df.to_numpy().T
        print(X_unspliced.shape)

        df = pd.read_csv(path + "spliced.csv")
        df = df.drop(df.columns[[0]], axis=1)
        X_spliced = df.to_numpy().T
        print(X_unspliced.shape)

        df = pd.read_csv(path + "velocity.csv")
        df = df.drop(df.columns[[0]], axis=1)
        velo_matrix = df.to_numpy().T

        adata = anndata.AnnData(X = csr_matrix(X_spliced),
                        layers = dict(
                            unspliced = csr_matrix(X_unspliced),
                            spliced = csr_matrix(X_spliced),
                        ))

        # scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=305)
        # adata = adata[::3,:300]
        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=301, log=True)
        try:
            adata = adata[::3,:300]
        except:
            raise ValueError("Feature number smaller than 300")
        
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode='stochastic') 
           
        velo_matrix = adata.layers["velocity"].copy()

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

        X_spliced = adata.X.toarray()

        pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
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

        # X_pca_pre is after pca
        v = StandardScaler().fit_transform(X_pre)
        v = torch.FloatTensor(v)

        # Simulation time label
        # y = adata.obs['sim_time'].to_numpy().reshape((-1, 1))
        # scaler = MinMaxScaler((0, 1))
        # scaler.fit(y)
        # y = torch.FloatTensor(scaler.transform(y).reshape(-1, 1))

        # Graph type label
        # y = torch.LongTensor(np.where(np.array(backbones) == bb)[0])

        edge_index = np.array(np.where(conn == 1))
        edge_index = torch.LongTensor(edge_index)

        adj = calculate_adj(conn, X_pca, velo_pca)
            
        assert adata.n_vars == 300
        
        data = Data(x=x, edge_index=edge_index, adj=adj, v=v, y_dpt = y_dpt, y_vdpt = y_vdpt)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class forebrainDataset_large(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(forebrainDataset_large, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['forebrainDataset_large.dataset']

    def download(self):
        pass

    def process(self):
        root = 'data/real_dataset/linear/'

        data_list = []
        
        path = root

        df = pd.read_csv(path + "unspliced.csv")
        df = df.drop(df.columns[[0]], axis=1)
        X_unspliced = df.to_numpy().T
        print(X_unspliced.shape)

        df = pd.read_csv(path + "spliced.csv")
        df = df.drop(df.columns[[0]], axis=1)
        X_spliced = df.to_numpy().T
        print(X_unspliced.shape)

        df = pd.read_csv(path + "velocity.csv")
        df = df.drop(df.columns[[0]], axis=1)
        velo_matrix = df.to_numpy().T

        adata = anndata.AnnData(X = csr_matrix(X_spliced),
                        layers = dict(
                            unspliced = csr_matrix(X_unspliced),
                            spliced = csr_matrix(X_spliced),
                        ))

        # scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=305)
        # adata = adata[::3,:300]
        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=301, log=True)
        
        if adata.n_vars > 300:
            adata = adata[:,:300]
        elif adata.n_vars < 300:
            raise ValueError("Feature number smaller than 300")
        
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode='stochastic') 
           
        velo_matrix = adata.layers["velocity"].copy()

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

        X_spliced = adata.X.toarray()

        pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
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

        # X_pca_pre is after pca
        v = StandardScaler().fit_transform(X_pre)
        v = torch.FloatTensor(v)

        # Simulation time label
        # y = adata.obs['sim_time'].to_numpy().reshape((-1, 1))
        # scaler = MinMaxScaler((0, 1))
        # scaler.fit(y)
        # y = torch.FloatTensor(scaler.transform(y).reshape(-1, 1))

        # Graph type label
        # y = torch.LongTensor(np.where(np.array(backbones) == bb)[0])

        edge_index = np.array(np.where(conn == 1))
        edge_index = torch.LongTensor(edge_index)

        adj = calculate_adj(conn, X_pca, velo_pca)
            
        assert adata.n_vars == 300
        
        data = Data(x=x, edge_index=edge_index, adj=adj, v=v, y_dpt = y_dpt, y_vdpt = y_vdpt)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)



class chromaffinDataset(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(chromaffinDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['chromaffinDataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        
        adata = anndata.read_h5ad('./data/real_dataset/chromaffin.h5ad')

        # scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=300)
        # scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        # scv.tl.velocity(adata, mode='stochastic')   

        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=301, log=True)
        
        if adata.n_vars > 300:
            adata = adata[:,:300]
        elif adata.n_vars < 300:
            raise ValueError("Feature number smaller than 300")

        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode='stochastic')     
             
        velo_matrix = adata.layers["velocity"].copy()
       
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

        X_spliced = adata.X.toarray()

        pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
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

        # X_pca_pre is after pca
        v = StandardScaler().fit_transform(X_pre)
        v = torch.FloatTensor(v)

        # Simulation time label
        # y = adata.obs['sim_time'].to_numpy().reshape((-1, 1))
        # scaler = MinMaxScaler((0, 1))
        # scaler.fit(y)
        # y = torch.FloatTensor(scaler.transform(y).reshape(-1, 1))

        # Graph type label
        # y = torch.LongTensor(np.where(np.array(backbones) == bb)[0])

        edge_index = np.array(np.where(conn == 1))
        edge_index = torch.LongTensor(edge_index)

        adj = calculate_adj(conn, X_pca, velo_pca)
            
        assert adata.n_vars == 300
        
        data = Data(x=x, edge_index=edge_index, adj=adj, v=v, y_dpt = y_dpt, y_vdpt = y_vdpt)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class peDataset_large(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(peDataset_large, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['peDataset_large.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        
        adata = anndata.read_h5ad('./data/real_dataset/endocrinogenesis_day15.h5ad')

        # scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=300)
        # scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        # scv.tl.velocity(adata, mode='stochastic')   

        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=301, log=True)
        
        if adata.n_vars > 300:
            adata = adata[:,:300]
        elif adata.n_vars < 300:
            raise ValueError("Feature number smaller than 300")

        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode='stochastic')     
             
        velo_matrix = adata.layers["velocity"].copy()
       
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

        X_spliced = adata.X.toarray()

        pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
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

        # X_pca_pre is after pca
        v = StandardScaler().fit_transform(X_pre)
        v = torch.FloatTensor(v)

        # Simulation time label
        # y = adata.obs['sim_time'].to_numpy().reshape((-1, 1))
        # scaler = MinMaxScaler((0, 1))
        # scaler.fit(y)
        # y = torch.FloatTensor(scaler.transform(y).reshape(-1, 1))

        # Graph type label
        # y = torch.LongTensor(np.where(np.array(backbones) == bb)[0])

        edge_index = np.array(np.where(conn == 1))
        edge_index = torch.LongTensor(edge_index)

        adj = calculate_adj(conn, X_pca, velo_pca)
            
        assert adata.n_vars == 300
        
        data = Data(x=x, edge_index=edge_index, adj=adj, v=v, y_dpt = y_dpt, y_vdpt = y_vdpt)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class peDataset(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(peDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['peDataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        
        adata = anndata.read_h5ad('./data/real_dataset/endocrinogenesis_day15.h5ad')

        # scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=300)
        # scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        # scv.tl.velocity(adata, mode='stochastic')   

        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=301, log=True)
        adata = adata[::5,:300]

        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode='stochastic')     
             
        velo_matrix = adata.layers["velocity"].copy()
       
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

        X_spliced = adata.X.toarray()

        pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
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

        # X_pca_pre is after pca
        v = StandardScaler().fit_transform(X_pre)
        v = torch.FloatTensor(v)

        # Simulation time label
        # y = adata.obs['sim_time'].to_numpy().reshape((-1, 1))
        # scaler = MinMaxScaler((0, 1))
        # scaler.fit(y)
        # y = torch.FloatTensor(scaler.transform(y).reshape(-1, 1))

        # Graph type label
        # y = torch.LongTensor(np.where(np.array(backbones) == bb)[0])

        edge_index = np.array(np.where(conn == 1))
        edge_index = torch.LongTensor(edge_index)

        adj = calculate_adj(conn, X_pca, velo_pca)
            
        assert adata.n_vars == 300
        
        data = Data(x=x, edge_index=edge_index, adj=adj, v=v, y_dpt = y_dpt, y_vdpt = y_vdpt)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
