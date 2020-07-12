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
from sklearn.neighbors import kneighbors_graph

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

def process_adata(adata, noise=0.0):
    # scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=305)
    # adata = adata[:,:300]
    
    scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=301, log=True)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode='stochastic') 

    print(adata.n_vars)

    # compute velocity
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
    y = adata.obs['sim_time'].to_numpy().reshape((-1, 1))
    scaler = MinMaxScaler((0, 1))
    scaler.fit(y)
    y = torch.FloatTensor(scaler.transform(y).reshape(-1, 1))

    # Graph type label
    # y = torch.LongTensor(np.where(np.array(backbones) == bb)[0])

    edge_index = np.array(np.where(conn == 1))
    edge_index = torch.LongTensor(edge_index)

    adj = calculate_adj(conn, X_pca, velo_pca)

    data = Data(x=x, edge_index=edge_index, y=y, adj=adj, v=v, y_dpt = y_dpt, y_vdpt = y_vdpt, noise=noise)
    return data

class SymsimBifurNoisy(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(SymsimBifurNoisy, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SymsimBifurNoisy.dataset']

    def download(self):
        pass

    def process(self):
        start = 25
        end = 75
        root = "data/symsim/"
        count = 2 * np.arange(start,end) + 1
        files = ['bifur/branch'+str(x) + '/' for x in count]
        data_list = []

        for file_name in files:
            print(self.root)
            print(file_name)
            print(root + file_name)
            X_unspliced = pd.read_csv(root + file_name + "unspliced_counts.txt", sep="\t",header=None).to_numpy().T
            X_spliced = pd.read_csv(root + file_name + "spliced_counts.txt", sep = "\t",header=None).to_numpy().T
            true_velo = pd.read_csv(root + file_name + "true_velo.txt", sep="\t",header=None).to_numpy().T
            true_time = pd.read_csv(root + file_name + "true_time.txt", lineterminator="\n",header=None).to_numpy().squeeze()
            X_obs = pd.DataFrame(data=true_time, index = ['cell_' + str(x) for x in range(X_unspliced.shape[0])], columns = ['sim_time'])

            # add noise
            X_spliced = technological_noise(X_spliced)
            X_unspliced = technological_noise(X_unspliced)

            adata = anndata.AnnData(X = csr_matrix(X_spliced),
                            obs = X_obs,
                            layers = dict(
                                unspliced = csr_matrix(X_unspliced),
                                spliced = csr_matrix(X_spliced),
                                true_velo = true_velo
                            ))

            data = process_adata(adata)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class SymsimTreeNoisy(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(SymsimTreeNoisy, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SymsimTreeNoisy.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        root = "data/symsim/"
        tree_files = ['tree/rand'+str(i)+'/' for i in range(1,101)]

        for file_name in tree_files:
            X_unspliced = pd.read_csv(root + file_name + "unspliced_counts.txt", sep="\t",header=None).to_numpy().T
            X_spliced = pd.read_csv(root + file_name + "spliced_counts.txt", sep = "\t",header=None).to_numpy().T
            true_velo = pd.read_csv(root + file_name + "true_velo.txt", sep="\t",header=None).to_numpy().T
            true_time = pd.read_csv(root + file_name + "cell_labels.txt", sep = "\t",lineterminator="\n").to_numpy()
            X_obs = pd.DataFrame(data=true_time, index = ['cell_' + str(x) for x in range(X_unspliced.shape[0])], columns = ['back_bone','sim_time'])

            # add noise
            X_spliced = technological_noise(X_spliced)
            X_unspliced = technological_noise(X_unspliced)

            adata = anndata.AnnData(X = csr_matrix(X_spliced),
                            obs = X_obs,
                            layers = dict(
                                unspliced = csr_matrix(X_unspliced),
                                spliced = csr_matrix(X_spliced),
                                true_velo = true_velo
                            ))

            data = process_adata(adata)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class SymsimLinearNoisy(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(SymsimLinearNoisy, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SymsimLinearNoisy.dataset']

    def download(self):
        pass

    def process(self):
        root = "data/symsim/"
        tree_files = ['linear/rand'+str(i)+'/' for i in range(1,51)]
        data_list = []

        for file_name in tree_files:
            X_unspliced = pd.read_csv(root + file_name + "unspliced_counts.txt", sep="\t",header=None).to_numpy().T
            X_spliced = pd.read_csv(root + file_name + "spliced_counts.txt", sep = "\t",header=None).to_numpy().T
            true_velo = pd.read_csv(root + file_name + "true_velo.txt", sep="\t",header=None).to_numpy().T
            true_time = pd.read_csv(root + file_name + "cell_labels.txt", sep = "\t",lineterminator="\n").to_numpy()
            X_obs = pd.DataFrame(data=true_time, index = ['cell_' + str(x) for x in range(X_unspliced.shape[0])], columns = ['back_bone','sim_time'])

            # add noise
            X_spliced = technological_noise(X_spliced)
            X_unspliced = technological_noise(X_unspliced)

            adata = anndata.AnnData(X = csr_matrix(X_spliced),
                            obs = X_obs,
                            layers = dict(
                                unspliced = csr_matrix(X_unspliced),
                                spliced = csr_matrix(X_spliced),
                                true_velo = true_velo
                            ))
        
            data = process_adata(adata)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])       
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class dyngenBifur(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(dyngenBifur, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dyngenBifur.dataset']

    def download(self):
        pass

    def process(self):
        root = 'data/dyngen/'
        backbones = ['bifur/bifurcating', 'trifur/trifurcating']
        num_cells = ['300','500']
        
        combined = [(bb, nc) for bb in backbones for nc in num_cells]
        data_list = []
        
        for item in combined:
            bb, nc = item
            path = root + bb + "_1_1_1_" + nc

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

            scv.pp.filter_and_normalize(adata, min_shared_counts=0, n_top_genes=305)

            adata = adata[:,:300]
            print(adata.n_vars)

            # compute velocity
            scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
            scv.tl.velocity(adata, mode='stochastic')  

            adata2 = adata.copy()
            # dpt
            print(np.argmin(adata.obs['sim_time'].to_numpy()))
            adata.uns['iroot'] = np.argmin(adata.obs['sim_time'].to_numpy())
            sc.tl.diffmap(adata)
            sc.tl.dpt(adata)
            #velo-dpt
            scv.tl.velocity_graph(adata2)
            scv.tl.velocity_pseudotime(adata2)
            y_dpt = adata.obs['dpt_pseudotime'].to_numpy()
            y_vdpt = adata2.obs['velocity_pseudotime'].to_numpy()    

            velo_matrix = adata.layers["velocity"].copy()

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
            y = adata.obs['sim_time'].to_numpy().reshape((-1, 1))
            scaler = MinMaxScaler((0, 1))
            scaler.fit(y)
            y = torch.FloatTensor(scaler.transform(y).reshape(-1, 1))

            # Graph type label
            # y = torch.LongTensor(np.where(np.array(backbones) == bb)[0])

            edge_index = np.array(np.where(conn == 1))
            edge_index = torch.LongTensor(edge_index)

            adj = calculate_adj(conn, X_pca, velo_pca)
            
            data = Data(x=x, edge_index=edge_index, y=y, adj=adj, v=v, y_dpt = y_dpt, y_vdpt = y_vdpt)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])       
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


