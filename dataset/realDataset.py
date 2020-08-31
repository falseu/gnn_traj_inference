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
from utils import technological_noise, calculate_adj
from backbone import nearest_neighbor

def process_adata(adata, noise=0.0):
    
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

    X_spliced = adata.X.toarray()

    pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
    X_pca = pipeline.fit_transform(X_spliced)

    conn, G = nearest_neighbor(X_pca, k=10, sigma=3)

    # X_spliced is original, X_pca is after pca
    x = X_spliced.copy()
    x = StandardScaler().fit_transform(x)
    x = torch.FloatTensor(x)

    edge_index = np.array(np.nonzero(conn))
    edge_index = torch.LongTensor(edge_index)

    adj = conn.copy()

    data = Data(x=x, edge_index=edge_index, adj=adj, y_dpt = y_dpt, y_vdpt = y_vdpt)
    print(x.shape)
    return data

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
        
        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=300, log=True)
        
        if adata.n_vars > 299:
            adata = adata[::2,:299]
        elif adata.n_vars < 299:
            raise ValueError("Feature number smaller than 300")

        data = process_adata(adata)
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

        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=300, log=True)
        
        if adata.n_vars > 299:
            adata = adata[:,:299]
        elif adata.n_vars < 299:
            raise ValueError("Feature number smaller than 300")
        
        data = process_adata(adata)
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

        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=300, log=True)
        
        if adata.n_vars > 299:
            adata = adata[:,:299]
        elif adata.n_vars < 299:
            raise ValueError("Feature number smaller than 300")

        data = process_adata(adata)
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

        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=300, log=True)
        
        if adata.n_vars > 299:
            adata = adata[:,:299]
        elif adata.n_vars < 299:
            raise ValueError("Feature number smaller than 300")

        data = process_adata(adata)
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

        scv.pp.filter_and_normalize(adata, flavor = 'cell_ranger', min_shared_counts=20, n_top_genes=300, log=True)
        adata = adata[::3,:299]

        data = process_adata(adata)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class paulDataset(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(paulDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['paulDataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
                
        cell_annot = pd.read_csv("./data/real_dataset/Paul_cell_meta.txt", sep="\t")
        expr = pd.read_csv("./data/real_dataset/Paul_expr.txt", sep="\t")
        adata = anndata.AnnData(X=expr.T, obs = cell_annot)
        sc.pp.filter_genes(adata, min_counts = 20)
        sc.pp.normalize_per_cell(adata)
        sc.pp.filter_genes_dispersion(adata, n_top_genes= 300)
        sc.pp.log1p(adata)
        adata = adata[:,:299]


        # dpt
        # adata.uns['iroot'] = 0
        # scv.pp.neighbors(adata)
        # sc.tl.diffmap(adata)
        # sc.tl.dpt(adata)

        # y_dpt = adata.obs['dpt_pseudotime'].to_numpy()

        X_spliced = adata.X

        pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
        X_pca = pipeline.fit_transform(X_spliced)

        directed_conn = kneighbors_graph(X_pca, n_neighbors=10, mode='connectivity', include_self=False).toarray()
        conn = directed_conn + directed_conn.T
        conn[conn.nonzero()[0],conn.nonzero()[1]] = 1

        # X_spliced is original, X_pca is after pca
        x = X_spliced.copy()
        x = StandardScaler().fit_transform(x)
        x = torch.FloatTensor(x)

        edge_index = np.array(np.where(conn == 1))
        edge_index = torch.LongTensor(edge_index)

        adj = conn.copy()
            
        assert adata.n_vars == 299
        
        data = Data(x=x, edge_index=edge_index, adj=adj)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

    
class oeHBCDataset(InMemoryDataset):

    def __init__(self, root='data/', transform=None, pre_transform=None):
        super(oeHBCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['oeHBCDataset.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        X = pd.read_csv("./data/oeHBC/oeHBC/GSE95601_oeHBCdiff_RSEM_eSet_fpkm_table.txt", sep="\t")
        clusters = pd.read_csv("./data/oeHBC/oeHBC/oeHBCdiff_clusterLabels.txt", sep='\t',header=None, index_col = 0, names=["cluster"])
        X = X.fillna(0)[list(clusters.index)]
        clusters = clusters['cluster'].astype('category')
        clusters = pd.DataFrame(clusters.cat.rename_categories(['Resting Horizontal Basal Cells (HBCs)', 'Immediate Neuronal Precursor 1 (INP1)', 'Globose Basal Cells (GBCs)', 'Mature Sustentacular Cells','Transitional HBC 2', 'Immature Sustentacular Cells', 'Transitional HBC 1', 'Immature Olfactory Sensory Neurons (iOSNs)', 'Immediate Neuronal Precursor 3 (INP3)', 'Microvillous Cells, type 1', 'Mature Olfactory Sensory Neurons (mOSNs)', 'Immediate Neuronal Precursor 2 (INP2)', 'Microvillous Cells, type 2']))
        adata = anndata.AnnData(X = X.T, obs = clusters)
        
        # sc.pp.filter_genes(adata, min_counts = 20) 
        # sc.pp.normalize_per_cell(adata)
        sc.pp.filter_genes_dispersion(adata, n_top_genes= 300)
        sc.pp.log1p(adata)
        adata = adata[:,:299]

        X_spliced = adata.X

        pipeline = Pipeline([('pca', PCA(n_components=30, svd_solver='arpack'))])
        X_pca = pipeline.fit_transform(X_spliced)

        directed_conn = kneighbors_graph(X_pca, n_neighbors=10, mode='connectivity', include_self=False).toarray()
        conn = directed_conn + directed_conn.T
        conn[conn.nonzero()[0],conn.nonzero()[1]] = 1

        # X_spliced is original, X_pca is after pca
        x = X_spliced.copy()
        x = StandardScaler().fit_transform(x)
        x = torch.FloatTensor(x)

        edge_index = np.array(np.where(conn == 1))
        edge_index = torch.LongTensor(edge_index)

        adj = conn.copy()
            
        data = Data(x=x, edge_index=edge_index, adj=adj)
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)