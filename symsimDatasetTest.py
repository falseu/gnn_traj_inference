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
from utils import process_adata, technological_noise, calculate_adj

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
            X_spliced = technological_noise(X_spliced, capture_rate=0.2)
            X_unspliced = technological_noise(X_unspliced, capture_rate=0.2)

            adata = anndata.AnnData(X = csr_matrix(X_spliced),
                            obs = X_obs,
                            layers = dict(
                                unspliced = csr_matrix(X_unspliced),
                                spliced = csr_matrix(X_spliced),
                                true_velo = true_velo
                            ))

            data = process_adata(adata, noise=0.2)
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
        tree_files = ['tree/rand'+str(i)+'/' for i in range(1,51)]

        for file_name in tree_files:
            X_unspliced = pd.read_csv(root + file_name + "unspliced_counts.txt", sep="\t",header=None).to_numpy().T
            X_spliced = pd.read_csv(root + file_name + "spliced_counts.txt", sep = "\t",header=None).to_numpy().T
            true_velo = pd.read_csv(root + file_name + "true_velo.txt", sep="\t",header=None).to_numpy().T
            true_time = pd.read_csv(root + file_name + "cell_labels.txt", sep = "\t",lineterminator="\n").to_numpy()
            X_obs = pd.DataFrame(data=true_time, index = ['cell_' + str(x) for x in range(X_unspliced.shape[0])], columns = ['back_bone','sim_time'])

            # add noise
            X_spliced = technological_noise(X_spliced, capture_rate=0.2)
            X_unspliced = technological_noise(X_unspliced, capture_rate=0.2)

            adata = anndata.AnnData(X = csr_matrix(X_spliced),
                            obs = X_obs,
                            layers = dict(
                                unspliced = csr_matrix(X_unspliced),
                                spliced = csr_matrix(X_spliced),
                                true_velo = true_velo
                            ))

            data = process_adata(adata, noise=0.2)
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
            X_spliced = technological_noise(X_spliced, capture_rate=0.2)
            X_unspliced = technological_noise(X_unspliced, capture_rate=0.2)

            adata = anndata.AnnData(X = csr_matrix(X_spliced),
                            obs = X_obs,
                            layers = dict(
                                unspliced = csr_matrix(X_unspliced),
                                spliced = csr_matrix(X_spliced),
                                true_velo = true_velo
                            ))
        
            data = process_adata(adata, noise=0.2)
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

            adata = adata[:,:600]
            print(adata.n_vars)

            data = process_adata(adata)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])       
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


