import matplotlib.pyplot as plt 
import numpy as np
from torch_geometric.data import Data
import torch 
from sklearn.metrics import mean_squared_error
import scvelo as scv 
import scanpy as sc
import anndata
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

def pca_op(X, n_comps = 2, standardize  = True):
    """\
    Calculate the PCA
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    if standardize:
        pipeline = Pipeline([('standardize', StandardScaler()), ('pca', PCA(n_components=n_comps))])
    else:
        pipeline = Pipeline([('pca', PCA(n_components=n_comps))])
    X_pca = pipeline.fit_transform(X)
    return X_pca

def umap_op(X, n_comps = 2):
    """\
    Calculate the umap
    """
    from umap import UMAP
    Umap = UMAP(n_components=n_comps)
    X_umap = Umap.fit_transform(X)
    return X_umap

def kendalltau(y_pred, y_label):
    from scipy.stats import kendalltau
    if isinstance(y_label, torch.Tensor):
        y_label = y_label.numpy().squeeze()
    else:
        y_label = y_label.squeeze()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy().squeeze()
    else:
        y_pred = y_pred.squeeze()
    tau, p_val = kendalltau(y_pred, y_label)
    return tau

def pearson(y_pred, y_label):
    if isinstance(y_label, torch.Tensor):
        y_label = y_label.numpy().squeeze()
    else:
        y_label = y_label.squeeze()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy().squeeze()
    else:
        y_pred = y_pred.squeeze()

    vx = y_pred - np.mean(y_pred)
    vy = y_label - np.mean(y_label)
    score = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return score

def scatter(model, data, figsize = (15,5), method = 'pca', coloring = "order", metric = "kendall_tau", knn=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # X should be something before pca
    if isinstance(data.x, torch.Tensor):
        X = data.x.numpy()
    elif isinstance(data.x, np.ndarray):
        X = data.x
    else:
        raise ValueError('tensor or numpy array')
    
    if isinstance(data.y, torch.Tensor):
        y = data.y.numpy().squeeze()
    elif isinstance(data.y, np.ndarray):
        y = data.y
    else:
        raise ValueError('tensor or numpy array')
    if method == 'pca':
        X_pca = pca_op(X, n_comps = 2, standardize=False)
    elif method == 'umap':
        X_pca = umap_op(X, n_comps = 2)
    else:
        raise ValueError("either pca or umap")
    
    data = data.to(device)
    pred,_,_ = model(data)

    pred = pred.detach().cpu().numpy().reshape(-1)

    if knn: 
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        ax3.set_title('knn')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if metric == "kendall_tau":
        loss = kendalltau(pred, y)
        ax1.set_title("Prediction, kendalltau="+str(loss)[:5] + " data noise="+str(data.noise[0]))
    elif metric == "pearson":
        loss = pearson(pred, y)
        ax1.set_title("Prediction, pearson="+str(loss)[:5] + " data noise="+str(data.noise[0]))

    else:
        loss = mean_squared_error(y, pred)
        ax1.set_title("Prediction, rmse="+str(loss)[:5])

    ax2.set_title("Ground Truth")

    if coloring == "order":
        y_sorted = sorted(y)
        y = [y_sorted.index(i) for i in y]

        pred_sorted = sorted(pred)
        pred = [pred_sorted.index(i) for i in pred]

    v1 = ax1.scatter(X_pca[:,0],X_pca[:,1], cmap = 'gnuplot', c=pred)
    fig.colorbar(v1, fraction=0.046, pad=0.04, ax = ax1)

    v2 = ax2.scatter(X_pca[:,0],X_pca[:,1], cmap = 'gnuplot', c=y)
    fig.colorbar(v1, fraction=0.046, pad=0.04, ax = ax2)

    if knn:
        edges = data.edge_index.cpu().numpy()
        for i in range(edges.shape[1]):
            ax3.plot(X_pca[edges[:,i]][:,0], X_pca[edges[:,i]][:,1])
    plt.show()