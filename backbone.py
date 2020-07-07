import numpy as np
import torch
import utils

def random_walk(X, y, t_max = 10000, i_max = 10, k = 10):
    from sklearn.neighbors import kneighbors_graph
    import random
    root_idices = np.argsort(y)[:10] # np.argmin(y)
    directed_conn = kneighbors_graph(X, n_neighbors = k, mode='connectivity', include_self=False).toarray()
    conn = directed_conn + directed_conn.T
    conn[conn.nonzero()[0],conn.nonzero()[1]] = 1
    for curr_idx in range(conn.shape[0]):
        neigh = np.where(conn[curr_idx,:]!=0)[0]
        p = 1/(np.sum(y[neigh] > y[curr_idx]) + 0.01)
        p_neigh = (y[neigh] > y[curr_idx]) * p
        conn[curr_idx, neigh] = p_neigh

    randwk = {}
    for i in range(i_max):
        t = 0
        randwk[i] = [random.choice(root_idices)]
        while(t < t_max):
            curr = randwk[i][-1]
            neighs = np.where(conn[curr,:]!=0)[0]
            if neighs.shape[0] != 0:
                # randwk[i].append(random.choices(neighs))
                randwk[i].append(random.choices(np.arange(conn.shape[0]), weights=conn[curr,:], k = 1)[0])
            else:
                break
    return randwk

def segmentation(random_wk, node_features, y, cell_backbone = None, wind_size = 10, step = 2):
    belongings = []
    segments = []
    features = []
    bbs = []
    pseudo_time = []
    
    for i in random_wk.keys():
        rwk_i = random_wk[i]
        curr_step = 0
        while(curr_step+wind_size < len(rwk_i)):
            segments.append(rwk_i[curr_step:curr_step+wind_size])
            belongings.append([i])
            curr_step += step
    
    segments = np.array(segments)
    belongings = np.array(belongings)

    for seg in range(segments.shape[0]):
        features.append([])
        bbs.append([])
        pseudo_time.append([])
        for idx in segments[seg,:]:
            features[-1].extend(node_features[idx,:])
            pseudo_time[-1].append(y[idx])
        #     if cell_backbone:
        #         bbs[-1].append(cell_backbone[idx])
        # unique, counts = np.unique(np.array(bbs[-1]), return_counts=True)
        # # print("seg: ", seg, ", bbs[-1]: ", unique[np.argmax(counts)], ", bbs: ", bbs[-1])
        # bbs[-1] = unique[np.argmax(counts)]    
    
    
    block_diagonal = []
    adj = np.zeros((segments.shape[0],segments.shape[0]))

    for i in random_wk.keys():
        indices = np.where(belongings.squeeze() == i)[0]
        adj_temp = np.eye(indices.shape[0], indices.shape[0])
        adj[indices[0]:indices[-1], indices[0]:indices[-1]+1] += adj_temp[1:, :]
        adj[indices[1]:indices[-1]+1, indices[0]:indices[-1]+1] += adj_temp[0:-1, :]    
    
    # if cell_backbone:
    #     return {"rwk": belongings, "segments": segments, "seg_features": np.array(features), "adjacency": adj, "seg_backbone": np.array(bbs), "pseudo_time": np.array(pseudo_time)}
    # else:
    return {"rwk": belongings, "segments": segments, "adjacency": adj, "seg_features": np.array(features), "pseudo_time": np.array(pseudo_time)}
    
def retrieve_conn_eigen_pool(seg, groups):
    """\
        seg: the segmentation result
        groups: the clustered result for segments, of the shape [seg,]
    """
    segments = seg['segments']
    adj = seg['adjacency']
    n_segs = segments.shape[0]

    n_groups = int(np.max(groups) + 1)
    adj_int = np.zeros(adj.shape)
    assign = np.zeros((n_groups, n_segs))

    for group in range(n_groups):
        print(group)
        indices = np.where(groups == group)[0]
        c_k = np.zeros((adj.shape[0], indices.shape[0]))
        c_k[indices, :] = np.eye(indices.shape[0])
        adj_k = np.matmul(np.matmul(c_k.T, adj),c_k)
        adj_int += np.matmul(np.matmul(c_k, adj_k),c_k.T)
        assign[group, indices] = 1   
    adj_ext = adj - adj_int

    adj_group = np.matmul(np.matmul(assign, adj_ext), assign.T)
    return adj_group

def retrieve_conn(seg, groups):
    segments = seg['segments']
    adj = seg['adjacency']
    n_segs = segments.shape[0]

    n_groups = int(np.max(groups) + 1)
    adj_group = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(n_groups):
            if i != j:
                seg_i = np.where(groups == i)[0]
                seg_j = np.where(groups == j)[0]
                mutual_adj = adj[seg_i,:][:,seg_j]
                adj_group[i,j] = adj_group[j,i] = np.sum(mutual_adj)
    
    return adj_group

def post_processing_graph(adj):
    import networkx as nx
    deg = np.sum(adj, axis = 1)
    norm_adj = 1 / np.sqrt(deg)[:,None] * adj * 1 / np.sqrt(deg)[None,:]
    G = nx.from_numpy_matrix(norm_adj, create_using=nx.Graph)
    T = nx.maximum_spanning_tree(G, weight = 'weight', algorithm = 'kruskal')
    T = nx.to_numpy_matrix(T)
    T = np.where(T!= 0, 1, 0)
    return T

def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        print( 'Your adjacency matrix contained redundant nodes.' )
    return g

    
def nearest_neighbor(features, k = 15, sigma = 3):
    from sklearn.neighbors import kneighbors_graph, NearestNeighbors
    from sklearn.metrics import pairwise_distances
    import random
    import networkx as nx
    from umap.umap_ import fuzzy_simplicial_set
    from scipy.sparse import coo_matrix


    directed_conn = kneighbors_graph(features, n_neighbors = k, mode='connectivity', include_self=False).toarray()
    conn = directed_conn + directed_conn.T
    dist = pairwise_distances(features, metric = "euclidean", n_jobs = 4)

    conn[conn.nonzero()[0],conn.nonzero()[1]] = np.exp(-0.5 / sigma * dist[conn.nonzero()[0],conn.nonzero()[1]] ** 2)
    G = nx.from_numpy_matrix(conn, create_using=nx.Graph)

    nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(features)
    knn_dists, knn_indices = nbrs.kneighbors(features)

    X = coo_matrix(([], ([], [])), shape=(features.shape[0], 1))

    connectivities = fuzzy_simplicial_set(
        X,
        n_neighbors = k,
        metric = None,
        random_state = None,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )

    if isinstance(connectivities, tuple):
        # In umap-learn 0.4, this returns (result, sigmas, rhos)
        connectivities = connectivities[0]
    connectivities = connectivities.toarray()
    print(np.allclose(connectivities, connectivities.T))
    G = nx.from_numpy_matrix(connectivities, create_using=nx.Graph)
    return connectivities, G

def leiden(conn, resolution = 0.05, random_state = 0, n_iterations = -1):
    try:
        import leidenalg as la
    except ImportError:
        raise ImportError(
            'Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip3 install leidenalg`.'
        )
        
    start = print('running Leiden clustering')

    partition_kwargs = {}
    # # convert adjacency matrix into igraph
    g = get_igraph_from_adjacency(conn)
    
    # Parameter setting
    partition_type = la.RBConfigurationVertexPartition
    partition_kwargs['weights'] = np.array(g.es['weight']).astype(np.float64)     
    partition_kwargs['n_iterations'] = n_iterations
    partition_kwargs['seed'] = random_state
    partition_kwargs['resolution_parameter'] = resolution
        
    # Leiden algorithm
    # part = la.find_partition(g, la.CPMVertexPartition, **partition_kwargs)
    part = la.find_partition(g, partition_type, **partition_kwargs)
    # groups store the length |V| array, the integer in each element(node) denote the cluster it belong
    groups = np.array(part.membership)

    n_clusters = int(np.max(groups) + 1)
    
    print('finished')
    return groups, n_clusters



def cell_clust_assign(seg, X, groups):
    segments = seg['segments']
    groups_cell = np.zeros((1, X.shape[0]))
    groups_cell = [[] for cell in range(X.shape[0])]
    counts = [[] for cell in range(X.shape[0])]
    count_group = [[] for group in np.unique(groups)]

    for seg_id in range(segments.shape[0]):
        # cell passed in the segment
        rand_wk = segments[seg_id]
        # group of the segment
        group_id = groups[seg_id]
        # the cells included in the group
        count_group[group_id].extend(rand_wk)

        # cell ids in this random walk
        for cell_id in rand_wk:
            if group_id not in groups_cell[cell_id]:
                # append the group into the list of the cell
                groups_cell[cell_id].append(group_id)
                # append one count for the group
                counts[cell_id].append(1)
            else:
                # add one count for corresponding group
                counts[cell_id][int(np.where(groups_cell[cell_id] == group_id)[0])] += 1
        
    # cell_bb store the backbone id of each cell
    cell_bb = np.zeros(X.shape[0])
    for cell_id in range(X.shape[0]):
        if len(groups_cell[cell_id]) == 0:
            cell_bb[cell_id] = np.inf
        else:
            cell_bb[cell_id] = groups_cell[cell_id][np.argmax(counts[cell_id])]

    # make sure that each backbone cluster has at least one cell assigned to it
    for group in np.unique(groups):
        indices = np.where(cell_bb == group)[0]
        if indices.shape[0] == 0:
            print("empty cluster: ", group)
            unique_cells, count_cells = np.unique(count_group[group], return_counts=True)
            unique_cells = unique_cells[np.argsort(count_cells)[::-1][:10]]
            cell_bb[unique_cells] = group

    return cell_bb

def backbone_finding(data, n_randwk = 300, window_size = 20, step_size = 10, n_neighs = 30, resolution = 0.3):
    X = data.x
    y = data.y.squeeze()
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    X_pca = utils.pca_op(X, n_comps=30, standardize=False)
    randwk = random_walk(X_pca, y, t_max = 10000, i_max = n_randwk, k = 5)
    seg = segmentation(randwk, X_pca[:,:4], cell_backbone=None, wind_size = window_size, step = step_size, y = y)
    seg_pca = utils.pca_op(seg['seg_features'], n_comps=30, standardize=False)
    conn, G = nearest_neighbor(seg_pca, k = n_neighs)
    groups, n_clusters = leiden(conn, resolution = resolution, n_iterations = -1)
    print('number of clusters: ', int(np.max(groups))+1)
    adj_groups = retrieve_conn(seg, groups)
    T = post_processing_graph(adj_groups)
    cell_bb = cell_clust_assign(seg = seg, X = X, groups = groups)
    
    results = {'seg': seg, 'seg_groups': groups, 'cell_groups': cell_bb, 'mst': T, 'origin_conn': adj_groups, 'X': X, 'y': y}
    return results


def plot_backbone(results, version = 'segment', figsize = (20,5)):
    import matplotlib.pyplot as plt
    groups = results['seg_groups']
    cell_bb = results['cell_groups']
    T = results['mst']
    adj_groups = results['origin_conn']
    seg = results['seg']
    X = results['X']
    y = results['y']

    fig = plt.figure(figsize = figsize)
    ax1, ax2 = fig.subplots(1,2)
    cmap = plt.get_cmap('tab20')
    mean_cluster = [[] for x in range(adj_groups.shape[0])]

    conn = adj_groups > 0.0001

    if version == 'segment':
        import umap
        
        Umap = umap.UMAP(n_components = 2, n_neighbors=30)
        X_umap = Umap.fit_transform(np.array(seg['seg_features']))
        
        cmap = plt.get_cmap('tab20')
        for i, cat in enumerate(np.unique(groups)):
            idx = np.where(groups == cat)[0]
            cluster = ax1.scatter(X_umap[idx,0], X_umap[idx,1], color = cmap(i), cmap = 'tab20')
            cluster.set_label("group"+str(cat))

            cluster = ax2.scatter(X_umap[idx,0], X_umap[idx,1], color = cmap(i), cmap = 'tab20')
            cluster.set_label("group"+str(cat))
            mean_cluster[int(cat)] = [np.mean(X_umap[idx,0]), np.mean(X_umap[idx,1])]    
        
        conn = adj_groups > 0.0001

        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        for i in range(conn.shape[0]):
            for j in range(conn.shape[1]):
                if conn[i,j] != 0:
                    ax1.plot([mean_cluster[i][0], mean_cluster[j][0]], [mean_cluster[i][1], mean_cluster[j][1]], 'r-')
        

        conn = T
        for i in range(adj_groups.shape[0]):
            for j in range(adj_groups.shape[1]):
                if conn[i,j] != 0:
                    ax2.plot([mean_cluster[i][0], mean_cluster[j][0]], [mean_cluster[i][1], mean_cluster[j][1]], 'r-')


    elif version == 'cell':
        X_pca = utils.pca_op(X, n_comps=30, standardize=False)

        ax1.scatter(X_pca[:,0], X_pca[:,1], color = 'gray')
        ax2.scatter(X_pca[:,0], X_pca[:,1], color = 'gray')

        mean_cluster = [[] for x in range(adj_groups.shape[0])]
        for i, cat in enumerate(np.unique(cell_bb)):
            idx = np.where(cell_bb == cat)[0]
            cluster = ax1.scatter(X_pca[idx,0], X_pca[idx,1], color = cmap(i), cmap = 'tab20')
            cluster.set_label("group"+str(cat))

            cluster = ax2.scatter(X_pca[idx,0], X_pca[idx,1], color = cmap(i), cmap = 'tab20')
            
            cluster.set_label("group"+str(cat))
            if cat != np.inf:
                mean_cluster[int(cat)] = [np.mean(X_pca[idx,0]), np.mean(X_pca[idx,1])]    
  
        # ax1.legend()
        # ax2.legend()
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')


        for i in range(conn.shape[0]):
            for j in range(conn.shape[1]):
                if conn[i,j] != 0:
                    ax1.plot([mean_cluster[i][0], mean_cluster[j][0]], [mean_cluster[i][1], mean_cluster[j][1]], 'r-')
        

        conn = T
        for i in range(adj_groups.shape[0]):
            for j in range(adj_groups.shape[1]):
                if conn[i,j] != 0:
                    ax2.plot([mean_cluster[i][0], mean_cluster[j][0]], [mean_cluster[i][1], mean_cluster[j][1]], 'r-')
