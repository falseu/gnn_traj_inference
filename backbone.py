import numpy as np

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

def segmentation(random_wk, node_features, cell_backbone = None, wind_size = 10, step = 2):
    belongings = []
    segments = []
    features = []
    bbs = []
    
    for i in random_wk.keys():
        rwk_i = random_wk[i]
        curr_step = 0
        while(curr_step+wind_size < len(rwk_i)):
            segments.append(rwk_i[curr_step:curr_step+wind_size])
            belongings.append([i])
            curr_step += step
    
    segments = np.array(segments)
    belongings = np.array(belongings)

    for seg in range(len(segments)):
        features.append([])
        bbs.append([])
        for idx in segments[seg,:]:
            features[-1].extend(node_features[idx,:])
            if cell_backbone:
                bbs[-1].append(cell_backbone[idx])
        unique, counts = np.unique(np.array(bbs[-1]), return_counts=True)
        # print("seg: ", seg, ", bbs[-1]: ", unique[np.argmax(counts)], ", bbs: ", bbs[-1])
        bbs[-1] = unique[np.argmax(counts)]
        
        adj = np.zeros((segments.shape[0],segments.shape[0]))

    for curr_seg in range(segments.shape[0]):
        rwk_i = belongings[curr_seg]
        indices = np.where(belongings.squeeze() == rwk_i)[0]
        if curr_seg == 0:
            adj[curr_seg, curr_seg + 1] = 1
        elif curr_seg == segments.shape[0] - 1:
            adj[curr_seg, curr_seg - 1] = 1
        else:
            adj[curr_seg, [curr_seg - 1, curr_seg + 1]] = 1       
    
    if cell_backbone:
        return {"rwk": belongings, "segments": segments, "seg_features": np.array(features), "adjacency": adj, "seg_backbone": np.array(bbs)}
    else:
        return {"rwk": belongings, "segments": segments, "adjacency": adj, "seg_features": np.array(features)}
    
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