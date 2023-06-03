import numpy as np
from detection_pipeline.utils.clustering_utils import get_adjacency_matrix

def cluster_similarity(conn, threshold, last_layer=False):
    
    A = np.zeros(conn.shape)
    B = np.zeros(conn.shape)
    for i in range(A.shape[0]):
        A[i] = conn[i] / np.sum(conn[i])
    if last_layer == True:  
        A = np.ones(conn.shape)
    B = np.transpose(B)
    conn = np.transpose(conn)
    for i in range(B.shape[0]):
        B[i] = conn[i] / np.sum(conn[i])
    B = np.transpose(B)
    conn = np.transpose(conn)

    C = np.zeros(conn.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if A[i][j] > threshold and B[i][j] > threshold :
                C[i][j] = 1
            else:
                C[i][j] = 0
    merge = -1*np.ones(C.shape[1])
    C = np.transpose(C)
    for i in range(len(merge)):
        if np.sum(C[i])>0:
            merge[i] = np.argmax(C[i])
    return merge

def create_dict(p_prev):
    my_dict = {}
    clusters = []
    p_prev_new = []
    for i in range(len(p_prev)):
        flag = -1
        for j in range(len(clusters)):
            if p_prev[i] == clusters[j]:
                flag = j
        if flag == -1:
            my_dict[p_prev[i]] = len(clusters)
            flag = my_dict[p_prev[i]]
            clusters.append(p_prev[i])
        p_prev_new.append(flag)
    return my_dict, p_prev_new

def revert_dict(my_dict, p_prev):
    values = list(my_dict.values())
    clus = list(my_dict.keys())
    p_prev_old = []
    for i in range(len(p_prev)):
        for j in range(len(values)):
            if values[j] == p_prev[i]:
                p_prev_old.append(int(clus[j]))
    return p_prev_old


def merge_clusters(network, mask, arch, partition, weighted, ignore_nodes, threshold):
    partition_merged = np.zeros(len(partition))
    num = 0
    for i in range(len(arch)):
        p = partition[num:num+arch[i]]
        
        if i==0:
            num_clusters = np.max(p) - np.min(p) + 1
            partition_merged[num:num+arch[i]] = p
            num += arch[i]
            par = p
        else:
            A = get_adjacency_matrix(network, mask, ignore_nodes, weighted=False, directed=False, layer_matrix=i-1)
            num_clusters = np.max(p) - np.min(p) + 1
            clus_max_prev = np.max(partition_merged[:num]) + 1
            clus = np.max(p) + 1

            my_dict, p_prev = create_dict(p_prev)

            conn = np.zeros((int(num_clusters_prev), int(num_clusters)))
            for j in range(arch[i-1]):
                for k in range(arch[i]):
                    if A[j][k] > 0.0:
                        conn[int(p_prev[j])][int(p[k])] += 1
            if i==len(arch)-1:
                merge = cluster_similarity(conn, threshold, last_layer=True)
            else:
                merge = cluster_similarity(conn, threshold)
            par = np.zeros(len(p))
            new_clusters = {}
            for j in range(len(p)):
                if merge[int(p[j])] != -1:
                    par[j] = merge[int(p[j])]
                else:
                    val = list(new_clusters.keys())
                    new = -1
                    for k in range(len(val)):
                        if p[j] == val[k]:
                            par[j] = new_clusters[p[j]]
                            new = 0
                    if new == -1:
                        new_clusters[p[j]] = clus_max_prev
                        my_dict[clus_max_prev] = clus_max_prev
                        par[j] = clus_max_prev
                        clus += 1
                        clus_max_prev += 1

            par = revert_dict(my_dict, par)
            partition_merged[num:num+arch[i]] = par
            num += arch[i]
        num_clusters_prev = num_clusters
        p_prev = par

    return partition_merged