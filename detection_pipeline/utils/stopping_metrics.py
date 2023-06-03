import numpy as np
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform

def ch_index(partition, A, similarity):
    loss = -1.0*metrics.calinski_harabasz_score(A, partition)
    return loss


def silhouette_index(partition, A, similarity):
    loss = -1.0*metrics.silhouette_score(A, partition, metric=similarity)
    return loss

def db_index(partition, A, similarity):
    loss = metrics.davies_bouldin_score(A, partition)
    return loss


def modularity_metric(partition, A, similarity):
    distance_matrix = pdist(A, metric=similarity)
    distance_matrix = squareform(distance_matrix)

    np.fill_diagonal(distance_matrix, 0)
    if np.sum(distance_matrix) == 0.0:
        distance_matrix = distance_matrix / 1.0
    else:
        distance_matrix = distance_matrix / np.sum(distance_matrix)

    p = np.zeros((A.shape[0], int(np.max(partition))+1))
    for i in range(A.shape[0]):
        p[i][int(partition[i])] = 1
    
    a1 = np.matmul(np.transpose(p),distance_matrix)
    a1 = np.matmul(a1,p)
    a1 = np.trace(a1)

    a = np.matmul(np.transpose(p),distance_matrix)
    a = np.matmul(a,np.ones((A.shape[0],1)))

    b = np.matmul(distance_matrix,p)
    b = np.matmul(np.ones((1,A.shape[0])),b)

    a2 = np.matmul(a,b)
    a2 = np.trace(a2)

    mod = a1-a2
    return  round(mod, 8)
