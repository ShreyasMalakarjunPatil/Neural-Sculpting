import numpy as np
import copy
from sklearn.cluster import AgglomerativeClustering

from utils.load import load_stopping_metric
from detection_pipeline.utils.clustering_utils import get_feature_vector
from detection_pipeline.utils.clusterability import check_clusterability

def cluster(args, network, new_arch, ignore_nodes, dev, mask=None):

    layers = len(new_arch)
    partition = np.zeros(np.sum(new_arch))
    num = 0

    for i in range(layers):
        
        if i < layers-1 :
            if i==0:
                direction = 'outgoing'
            else:
                direction = args.feature_direction
            
            A = get_feature_vector(network, mask, new_arch, i, ignore_nodes, 
                                    args.clustering_distance_measure, args.feature_weighted, 
                                    direction, args.feature_step_size)
            
            
            num_clusters_clusterability = check_clusterability(A, args.clusterability_criterion, new_arch, i, 
                                                args.clustering_distance_measure, args.clustering_linkage, args.clusterability_step_size, direction)
            num_clusters = None

            if num_clusters == None:

                A = get_feature_vector(network, mask, new_arch, i, ignore_nodes, 
                                    args.clustering_distance_measure, args.feature_weighted, 
                                    direction, args.feature_step_size)
                loss = []
                for k in range(2, A.shape[0]+1):
                    if k<A.shape[0]:
                        clustering = AgglomerativeClustering(n_clusters=k, affinity=args.clustering_distance_measure, linkage=args.clustering_linkage, distance_threshold=None, compute_distances=False).fit(A)
                        par = clustering.labels_
                        loss.append(load_stopping_metric(args.stopping_criteria)(par, A, args.clustering_distance_measure))
                    elif A.shape[0]==2:
                        clustering = AgglomerativeClustering(n_clusters=k, affinity=args.clustering_distance_measure, linkage=args.clustering_linkage, distance_threshold=None, compute_distances=False).fit(A)
                        par = clustering.labels_
                        loss.append(0.0)

                print('Num clus error :',loss)
                smallest_loss = 10000.0
                for k1 in range(2,A.shape[0]):
                    if loss[k1-2] <= smallest_loss:
                        smallest_loss = loss[k1-2]
                        k = k1
                
                num_clusters = k

                if len(loss)==0:
                    num_clusters=1

            if num_clusters == 2 or num_clusters == A.shape[0]-1 or smallest_loss>-args.mm_threshold:
                if num_clusters_clusterability != None:
                    num_clusters = num_clusters_clusterability

            print('Num Clusters :', num_clusters)
            
            if num_clusters != 1:
                A = get_feature_vector(network, mask, new_arch, i, ignore_nodes, 
                                    args.clustering_distance_measure, args.feature_weighted, 
                                    direction, args.feature_step_size)
                clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity=args.clustering_distance_measure, linkage=args.clustering_linkage, distance_threshold=None, compute_distances=False).fit(A)
                par = clustering.labels_
            else:
                par = np.zeros(new_arch[i])

            partition[num:num+new_arch[i]] = par
            num += new_arch[i]   
        
        else:
            partition[num:num+new_arch[i]] = np.arange(new_arch[i])
        
        
    
    return partition