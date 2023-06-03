import numpy as np
from sklearn.cluster import AgglomerativeClustering

def check_separate_cluster(A, method, arch, layer, step_size, similarity, linkage, direction='outgoing'):
    
    if method == 'compare_expected':
        Ac = np.transpose(A)
        Ac = Ac[:np.sum(arch[:layer+1+step_size]) - np.sum(arch[:layer+1])][:]
        Ac = np.transpose(Ac)

        num_clusters = None
        distance_threhsold = None
        difference = None
        
        clustering = AgglomerativeClustering(n_clusters=Ac.shape[0]-1, affinity=similarity, linkage=linkage, distance_threshold=None, compute_distances=False).fit(Ac)
        par = clustering.labels_

        p,count = np.unique(par, return_counts=True)
        p = p[np.argmax(count)]
        lkj = 0
        for i in range(A.shape[0]):
            if par[i] == p and lkj==0:
                if direction == 'outgoing':
                    a1 = Ac[i]
                    N = np.sum(arch[:layer+1+step_size]) - np.sum(arch[:layer+1])
                if direction == 'incomming':
                    a1 = A[i]
                    N = np.sum(arch[:layer]) 
                    if layer-step_size >0:
                        N = N -  np.sum(arch[:layer-step_size])
                if direction == 'bidirectional':
                    a1 = A[i]
                    N = np.sum(arch[:layer+1+step_size]) - np.sum(arch[:layer+1]) + np.sum(arch[:layer]) 
                    if layer-step_size >0:
                        N = N -  np.sum(arch[:layer-step_size])
                lkj+=1
            elif par[i] == p:
                if direction == 'outgoing':
                    a2 = Ac[i]
                if direction == 'incomming':
                    a2 = A[i]
                if direction == 'bidirectional':
                    a2 = A[i]

        ki = np.sum(a1>0.0)
        kj = np.sum(a2>0.0)
        kij = np.dot(a1.astype(float),a2.astype(float))

        x = ki*kj/N**2
        e_kij = N*x
        s_kij = np.sqrt(N*x*(1.0-x))


        if kij<max(ki,kj) and kij <= e_kij:
            num_clusters = A.shape[0]
        elif kij < 1: 
            num_clusters = A.shape[0]

        if s_kij!= 0.0:
            difference = (-kij + e_kij)/s_kij
        else:
            difference = (-kij + e_kij)
        
    return num_clusters, difference

def check_single_cluster(A, method, arch, layer,step_size, similarity, linkage, direction='outgoing'):

    if method == 'compare_expected':

        Ac = A
        num_clusters = None
        distance_threhsold = None
        difference = None
        
        clustering = AgglomerativeClustering(n_clusters=2, affinity=similarity, linkage=linkage, distance_threshold=None, compute_distances=False).fit(Ac)
        par = clustering.labels_
        A1 = []
        A2 = []
        for i in range(A.shape[0]):
            if par[i] == 0:
                if direction == 'outgoing':
                    A1.append(Ac[i])
                if direction == 'incomming':
                    A1.append(A[i])
                if direction == 'bidirectional':
                    A1.append(A[i])
            else:
                if direction == 'outgoing':
                    A2.append(Ac[i])
                if direction == 'incomming':
                    A2.append(A[i])
                if direction == 'bidirectional':
                    A2.append(A[i])
        if direction == 'outgoing':
            N = np.sum(arch[:layer+1+step_size]) - np.sum(arch[:layer+1])
        if direction == 'incomming':
            N = np.sum(arch[:layer]) 
            if layer-step_size >0:
                N = N -  np.sum(arch[:layer-step_size])
        if direction == 'bidirectional':
            N = np.sum(arch[:layer+1+step_size]) - np.sum(arch[:layer+1]) + np.sum(arch[:layer]) 
            if layer-step_size >0:
                N = N -  np.sum(arch[:layer-step_size])
        a1 = A1[0]
        a2 = A2[0]
        for i in range(len(a1)):
            if a1[i]==0:
                for j in range(1,len(A1)):
                    if A1[j][i] > 0:
                        a1[i] = 1
        for i in range(len(a2)):
            if a2[i]==0:
                for j in range(1,len(A2)):
                    if A2[j][i] > 0:
                        a2[i] = 1
        ki = np.sum(a1>0.0)
        kj = np.sum(a2>0.0)
        kij = np.dot(a1.astype(float),a2.astype(float))

        
        x = ki*kj/N**2
        e_kij = N*x
        s_kij = np.sqrt(N*x*(1.0-x))

        if kij>=1 and kij >= e_kij:
            num_clusters = 1
        elif kij >= max(ki,kj): 
            num_clusters = 1

        if s_kij!= 0.0:
            difference = (kij - e_kij)/s_kij
        else:
            difference = (kij - e_kij)
        
    return num_clusters, difference

def check_clusterability(A, method, arch, layer, similarity, linkage,step_size, direction='outgoing'):

    num_clusters = None

    if A.shape[0]!=1:
        num_clusters_sep, difference_seperate = check_separate_cluster(A, method, arch, layer,step_size, similarity, linkage,direction)
        num_clusters_sin, difference_single = check_single_cluster(A, method, arch, layer,step_size, similarity, linkage, direction)
        
        if num_clusters_sep != None and num_clusters_sin != None:
            print('Clusterability differences :',difference_seperate, difference_single)
            if difference_single >= difference_seperate:
                num_clusters = num_clusters_sin
            else:
                num_clusters = num_clusters_sep

        elif num_clusters_sep != None:
            num_clusters = num_clusters_sep

        elif num_clusters_sin != None:
            num_clusters = num_clusters_sin

    else:
        num_clusters = 1

    return num_clusters
