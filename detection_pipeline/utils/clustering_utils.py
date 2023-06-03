import numpy as np
import torch

def get_adjacency_matrix(net, mask = None, ignore_nodes = None, weighted=False, directed=False, layer_matrix=None, direction='outgoing'):

    dim = []
    for i in range(len(mask)):
        if i==0:
            dim.append(len(np.transpose(mask[i].detach().cpu().numpy())))
        dim.append(len(mask[i].detach().cpu().numpy()))

    A = np.zeros((np.sum(dim), np.sum(dim)))
    layer = 0
    i = 0
    d = 0
    
    for p in net.parameters():
        if len(p.data.size()) != 1:
            if mask!=None:
                if weighted:
                    mat = p.data.abs_().detach().cpu().numpy()*mask[layer].detach().cpu().numpy()
                else:
                    mat = mask[layer].detach().cpu().numpy()
            else:
                mat = p.data.abs_().detach().cpu().numpy()

            d+=dim[i]
            if direction=='outgoing':
                A[d-dim[i]:d,d:d+dim[i+1]] = np.transpose(mat)
                if directed == False:
                    A[d:d+dim[i+1],d-dim[i]:d] = mat
            if direction=='incomming':
                A[d:d+dim[i+1],d-dim[i]:d] = mat
                if directed == False:
                    A[d-dim[i]:d,d:d+dim[i+1]] = np.transpose(mat)       
            layer+=1
            i+=1

    if ignore_nodes is None:
        return A
    else:
        if layer_matrix == None:
            A1 = []
            for i in range(A.shape[0]):
                if ignore_nodes[i] == False:
                    A1.append(A[i]) 
            A = np.transpose(np.array(A1))
            A1 = []
            for i in range(A.shape[0]):
                if ignore_nodes[i] == False:
                    A1.append(A[i]) 
            A = np.transpose(A1)
        else:
            A1 = []
            if layer_matrix == 0:
                begining = 0
            else:
                begining = np.sum(dim[:layer_matrix])
            for i in range(begining, np.sum(dim[:layer_matrix+1])):
                if ignore_nodes[i] == False:
                    A1.append(A[i])
            A = np.transpose(np.array(A1))
            A1 = []
            for i in range(np.sum(dim[:layer_matrix+1]), np.sum(dim[:layer_matrix+2])):
                if ignore_nodes[i] == False:
                    A1.append(A[i])
            A = np.transpose(A1)
    
        return A

def correct_arch(arch, net, mask):

    ignore_nodes = np.zeros(np.sum(arch))
    A = get_adjacency_matrix(net, mask, ignore_nodes = None, weighted=False, directed=False, layer_matrix=None)

    for i in range(10):
        for j in range(arch[0]):
            if np.sum(A[j][j:]) == 0:
                ignore_nodes[j] = 1
        for j in range(arch[0], np.sum(arch) - arch[-1]):
            if np.sum(A[j][0:j]) == 0 or np.sum(A[j][j:]) == 0:
                ignore_nodes[j] = 1
        for j in range(A.shape[0]):
            if ignore_nodes[j] == 1:
                A[j] = np.zeros(A[j].shape)
                A = np.transpose(A)
                A[j] = np.zeros(A[j].shape)
                A = np.transpose(A)
    
    new_arch = []
    s = 0
    for i in range(len(arch)):
        #s = arch[i]
        new_arch.append(int(arch[i] - np.sum(ignore_nodes[s:s+arch[i]])))
        s += arch[i]
    
    print('Condensed Architecture :', new_arch)

    return new_arch, ignore_nodes > 0

def get_feature_vector(net, mask, new_arch, layer, ignore_nodes, distance_measure, weighted=False, direction='outgoing', step_size=1):

    if direction=='outgoing':

        A = get_adjacency_matrix(net, mask, ignore_nodes, weighted, directed=True, layer_matrix=None, direction='outgoing')
        A1, A2 = A, A
        for i in range(step_size-1):
            A2 = np.matmul(A2, A1)
            A = A + A2
    
        A = A[int(np.sum(new_arch[:layer])):int(np.sum(new_arch[:layer+1]))][:]
        A = np.transpose(A)
        A = A[int(np.sum(new_arch[:layer+1])):][:]
        A = np.transpose(A)

    if direction=='incomming':

        A = get_adjacency_matrix(net, mask, ignore_nodes, weighted, directed=True, layer_matrix=None, direction='incomming')
        A1, A2 = A, A
        for i in range(step_size-1):
            A2 = np.matmul(A2, A1)
            A = A + A2
    
        A = A[int(np.sum(new_arch[:layer])):int(np.sum(new_arch[:layer+1]))][:]

        A = np.transpose(A)
        A = A[:int(np.sum(new_arch[:layer]))][:]
        A = np.transpose(A)

    if direction == 'bidirectional':

        Ao = get_adjacency_matrix(net, mask, ignore_nodes, weighted, directed=True, layer_matrix=None, direction='outgoing')
        A1, A2 = Ao, Ao
        for i in range(step_size-1):
            A2 = np.matmul(A2, A1)
            Ao = Ao + A2
    
        Ao = Ao[int(np.sum(new_arch[:layer])):int(np.sum(new_arch[:layer+1]))][:]

        Ai = get_adjacency_matrix(net, mask, ignore_nodes, weighted, directed=True, layer_matrix=None, direction='incomming')
        A1, A2 = Ai, Ai
        for i in range(step_size-1):
            A2 = np.matmul(A2, A1)
            Ai = Ai + A2
    
        Ai = Ai[int(np.sum(new_arch[:layer])):int(np.sum(new_arch[:layer+1]))][:]

        A = np.zeros(Ai.shape)
        for i in range(A.shape[0]):
            A[i][:int(np.sum(new_arch[:layer]))] = Ai[i][:int(np.sum(new_arch[:layer]))]
            A[i][int(np.sum(new_arch[:layer])):] = Ao[i][int(np.sum(new_arch[:layer])):]

        
    
    if distance_measure == 'cosine' or weighted==False:
        A = (A > 0.0).astype(float) 
    if distance_measure == 'jaccard':
        A = (A > 0.0).astype(float)
    return A

def partition_convert(partition):
    my_dict = {}
    for i in range(partition.shape[0]):
        my_dict[int(i)] = int(partition[i])
    return my_dict

def get_bias_mask(weight_masks, dev):
    bias_masks = []
    for i in range(len(weight_masks)):
        mask = torch.ones(len(weight_masks[i]))
        for j in range(len(weight_masks[i])):
            if torch.sum(weight_masks[i][j]) == 0:
                mask[j] = torch.tensor(0.0)
        mask.to(dev)
        bias_masks.append(mask)
    return bias_masks