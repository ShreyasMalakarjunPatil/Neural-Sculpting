from fcntl import F_SEAL_SEAL
from math import dist
from pickle import TRUE
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import os
import seaborn as sns
from detection_pipeline.utils import clustering_utils
import torch

def get_ellipse_coord(partition, pos):

    num = 1 + np.max(list(partition.values()))
    X_max = -100*np.ones(num)
    Y_max = -100*np.ones(num)
    X_min = 100*np.ones(num)
    Y_min = 100*np.ones(num)

    for i in range(len(partition)):

        if pos[int(i)][0] > X_max[partition[int(i)]] :
            X_max[partition[int(i)]] = pos[int(i)][0]
        if pos[int(i)][1] > Y_max[partition[int(i)]] :
            Y_max[partition[int(i)]] = pos[int(i)][1]

        if pos[int(i)][0] < X_min[partition[int(i)]] :
            X_min[partition[int(i)]] = pos[int(i)][0]
        if pos[int(i)][1] < Y_min[partition[int(i)]] :
            Y_min[partition[int(i)]] = pos[int(i)][1]

    return X_max, Y_max, X_min, Y_min
    

def define_layered_position(arch, partition, mod):
    mn = []
    pos = {}
    n = 0
    lkj = 0
    flag = True
    for i in range(len(arch)):
        y = i
        my_dict = {}
        num_points = {}
        np1 = 0
        layer_means = np.zeros(arch[i])
        for j in range(arch[i]):
            fl = -1
            ke = list(my_dict.keys())
            for h in range(len(ke)):
                if ke[h] == partition[n]:
                    fl = 0
            if fl == 0:
                abc = my_dict[partition[n]]
                number = num_points[partition[n]]
                num_points[partition[n]] += 2
            else:
                #if partition[n] == 0:
                #    my_dict[partition[n]] = 2
                #elif partition[n] == 1:
                #    my_dict[partition[n]] = 1
                #elif partition[n] == 2:
                #    my_dict[partition[n]] = 0
                    
                #if partition[n] == 1:
                #    my_dict[partition[n]] = 2
                #elif partition[n] == 2:
                #    my_dict[partition[n]] = 1
                #else:
                my_dict[partition[n]] = partition[n]
                abc = my_dict[partition[n]]
                num_points[partition[n]] = 0
                number = num_points[partition[n]]
                num_points[partition[n]] += 2
                np1 += 1
            pos[n] = (float(abc*arch[i]+number), y)
            layer_means[j] = float(float(abc)*float(arch[i])+float(number))
            n+=1
        if arch[i]==1:
            mn.append(1.0)
        else:
            mn.append(np.array(layer_means).mean())

    layer = arch[0]
    lkj = 0
    for i in range(n):
        if i>=layer:
            lkj+=1
            layer += arch[lkj]
        x,y = pos[i]
           
        if lkj==0:
            if partition[i] == 0:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 1:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 2:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            else:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
        
        if lkj==1:
            if partition[i] == 0:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 1:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 2:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            else:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)

        if lkj==2:
            if partition[i] == 0:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 1:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 2:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            else:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
        
        if lkj==3:
            if partition[i] == 0:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 1:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 2:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            else:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)

        if lkj==4:
            if partition[i] == 0:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 1:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 2:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            else:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)

        if lkj==5:
            if partition[i] == 0:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 1:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 2:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            else:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)

        if lkj==6:
            if partition[i] == 3:
                pos[i] = ((x - mn[lkj]-2)*8/mn[lkj], y)
            elif partition[i] == 4:
                pos[i] = ((x - mn[lkj])*8/mn[lkj], y)
            elif partition[i] == 5:
                pos[i] = ((x - mn[lkj]+2)*8/mn[lkj], y)
            else:
                pos[i] = ((x - mn[lkj])*16/mn[lkj], y)
          
    return pos, n


def visualize(A, partition, path, arch,  original_arch, args):
    
    p = list(partition.values())
    num = len(np.unique(p))
    mod = args.modularity[1:-1]

    G = nx.from_numpy_array(A)
    pos, n = define_layered_position(arch, partition, mod)

    X_max, Y_max, X_min, Y_min = get_ellipse_coord(partition, pos)
    plt.figure()
    ax = plt.gca()    
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    for i in range(len(X_max)):
        xy = (X_min[i]-0.6, Y_min[i]-0.4)
        width = X_max[i] - X_min[i] + 1.25
        height = Y_max[i] - Y_min[i] + 0.75
        lkj = cmap(int(i))
        lkj = (lkj[0]/1.65, lkj[1]/1.65, lkj[2]/1.65, lkj[3]/1.65)
        ellipse = Rectangle(xy, width, height, angle=0, edgecolor='k', facecolor =  lkj, zorder=0)

        ax.add_patch(ellipse)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=200,cmap=cmap, node_color=list(partition.values()))
    
    a = arch
    arch = original_arch
    plt.suptitle('Modularity' + str(mod) + ': Arch' + str(arch) + ': Density = ' +str(round(100.0-args.prune_perc, 2)) + '%', y=0.93, fontsize=13)

    plt.title('$lr =$ '+str(args.lr) + ', batch size =' +str(args.batch_size) + ', epochs =' +str(args.epochs) + ', seed =' + str(args.seed)+ ', arch =' + str(a) +'\n $P_u$ ='+str(args.unit_pruning_gradient) +', $P_e$ ='+str(args.pruning_gradient)
    ,  y = -0.135, fontsize=11)
    
    nx.drawing.nx_pylab.draw_networkx_edges(G, pos, edge_color='k', alpha=0.9)

    direct =  args.visualization_path + str(args.modularity) + '/' + str(arch) + '/' 
    #direct =  args.visualization_path + str(args.clustering_linkage) + '_' + str(args.clustering_distance_measure) + '_' + str(args.stopping_criteria) + '/'  + str(args.modularity) + '/' + str(arch) + '/' 
    isdir = os.path.isdir(direct)
    if not isdir:
        os.makedirs(direct)
    plt.savefig(direct+ 'Modularity' + str(args.modularity) + 'Arch' + str(original_arch) +'$lr =$ '+str(args.lr) + ', batch size =' +str(args.batch_size)  + 'Epochs' +str(args.epochs) + 'seed' + str(args.seed) + str(args.feature_step_size) +'.png',bbox_inches='tight')