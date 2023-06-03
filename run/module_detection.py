import torch
from utils import load
import numpy as np
import pickle as pkl
from detection_pipeline.cluster import cluster
from detection_pipeline.merge_clusters import merge_clusters
from detection_pipeline.visualize import visualize
from detection_pipeline.utils.clustering_utils import correct_arch, partition_convert, get_adjacency_matrix
import copy
import os

def get_unit_edge_model(args,dev):
    direct_models, direct_masks, direct_results = load.create_directory(args)
    args.pruning_gradient = args.pruning_gradient[0]
    with open(direct_results + 'prune' +args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.pruning_gradient) + str(None) + str(args.gamma) +'_Results'+ str(args.seed) + '.pkl', "rb") as fout:
        re = pkl.load(fout)
    args.prune_perc = re[-1]
    print('Unit Edge Pruning :', args.unit_pruning_gradient, args.pruning_gradient, args.prune_perc)

    model = load.load_model(args.arch,False)(args.arch, 'relu', False).to(dev)
    model.load_state_dict(torch.load(direct_models + 'prune' +args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs)+ str(args.unit_pruning_gradient) + str(args.pruning_gradient) + str(None) + str(args.gamma) +'_Model' + str(args.seed) + '.pth'))

    with open(direct_masks + 'prune' +args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs)+ str(args.unit_pruning_gradient) + str(args.pruning_gradient) + str(None) + str(args.gamma) +'_Mask'+ str(args.seed) + '.pkl', "rb") as fout:
        weight_masks = pkl.load(fout)
    
    return model, weight_masks

def run(args):
    load.set_seed(args.seed)
    dev = load.load_device(args.gpu)

    model, weight_masks = get_unit_edge_model(args, dev)
    original_arch = copy.deepcopy(args.arch)
    new_arch, ignore_nodes = correct_arch(args.arch, model, weight_masks)

    partition = cluster(args, model, new_arch, ignore_nodes, dev, mask=weight_masks)

    partition = merge_clusters(model, weight_masks, new_arch, partition, args.feature_weighted, ignore_nodes, args.cluster_merging_threshold)
    par = partition
    partition = partition_convert(partition)

    A = get_adjacency_matrix(model, weight_masks, ignore_nodes)
    visualize(A, partition, args.visualization_path, new_arch, original_arch, args)
