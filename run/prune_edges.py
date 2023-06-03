import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import copy
import os

from utils import train
from utils import prune
from models import utils
from utils import load

def accept_network(accuracy,acc_thresh):
    if accuracy >= acc_thresh:
        return True
    else:
        return False

def run(args):
    torch.manual_seed(args.seed)
    dev = load.load_device(args.gpu)

    train_loader, validation_loader, test_loader = load.load_data(args.dataset_path, args.task, args.modularity, args.batch_size, args.dataset_noise)
    model = load.load_model(args.arch,args.bn)(args.arch, args.hidden_layer_activation, args.bn, args.dropout).to(dev)
    loss = load.load_loss(args.loss)

    opt, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=- 1)

    (model, train_curve, test_curve, accuracy, test_loss, test_accuracy, epochs) = train.train_network(model, loss, optimizer, train_loader, validation_loader, test_loader, dev, args.epochs, scheduler, args.task)

    results = []
    results.append(train_curve)
    results.append(test_curve)
    results.append(accuracy)
    results.append(test_loss)
    results.append(test_accuracy)

    direct_models, direct_masks, direct_results = load.create_directory(args)
    p_min, prune_done, follow_schedule, p, prev_prune_perc, pruning_steps, args.pruning_gradient, epochs = load.initialize_edge_pruning(args)
    i = 1
    
    model1 = load.load_model(args.arch,args.bn)(args.arch, args.hidden_layer_activation, args.bn).to(dev)
    model1.load_state_dict(model.state_dict())

    prev_model = copy.deepcopy(model1)
    prev_results = None
    prev_weight_masks, prev_bias_masks = None, None

    while prune_done == False:

        if follow_schedule == True:
            prune_perc = prune.pruning_schedule(args.pruning_schedule, i, iterations = args.pruning_iterations, gradient = args.pruning_gradient)
            p = prune_perc - prev_prune_perc
            print('Pruning Step :', p)
            print('Pruning Percentage :', prune_perc)
        else:
            prune_perc = prev_prune_perc + p
            print('Moved to Linear :', p)
            print('Pruning Percentage : ', prune_perc)
        
        pruning_steps.append(p)
        if prune_perc>=100.0:
            checks_passed=False
        else:
            weight_masks, bias_masks, density, checks_passed = prune.magnitude_prune(model1, prune_perc, dev)
        print('Checks Passed =', checks_passed)

        if checks_passed == True:

            model1.set_masks(weight_masks, bias_masks)
            
            optimizer = opt(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=- 1)

            (model1, train_curve, test_curve, accuracy, test_loss, test_accuracy, _) = train.train_network(model1, loss, optimizer, train_loader, validation_loader, test_loader, dev, epochs, scheduler, args.task)
            
            
            results = []
            results.append(train_curve)
            results.append(test_curve)
            results.append(accuracy)
            results.append(test_loss)
            results.append(test_accuracy)
            results.append(pruning_steps)
            results.append(round(100.0*(1.0-density.detach().cpu().numpy()), 3))
            re = results

            accepted = accept_network(test_accuracy, args.prune_acc_threshold)
            print('NN Accepted :', accepted, test_accuracy)
        else:
            accepted = False
            print('NN Rejected due to checks failed :', accepted, test_accuracy)

        if accepted == False:
            if p > p_min:
                model1 = copy.deepcopy(prev_model)
                model1.to(dev)
                results = prev_results
                weight_masks, bias_masks = copy.deepcopy(prev_weight_masks), copy.deepcopy(prev_bias_masks)
                follow_schedule = False
                p = p/2
                if p < p_min: 
                    p = p_min
            else:
                torch.save(prev_model.state_dict(), direct_models + args.experiment + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.pruning_gradient) + str(args.pruning_iterations) + str(args.gamma) +'_Model' + str(args.seed) +'.pth')
        
                with open(direct_results+ args.experiment  + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.pruning_gradient) + str(args.pruning_iterations) + str(args.gamma) +'_Results' + str(args.seed) +'.pkl', "wb") as fout:
                    pkl.dump(prev_results, fout, protocol=pkl.HIGHEST_PROTOCOL)
        
                with open(direct_masks + args.experiment  + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.pruning_gradient) + str(args.pruning_iterations) + str(args.gamma) +'_Mask' + str(args.seed) +'.pkl', "wb") as fout:
                    pkl.dump(prev_weight_masks, fout, protocol=pkl.HIGHEST_PROTOCOL)

                prune_done = True

        elif accepted == True :
            prev_model = copy.deepcopy(model1)
            prev_results = copy.deepcopy(results)
            prev_weight_masks, prev_bias_masks = copy.deepcopy(weight_masks), copy.deepcopy(bias_masks)
            prev_prune_perc = prune_perc
        i+=1