import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import copy
import os

from utils import train
from utils import prune_units
from utils import prune
from models import utils
from utils import load

def accept_network(accuracy,acc_thresh):
    if accuracy >= acc_thresh:
        return True
    else:
        return False

def run(args):

    #### set seed value for torch and numpy, get the device
    load.set_seed(args.seed)
    dev = load.load_device(args.gpu)

    #### Load model, data, loss, optimizer and learning rate scheduler
    train_loader, validation_loader, test_loader = load.load_data(args.dataset_path, args.task, args.modularity, args.batch_size, args.dataset_noise)
    model = load.load_model(args.arch,args.bn)(args.arch, args.hidden_layer_activation, args.bn, args.dropout).to(dev)

    loss = load.load_loss(args.loss)
    opt, opt_kwargs = load.optimizer(args.optimizer)

    if args.prune_units == 1:

        optimizer = opt(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=- 1)
        epochs = args.epochs
        (model, train_curve, test_curve, accuracy, test_loss, test_accuracy, epochs) = train.train_network(model, loss, optimizer, train_loader, validation_loader, test_loader, dev, epochs, scheduler, args.task)

        results = []
        results.append(train_curve)
        results.append(test_curve)
        results.append(accuracy)
        results.append(test_loss)
        results.append(test_accuracy)

        (direct_models, direct_masks, direct_results) = load.create_directory(args)

        model1 = load.load_model(args.arch,args.bn)(args.arch, args.hidden_layer_activation, args.bn).to(dev)
        model1.load_state_dict(model.state_dict())

        (num_layers, prune_done, follow_schedule, pruning_steps, p, prev_prune_perc, prune_perc, p_min) = load.initialize_unit_pruning(args)
        print('p_min', p_min)

        i = 1
        weight_masks = None
        bias_masks = None
        prev_model = None
        epochs = args.epochs
        E = []

        while prune_done==False:

            if follow_schedule==True:
                prune_perc = prune_units.pruning_schedule(args.unit_pruning_schedule, i, args.unit_pruning_iterations, args.unit_pruning_gradient)
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
                weight_masks, bias_masks, density, checks_passed = prune_units.prune_units(args, model1, prune_perc, dev, layer=None, mask=weight_masks)

            if checks_passed == True:
                model1.set_masks(weight_masks, bias_masks)
                optimizer = opt(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=- 1)
                epochs = args.epochs
                
                (model1, train_curve, test_curve, accuracy, test_loss, test_accuracy, e) = train.train_network(model1, loss, optimizer, train_loader, validation_loader, test_loader, dev, epochs, scheduler, args.task)

                results = []
                results.append(train_curve)
                results.append(test_curve)
                results.append(accuracy)
                results.append(test_loss)
                results.append(test_accuracy)
                results.append(pruning_steps)
                
                re = results

                accepted = accept_network(test_accuracy, args.prune_acc_threshold)
                print('Sparse NN Accepted :', accepted, test_accuracy)
            else:
                accepted = False
                print('NN Rejected due to checks failed')

            if accepted == False:
                if p > p_min:
                    if prev_model!=None:
                        model1 = copy.deepcopy(prev_model)
                        results = prev_results
                        weight_masks, bias_masks = copy.deepcopy(prev_weight_masks), copy.deepcopy(prev_bias_masks)
                    else:
                        model1 = copy.deepcopy(model)
                        weight_masks = None
                        bias_masks = None
                    
                    follow_schedule = False
                    p = p/2
                    if p < p_min: 
                        p = p_min
                else:

                    d,_ = prune_units.count(prev_weight_masks)
                    prev_prune_perc = round(100.0*(1.0-d.detach().cpu().numpy()), 3)

                    torch.save(prev_model.state_dict(), direct_models + args.experiment + args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.unit_pruning_iterations) + str(args.gamma) + '_Model' + str(args.seed) + '.pth')
                    
                    prev_results.append(prev_prune_perc)
                    with open(direct_results + args.experiment + args.unit_pruning_metric  + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.unit_pruning_iterations) + str(args.gamma) + '_Results' + str(args.seed) + '.pkl', "wb") as fout:
                        pkl.dump(prev_results, fout, protocol=pkl.HIGHEST_PROTOCOL)
        
                    with open(direct_masks + args.experiment + args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.unit_pruning_iterations) + str(args.gamma) + '_Mask'+ str(args.seed) + '.pkl', "wb") as fout:
                        pkl.dump(prev_weight_masks, fout, protocol=pkl.HIGHEST_PROTOCOL)
                
                    prune_done = True

            elif accepted == True :
                E.append(e)
                results.append(np.max(E))
                prev_model = copy.deepcopy(model1)
                prev_results = copy.deepcopy(results)
                prev_weight_masks, prev_bias_masks = copy.deepcopy(weight_masks), copy.deepcopy(bias_masks)
                prev_prune_perc = prune_perc

            i+=1

    else:

        (direct_models, direct_masks, direct_results) = load.create_directory(args)
        print('Unit Pruning Gradients :', args.unit_pruning_gradient)

        pruning_gradient = args.pruning_gradient

        epochs = int(args.epochs)

        (direct_models, direct_masks, direct_results) = load.create_directory(args)
        (model, weight_masks1, bias_masks1, prune_perc_init1, prev_prune_perc1, p_min) = load.initialize_edge_unit_pruning(args, direct_models, direct_masks, dev)
        print(prune_perc_init1, prev_prune_perc1, p_min)

        model.set_masks(weight_masks1, bias_masks1)
        optimizer = opt(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=- 1)
        (model, train_curve, test_curve, accuracy, test_loss, test_accuracy, _) = train.train_network(model, loss, optimizer, train_loader, validation_loader, test_loader, dev, epochs, scheduler, args.task)

        for lkj in range(len(pruning_gradient)):

            model1 = copy.deepcopy(model)
            args.pruning_gradient = pruning_gradient[lkj]
            prune_perc_init = copy.deepcopy(prune_perc_init1)
            prev_prune_perc = copy.deepcopy(prev_prune_perc1)
            prune_done = False
            follow_schedule = True
            p = args.pruning_gradient
            pruning_steps = []

            i = 1
            epochs = int(args.epochs)
            prev_model = copy.deepcopy(model1)
            prev_results = None
            prev_weight_masks, prev_bias_masks = copy.deepcopy(weight_masks1), copy.deepcopy(bias_masks1)


            while prune_done == False:

                if follow_schedule == True:
                    prune_perc = prune_units.edge_pruning_step_schedule(args.pruning_schedule, i, prune_perc_init, iterations = args.pruning_iterations, gradient = args.pruning_gradient)
                    p = prune_perc - prev_prune_perc
                    print('Edge Pruning Step :', p)
                    print('Edge Pruning Percentage :', prune_perc)
                else:
                    prune_perc = prev_prune_perc + p
                    print('Edge Moved to Linear :', p)
                    print('Edge Pruning Percentage : ', prune_perc)

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
                    re = results

                    accepted = accept_network(test_accuracy, args.prune_acc_threshold)
                    print('NN Accepted :', accepted, test_accuracy)
                else:
                    accepted = False
                    print('NN Rejected due to checks failed')

                if accepted == False:
                    if p > p_min:
                        if prev_results != None:
                            model1 = copy.deepcopy(prev_model)
                            model1.to(dev)
                            results = prev_results
                            weight_masks, bias_masks = copy.deepcopy(prev_weight_masks), copy.deepcopy(prev_bias_masks)
                        else:
                            model1 = copy.deepcopy(prev_model)
                            model1.to(dev)
                            weight_masks, bias_masks = copy.deepcopy(prev_weight_masks), copy.deepcopy(prev_bias_masks)
                        follow_schedule = False
                        p = p/2
                        if p < p_min: 
                            p = p_min
                    else:
                        d,_ = prune_units.count(prev_weight_masks)
                        prev_prune_perc = round(100.0*(1.0-d.detach().cpu().numpy()), 10)

                        torch.save(prev_model.state_dict(), direct_models + args.experiment + args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(pruning_gradient[lkj]) + str(args.pruning_iterations) + str(args.gamma) +'_Model' + str(args.seed)  + '.pth')
                        if prev_results != None:
                            prev_results.append(prev_prune_perc)
                        else:
                            prev_results = []
                            prev_results.append(prev_prune_perc)
                        with open(direct_results + args.experiment + args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(pruning_gradient[lkj]) + str(args.pruning_iterations) + str(args.gamma) +'_Results'+ str(args.seed)  +'.pkl', "wb") as fout:
                            pkl.dump(prev_results, fout, protocol=pkl.HIGHEST_PROTOCOL)
        
                        with open(direct_masks + args.experiment + args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(pruning_gradient[lkj]) + str(args.pruning_iterations) + str(args.gamma) +'_Mask'+ str(args.seed)  +'.pkl', "wb") as fout:
                            pkl.dump(prev_weight_masks, fout, protocol=pkl.HIGHEST_PROTOCOL)
                
                        prune_done = True

                elif accepted == True :
                    prev_model = copy.deepcopy(model1)
                    prev_results = copy.deepcopy(results)
                    prev_weight_masks, prev_bias_masks = copy.deepcopy(weight_masks), copy.deepcopy(bias_masks)
                    prev_prune_perc = prune_perc

                i+=1