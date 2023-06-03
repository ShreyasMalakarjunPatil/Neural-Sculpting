import torch
import numpy as np
import copy
from utils.prune import get_bias_mask
from utils import load
import os
import pickle as pkl

def pruning_schedule(schedule, i, iterations = None, gradient = None):
    if schedule == 'linear':
        return gradient*i
    if schedule == 'exponential':
        pp = 0.0
        for j in range(i):
            pp += gradient/(2**j)
        return pp
    if schedule == 'cosine':
        return 100.0 - 50.0*( 1 + np.cos(np.pi*i/iterations) )
    if schedule == 'cubic':
        return 100.0 - 100.0 * ( 1.0 - i/iterations )**3
    if schedule == 'lth':
        d = 100.0
        for j in range(i):
            d = d - d * gradient / 100.0
        return 100.0 - d

def edge_pruning_step_schedule(schedule, i, p_init, iterations = None, gradient=None):
    if schedule == 'linear':
        return p_init + gradient*i
    if schedule == 'lth':
        d = gradient
        pp = gradient
        for j in range(i-1):
            d = d*(100.0-gradient) / 100.0
            pp+=d
        return p_init + pp
    if schedule == 'cubic':
        d = 0
        for j in range(i):
            d += (300.0/iterations)*(1-j/iterations)*(1-j/iterations)*(1-j/iterations)
        return p_init + d

def num_param(net):
    num = 0
    prev = net[0]
    for i in range(1,len(net)):
        num += prev*net[i]
        prev = net[i]
    return num

def count(wm):
    num = 0
    n = 0
    for i in range(len(wm)):
        n+=torch.sum(wm[i])
        num+=torch.numel(wm[i])
    print('Prune Ratio :',1.0 - n/num)
    checks_passed = True
    w = np.transpose(wm[0].detach().cpu().numpy())
    w = np.sum(w, axis=1) > 0
    if np.sum(w) < len(w):
        checks_passed = False
    w = wm[-1].detach().cpu().numpy()
    w = np.sum(w, axis=1) > 0
    if np.sum(w) < len(w):
        checks_passed = False

    w = copy.deepcopy(wm)
    for i in range(len(w)):
        print('Layer Units', np.sum(np.sum(w[i].detach().cpu().numpy(), axis=1) > 0))
    return n/num, checks_passed

def get_act_gradient(network, mask, dataloader, dev, loss):
    net = copy.deepcopy(network)
    net.to(dev)
    if mask!=None:
        bias_masks = get_bias_mask(mask, dev)
        net.set_masks(mask, bias_masks)
    
    net.train()
    op = []
    for i in range(len(mask)-1):
        op.append([])

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(dev)
        target = target.to(dev)
        output = net.forward2(data.float())
        batch_loss = loss(output[-1], target, dev)
        batch_loss.backward()
        output = net.get_op()
        for i in range(1,len(output)-1):
            op[i-1].append((output[i].grad.abs().detach().cpu()*output[i].abs().detach().cpu()).numpy()[0])

    scores = []
    for i in range(len(mask)-1):
        scores+=list(np.mean(op[i], axis=0))
    return scores

def get_sum_gradient(network, mask, dataloader, dev, loss):
    net = copy.deepcopy(network)
    net.to(dev)
    if mask!=None:
        bias_masks = get_bias_mask(mask, dev)
        net.set_masks(mask, bias_masks)
    
    net.eval()
    op = []
    for i in range(len(mask)-1):
        op.append([])

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(dev)
        target = target.to(dev)
        output = net.forward2(data.float())
        batch_loss = torch.sum(output[-1])
        batch_loss.backward()
        output = net.get_op()
        for i in range(1,len(output)-1):
            op[i-1].append((output[i].grad.abs().detach().cpu()*output[i].abs().detach().cpu()).numpy()[0])

    scores = []
    for i in range(len(mask)-1):
        scores.append(np.mean(op[i], axis=0))
    return scores


def get_outputs(network, mask, dataloader, dev, ignore_nodes):
    if mask!=None:
        bias_masks = get_bias_mask(mask, dev)
        network.set_masks(mask, bias_masks)
    
    all_output_initial = []
    all_output= []
    lkj = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(dev)
            target = target.to(dev)
            output = network(data.float())
            op = np.zeros(len(ignore_nodes))
            start = 0
            all_output_initial.append(output)

    for j in range(len(all_output_initial)):
        output = all_output_initial[j]
        op = np.zeros(len(ignore_nodes))
        start = 0
        for i in range(len(output)):
            if i < len(output) - 1:
                inds = output[i] 
            else: 
                inds = output[i] 
            output[i] = inds.float()
            op[start:output[i].size()[1]+start] = output[i].detach().cpu().numpy().flatten()
            start = start + output[i].size()[1]
        output = op

        lkj+=1
        op = []
        for i in range(len(output)):
            if ignore_nodes[i] == False:
                op.append(output[i])
        output = op
        all_output.append(output)

    return all_output

def get_weights(network, arch):
    weights = []
    for p in network.parameters():
        if len(p.data.size()) != 1:
            weights.append(p.data.abs().detach().cpu().numpy())
    return weights

def get_scores(unit_pruning_metric, network, dev, arch, val_loader=None, mask=None, loss=None):

    if unit_pruning_metric == 'activation_variance':

        ig = np.zeros(np.sum(arch)) > 0
        all_output = get_outputs(network, mask, val_loader, dev, ig)
        all_output = np.transpose(np.array(all_output))
        scores = np.std(all_output, axis=1)
        scores = scores[arch[0]:np.sum(arch[:-1])]
    
    elif unit_pruning_metric == 'average_weight_magnitude':

        weights = get_weights(network, arch)
        scores = []
        lkj = 0
        for i in range(len(weights)-1):
            for j in range(len(weights[i])):
                s1 = np.array(weights[i][j])
                s2 = np.array(np.transpose(weights[i+1])[j])
                s = np.concatenate((s1, s2))
                if np.sum(s) > 0.0:
                    scores.append(np.sum(s)/np.sum(s>0.0))
                else:
                    scores.append(0.0)
                lkj+=1

    elif unit_pruning_metric == 'average_outgoing_weight_magnitude':

        weights = get_weights(network, arch)
        scores = []
        lkj = 0
        for i in range(len(weights)-1):
            for j in range(len(weights[i])):
                s = np.array(np.transpose(weights[i+1])[j])
                if np.sum(s) > 0.0:
                    scores.append(np.sum(s)/np.sum(s>0.0))
                else:
                    scores.append(0.0)
                lkj+=1
    
    elif unit_pruning_metric == 'average_activation_magnitude':

        ig = np.zeros(np.sum(arch)) > 0
        all_output = get_outputs(network, mask, val_loader, dev, ig)
        all_output = np.transpose(np.array(all_output))
        scores = np.mean(all_output, axis=1)
        scores = scores[arch[0]:np.sum(arch[:-1])]

    elif unit_pruning_metric == 'average_activation_weight_magnitude':
        
        weights = get_weights(network, arch)
        scores = []
        lkj = 0
        for i in range(len(weights)-1):
            for j in range(len(weights[i])):
                s = np.array(np.transpose(weights[i+1])[j])
                if np.sum(s) > 0.0:
                    scores.append(np.sum(s)/np.sum(s>0.0))
                else:
                    scores.append(0.0)
                lkj+=1
        
        ig = np.zeros(np.sum(arch)) > 0
        all_output = get_outputs(network, mask, val_loader, dev, ig)
        all_output = np.transpose(np.array(all_output))
        scores2 = np.mean(all_output, axis=1)
        scores2 = scores2[arch[0]:np.sum(arch[:-1])]
        for i in range(len(scores)):
            scores[i] = scores[i] * scores2[i]
    
    elif unit_pruning_metric == 'activation_weight_path_product':

        weights = get_weights(network, arch)

        ig = np.zeros(np.sum(arch)) > 0
        all_output = get_outputs(network, mask, val_loader, dev, ig)
        all_output = np.transpose(np.array(all_output))
        activation = np.mean(all_output, axis=1)[:np.sum(arch[:-1])]

        unit_number = 0
        for i in range(len(weights)):
            w = np.transpose(weights[i])
            for j in range(w.shape[0]):
                w[j] = w[j]*activation[unit_number]
                unit_number+=1
            w = np.transpose(w)
            weights[i] = w
        scores = []

        for i in range(1,len(arch)-1):
            w = copy.deepcopy(weights)
            for j in range(i):
                if j==0:
                    w1 = np.transpose(w[j])
                else:
                    w1 = np.matmul(w1, np.transpose(w[j]))

            for j in range(i,len(arch)-1):
                if j==i:
                    w2 = np.transpose(w[j])
                else:
                    w2 = np.matmul(w2, np.transpose(w[j]))
            w1 = np.transpose(w1)
            w1 = np.sum(w1,axis=1)
            w2 = np.sum(w2,axis=1)
            w = w1*w2
            
            num = np.sum(np.sum(mask[i-1].detach().cpu().numpy(), axis=1)>0)
            scores.append(w)
        scores = np.array(scores).flatten()

    elif unit_pruning_metric == 'activation_gradient_product':
        scores = get_act_gradient(network, mask, val_loader, dev, loss)
        scores = np.array(scores).flatten()

    elif unit_pruning_metric == 'activation_sum_gradient_product':
        scores = get_sum_gradient(network, mask, val_loader, dev, loss)
        scores = np.array(scores).flatten()
    
    elif unit_pruning_metric == 'random':
        scores = np.absolute(np.random.uniform(size=np.sum(arch[1:-1])))
        s = []
        for i in range(len(mask)-1):
            m = mask[i].detach().cpu().numpy()
            m = np.sum(m, axis=1) > 0.0
            s.append(m)
        s = np.array(s).flatten()
        scores = scores*s


    return scores
        
def get_ignore_nodes(arch, mask=None):
    ignore_nodes = np.ones(np.sum(arch))
    ignore_nodes[:arch[0]] = 0
    ignore_nodes[np.sum(arch[:-1]):] = 0

    if mask !=None:
        for i in range(len(mask)-1):
            m = np.sum(mask[i].detach().cpu().numpy(), axis = 1)
            ignore_nodes[np.sum(arch[:i+1]):np.sum(arch[:i+2])] = m 
    
    ignore_nodes = ignore_nodes <= 0
    return ignore_nodes

def prune_units(args, network, prune_perc, dev, layer=None, mask=None):
    
    train_loader, val_loader, test_loader = load.load_data(args.dataset_path, args.task, args.modularity, 1, args.dataset_noise)
    loss = load.load_loss(args.loss)
    ignore_nodes = get_ignore_nodes(args.arch, mask)

    if mask != None:
        weight_mask = mask
    else:
        weight_mask = []
        for p in network.parameters():
            if len(p.data.size()) > 1:
                weight_mask.append(torch.ones(p.data.size()))
        mask = weight_mask

    scores = get_scores(args.unit_pruning_metric, network, dev, args.arch, val_loader, mask, loss)

    num = []
    for i in range(len(weight_mask)):
        w = weight_mask[i].detach().cpu().numpy()
        num.append(np.sum(np.sum(w, axis=1)>0))
    

    if args.unit_pruning_scale == 'global':
        if args.unit_pruning_metric_normalized == 1:
            prev_num_units = args.arch[0]
            for i in range(len(args.arch)-2):
                if i == 0:
                    begin = 0
                else:
                    begin = np.sum(args.arch[1:1+i])
                st = []
                for j in range(begin, np.sum(args.arch[1:2+i])):
                    if scores[j]>0.0:
                        st.append(scores[j])
                
                if args.unit_pruning_norm == 'sum_scores':
                    norm_factor = np.sum(st)
                    for j in range(begin, np.sum(args.arch[1:2+i])):
                        scores[j] = scores[j] / norm_factor

                elif args.unit_pruning_norm == 'mean_std':
                    norm_mean = np.mean(st)
                    norm_std = np.std(st)
                    for j in range(begin, np.sum(args.arch[1:2+i])):
                        if scores[j]>0.0:
                            scores[j] = (scores[j] - norm_mean) / norm_std
                        else:
                            scores[j] = -100000000.0
                
                elif args.unit_pruning_norm == 'mean':
                    norm_mean = np.mean(st)
                    for j in range(begin, np.sum(args.arch[1:2+i])):
                        if scores[j]>0.0:
                            scores[j] = (scores[j] - norm_mean)
                        else:
                            scores[j] = -100000000.0

                elif args.unit_pruning_norm == 'std':
                    norm_std = np.std(st)
                    for j in range(begin, np.sum(args.arch[1:2+i])):
                        scores[j] = scores[j] / norm_std
                
                elif args.unit_pruning_norm == 'prev_num_units':
                    norm_factor = prev_num_units
                    prev_num_units = num[i]
                    for j in range(begin, np.sum(args.arch[1:2+i])):
                        scores[j] = scores[j] / norm_factor

                elif args.unit_pruning_norm == 'num_units':
                    norm_factor = num[i]
                    for j in range(begin, np.sum(args.arch[1:2+i])):
                        scores[j] = scores[j] * norm_factor

                else:
                    scores = scores
                        
        threshold = np.percentile(np.array(scores), prune_perc)
        pruned_inds = scores > threshold
        lkj = 0
        for i in range(len(args.arch)-2):
            for j in range(len(weight_mask[i])):
                if pruned_inds[lkj] == False :
                    weight_mask[i][j] = torch.zeros(weight_mask[i][j].size())
                    for k in range(len(weight_mask[i+1])):
                        weight_mask[i+1][k][j] = 0
                lkj+=1

    else:
        if layer == 0:
            begin = 0
        else:
            begin = np.sum(args.arch[1:1+layer])
        threshold = np.percentile(np.array(scores[begin:np.sum(args.arch[1:2+layer])]), prune_perc)
        pruned_inds = scores > threshold
        lkj = begin
        for j in range(len(weight_mask[layer])):
            if pruned_inds[lkj] == False:
                weight_mask[layer][j] = torch.zeros(weight_mask[layer][j].size())
                for k in range(len(weight_mask[layer+1])):
                    weight_mask[layer+1][k][j] = 0
            lkj+=1

    bias_masks = get_bias_mask(weight_mask, dev)
    density, checks_passed = count(weight_mask)
    checks_passed = True
    num = []
    for i in range(len(weight_mask)):
        w = weight_mask[i].detach().cpu().numpy()
        num.append(np.sum(np.sum(w, axis=1)>0))

    return weight_mask, bias_masks, density, checks_passed