import torch
import numpy as np
import copy

def pruning_schedule(schedule, i, iterations = None, gradient = None):
    if schedule == 'linear':
            return gradient*i
    if schedule == 'cosine':
        return 100.0 - 50.0*( 1 + np.cos(np.pi*i/iterations) )
    if schedule == 'cubic':
        return 100.0 - 100.0 * ( 1.0 - i/iterations )**3
    if schedule == 'lth':
        d = 100.0
        for j in range(i):
            d = d - d * gradient / 100.0
        return 100.0 - d


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
    input_units_failed = []
    if np.sum(w) < len(w):
        checks_passed = False
        for i in range(len(w)):
            if w[i] == 0:
                input_units_failed.append(i)
    w = wm[-1].detach().cpu().numpy()
    w = np.sum(w, axis=1) > 0
    outut_units_failed = []
    if np.sum(w) < len(w):
        checks_passed = False
        for i in range(len(w)):
            if w[i] == 0:
                outut_units_failed.append(i)
    return n/num, checks_passed, input_units_failed, outut_units_failed

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

def magnitude_prune(network, prune_perc, dev):
    net = copy.deepcopy(network)
    net.to(dev)
    net.eval()

    scores = []
    for p in net.parameters():
        if len(p.data.size()) != 1:
            scores.append(p.data.abs_())

    all_weights = []
    for i in range(len(scores)):
        all_weights += list(scores[i].cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), prune_perc)

    weight_masks = []
    for i in range(len(scores)):
        pruned_inds = scores[i] > threshold
        weight_masks.append(pruned_inds.float())

    density, checks_passed, inpf, outf = count(weight_masks)
    checks_passed = True
    bias_masks = get_bias_mask(weight_masks, dev)
    del net
    return weight_masks, bias_masks, density, checks_passed

def magnitude_prune_layer(network, prune_perc, dev, layer=None):
    net = copy.deepcopy(network)
    net.to(dev)
    net.eval()

    scores = []
    for p in net.parameters():
        if len(p.data.size()) != 1:
            scores.append(p.data.abs_())

    all_weights = []
    for i in range(len(scores)):
        if i == layer:
            all_weights += list(scores[i].cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), prune_perc)

    weight_masks = []
    for i in range(len(scores)):
        if i==layer:
            pruned_inds = scores[i] > threshold
            weight_masks.append(pruned_inds.float())
        else:
            weight_masks.append(torch.ones(scores[i].size()))
        
    density, checks_passed, inpf, outf = count(weight_masks)
    checks_passed = True
    bias_masks = get_bias_mask(weight_masks, dev)
    del net
    return weight_masks, bias_masks, density, checks_passed
