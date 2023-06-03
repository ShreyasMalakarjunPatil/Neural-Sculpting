import torch
import torch.nn as nn

def bce_loss(outputs, labels, dev):

    one = outputs*labels
    zero = (torch.ones(labels.size()).to(dev)-labels)*(torch.ones(labels.size()).to(dev) - outputs)
    outs = one+zero
    
    l = torch.sum(-torch.log(outs))/(torch.numel(labels))
    if torch.isinf(l):
        outs = outs + (outs<=0.0).float()*0.00000001
        l = torch.sum(-torch.log(outs))/(torch.numel(labels))
    return l

def ce_loss(outputs, labels, dev):
    loss = nn.CrossEntropyLoss()
    return loss(outputs, labels)

def mean_squared_error(outputs, labels, dev):

    return torch.sum((outputs - labels)**2)/torch.numel(labels)