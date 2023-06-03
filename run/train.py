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

def run(args):

    print('Train :', args.lr, args.batch_size, args.arch, args.epochs)

    load.set_seed(args.seed)
    dev = load.load_device(args.gpu)

    direct = args.result_path + str(args.modularity) + '/' + str(args.arch) + '/' 
    isdir = os.path.isdir(direct)

    if not isdir:
        os.makedirs(direct)

    #### Load model, data, loss, optimizer and learning rate scheduler
    train_loader, validation_loader, test_loader = load.load_data(args.dataset_path, args.task, args.modularity, args.batch_size, args.dataset_noise)
    
    model = load.load_model(args.arch,args.bn)(args.arch, args.hidden_layer_activation, args.bn, args.dropout).to(dev)
    #torch.save(model.state_dict(), direct + args.experiment + str(args.modularity) + str(args.lr) + str(args.batch_size) + str(args.epochs) + str(args.arch)  + str(args.gamma) + '_Init_Model' + str(args.seed) + '.pth')

    loss = load.load_loss(args.loss)
    opt, opt_kwargs = load.optimizer(args.optimizer)

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
    results.append(epochs)

    with open(direct + args.experiment + str(args.modularity) + str(args.lr) + str(args.batch_size) + str(epochs) + str(args.arch)  + str(args.gamma) + '_Results' + str(args.seed) + '.pkl', "wb") as fout:
        pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)
    
    torch.save(model.state_dict(), direct + args.experiment + str(args.modularity) + str(args.lr) + str(args.batch_size) + str(epochs) + str(args.arch)  + str(args.gamma) + '_Model' + str(args.seed) + '.pth')
                    