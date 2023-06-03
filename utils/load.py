
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import datasets, transforms
import pickle as pkl
from utils import loss, prune_units
from models import mlp
import os
import numpy as np
from detection_pipeline.utils.stopping_metrics import ch_index, silhouette_index, db_index, modularity_metric
from utils.prune import get_bias_mask

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_directory(args):

    direct_models = args.result_path  + str(args.modularity) + '/Models/' +str(args.arch) + '/'
    isdir = os.path.isdir(direct_models)

    if not isdir:
        os.makedirs(direct_models)

    direct_results = args.result_path + str(args.modularity) + '/Results/' +str(args.arch)+'/'
    isdir = os.path.isdir(direct_results)

    if not isdir:
        os.makedirs(direct_results)

    direct_masks = args.result_path + str(args.modularity) + '/Masks/' +str(args.arch)+'/'
    isdir = os.path.isdir(direct_masks)

    if not isdir:
        os.makedirs(direct_masks)

    return direct_models, direct_masks, direct_results

def initialize_unit_pruning(args, layer=None):

    num_layers = len(args.arch)-2

    if args.unit_pruning_scale == 'local':

        prune_done = np.zeros(num_layers) > 0
        follow_schedule = np.ones(num_layers) > 0
        p = np.zeros(num_layers)
        prev_prune_perc = np.zeros(num_layers)
        prune_perc = np.zeros(num_layers)

        p_min = []
        for i in range(1,len(args.arch)-1):
            p_min.append(100.0/float(args.arch[i]))

    elif args.unit_pruning_scale == 'local_global':

        prune_done = False
        follow_schedule = True
        p = 0.0
        prev_prune_perc = 0.0
        prune_perc = 0.0

        p_min = 100.0/args.arch[layer+1]

    else:
        prune_done = False
        follow_schedule = True
        p = 0.0
        prev_prune_perc = 0.0
        prune_perc = 0.0

        p_min = 100.0/np.sum(args.arch[1:-1])

    pruning_steps = []

    return num_layers, prune_done, follow_schedule, pruning_steps, p, prev_prune_perc, prune_perc, p_min

def initialize_edge_unit_pruning(args, direct_models, direct_masks, dev):


    model = load_model(args.arch,args.bn)(args.arch, args.hidden_layer_activation, args.bn).to(dev)
    model.load_state_dict(torch.load(direct_models + 'prune' + args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.unit_pruning_iterations) + str(args.gamma) +'_Model' + str(args.seed) + '.pth'))

    with open(direct_masks + 'prune'  + args.unit_pruning_metric + args.task + str(args.lr) +str(args.batch_size)+str(args.epochs) + str(args.unit_pruning_gradient) + str(args.unit_pruning_iterations) + str(args.gamma) + '_Mask' + str(args.seed) + '.pkl', "rb") as fout:
        weight_masks = pkl.load(fout)

    d,_ = prune_units.count(weight_masks)
    prune_perc_init = 100.0*(1.0-d)
    prev_prune_perc = prune_perc_init


    num_weights = 0
    for i in range(len(args.arch)-1):
        num_weights += args.arch[i]*args.arch[i+1]
    p_min = 100.0/float(num_weights)

    bias_masks = get_bias_mask(weight_masks, dev)

    return model, weight_masks, bias_masks, prune_perc_init, prev_prune_perc, p_min


def initialize_edge_pruning(args):

    num_weights = 0
    for i in range(len(args.arch)-1):
        num_weights += args.arch[i]*args.arch[i+1]
    p_min = 100.0/float(num_weights)

    prune_done = False
    follow_schedule = True
    p = args.pruning_gradient[0]
    prev_prune_perc = 0.0
    pruning_steps = []
    
    epochs = int(args.epochs)
    return p_min, prune_done, follow_schedule, p, prev_prune_perc, pruning_steps, args.pruning_gradient[0], epochs

def load_device(gpu):
    use_cuda = torch.cuda.is_available()
    print('Use Cuda',use_cuda)
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

def AddGaussianNoise(tensor, std):
    return tensor + torch.randn(tensor.size()) * std

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None, std=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.std = std
    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x, self.std)
        y = self.tensors[1][index]
        return x, y
    def __len__(self):
        return self.tensors[0].size(0)

def load_data(dataset_path, task, modularity, batch_size, dataset_noise=False):

    if task == 'boolean':
        dataset_name = dataset_path + task + str(modularity) + '.pkl'
        with open(dataset_name, "rb") as fout:
            dataset = pkl.load(fout)
        X_train = dataset[0]
        Y_train = dataset[1]

        if dataset_noise:
            train_dataset = CustomTensorDataset(tensors=(X_train, Y_train), transform=AddGaussianNoise, std=0.1)
            X_val = X_train 
            Y_val = Y_train 
            val_dataset = CustomTensorDataset(tensors=(X_val, Y_val))
        else:
            train_dataset = CustomTensorDataset(tensors=(X_train, Y_train))
            val_dataset = CustomTensorDataset(tensors=(X_train, Y_train))

        test_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    num_workers = 2)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=2)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=2)
    
    elif task == 'mnist_10way':
        transforms1 = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

        train_data = datasets.MNIST('../../datasets/', train=True, transform=transforms1, download=True)

        test_data = datasets.MNIST('../../datasets/', train=False, transform=transforms1, download=True)

        kwargs = {'num_workers': 4, 'pin_memory': True} 

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                             shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                             shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                             shuffle=False, **kwargs)
    elif task == 'mnist':
        dataset_name = dataset_path + task + str(modularity) + 'train.pkl'
        with open(dataset_name, "rb") as fout:
            dataset = pkl.load(fout)
        X_train = dataset[0]
        Y_train = dataset[1]

        train_data = torch.utils.data.TensorDataset(X_train, Y_train)

        dataset_name = dataset_path + task + str(modularity) + 'test.pkl'
        with open(dataset_name, "rb") as fout:
            dataset = pkl.load(fout)
        X_test = dataset[0]
        Y_test = dataset[1]

        test_data = torch.utils.data.TensorDataset(X_test, Y_test)

        kwargs = {'num_workers': 4, 'pin_memory': True} 

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                             shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                             shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                             shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader

def optimizer(optimizer):
    optimizers = {
        'adam' : (optim.Adam, {}),
        'sgd' : (optim.SGD, {}),
        'momentum' : (optim.SGD, {'momentum' : 0.9, 'nesterov' : True}),
        'rms' : (optim.RMSprop, {})
    }
    return optimizers[optimizer]

def load_model(arch, bn=False):
    depth = len(arch)-1
    if bn==False:
        models = {
            2 : mlp.mlp2,
            3 : mlp.mlp3,
            4 : mlp.mlp4,
            5 : mlp.mlp5,
            6 : mlp.mlp6,
            7 : mlp.mlp7,
            8 : mlp.mlp8,
            9 : mlp.mlp9,
            10 : mlp.mlp10
            }

    return models[depth]

def load_loss(loss_function):
    losses = {
            'BCE' : loss.bce_loss,
            'MSE' : loss.mean_squared_error,
            'CE' : loss.ce_loss
        }
    return losses[loss_function]


def load_stopping_metric(stopping_criteria):
    losses = {
            'ch-index' : ch_index, 
            'silhouette' : silhouette_index, 
            'db-index' : db_index,
            'modularity-metric' : modularity_metric
        }
    return losses[stopping_criteria]

    