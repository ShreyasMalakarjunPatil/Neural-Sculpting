import torch
import numpy as np
import copy
from torch.autograd import Variable

def to_var(x, requires_grad = False, volatile = False):
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda:0"))
    return Variable(x, requires_grad = requires_grad, volatile = volatile)