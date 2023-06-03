import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.layers import MaskedLinear, MaskedConv2d
import numpy as np
from torch.autograd import Variable

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class mlp2(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp2, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)

        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        self.lin2 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()
    
    def get_op(self):
        return self.op
        
    def forward(self,x):
        op = []
        op.append(x)
        x = self.act1(self.lin1(x))

        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)
        x = self.lin2(x)
        x = self.prob(x)
        op.append(x)
        return op

    def forward2(self,x):
        op = []
        op.append(x)
        x = self.act1(self.lin1(x))
        x.retain_grad()

        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)
        x = self.lin2(x)
        x = self.prob(x)
        op.append(x)
        self.op = op
        return op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1

class mlp3(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp3, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)
        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        input_dim = arch[1]
        width = arch[2]
        self.lin2 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act2 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act2 = nn.Sigmoid()
        elif activation == 'ste':
            self.act2 = StraightThroughEstimator()

        self.lin3 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()
        
    def forward(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin3(x)
        x = self.prob(x)
        op.append(x)

        return op

    def forward2(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin3(x)
        x = self.prob(x)
        op.append(x)
        self.op = op

        return op

    def get_op(self):
        return self.op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1


class mlp4(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp4, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)
        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        input_dim = arch[1]
        width = arch[2]
        self.lin2 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act2 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act2 = nn.Sigmoid()
        elif activation == 'ste':
            self.act2 = StraightThroughEstimator()

        input_dim = arch[2]
        width = arch[3]
        self.lin3 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act3 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act3 = nn.Sigmoid()
        elif activation == 'ste':
            self.act3 = StraightThroughEstimator()

        self.lin4 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()
        
    def forward(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin4(x)
        x = self.prob(x)
        op.append(x)
        
        return op

    def forward2(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin4(x)
        x = self.prob(x)
        op.append(x)

        self.op = op
        
        return op

    def get_op(self):
        return self.op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def assign_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.normal_(m.weight, mean = 0.0, std=2.0)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean = 0.0, std = 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1



class mlp5(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp5, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)
        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        input_dim = arch[1]
        width = arch[2]
        self.lin2 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act2 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act2 = nn.Sigmoid()
        elif activation == 'ste':
            self.act2 = StraightThroughEstimator()

        input_dim = arch[2]
        width = arch[3]
        self.lin3 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act3 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act3 = nn.Sigmoid()
        elif activation == 'ste':
            self.act3 = StraightThroughEstimator()

        input_dim = arch[3]
        width = arch[4]
        self.lin4 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act4 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act4 = nn.Sigmoid()
        elif activation == 'ste':
            self.act4 = StraightThroughEstimator()

        self.lin5 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()

    def get_op(self):
        return self.op
        
    def forward(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin5(x)
        x = self.prob(x)
        op.append(x)        
        return op

    def forward2(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin5(x)
        x = self.prob(x)
        op.append(x)

        self.op = op
        
        return op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1


class mlp6(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp6, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)
        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        input_dim = arch[1]
        width = arch[2]
        self.lin2 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act2 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act2 = nn.Sigmoid()
        elif activation == 'ste':
            self.act2 = StraightThroughEstimator()

        input_dim = arch[2]
        width = arch[3]
        self.lin3 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act3 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act3 = nn.Sigmoid()
        elif activation == 'ste':
            self.act3 = StraightThroughEstimator()

        input_dim = arch[3]
        width = arch[4]
        self.lin4 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act4 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act4 = nn.Sigmoid()
        elif activation == 'ste':
            self.act4 = StraightThroughEstimator()

        input_dim = arch[4]
        width = arch[5]
        self.lin5 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act5 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act5 = nn.Sigmoid()
        elif activation == 'ste':
            self.act5 = StraightThroughEstimator()

        self.lin6 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()

    def get_op(self):
        return self.op
        
    def forward(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin6(x)
        x = self.prob(x)
        op.append(x)
        
        return op

    def forward2(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin6(x)
        x = self.prob(x)
        op.append(x)

        self.op = op
        
        return op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1

class mlp7(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp7, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)
        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        input_dim = arch[1]
        width = arch[2]
        self.lin2 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act2 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act2 = nn.Sigmoid()
        elif activation == 'ste':
            self.act2 = StraightThroughEstimator()

        input_dim = arch[2]
        width = arch[3]
        self.lin3 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act3 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act3 = nn.Sigmoid()
        elif activation == 'ste':
            self.act3 = StraightThroughEstimator()

        input_dim = arch[3]
        width = arch[4]
        self.lin4 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act4 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act4 = nn.Sigmoid()
        elif activation == 'ste':
            self.act4 = StraightThroughEstimator()

        input_dim = arch[4]
        width = arch[5]
        self.lin5 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act5 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act5 = nn.Sigmoid()
        elif activation == 'ste':
            self.act5 = StraightThroughEstimator()

        input_dim = arch[5]
        width = arch[6]
        self.lin6 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act6 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act6 = nn.Sigmoid()
        elif activation == 'ste':
            self.act6 = StraightThroughEstimator()

        self.lin7 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()

    def get_op(self):
        return self.op
        
    def forward(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)
        
        x = self.act6(self.lin6(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin7(x)
        x = self.prob(x)
        op.append(x)
        
        return op

    def forward2(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act6(self.lin6(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin7(x)
        x = self.prob(x)
        op.append(x)

        self.op = op
        
        return op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1

class mlp8(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp8, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)
        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        input_dim = arch[1]
        width = arch[2]
        self.lin2 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act2 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act2 = nn.Sigmoid()
        elif activation == 'ste':
            self.act2 = StraightThroughEstimator()

        input_dim = arch[2]
        width = arch[3]
        self.lin3 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act3 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act3 = nn.Sigmoid()
        elif activation == 'ste':
            self.act3 = StraightThroughEstimator()

        input_dim = arch[3]
        width = arch[4]
        self.lin4 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act4 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act4 = nn.Sigmoid()
        elif activation == 'ste':
            self.act4 = StraightThroughEstimator()

        input_dim = arch[4]
        width = arch[5]
        self.lin5 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act5 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act5 = nn.Sigmoid()
        elif activation == 'ste':
            self.act5 = StraightThroughEstimator()

        input_dim = arch[5]
        width = arch[6]
        self.lin6 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act6 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act6 = nn.Sigmoid()
        elif activation == 'ste':
            self.act6 = StraightThroughEstimator()

        input_dim = arch[6]
        width = arch[7]
        self.lin7 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act7 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act7 = nn.Sigmoid()
        elif activation == 'ste':
            self.act7 = StraightThroughEstimator()

        self.lin8 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()

    def get_op(self):
        return self.op
        
    def forward(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)
        
        x = self.act6(self.lin6(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act7(self.lin7(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin8(x)
        x = self.prob(x)
        op.append(x)
        
        return op

    def forward2(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act6(self.lin6(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act7(self.lin7(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin8(x)
        x = self.prob(x)
        op.append(x)

        self.op = op
        
        return op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1

class mlp9(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp9, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)
        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        input_dim = arch[1]
        width = arch[2]
        self.lin2 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act2 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act2 = nn.Sigmoid()
        elif activation == 'ste':
            self.act2 = StraightThroughEstimator()

        input_dim = arch[2]
        width = arch[3]
        self.lin3 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act3 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act3 = nn.Sigmoid()
        elif activation == 'ste':
            self.act3 = StraightThroughEstimator()

        input_dim = arch[3]
        width = arch[4]
        self.lin4 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act4 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act4 = nn.Sigmoid()
        elif activation == 'ste':
            self.act4 = StraightThroughEstimator()

        input_dim = arch[4]
        width = arch[5]
        self.lin5 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act5 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act5 = nn.Sigmoid()
        elif activation == 'ste':
            self.act5 = StraightThroughEstimator()

        input_dim = arch[5]
        width = arch[6]
        self.lin6 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act6 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act6 = nn.Sigmoid()
        elif activation == 'ste':
            self.act6 = StraightThroughEstimator()

        input_dim = arch[6]
        width = arch[7]
        self.lin7 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act7 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act7 = nn.Sigmoid()
        elif activation == 'ste':
            self.act7 = StraightThroughEstimator()

        input_dim = arch[7]
        width = arch[8]
        self.lin8 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act8 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act8 = nn.Sigmoid()
        elif activation == 'ste':
            self.act8 = StraightThroughEstimator()

        self.lin9 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()

    def get_op(self):
        return self.op
        
    def forward(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)
        
        x = self.act6(self.lin6(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act7(self.lin7(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act8(self.lin8(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin9(x)
        x = self.prob(x)
        op.append(x)
        
        return op

    def forward2(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act6(self.lin6(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act7(self.lin7(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act8(self.lin8(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin9(x)
        x = self.prob(x)
        op.append(x)

        self.op = op
        
        return op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1

class mlp10(nn.Module):
    def __init__(self, arch, activation, batch_norm=False, dropout=None):
        super(mlp10, self).__init__()
        if dropout == None:
            self.dropout_true = False
        else:
            self.dropout_true = True
            self.dropout  = nn.Dropout(dropout)
        input_dim = arch[0]
        output_dim = arch[-1]

        width = arch[1]
        self.lin1 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act1 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif activation == 'ste':
            self.act1 = StraightThroughEstimator()

        input_dim = arch[1]
        width = arch[2]
        self.lin2 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act2 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act2 = nn.Sigmoid()
        elif activation == 'ste':
            self.act2 = StraightThroughEstimator()

        input_dim = arch[2]
        width = arch[3]
        self.lin3 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act3 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act3 = nn.Sigmoid()
        elif activation == 'ste':
            self.act3 = StraightThroughEstimator()

        input_dim = arch[3]
        width = arch[4]
        self.lin4 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act4 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act4 = nn.Sigmoid()
        elif activation == 'ste':
            self.act4 = StraightThroughEstimator()

        input_dim = arch[4]
        width = arch[5]
        self.lin5 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act5 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act5 = nn.Sigmoid()
        elif activation == 'ste':
            self.act5 = StraightThroughEstimator()

        input_dim = arch[5]
        width = arch[6]
        self.lin6 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act6 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act6 = nn.Sigmoid()
        elif activation == 'ste':
            self.act6 = StraightThroughEstimator()

        input_dim = arch[6]
        width = arch[7]
        self.lin7 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act7 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act7 = nn.Sigmoid()
        elif activation == 'ste':
            self.act7 = StraightThroughEstimator()

        input_dim = arch[7]
        width = arch[8]
        self.lin8 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act8 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act8 = nn.Sigmoid()
        elif activation == 'ste':
            self.act8 = StraightThroughEstimator()
        
        input_dim = arch[8]
        width = arch[9]
        self.lin9 = MaskedLinear(input_dim,width)
        if activation == 'relu':
            self.act9 = nn.ReLU()
        elif activation == 'sigmoid':
            self.act9 = nn.Sigmoid()
        elif activation == 'ste':
            self.act9 = StraightThroughEstimator()

        self.lin10 = MaskedLinear(width,output_dim)

        self.prob = nn.Sigmoid()
        self._initialize_weights()

    def get_op(self):
        return self.op
        
    def forward(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)
        
        x = self.act6(self.lin6(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act7(self.lin7(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act8(self.lin8(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act9(self.lin9(x))
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin10(x)
        x = self.prob(x)
        op.append(x)
        
        return op

    def forward2(self,x):
        op = []
        op.append(x)

        x = self.act1(self.lin1(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act2(self.lin2(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act3(self.lin3(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act4(self.lin4(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act5(self.lin5(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act6(self.lin6(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act7(self.lin7(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.act8(self.lin8(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)
        
        x = self.act9(self.lin9(x))
        x.retain_grad()
        op.append(x)
        if self.dropout_true == True:
            x = self.dropout(x)

        x = self.lin10(x)
        x = self.prob(x)
        op.append(x)

        self.op = op
        
        return op

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1