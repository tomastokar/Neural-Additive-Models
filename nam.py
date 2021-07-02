from numpy.lib.function_base import append
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ExU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(in_dim))
        self.init_params()

    
    def init_params(self):
        self.weight = nn.init.normal_(self.weight, mean=4., std=.5)
        self.bias = nn.init.normal_(self.bias, std=.5)

    
    def forward(self, x):
        out = torch.matmul((x - self.bias), torch.exp(self.weight))
        out = torch.clamp(out, 0, 1)
        return out


class ReLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_params()


    def init_params(self):        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.normal_(self.linear.bias, std=.5)


    def forward(self, x):
        out = self.linear(x)
        out = F.relu(out)
        return out



class FeatureNet(nn.Module):
    def __init__(self, hidden_sizes, dropout_rate = .2, use_exu = True):
        super(FeatureNet, self).__init__()
        layers = [
            ExU(1, hidden_sizes[0]) if use_exu else ReLU(1, hidden_sizes[0])
        ]
        input_size = hidden_sizes[0]
        for s in hidden_sizes[1:]:
            layers.append(ReLU(input_size, s))
            layers.append(nn.Dropout(dropout_rate))
            input_size = s
        layers.append(nn.Linear(input_size, 1, bias = False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class NAM(nn.Module):
    def __init__(self, no_features, hidden_sizes, dropout_rate = .2, feature_dropout = 0.0, use_exu = False):
        super(NAM, self).__init__()
        self.no_features = no_features
        feature_nets = [FeatureNet(hidden_sizes, dropout_rate, use_exu) for _ in range(no_features)]
        self.feature_nets = nn.ModuleList(feature_nets)
        self.feature_drop = nn.Dropout(feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1,), requires_grad=True)
            
    def forward(self, x):
        y = []
        for i in range(self.no_features):
            o = self.feature_nets[i](x[:,i].unsqueeze(1))
            y.append(o)
        y = torch.cat(y, 1)
        y = self.feature_drop(y)
        out = torch.sum(y, axis = -1) + self.bias
        out = torch.sigmoid(out)
        return out, y