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
        self.bias = nn.init.normal_(self.bias, mean=4., std=.5)

    
    def forward(self, x):
        out = torch.matmul((x - self.bias), torch.exp(self.weight)))
        out = F.relu(out)

	return out



class FeatureNet(nn.Module):
    def __init__(self, hidden_sizes, dropout_rate = .2):
        super(FeatureNet, self).__init__()
        input_size = 1
        layers = []
        for s in hidden_sizes:
            layers.append(ExU(input_size, s))
            layers.append(nn.Dropout(dropout_rate))
            input_size = s
        layers.append(Exu(input_size, 1))
        self.layers = nn.Sequetial(*layers))

    def forward(self, x):
        return self.layers(x)


class NAM(nn.Module):
    def __init__(self, no_features, hidden_sizes, dropout_rate = .2):
        super(NAM, self).__init__()
        self.no_features = no_features
        feature_nets = [FeatureNet(hidden_sizes, dropout_rate) for _ in range(no_features)]]
        self.feature_nets = nn.ModuleList(feature_nets)
        self.summation = nn.Linear(no_features, 1)
    
    def forward(self, x):
        y = []
        for i in range(no_features):
            y.append(self.feature_nets[i](x[:,i]))
        y = torch.cat(y, 1)
        out = self.summation(y)
        out = F.sigmoid(y)
        

 	return out
