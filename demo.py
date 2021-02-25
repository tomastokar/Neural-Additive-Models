import torch
import numpy as np
import pandas as pd
import pickle as pkl
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader
from nam import NAM

class CompasData(Dataset):
    def __init__(self, d):
        n, m = d.shape
        self.n = n
        self.m = m - 1
        self.X = torch.tensor(data[:,:-1])
        self.y = torch.tensor(data[:, -1])        

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, data, max_epochs = 10, batch_size = 32, learning_rate = 1e-3, weight_decay = 5e-4, verbosity = 20):
    d = CompasData(data)

    # Data loader
    loader = DataLoader(
        d, 
        shuffle = True, 
        batch_size = batch_size
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate,
        weight_decay = weight_decay
    )    
    
    loss = nn.BCELoss()
    for epoch in range(max_epochs):
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()                        
            y_, _ = model(x)
            err = loss(y_, y.reshape(-1, 1))
            
            err.backward()
            optimizer.step()
            
            if verbosity is not None:
                if i % verbosity == 0:
                    print(epoch, i, err.item())


def plot_roc_curve(preds, targets):
    fpr, tpr, _ = roc_curve(targets, preds)      
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.clf()
    plt.figure()
    plt.plot(fpr, tpr, '-', color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc,)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


data = pd.read_csv('./data/compas/compas.csv').values
data_train, data_test = train_test_split(data, test_size=0.1)

model = NAM(6, [16, 16, 16], 0.0)
model = model.double()

train_model(model, data_train, batch_size = 128, max_epochs=100, verbosity=10, learning_rate=1e-2)

d = CompasData(data_test)
y_, p = model(d.X)

preds = y_.flatten().detach().numpy()
targets = d.y.numpy()
plot_roc_curve(preds, targets)