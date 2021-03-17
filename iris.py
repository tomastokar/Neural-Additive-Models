import torch
import pandas as pd
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from plots import plot_roc_curve
from nam import NAM, FFNN

class IrisData(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        n, m = X.shape
        self.n = n
        self.m = m - 1
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)        

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, data, max_epochs = 10, batch_size = 32, learning_rate = 1e-3, weight_decay = 5e-4, verbosity = 20):
    # Data loader
    loader = DataLoader(
        data, 
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



def main():
    iris = load_iris()
    X, y = iris['data'], iris['target']    
    y = (y == 1) * 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    data_train = IrisData(X_train, y_train)
   
    # model = NAM(6, [128, 128, 128], 0.01)
    model = FFNN(4, [16] * 3)
    model = model.double()

    train_model(
        model, 
        data_train, 
        batch_size = 16, 
        max_epochs=1000, 
        verbosity=5, 
        learning_rate=0.01
    )

    data_test = IrisData(X_test, y_test)
    y_, p = model(data_test.X)

    preds = y_.flatten().detach().numpy()
    targets = data_test.y.numpy()
    plot_roc_curve(preds, targets)

if __name__ == '__main__':
    main()