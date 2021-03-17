import time
import torch
import pandas as pd
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from plots import plot_roc_curve
from nam import NAM, FFNN


class CompasData(Dataset):
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
    no_batches = len(loader)
    for epoch in range(max_epochs):
        start = time.time()
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()                        
            y_, _ = model(x)
            err = loss(y_, y)
            
            err.backward()
            optimizer.step()
            
            if verbosity is not None:
                if i % verbosity == 0:
                    print('Epoch: {0}/{1};\t Batch: {2}/{3};\t Err: {4:1.3f}'.format(epoch, max_epochs, i, no_batches, err.item()))

        print('\n\t Epoch finished in {:1.2f} seconds!\n'.format(time.time() - start))


def main():
    data = pd.read_csv('./data/compas/compas.csv')    
    data = data.values
    X, y = data[:,:-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    data_train = CompasData(X_train, y_train)
    data_test = CompasData(X_test, y_test)
   
    model = NAM(6, [64, 64, 32], dropout_rate = 0.1, feature_dropout=0.0, use_exu=False)
    # model = FFNN(6, [100] * 10, dropout_rate=0.1)
    model = model.double()

    train_model(
        model, 
        data_train, 
        batch_size = 16, 
        max_epochs=10, 
        verbosity=20, 
        learning_rate=2e-4
    )

    model.eval()
    y_, _ = model(data_test.X)

    preds = y_.flatten().detach().numpy()
    targets = data_test.y.numpy()
    plot_roc_curve(preds, targets)

if __name__ == '__main__':
    main()