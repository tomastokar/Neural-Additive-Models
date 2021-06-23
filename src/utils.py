import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CompasData(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        n, m = X.shape
        self.n = n
        self.m = m
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)        

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, data, max_epochs = 10, batch_size = 32, learning_rate = 1e-3, weight_decay = 5e-4, output_penalty = 0., verbosity = 20):
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
    
    loss = nn.BCELoss(reduction='none')
    no_batches = len(loader)
    for epoch in range(max_epochs):
        start = time.time()
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()                        
            y_, p_ = model(x)
            pen = torch.norm(p_, dim = 1)
            err = loss(y_, y) + output_penalty * pen
            err = err.mean()
            
            err.backward()
            optimizer.step()
            
            if verbosity is not None:
                if i % verbosity == 0:
                    print('Epoch: {0}/{1};\t Batch: {2}/{3};\t Err: {4:1.3f}'.format(epoch + 1, max_epochs, i + 1, no_batches, err.item()))

        print('\n\t Epoch finished in {:1.2f} seconds!\n'.format(time.time() - start))


def eval_model(model, data):
    model.eval()
    with torch.no_grad():
        y_, p_ = model(data.X)   
        y_ = y_.flatten().numpy()
        p_ = p_.numpy()
    return y_, p_

