import pandas as pd
from sklearn.model_selection import train_test_split
from plots import plot_roc_curve
from nam import NAM, FFNN
from utils import CompasData, train_model


def main():
    data = pd.read_csv('./data/compas/compas.csv')    
    data = data.values
    X, y = data[:,:-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    data_train = CompasData(X_train, y_train)
    data_test = CompasData(X_test, y_test)
   
    model = NAM(
        6, 
        [64, 64, 32],
        dropout_rate = 0.1, 
        feature_dropout=0.0, 
        use_exu=False
    )
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
    y_, p_ = model(data_test.X)

    preds = y_.flatten().detach().numpy()
    partial = p_.flatten().detach().numpy()
    targets = data_test.y.numpy()
    plot_roc_curve(preds, targets)

if __name__ == '__main__':
    main()