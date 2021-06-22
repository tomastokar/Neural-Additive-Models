import pandas as pd
from sklearn.model_selection import train_test_split
from nam import NAM, FFNN
from utils import CompasData, train_model, eval_model


def main():
    data = pd.read_csv('./data/compas/compas.csv')   
    cols = data.columns

    X = data.iloc[:,:-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    data_train = CompasData(X_train, y_train)
    data_test = CompasData(X_test, y_test)
   
    model = NAM(
        6, 
        [64, 64, 32],
        dropout_rate = 0.1, 
        feature_dropout=0.05, 
        use_exu=False
    )
    model = model.double()

    train_model(
        model, 
        data_train, 
        batch_size = 16, 
        max_epochs=10, 
        verbosity=20, 
        learning_rate=2e-4,
        weight_decay=0.,
        output_penalty=0.2
    )

    eval_model(
        model,
        data_test,
        feature_names = cols
    )

if __name__ == '__main__':
    main()