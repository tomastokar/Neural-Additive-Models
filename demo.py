import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from nam import NAM
from utils import CompasData, train_model, eval_model
from plots import plot_roc_curve, plot_shape_functions

def main():
    data = pd.read_csv('./data/compas/compas.csv')  
    cols = data.columns
    X = data.iloc[:,:-1].values
    y = data.iloc[:, -1].values

    preds, partials = [], []
    cv = KFold(n_splits = 5, shuffle=False)
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):

        print('\t===== Fold no. {} =====\n'.format(i + 1))
    
        data_train = CompasData(
            X[train_idx], 
            y[train_idx]
        )
        
        data_test = CompasData(
            X[test_idx], 
            y[test_idx]
        )        
    
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

        y_, p_ = eval_model(
            model, 
            data_test
        )

        preds.append(y_)
        partials.append(p_)

    preds = np.concatenate(preds)
    partials = np.concatenate(partials)
    
    plot_roc_curve(preds, y)
    plot_shape_functions(partials, X, cols[:-1])    

if __name__ == '__main__':
    main()