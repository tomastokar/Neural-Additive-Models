import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from nam import NAM
from utils import CompasData, train_model, eval_model
from plots import plot_roc_curve, plot_shape_functions
from responsibly.dataset import COMPASDataset

def load_data():
    # Fetch data and subset columns
    cols = ['sex', 'age','race', 'length_of_stay', 'priors_count', 'c_charge_degree', 'is_recid']
    compas = COMPASDataset()
    compas = compas.df
    compas = compas[cols]

    # Get length of stay in days
    compas['length_of_stay'] /= np.timedelta64(1, 'D')
    compas['length_of_stay'] = np.ceil(compas['length_of_stay'])

    # Rename column
    compas = compas.rename(columns = {'c_charge_degree' : 'charge_degree'})
    return compas


def encode_data(data):
    encoders = {}
    for col in ['race', 'sex', 'charge_degree']:
        encoders[col] = LabelEncoder().fit(data[col])
        data.loc[:,col] = encoders[col].transform(data[col])    
    return data, encoders


def main():
    data = load_data()
    cols = data.columns

    data, encoders = encode_data(data)
    X = data.iloc[:,:-1].values
    y = data.iloc[:, -1].values

    predicts, partials, features, targets = [], [], [], []
    cv = KFold(n_splits = 3, shuffle=False)
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

        predicts.append(y_)
        partials.append(p_)
        features.append(X[test_idx])
        targets.append(y[test_idx])
    
    plot_roc_curve(predicts, targets)
    plot_shape_functions(partials, features, cols[:-1])    

if __name__ == '__main__':
    main()