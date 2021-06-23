import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from nam import NAM
from utils import CompasData, train_model, eval_model
from plots import plot_roc_curves, plot_shape_functions
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


def decode_data(data, encoders):
    for col, encoder in encoders.items():
        data.loc[:,col] = encoder.inverse_transform(data[col])
    return data

    

def main():
    data = load_data()
    cols = data.columns

    features, response = cols[:-1], cols[-1]
    data, encoders = encode_data(data)

    no_replicates = 20
    no_testings = 500
    results = []
    for i in range(no_replicates):        
        print('\t===== Replicate no. {} =====\n'.format(i + 1))

        d_train, d_test = train_test_split(
            data, 
            test_size=no_testings
        )    

        data_train = CompasData(
            d_train[features].values, 
            d_train[response].values
        )

        data_test = CompasData(
            d_test[features].values, 
            d_test[response].values
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
            max_epochs=20, 
            verbosity=20, 
            learning_rate=2e-4,
            weight_decay=0.,
            output_penalty=0.2
        )

        y_, p_ = eval_model(
            model, 
            data_test
        )
        
        res = (
            pd
            .DataFrame(p_, columns = features, index = d_test.index)
            .add_suffix('_partial')
            .join(d_test)
            .assign(is_recid_proba = y_)   
            .assign(replicate = i)         
        )
        
        results.append(res)
    
    results = pd.concat(results)
    results = decode_data(results, encoders)

    plot_roc_curves(results, 'is_recid_proba', 'is_recid')
    plot_shape_functions(results, features)    

if __name__ == '__main__':
    main()