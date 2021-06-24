import yaml
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from nam import NAM
from src.utils import TabularData, train_model, eval_model
from src.plots import plot_roc_curves, plot_shape_functions
from sklearn.datasets import fetch_california_housing


def load_data():
    california = fetch_california_housing()

    names = california.feature_names
    price = california.target

    california = (
        pd
        .DataFrame(california.data, columns=names)
        .assign(price = price)    
    )

    return california

    
def main(args):
    data = load_data()
    cols = data.columns

    results = []
    features, response = cols[:-1], cols[-1]    
    for i in range(args['no_replicates']):        
        print('\t===== Replicate no. {} =====\n'.format(i + 1))

        d_train, d_test = train_test_split(
            data, 
            test_size=args['test_size']
        )    

        data_train = TabularData(
            d_train[features].values, 
            d_train[response].values
        )

        data_test = TabularData(
            d_test[features].values, 
            d_test[response].values
        )        
    
        model = NAM(**args['model'])
        model = model.double()

        train_model(model, data_train, **args['training']) 
        y_, p_ = eval_model(model, data_test)
        
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

    plot_roc_curves(
        results, 
        'is_recid_proba', 
        'is_recid', 
        './results/compas_roc.png')

    plot_shape_functions(
        results, 
        features, 
        **args['plotting']
    )    


if __name__ == '__main__':
    with open('./config.yml', 'r') as f:
        args = yaml.safe_load(f)
        args = args['compas']
    main(args['compas'])