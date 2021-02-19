import pickle as pkl
import numpy as np
import pandas as pd
import sqlite3

from sklearn.preprocessing import LabelEncoder

def fetch_data(fname):
    # Establish connection
    conn = sqlite3.connect(fname)    

    # Get table names
    cursor = conn.cursor()
    tables = [n[0] for n in cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]

    # Extract data
    data = {}
    for table in tables:
        data[table] = pd.read_sql_query('SELECT * FROM {}'.format(table), conn)
    conn.close()

    return data


def calc_stay_length(data):  
    # Data copy
    d = data.copy()  

    # Lenght of stays
    dt = pd.to_datetime(d['c_jail_out']) - pd.to_datetime(d['c_jail_in'])
    dt = dt / np.timedelta64(1, 'D')  
    dt = np.ceil(dt)

    # Add and remove columns    
    d.drop(columns = ['c_jail_in', 'c_jail_out'], inplace = True)
    d.insert(1, 'length_of_stay', dt)
    
    return d


def get_charge_degree(data):
    # Copy data
    d = data.copy()

    # Get degree
    deg = d['c_charge_degree'].map(
        {
            '(F1)' : 'F', 
            '(F2)' : 'F', 
            '(F3)' : 'F', 
            '(F5)' : 'F',
            '(F6)' : 'F', 
            '(F7)' : 'F',
            '(M1)' : 'M',
            '(M2)' : 'M',
            '(M03)' : 'M',
            '(X)' : 'X', 
            '(CT)' : 'X', 
            '(NI0)' : 'X', 
            '(TCX)' : 'X', 
            '(F6)' : 'X', 
            '(CO3)' : 'X'
        }
    )

    # Add and remove columns
    d.drop(columns = ['c_charge_degree'])
    d.insert(1, 'charge_degree', deg)
    
    return d


def main():    
    # Fetch and pre-process data
    d = fetch_data('./data/compas/compas.db')['people']
    d = calc_stay_length(d)
    d = get_charge_degree(d)

    # Subset columns and drop NaNs
    cols = ['age', 'race', 'sex', 'priors_count', 'charge_degree', 'length_of_stay', 'is_recid']
    d = d[cols]
    d = d[d['charge_degree'] != 'X']
    d = d[d['is_recid'] >= 0]
    d = d.dropna()

    # Encode data
    encoders = {}
    for col in ['race', 'sex', 'charge_degree']:
        encoders[col] = LabelEncoder().fit(d[col])
        d[col] = encoders[col].transform(d[col])
    
    # Write to file
    d.to_csv('./data/compas/compas.csv', index = False)

    # Save encoders
    with open('./data/compas/encoders.pkl', 'wb') as f:
        pkl.dump(d, f)


if __name__ == '__main__':
    main()
