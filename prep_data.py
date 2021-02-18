import numpy as np
import pandas as pd
import sqlite3
#import sqlalchemy 
import pickle as pkl

conn = sqlite3.connect('./data/compas/compas.db')    

#Now in order to read in pandas dataframe we need to know table name
cursor = conn.cursor()
tables = [n[0] for n in cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]


data = {}
for table in tables:
    data[table] = pd.read_sql_query('SELECT * FROM {}'.format(table), conn)
conn.close()

cols = ['age', 'race', 'sex', 'priors_count', 'c_charge_degree', 'c_jail_in', 'c_jail_out', 'is_violent_recid']
d = data['people'][cols].copy()
d.loc[:,'c_charge_degree'] = d['c_charge_degree'].map(
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

d = d[d['c_charge_degree'] != 'X']
print(d['c_charge_degree'].value_counts())
#print(d.head())

dt = pd.to_datetime(d['c_jail_out']) - pd.to_datetime(d['c_jail_in'])
dt = dt / np.timedelta64(1, 'D')
dt = np.ceil(dt)
d['length_of_stay'] = dt
print(d.head())


#with open('./data/compas/compas.pkl', 'wb') as f:
#    pkl.dump(data, f)


