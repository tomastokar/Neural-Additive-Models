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

with open('./data/compas/compas.pkl', 'wb') as f:
    pkl.dump(data, f)


