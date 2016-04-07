import sqlite3 as sqlite
import pandas as pd
import pandas.io.sql as psql

f = open("data.csv")
eval = pd.read_csv(f,sep=',', header = 'infer', low_memory=False)