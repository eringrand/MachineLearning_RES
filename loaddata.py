import sqlite3 as sqlite
import pandas as pd
import pandas.io.sql as psql

f = open("data.csv")
train = pd.read_csv(f,sep=',', header = 'infer', low_memory=False)

# generating a matrix out of dataframe
trainpivot = trainsub.pivot(index='user_id',columns='sid', values='plays')
# creating the mapping of user index to user id
user_index = pd.DataFrame(trainpivot.index).reset_index()
user_index.columns = [['user_index','user_id']]
trainsub = pd.merge(trainsub, user_index, on='user_id')
testsub = pd.merge(testsub, user_index, on='user_id')
# creating the mapping of song index to song id
song_index = pd.DataFrame(trainpivot.columns).reset_index()
song_index.columns = [['song_index','sid']]
trainsub = pd.merge(trainsub, song_index, on='sid')
testsub = pd.merge(testsub, song_index, on='sid')

# Generating the default M_train and M_test matrices
M_train = trainpivot.as_matrix()
M_train = np.nan_to_num(M_train)

testpivot = testsub.pivot(index='user_id',columns='sid', values='plays')
M_test = testpivot.as_matrix()
M_test = np.nan_to_num(M_test)