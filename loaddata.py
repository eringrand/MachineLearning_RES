#import sqlite3 as sqlite
#import pandas as pd
#import pandas.io.sql as psql

import pandas as pd
import numpy as np
import random
import math

def importfiles(train, test):
    f1 = open(train)
    f2 = open(test)

    data_df = pd.read_csv(f1, sep=',', header = 'infer', low_memory=False)
    label_df = data_df[['label']]
    data_df = data_df.drop('label', 1)

    data_df =  data_df.drop(['0','20', '5', '7', '8', '9', '14', '16', '17', 
                             '56', '57', '58', '23', '25', '18', '26'],  axis=1)
    data_df = data_df.drop(['29', '31', '32', '35'],  axis=1)
    data = np.array(data_df)
    label = np.array(label_df)
    
    quiz_df = pd.read_csv(f2, sep=',', header = 'infer', low_memory=False)
    quiz_df = quiz_df.drop(['0','20', '5', '7', '8', '9', '14', '16', '17', 
                            '56', '57', '58', '23', '25', '18', '26'],  axis=1)
    quiz_df = quiz_df.drop(['31', '32', '35', '29'],  axis=1)

    quiz = np.array(quiz_df)
    
    return data, label, quiz
    

def splitdata(data, label):
    n = data.shape[0]
    train_num = int(math.ceil(n*.8))
    indices = random.sample(xrange(n), n)

    training_idx, test_idx = indices[train_num:], indices[:train_num]
    training, test = data[training_idx,:], data[test_idx,:]
    lab, testlabel = label[training_idx,:], label[test_idx,:]
    
    return training, lab, test, testlabel


