
# coding: utf-8

# In[9]:

import random
import math
import numpy as np
import pandas as pd
import operator
from collections import defaultdict	
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


# In[12]:

names = [
#    "Nearest Neighbors", 
#    "Linear SVM", 
#    "RBF SVM", 
    "Decision Tree",
    "Random Forest", 
 #   "AdaBoost", 
    "Naive Bayes", 
    "Linear Discriminant Analysis",
    "Quadratic Discriminant Analysis"
    ]

classifiers = [
#    KNeighborsClassifier(3),
 #   SVC(kernel="linear", C=0.025),
 #   SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=None),
    RandomForestClassifier(max_depth=None, n_estimators=50, max_features=None),
#    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()
]


use = [
 #   1,
  #  2,
  #  2,
    1,
    1,
 #   1,
 #   2,
 #   2,
 #   2
]


# In[3]:

import pandas as pd
import numpy as np
import math
import random
from sklearn.preprocessing import StandardScaler
        

f1 = open("trdata.csv")
data_df = pd.read_csv(f1, sep=',', header = None, low_memory=False)

f2 = open("testdata.csv")
quiz_df = pd.read_csv(f2, sep=',', header = None, low_memory=False)

f3 = open("trlabel.csv")
label_df = pd.read_csv(f3, sep=',', header = None, low_memory=False)
label = np.array(label_df)


data_end = np.shape(data_df)[0]
df = data_df.append(quiz_df)
data = np.array(df)
data = StandardScaler().fit_transform(data)
data, quiz = data[:data_end], data[data_end:]


#splitting to features and labels by training and test
def splitdata(data, label):
    n = data.shape[0]
    train_num = int(math.ceil(n*.8))
    indices = random.sample(xrange(n), n)

    training_idx, test_idx = indices[train_num:], indices[:train_num]
    training, test = data[training_idx,:], data[test_idx,:]
    lab, testlabel = label[training_idx,:], label[test_idx,:]
    
    return training, lab, test, testlabel

data = np.array(data_df)
quiz = np.array(quiz_df)

train, lab, test, testlabel = splitdata(data, label)

lab = np.ravel(lab)
testlabel = np.ravel(testlabel)


# In[4]:

import pandas as pd
import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler

def importfiles(train, test):
    f1 = open(train)
    f2 = open(test)

    ignore_features = ['29', '31', '32', '35']
    #ignore_features = ['25', '26', '18', '58', '20', '23', '29', '31', '32', '35']
    toencode = {}

    f = open("field_types.txt", 'r')
    for line in f.readlines():
        line = line.strip().split(" ")
        if line[1] != "numeric" and line[0] not in ignore_features:
        #if line[0] not in ignore_features:
            toencode[line[0]] = len(line) - 1
            
    f1 = open("data/data.csv")
    data_df = pd.read_csv(f1, sep=',', header = 'infer', low_memory=False)
    data_df = data_df.drop(ignore_features, axis=1)
    label_df = data_df[['label']]
    data_df = data_df.drop('label', axis=1)
    label = np.array(label_df)

    f2 = open("data/quiz.csv")
    quiz_df = pd.read_csv(f2, sep=',', header = 'infer', low_memory=False)
    quiz_df = quiz_df.drop(ignore_features, axis=1)

    data_end = np.shape(data_df)[0]
    df = data_df.append(quiz_df)
    df_trans = pd.get_dummies(df, columns=toencode.keys(), dummy_na=False)
    data = np.array(df_trans)

    print np.shape(df_trans)

    #data = StandardScaler().fit_transform(data)
    data, quiz = data[:data_end], data[data_end:]
    return data, label, quiz
    

def splitdata(data, label):
    n = data.shape[0]
    train_num = int(math.ceil(n*.8))
    indices = random.sample(xrange(n), n)

    training_idx, test_idx = indices[train_num:], indices[:train_num]
    training, test = data[training_idx,:], data[test_idx,:]
    lab, testlabel = label[training_idx,:], label[test_idx,:]
    
    return training, lab, test, testlabel



data, label, quiz = importfiles("data/data.csv", "data/quiz.csv")
train_erin, lab_erin, test_erin, testlabel_erin = splitdata(data, label)
testlabel_erin = np.ravel(testlabel_erin)
lab_erin = np.ravel(lab_erin)


# In[5]:

preds_dic = {}


# In[13]:

def eval(preds, testlabels):
    if len(preds) == len(testlabels):
        error = np.count_nonzero(preds != testlabels) / np.float(len(preds))
        return 1 - error

    
# iterate over classifiers
for name, clf, use in zip(names, classifiers, use):
    print name
    # if use == 1:
    preds = clf.fit(train_erin, lab_erin).predict(test_erin)
    preds_dic[name] = preds
    print eval(preds, testlabel_erin)
    #if use == 2:
    #preds = clf.fit(train, lab).predict(test)
    #preds_dic[name] = preds
    #print eval(preds, testlabel)


# In[ ]:




# In[ ]:

def avg_lab_ens(preds_dic, testlabel):
    avg_lab = np.zeroes(len(testlabel))
    for key, value in preds_dic:
        avg_lab = avg_lab + value
    new_lab = avg_lab / abs(avg_lab)
    print eval(new_lab, testlabel)
    return new_lab
        
        
    


# In[ ]:

def writequizpreds(preds):
    x = []
    for i in xrange(len(preds)):
        p = preds[i]
        x.append((i+1, p))
    preds_df = pd.DataFrame(x, columns=['Id', 'Prediction'])
    preds_df.to_csv('data/out.csv', index=False)
    return


# In[ ]:




# In[ ]:



