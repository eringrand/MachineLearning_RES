from loaddata import importfiles, splitdata
from testerror import eval
import numpy as np
import math

data, label, quiz = importfiles("data.csv", "quiz.csv")
train, lab, test, testlabel = splitdata(data, label)


def knn(train, trainlabels, test, k):
    if k/2 == 0.0:
        print "K must be an odd number to break ties."
        return None
    norms = np.square(np.linalg.norm(train, axis = 1))
    cross = test.dot(train.T)
    distances = -2*cross + norms
    indicies = np.argmin(distances, axis = 1)
    preds = train_label[indices]
    return preds


preds = knn(train, lab, test, 3)
print eval(preds, testlabel)