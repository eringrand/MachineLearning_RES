# We shall use binary classification accuracy (i.e., \(1 - \text{error rate}\)) as the performance metric.
import numpy as np

def eval(preds, testlabels):
	p = []
	for i, pred in preds:
		p.append([pred])

	if len(p) == len(testlabels):
		error = np.count_nonzero(p != testlabels) / np.float(len(p))
		return 1 - error
