import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import assert_all_finite
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.utils.fixes import bincount
# from sklearn.exceptions import UndefinedMetricWarning
from IPython import embed

def log_loss_per_instance(y_true, y_pred, eps=1e-15,labels=None):
    """Log loss, aka logistic loss or cross-entropy loss.
    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of the true labels given a probabilistic classifier's
    predictions. The log loss is only defined for two or more labels.
    For a single sample with true label yt in {0,1} and
    estimated probability yp that yt = 1, the log loss is
        -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
    Read more in the :ref:`User Guide <log_loss>`.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier's
        predict_proba method. If ``y_pred.shape = (n_samples,)``
        the probabilities provided are assumed to be that of the
        positive class. The labels in ``y_pred`` are assumed to be
        ordered alphabetically, as done by
        :class:`preprocessing.LabelBinarizer`.
    eps : float
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).    
    labels : array-like, optional (default=None)
        If not provided, labels will be inferred from y_true. If ``labels``
        is ``None`` and ``y_pred`` has shape (n_samples,) the labels are
        assumed to be binary and are inferred from ``y_true``.
        .. versionadded:: 0.18
    Returns
    -------
    loss : float
    Examples
    --------
    >>> log_loss(["spam", "ham", "ham", "spam"],  # doctest: +ELLIPSIS
    ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
    0.21616...
    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.
    Notes
    -----
    The logarithm used is the natural logarithm (base-e).
    """
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true)

    lb = LabelBinarizer()

    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)

    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError('y_true contains only one label ({0}). Please '
                             'provide the true labels explicitly through the '
                             'labels argument.'.format(lb.classes_[0]))
        else:
            raise ValueError('The labels array needs to contain at least two '
                             'labels for log_loss, '
                             'got {0}.'.format(lb.classes_))

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels,
                                       transformed_labels, axis=1)

    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError("y_true and y_pred contain different number of "
                             "classes {0}, {1}. Please provide the true "
                             "labels explicitly through the labels argument. "
                             "Classes found in "
                             "y_true: {2}".format(transformed_labels.shape[1],
                                                  y_pred.shape[1],
                                                  lb.classes_))
        else:
            raise ValueError('The number of classes in labels is different '
                             'from that in y_pred. Classes found in '
                             'labels: {0}'.format(lb.classes_))    

    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)
    print(np.array(loss)[[np.newaxis]].T[0])
    return np.array(loss)[[np.newaxis]].T


def sigmoid(x,deriv=False):
	"""  This function calculates sigmoid of x"""
	if(deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))


class FeedFowardNetwork():
	""" This class is responsable for training a FFN with 3 layers """

	def __init__(self, numHiddenUnits = 100, gradientMethod = "GD", learningRate = 0.01, numIter=4):		
		self._numOutputs = 10
		self._numHiddenUnits = numHiddenUnits
		self._gradientMethod = gradientMethod
		self._learningRate = learningRate
		self._numIter = numIter

		np.random.seed(1)

	def fit(self, X, y):		
		# Start random weights
		self._synLayer1 = 2 * np.random.random((X.shape[1],self._numHiddenUnits)) - 1
		self._synLayer2 = 2 * np.random.random((self._numHiddenUnits,10)) - 1			

		for i in xrange(self._numIter):

			# if(gradientMethod!="GD"){
			# 	random_minibatch = np.random.choice(np.arange(num_train), batch_size)
			# 	X_batch = X[random_minibatch]
			# 	y_batch = y[random_minibatch]
			# }

			# Foward propagate
			l1 = sigmoid(np.dot(X,self._synLayer1))
			l2 = sigmoid(np.dot(l1,self._synLayer2))
			print("foward prop:")
			print(l2[0:4])

			# Calculate loss and deltas for both layers
			loss = self.calculateLoss(y,l2)
			print("loss: ")
			print(loss[0:10])			
			l2_delta = loss * sigmoid(l2,deriv=True)
			# print("l2 delta")
			# print(l2_delta[0:10])			
			l1_loss = l2_delta.dot(self._synLayer2.T)
			l1_delta = l1_loss * sigmoid(l1,deriv=True)			
			print("l1 delta")
			print(l1_delta[0:2])
			
			# if(i %10 == 0) :
			print 'Iteration %d of %d: loss %f' % (i, self._numIter, np.mean(loss))


			# print("weights before update l2: ")
			# print(self._synLayer2[0:2])
			print("weights before update l1: ")
			print(self._synLayer1[0:2])

			# Update weights accordingly 			
			self._synLayer2+= l1.T.dot(self._learningRate * l2_delta)
			# print("l2 updates: ")
			# print(l1.T.dot(self._learningRate * l2_delta)[0:5])
			self._synLayer1+= X.T.dot(self._learningRate * l1_delta)
			print("l1 updates: ")
			print(X.T.dot(self._learningRate * l1_delta)[0])
			# embed()

	def calculateLoss(self,y_true,pred):
		# #TODO change this to correct loss function
		return log_loss_per_instance(y_true,pred)

def main():	
	np.set_printoptions(precision=4)
	np.set_printoptions(suppress=True)
	data = pd.read_csv("../data/data_tp1",header=None,sep=",").rename(columns={0: "label"})	
	ffn = FeedFowardNetwork()
	X = data[[c for c in data.columns if c !="label"]]
	X = StandardScaler().fit_transform(X)	
	X = np.c_[np.ones(X.shape[0]),X]
	y = data["label"]
	
	ffn.fit(X,y)

if __name__ == "__main__":
	main()