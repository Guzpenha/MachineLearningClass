import numpy as np
import pandas as pd

from sklearn.metrics import log_loss

from IPython import embed

def sigmoid(x,deriv=False):
	"""  This function calculates sigmoid of x"""
	if(deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))


class FeedFowardNetwork():
	""" This class is responsable for training a FFN with 3 layers """

	def __init__(self, numHiddenUnits = 50, gradientMethod = "GD", learningRate = 0.5, numIter=3):
		self._numOutputs = 10
		self._numHiddenUnits = numHiddenUnits
		self._gradientMethod = gradientMethod
		self._learningRate = learningRate
		self._numIter = numIter

		np.random.seed(1)

	def fit(self, X, y):
		# Start random weights
		self._synLayer1 = 2*np.random.random((X.shape[1],self._numHiddenUnits)) - 1
		self._synLayer2 = 2*np.random.random((self._numHiddenUnits,10)) - 1

		for i in xrange(self._numIter):

			# if(gradientMethod!="GD"){
			# 	random_minibatch = np.random.choice(np.arange(num_train), batch_size)
			# 	X_batch = X[random_minibatch]
			# 	y_batch = y[random_minibatch]
			# }

			# Feedfoward
			l1 = sigmoid(np.dot(X,self._synLayer1))
			l2 = sigmoid(np.dot(l1,self._synLayer2))

			# Calculate loss  and deltas for both layers
			loss = self.calculateLoss(y,l2)
			l2_delta = loss * sigmoid(l2,deriv=True)
			print(np.sum(l2_delta))
			l1_loss = l2_delta.dot(self._synLayer2.T)
			l1_delta = l1_loss * sigmoid(l1,deriv=True)
			print(np.sum(l1_delta))
			
			# if(i %10 == 0) :
			print 'Iteration %d of %d: loss %f' % (i, self._numIter, loss)


			# Update weights accordingly 			
			self._synLayer2+= l1.T.dot(self._learningRate * l2_delta)
			self._synLayer1+= X.T.dot(self._learningRate * l1_delta)


	def calculateLoss(self,y,pred):
		#TODO change this to correct loss function		
		return log_loss(y,pred)

def main():
	data = pd.read_csv("../data/data_tp1",header=None,sep=",").rename(columns={0: "label"})	
	ffn = FeedFowardNetwork()
	X = data[[c for c in data.columns if c !="label"]]
	X = np.c_[np.ones(X.shape[0]),X]	
	y = data["label"]

	ffn.fit(X,y)
	# embed()

if __name__ == "__main__":
	main()