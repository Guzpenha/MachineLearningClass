import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from IPython import embed

def sigmoid(x,deriv=False):
	"""  This function calculates sigmoid of x"""	
	if(deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))

class FeedFowardNetwork():
	""" This class is for training a FFN with 2 layers """

	def __init__(self, numHiddenUnits = 100, gradientMethod = "MBGD", learningRate = 0.01, numIter=10000,batchSize=10):		
		self._numOutputs = 10
		self._numHiddenUnits = numHiddenUnits
		self._gradientMethod = gradientMethod
		self._learningRate = learningRate
		self._numIter = numIter
		self._batchSize = batchSize

		# np.random.seed(1)

	def fit(self, X, y, verbose =False):		
		# Start random weights
		self._synLayer1 = 2 * np.random.random((X.shape[1],self._numHiddenUnits)) - 1
		self._synLayer2 = 2 * np.random.random((self._numHiddenUnits,10)) - 1			
		self._X = X
		self._y = y

		for i in xrange(self._numIter):

			if(self._gradientMethod!="GD"):
				batch_size=self._batchSize
				
				if(self._gradientMethod == "SGD"):
					batch_size=1

				random_minibatch = np.random.choice(np.arange(X.shape[1]), batch_size)
				X = self._X[random_minibatch]
				y = self._y[random_minibatch]
						
			# Foward propagate
			l1 = sigmoid(np.dot(X,self._synLayer1))
			l2 = sigmoid(np.dot(l1,self._synLayer2))


			# Calculate loss and deltas for both layers
			loss = self.calculateLossDerivative(l2,y)
			l2_delta = loss 		
			l1_loss = loss.dot(self._synLayer2.T)
			l1_delta = l1_loss * sigmoid(l1,deriv=True)
			

			if(verbose and i %100 == 0) :
				if(self._gradientMethod!="GD"):
					l1_test = sigmoid(np.dot(self._X,self._synLayer1))
					l2_test = sigmoid(np.dot(l1_test,self._synLayer2))					
					print 'Iteration %d of %d: loss %f' % (i, self._numIter, self.calculateLoss(self._y,l2_test))
				else:				
					print 'Iteration %d of %d: loss %f' % (i, self._numIter, self.calculateLoss(y,l2))

			# Update weights accordingly 			
			self._synLayer2+= self._learningRate * (l1.T.dot(l2_delta))
			self._synLayer1+= self._learningRate * (X.T.dot(l1_delta))
			# embed()

	def calculateLoss(self,y_true,pred):		
		return log_loss(y_true,pred)

	def calculateLossDerivative(self,pred,y_true):
		y_true_one_hot = []
		for y in y_true:
			y_true_one_hot.append([0]*10)
			y_true_one_hot[-1][y] = 1
		return np.array(y_true_one_hot) - pred 

def main():	
	np.set_printoptions(precision=4)
	np.set_printoptions(suppress=True)
	
	# Read and prepare data
	data = pd.read_csv("../data/data_tp1",header=None,sep=",").rename(columns={0: "label"})	
	X = data[[c for c in data.columns if c !="label"]]
	X = StandardScaler().fit_transform(X)	
	X = np.c_[np.ones(X.shape[0]),X]
	y = data["label"]

	# Fit the data using a FeedFowardNetwork
	ffn = FeedFowardNetwork()	
	ffn.fit(X,y,verbose=True)

if __name__ == "__main__":
	main()