from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.utils import shuffle

import numpy as np

import pandas as pd


class FeedFowardNetwork():
	""" This class is a 2 layer FFN used for MNIST digit recognition problem"""

	def __init__(self, numHiddenUnits = 100, gradientMethod = "GD", learningRate = 0.01, numIter=100,batchSize=10):		
		self._numHiddenUnits = numHiddenUnits
		self._gradientMethod = gradientMethod
		self._learningRate = learningRate
		self._numIter = numIter
		self._batchSize = batchSize

		np.random.seed(1)

	def fit(self, X, y, verbose =False):		
		""" This function will fit the weights (synLayers) of a FNN using X and y (must have 10 classes)"""

		# Start random weights
		self._synLayer1 = 2 * np.random.random((X.shape[1],self._numHiddenUnits)) - 1
		self._synLayer2 = 2 * np.random.random((self._numHiddenUnits,10)) - 1			
		self._X = X
		self._y = y				

		# numIter is the number of epochs (number of times we go through all data)
		for i in xrange(self._numIter):

			self._X, self._y = shuffle(self._X, self._y)

			# Depending on the method we use only certain amount of X
			if(self._gradientMethod=="GD"):
				batch_size = X.shape[0]
			elif(self._gradientMethod=="SGD"):
				batch_size = 1
			elif(self._gradientMethod=="MB-GD"):
				batch_size = self._batchSize		

			for batch_step in range(0,self._X.shape[0]/batch_size):

				X = self._X[batch_step:batch_step+batch_size]
				y = self._y[batch_step:batch_step+batch_size]
							
				# Foward propagate
				l1 = self.sigmoid(np.dot(X,self._synLayer1))
				l2 = self.sigmoid(np.dot(l1,self._synLayer2))

				# Calculate loss and deltas for both layers
				loss = self.calculateLossDerivative(l2,y)
				l2_delta = loss 		
				l1_loss = loss.dot(self._synLayer2.T)
				l1_delta = l1_loss * self.sigmoid(l1,deriv=True)

				# Update weights accordingly
				self._synLayer2+= self._learningRate * (l1.T.dot(l2_delta))
				self._synLayer1+= self._learningRate * (X.T.dot(l1_delta))
				
			if(verbose and i%10==0):				
				l1_test = self.sigmoid(np.dot(self._X,self._synLayer1))
				l2_test = self.sigmoid(np.dot(l1_test,self._synLayer2))
				print("Cross Entropy empirical error at epoch "+str(i)+": "+str(self.calculateLoss(self._y,l2_test)))

	def calculateLoss(self,y_true,pred):	
		""" This functions calculates the cross entropy for analytical purposes"""
		return log_loss(y_true,pred)

	def calculateLossDerivative(self,pred,y_true):
		"""  This function calculates the cross entropy derivative regarding the closest weights """	
		y_true_one_hot = []
		for y in y_true:
			y_true_one_hot.append([0]*10)
			y_true_one_hot[-1][y] = 1
		return np.array(y_true_one_hot) - pred 

	def sigmoid(self, x,deriv=False):
		"""  This function calculates sigmoid of x"""	
		x = np.clip(x,-500,500)

		if(deriv==True):
			return x*(1-x)

		return 1/(1+np.exp(-x))

def main():	
	# Read and prepare data
	data = pd.read_csv("../data/data_tp1",header=None,sep=",").rename(columns={0: "label"})
	X = data[[c for c in data.columns if c !="label"]]
	X = StandardScaler().fit_transform(X) #scaling features
	X = np.c_[np.ones(X.shape[0]),X] #adding biases
	y = data["label"]

	# Fit the data using a FeedFowardNetwork
	ffn = FeedFowardNetwork()
	ffn.fit(X,y,verbose=True)

if __name__ == "__main__":
	main()