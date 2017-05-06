import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from IPython import embed

def sigmoid(x,deriv=False):
	"""  This function calculates sigmoid of x"""	
	if(deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))

class FeedFowardNetwork():
	""" This class is for training a FFN with 2 layers """

	def __init__(self, numHiddenUnits = 100, gradientMethod = "MBGD", learningRate = 0.01, numIter=1000,batchSize=50):		
		self._numOutputs = 10
		self._numHiddenUnits = numHiddenUnits
		self._gradientMethod = gradientMethod
		self._learningRate = learningRate
		self._numIter = numIter
		self._batchSize = batchSize

		np.random.seed(1)

	def fit(self, X, y, verbose =False):		
		# Start random weights
		self._synLayer1 = 2 * np.random.random((X.shape[1],self._numHiddenUnits)) - 1
		self._synLayer2 = 2 * np.random.random((self._numHiddenUnits,10)) - 1			
		self._X = X
		self._y = y		

		for i in xrange(self._numIter):

			self._X, self._y = shuffle(self._X, self._y)

			if(self._gradientMethod=="GD"):
				batch_size = X.shape[0]
			elif(self._gradientMethod=="SGD"):
				batch_size = 1
			elif(self._gradientMethod=="MBGD"):
				batch_size = self._batchSize		

			for batch_step in range(0,self._X.shape[0]/batch_size):

				X = self._X[batch_step:batch_step+batch_size]
				y = self._y[batch_step:batch_step+batch_size]
							
				# Foward propagate
				l1 = sigmoid(np.dot(X,self._synLayer1))
				l2 = sigmoid(np.dot(l1,self._synLayer2))

				# Calculate loss and deltas for both layers
				loss = self.calculateLossDerivative(l2,y)
				l2_delta = loss 		
				l1_loss = loss.dot(self._synLayer2.T)
				l1_delta = l1_loss * sigmoid(l1,deriv=True)				

				# Update weights accordingly 			
				self._synLayer2+= self._learningRate * (l1.T.dot(l2_delta))
				self._synLayer1+= self._learningRate * (X.T.dot(l1_delta))
				# embed()

			# if(verbose and i%100 ==0):
			if(verbose):
				if(self._gradientMethod!="GD"):
					l1_test = sigmoid(np.dot(self._X,self._synLayer1))
					l2_test = sigmoid(np.dot(l1_test,self._synLayer2))					
					print 'Epoch %d of %d: loss %f' % (i, self._numIter, self.calculateLoss(self._y,l2_test))
				else:				
					print 'Epoch %d of %d: loss %f' % (i, self._numIter, self.calculateLoss(y,l2))

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