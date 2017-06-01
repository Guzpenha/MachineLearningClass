import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from IPython import embed
from scipy.stats import mode

class DecisionStumpCategorical():
	
	def __init__(self):
		self.column_pred = None			
		self.predictions = []

	def fit(self,X,y,sample_weight):
		self.y = y		
		best_error = 10		

		for c in range(X.shape[1]):
			cut_zeros = X[:,c] == 0
			cut_ones = X[:,c] == 1
			mode_0 = mode(y[cut_zeros])[0][0]
			mode_1 = mode(y[cut_ones])[0][0]

			pred = np.ones(X.shape[0])
			pred[cut_zeros] = mode_0
			pred[cut_ones] = mode_1
			pred_map = (mode_0,mode_1)			
			error = (pred != y).dot(sample_weight)			

			if(error < best_error):
				best_error = error
				self.column_pred = c				
				self.predictions = pred_map
						
		return self

	def predict(self,X):
		pred = np.ones(X.shape[0])
		pred[X[:,self.column_pred] == 0] = self.predictions[0]
		pred[X[:,self.column_pred] == 1] = self.predictions[1]
		return pred

class AdaBoostDecisionStump():
	
	def __init__(self, n_estimators = 50):
		self.sample_weights = []
		self.hypothesis = []
		self.hypothesis_weights = []
		self.n_estimators = n_estimators
		
	def fit(self,X,y,verbose=False):		
		#All instances have equal initial weight
		self.sample_weights = np.ones(X.shape[0])/X.shape[0]		

		for i in range(self.n_estimators):			
			dsc = DecisionStumpCategorical()
			# dsc = DecisionTreeClassifier(max_depth=1)
			h = dsc.fit(X,y,sample_weight=self.sample_weights)
			error = (h.predict(X) != y).dot(self.sample_weights)
			alpha = 0.5 * (np.log((1 - error)/error)) # calculate alpha

			self.hypothesis.append(h)
			self.hypothesis_weights.append(alpha)

			self.sample_weights = self.sample_weights * np.exp(-alpha * h.predict(X) * y)
			self.sample_weights = (self.sample_weights/self.sample_weights.sum()).as_matrix()

	def predict(self,X):
		pred = np.zeros(X.shape[0])
		for (h, alpha) in zip(self.hypothesis,self.hypothesis_weights):
			pred+= h.predict(X) * alpha	
		return np.sign(pred)

def main():	
	# Read and prepare data
	data = pd.read_csv("../data/tictactoe.csv",header=None,sep=",").rename(columns={9: "label"})
	
	#Preprocess data	
	data["label"] = data.apply(lambda r: 1 if r.label == "positive" else -1,axis=1)
	X = pd.get_dummies(data[[c for c in data.columns if c !="label"]]).as_matrix()	
	y = data["label"]
	
	# Fit the data using the AdaBoostDecisionStump
	for n_est in range(1,20):
		bds = AdaBoostDecisionStump(n_estimators=n_est)
		# bds = AdaBoostClassifier(n_estimators=n_est)
		bds.fit(X,y)
		pred = bds.predict(X)
		print("Acuracia: " +str((pred==y).sum()/float(len(pred))))
	# embed()

if __name__ == "__main__":
	main()