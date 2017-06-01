import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from IPython import embed

class DecisionStumpCategorical():
	
	def __init__(self):
		self.column_pred = None	

	def fit(self,X,y,sample_weight):			
		best_error = 10
		best_h_column = None			
		for c in range(X.shape[1]):
			X[X[:,c] == 0,c] = -1
			error = (X[:,c] != y).dot(sample_weight)			
			if(error < best_error):
				best_error = error
				best_h_column = c		
		self.column_pred = best_h_column
		return self

	def predict(self,X):		
		X[X[:,self.column_pred] == 0,self.column_pred] = -1
		return X[:,self.column_pred]

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
	for n_est in range(1,10):
		bds = AdaBoostDecisionStump(n_estimators=n_est)
		# bds = AdaBoostClassifier(n_estimators=n_est)
		bds.fit(X,y)
		pred = bds.predict(X)
		print("Acuracia: " +str((pred==y).sum()/float(len(pred))))
	# embed()

if __name__ == "__main__":
	main()