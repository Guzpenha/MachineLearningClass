import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin

class CategoricalStump():
	
	def __init__(self):
		self.column_pred = None
		self.inverted = None
		self.constant = False

	def fit(self,X_input,y,sample_weight):
		X = X_input.copy()
		best_error = float("inf")
		best_h_column = None

		#stumps that predict the column values
		for c in range(X.shape[1]):
			X[X[:,c] == 0,c] = -1
			error = (X[:,c] != y).dot(sample_weight)
			if(error < best_error):
				best_error = error
				best_h_column = c
				self.inverted = False

		#stumps that predict the inverse of the column values
		for c in range(X.shape[1]):						
			X[X[:,c] == 1,c] = -1
			X[X[:,c] == 0,c] =  1
			error = (X[:,c] != y).dot(sample_weight)			
			if(error < best_error):
				best_error = error
				best_h_column = c
				self.inverted = True

		#constant 1 prediction
		error = (np.ones(X.shape[0]) != y).dot(sample_weight)
		if(error < best_error):
			self.constant = 1
		#constant -1 prediction
		error = (np.ones(X.shape[0])*-1 != y).dot(sample_weight)
		if(error < best_error):
			self.constant = -1

		self.column_pred = best_h_column
		return self

	def predict(self,X_input):
		X = X_input.copy()
		if(self.constant == 1):
			return np.ones(X.shape[0])
		elif(self.constant == -1):
			return np.ones(X.shape[0]) *-1

		if(self.inverted):
			X[X[:,self.column_pred] == 1,self.column_pred] = -1
			X[X[:,self.column_pred] == 0,self.column_pred] =  1
		else:
			X[X[:,self.column_pred] == 0,self.column_pred] = -1
		return X[:,self.column_pred]

class AdaBoostCategoricalClassifier(BaseEstimator,ClassifierMixin):
	
	def __init__(self, n_estimators = 500):
		self.sample_weights = []
		self.estimators = []
		self.estimators_weights = []
		self.n_estimators = n_estimators
		
	def fit(self,X,y,):
		#All instances have equal initial weight
		self.sample_weights = np.ones(X.shape[0])/X.shape[0]		

		for i in range(self.n_estimators):			
			dsc = CategoricalStump()			
			h = dsc.fit(X,y,sample_weight=self.sample_weights)			
			error = (h.predict(X) != y).dot(self.sample_weights)
			alpha = 0.5 * (np.log((1 - error)/error)) 

			self.estimators.append(h)
			self.estimators_weights.append(alpha)

			self.sample_weights = self.sample_weights * np.exp(-alpha * h.predict(X) * y)
			self.sample_weights = (self.sample_weights/self.sample_weights.sum()).as_matrix()			

	def predict(self,X):
		pred = np.zeros(X.shape[0])
		for (h, alpha) in zip(self.estimators,self.estimators_weights):
			pred+= h.predict(X) * alpha	
		return np.sign(pred)

def main():	
	data = pd.read_csv("../data/tictactoe.csv",header=None,sep=",").rename(columns={9: "label"})
	
	#Preprocess data
	data["label"] = data.apply(lambda r: 1 if r.label == "positive" else -1,axis=1)
	X = pd.get_dummies(data[[c for c in data.columns if c !="label"]]).as_matrix().astype(int)
	y = data["label"]

	# Fit the data using the AdaBoostCategoricalClassifier
	for n_estimators in [100,500,1000,1500]:
		print("CV accuracy scores for "+str(n_estimators)+": ")
		clf = AdaBoostCategoricalClassifier(n_estimators=n_estimators)
		scores = cross_val_score(clf, X, y, cv=5,scoring='accuracy')
		print(scores)

	# clf.fit(X,y)
	# pred = clf.predict(X)
	# print("Acuracia : " +str((pred==y).sum()/float(len(pred))))

if __name__ == "__main__":
	main()