import numpy as np
import pandas as pd

from IPython import embed

class DecisionStumpCategorical():
	
	def __init__(self):
		pass

	def fit(self,stumpFeature,predValue,verbose=False):
		self.stumpFeature = stumpFeature
		self.predValue = predValue
		
		return self

	def predict(self,X):
		y = pd.DataFrame(X).apply(lambda r,v=self.predValue,f=self.stumpFeature: 1 if str(r[f])==v else -1,axis=1)
		return y

class AdaBoostDecisionStump():
	
	def __init__(self, n_estimators = 50):
		self.sample_weights = []
		self.hypothesis = []
		self.hypothesis_weights = []
		self.n_estimators = n_estimators

	def best_single_stump(self,X,y):
		min_error_sum = 1000
		min_error = []
		h = None
		for feature in range(X.shape[1]):
			for value in np.unique(X[:,feature]):				
				dsc = DecisionStumpCategorical()
				h = dsc.fit(feature,value)
				pred = h.predict(X)
				error = (pred != y).dot(self.sample_weights)
				if(error.mean()<min_error_sum):
					min_error_sum = error.mean()
					min_error = error
					min_h = h
		return min_h, min_error
		
	def fit(self,X,y,verbose=False):		
		#All instances have equal initial weight
		self.sample_weights = np.ones(X.shape[0])/X.shape[0]

		for i in range(self.n_estimators):
			h, error = self.best_single_stump(X,y)					
			alpha = 0.5 * (np.log(1 - error.mean()) - np.log(error.mean())) # calculate alpha

			self.hypothesis.append(h)
			self.hypothesis_weights.append(alpha)

			self.sample_weights = self.sample_weights * np.exp(-alpha * h.predict(X) * y)
			self.sample_weights = self.sample_weights/self.sample_weights.sum()
	
	def predict(self,X):
		pred = np.zeros(X.shape[0])
		for (h, alpha) in zip(self.hypothesis,self.hypothesis_weights):
			pred+= h.predict(X) * alpha
		# embed()
		return np.sign(pred)

def main():	
	# Read and prepare data
	data = pd.read_csv("../data/tictactoe.csv",header=None,sep=",").rename(columns={9: "label"})
	
	#Preprocess data	
	data["label"] = data.apply(lambda r: 1 if r.label == "positive" else -1,axis=1)
	X = data[[c for c in data.columns if c !="label"]].as_matrix()	
	y = data["label"]

	# Fit the data using the AdaBoostDecisionStump
	bds = AdaBoostDecisionStump(n_estimators=20)
	bds.fit(X,y,verbose=True)
	pred = bds.predict(X)
	print((pred==y).sum())
	print(len(pred))
	# embed()

if __name__ == "__main__":
	main()