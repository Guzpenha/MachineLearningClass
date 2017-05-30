import numpy as np
import pandas as pd

from IPython import embed

class DecisionStumpCategorical():
	
	def __init__(self):
		pass
	
	def fit(self,stumpFeature,predValue,verbose=False):
		#labels must be the size of distinct values in stumpFeature
		self.stumpFeature = stumpFeature
		self.predValue = predValue

	def predict(self,X):
		y = pd.DataFrame(X[self.stumpFeature]).apply(lambda r,v=self.predValue: 1 if str(r)==v else -1,axis=1)
		return y

class BoostingDecisionStump():
	
	def __init__(self):
		self.samples_weights = []
		self.hypothesis = []
		self.hypothesis_weights = []


	def best_single_stump(X,y):
		#para cada variavel
		#	para cada valor que a variavel possui
			dsc = DecisionStumpCategorical(feature,valor)
			h = dsc.fit(X)
			pred = h.predict(X)
			error = #soma de erros de prediç~ao * self.sample_weights

		#retorna h com menor error 
		
	def fit(self,X,y,verbose=False):		
		#All instances have equal initial weight
		self.samples_weights = [1.0/X.shape[0]] * X.shape[0]

		h = self.best_single_stump(X,y)

		error = h.predict()
		alpha = 0.5 * np.log((1 - error)/error) # calculate alpha

		self.hypothesis.append(h)
		self.hypothesis_weights.append(alpha)
		embed()
	
	def predict(self,X):
		pass

def main():	
	# Read and prepare data
	data = pd.read_csv("../data/tictactoe.csv",header=None,sep=",").rename(columns={9: "label"})
	
	#Preprocess data	
	data["label"] = data.apply(lambda r: 1 if r.label == "positive" else -1,axis=1)
	X = data[[c for c in data.columns if c !="label"]].as_matrix()	
	y = data["label"]

	# Fit the data using the BoostingDecisionStump
	bds = BoostingDecisionStump()
	bds.fit(X,y,verbose=True)

if __name__ == "__main__":
	main()