import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor(object):
	def __init__(self,path = "tsec/data/2030.csv"):
		self.path = path

	def show(self):
		df = pd.read_csv(self.path)
		y = df.iloc[0:,6].values
		for i in range(len(y)-1):
			if y[i] >= y[i+1]:
				y[i] = 0
			else:
				y[i] = 1
		X = df.iloc[0:,[0,1,3,4,5,6]].values

		print(X)



def main():
	p = DataProcessor()
	p.show()



main()