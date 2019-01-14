import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor(object):
	def __init__(self,path = "tsec/data/2030.csv"):
		self.path = path

	def show(self):
		df = pd.read_csv(self.path)
		print(df.tail())


def main():
	p = DataProcessor()
	p.show()



main()