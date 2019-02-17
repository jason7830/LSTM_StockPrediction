import os
import numpy as np
import csv

class Indicators():
	def __init__(self , path):
		self.path = path
		self.list = []
		with open(path) as data:
			for row in csv.reader(data):
				self.list.append(row)


	#n日RSI
	def rsi(self , n): 
		#漲/跌
		incs = [0]
		decs = [0]
		for i in range(1,len(self.list)):
			#(float(self.list[i][7]) > 0)
			t = float(self.list[i][7])
			if t >= 0 :
				incs.append(t)
				decs.append(0)
			else:
				incs.append(0)
				decs.append(t*-1)
			if i >= n:
				rs = (np.mean(incs[i-n+1:i+1])/(np.mean(incs[i-n+1:i+1]) + np.mean(decs[i-n+1:i+1])))
				rsi = 100 - 100 / (1 + rs)
				print(rsi)

	#n日移動平均線			
	def ma(self , n):
		closed = []
		for i in range(0,len(self.list)):
			closed.append(float(self.list[i][6]))
			if i >= n-1 :
				ma = np.mean(closed[i-n+1:i+1])
				print(ma)

	def ema(self , n):
		di = []
		ema = [0] * 11
		for i in range(len(self.list)):
			di.append((float(self.list[i][4])+float(self.list[i][5])+float(self.list[i][6])*2)/4)
			if i == n-1:
				ema.append(np.mean(di))
			if i >= n:
				ema.append((ema[i-1] * (n-2) + di[i] * 2)/n)


			


indicator = Indicators("data/2330.csv")
indicator.ema(6)