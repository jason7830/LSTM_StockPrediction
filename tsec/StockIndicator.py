import os
import numpy as np
import csv

class Indicators():
	def __init__(self , path):
		self.path = path
		self.list = [0]
		self.incs = [0]
		self.decs = [0]
		self.closed = []
		with open(path) as data:
			for row in csv.reader(data):
				self.list.append(row)
		for i in range(1,len(self.list)):
			self.closed.append(float(self.list[i][7]))
			if i < 1 :
				continue
			t = float(self.list[i][7])
			if t >= 0:
				self.incs.append(t)
				self.decs.append(0)
			else:
				self.incs.append(0)
				self.decs.append(t*-1)


	#n日RSI
	def rsi(self , n): 
		#漲/跌
		incs = [0]
		decs = [0]
		for i in range(n-1,len(self.list)):
			rs = (np.mean(self.incs[i-n+1:i+1])/(np.mean(self.incs[i-n+1:i+1]) + np.mean(self.decs[i-n+1:i+1])))
			rsi = 100 - 100 / (1 + rs)
			print(rsi)

	def wrsi(self,n):
		#Upt-1
		Upt = 0
		up = np.mean(self.incs[1:7]) 
		Dnt = 0
		dn = np.mean(self.decs[1:7])
		#print(up,dn,self.decs[1:7])
		for i in range(n+1,len(self.list)):
			Upt = up
			up = Upt + 1 / n * (self.incs[i] - Upt)
			Dnt = dn
			dn = Dnt + 1 / n * (self.decs[i] - Dnt)
			rs = up / dn
			wrsi = 100 * up /  (dn + up)
			print(up,dn,wrsi)


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
indicator.wrsi(6)