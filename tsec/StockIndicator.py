import os
import numpy as np
import csv

class Indicators():
	def __init__(self , path):
		self.path = path
		self.list = []
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

	#n日長短ema,return: [emans , emanl , diff]
	def ema(self , ns , nl):
		di = []
		dif = [0] * (nl)
		emans = [0] * (ns-1)
		emanl = [0] * (nl-1)
		for i in range(len(self.list)):
			#print(i,self.list[i][4],self.list[i][5],self.list[i][6])
			di.append((float(self.list[i][4])+float(self.list[i][5])+float(self.list[i][6])*2)/4)
			if i == ns-1:
				emans.append(np.mean(di))
			if i >= ns:
				emans.append((emans[i-1] * (ns - 2) + di[i] * 2) / ns)
				if i>= nl:
					emanl.append((emanl[i-1] * (nl - 2) + di[i] * 2) / nl)
					dif.append(emans[i] - emanl[i])
			if i == nl-1:
				emanl.append(np.mean(di))
		return [emans , emanl , dif]

	def macd(self , ns , nl ,n):
		emans , emanl , dif = self.ema(ns,nl)
		macdn = [0] * (nl-1)
		for i in range(nl,len(self.list)):
			pass
			if i == nl:
				macdn.append(np.mean(dif[i-n+1:i]))
			macdn.append((macdn[i-1] * (n-1) + dif[i] * 2) / (n+1))
		print(macdn[26:40])

			


indicator = Indicators("data/2330.csv")
indicator.macd(12,26,9)