# -*- coding: utf-8 -*-

import os
import numpy as np
import csv
import pandas as pd
import argparse

"""
list[0], # 日期
list[1], # 成交股數
list[2], # 成交金額
list[3], # 開盤價
list[4], # 最高價
list[5], # 最低價
list[6], # 收盤價
list[7], # 漲跌價差
list[8], # 成交筆數
"""

class Indicator():
	def __init__(self , path):
		self.path = path
		self.list = []
		self.incs = [0]
		self.decs = [0]
		self.high = []
		self.low = []
		self.closed = []
		self.volume = []
		with open(path,'r') as data:
			for row in csv.reader(data):
				self.list.append(row)
		for i in range(len(self.list)):
			self.volume.append(float(self.list[i][1]))
			self.high.append(float(self.list[i][4]))
			self.low.append(float(self.list[i][5]))
			self.closed.append(float(self.list[i][6]))

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
			self.rsi = 100 - 100 / (1 + rs)
		return self.rsi

	def wrsi(self,n):
		#Upt-1
		Upt = 0
		up = np.mean(self.incs[1:7]) 
		Dnt = 0
		dn = np.mean(self.decs[1:7])
		#print(up,dn,self.decs[1:7])
		wrsi = [0] * (n+1)
		for i in range(n+1,len(self.list)):
			Upt = up
			up = Upt + 1 / n * (self.incs[i] - Upt)
			Dnt = dn
			dn = Dnt + 1 / n * (self.decs[i] - Dnt)
			#rs = up / dn
			wrsi.append(100 * up /  (dn + up))
		return wrsi


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
		macdn = [0] * (nl+n-2)
		for i in range(nl+n-1,len(self.list)):
			pass
			if i == (nl+n-1): # n日內 => [i-n+1:i+1]
				macdn.append(np.mean(dif[i-n+1:i+1]))
			macdn.append((macdn[i-1] * (n-1) + dif[i] * 2) / (n+1))
		return [emans , emanl , macdn] 

	def KD(self , n):
		pass
		k = [0] * (n-2)
		d = [0] * (n-2)
		k.append(50)
		d.append(50)
		for i in range(n-1,len(self.list)):
			pass
			nlow = min(self.low[i-n+1:i+1])
			nhigh = max(self.high[i-n+1:i+1])
			rsv = (self.closed[i] - nlow) / (nhigh - nlow) * 100
			k.append((k[i-1] * 2 + rsv) / 3)
			d.append((d[i-1] * 2 + k[i]) / 3)
		return [k , d]

def generateCSV(file):
	indicator = Indicator(file)		
	ema6 , ema12 ,dif  = indicator.ema(6,12)
	ema12 , ema26 , macd12_26_9 = indicator.macd(12,26,9)
	wrsi = indicator.wrsi(6)
	K , D = indicator.KD(9)
	dict_indicators = {
		"closed" : indicator.closed,
		"volume": indicator.volume,
		"ema6" : ema6,
		"ema12" : ema12,
		"ema26" : ema26,
		"macd" : macd12_26_9,
		"wrsi6" : wrsi,
		"K9": K,
		"D9" : D 
	}
	pd.DataFrame(dict_indicators).to_csv(file[:-4]+"_indicators.csv",sep=',',encoding='utf-8')
	print("{} Indicators has generated.".format(file))

def main():
    parser = argparse.ArgumentParser(description='Stock Indicators.')
    parser.add_argument('-f','--file', nargs='*',
        help='Stock to generate indicators.')
    parser.add_argument('-d','--dir', nargs='*',
        help='All csv files in dir to be generate indicators.')
    args = parser.parse_args()
    if args.dir != None:
        file_names = os.listdir(args.dir[0])
        for file_name in file_names:
            if not file_name.endswith('.csv'):
                continue
            generateCSV("{}/{}".format(args.dir[0],file_name))
    elif args.file != None:
        generateCSV(args.file[0])
    else:
        print('Enter atleast one arg, see help by "-h".')


if __name__ == "__main__":
	main()
"""
indicator = Indicators("data/2330.csv")		
ema6 , ema12 ,dif  = indicator.ema(6,12)
ema12 , ema26 , macd12_26_9 = indicator.macd(12,26,9)
wrsi = indicator.wrsi(6)
K , D = indicator.KD(9)



cols = ["EMA","wRSI","K","D"]
dict = {
	"closed" : indicator.closed,
	"volume": indicator.volume,
	"ema6" : ema6,
	"ema12" : ema12,
	"ema26" : ema26,
	"macd" : macd12_26_9,
	"wrsi6" : wrsi,
	"K9": K,
	"D9" : D 
}

df = pd.DataFrame(dict)
df.to_csv("data/2330_indicators.csv", sep=',', encoding='utf-8')

"""
