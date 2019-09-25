# -*- coding: utf-8 -*-
import requests
from requests.exceptions import ConnectionError
import os
from os import mkdir
from os.path import isdir
from datetime import datetime
import argparse
import logging
import csv
import re
class Crawler():
    def __init__(self,dir='data'):
        # save in directory 'data'
        if not isdir(dir):
            mkdir(dir)
        self.dir = dir
    
    def writeCSV(self,stock_no,row):
        with open("{}/{}.csv".format(self.dir,stock_no),'a') as f:
            writer = csv.writer(f,lineterminator='\n')
            writer.writerow(row)

    def getStockByNo(self,date,no):
        twse = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        para = {
            'response' : 'csv',
            'date' : date,
            'stockNo' : no
        }
        result = requests.get(twse,params=para)
        if not result.ok:
            raise ConnectionError
        for s in reversed(result.text.split('\n')[2:-6]):
            #row cleanup and split
            row = re.sub('[",]','',s[:-2].replace(',"','_')).split('_')
            self.writeCSV(no,row)

def subOneMonth(date):
    #datetime backward
    tu = date.timetuple()
    year = tu[0] 
    month = tu[1] - 1
    if month == 0:
        year -= 1
        month = 12
    return datetime(year,month,tu[2])

def pushMsg(msg,type=0):
    #push and log msg
    print(msg)
    if type == 0 :
        logging.info(msg)
    else:
        logging.error(msg)


def main():
    #logs
    logging.basicConfig(filename='craw.log',level=logging.INFO,format='%(asctime)s - [%(levelname)s] - %(message)s')
    parser = argparse.ArgumentParser(description='Crawl data at assigned day')
    parser.add_argument('-day', nargs='?',
        help='assigned day (format: YYYY-MM), default is today')
    parser.add_argument('-bday', nargs='?',
        help='crawl back from today to assigned day, maxium backs to 2010')
    parser.add_argument('-e', '--end', action='store_true',
        help='crawl back from today to assigned day, maxium backs to 2010')
    parser.add_argument('-no' , type=int ,
        help='Sotck Number')
    args = parser.parse_args()

    # Day only accept 0 or 2 arguments
    day = None
    try:
        #set day to assigned fisrt day or by default day which is today
        if args.day == None:
            day = datetime.today()
        else :
            day = datetime(int(args.day[:4]),int(args.day[5:]),1)

        crawler = Crawler()

        if args.end or args.bday != None:
            #crawl backwards to 2010-1 or assigned date
            if args.end:
                b_day = datetime(2009,12,31)
            else:
                day = datetime(int(args.day[:4]),int(args.day[5:]),1)

            while(day>b_day):
                d = str(day.date())
                crawler.getStockByNo(d.replace('-',''),args.no)
                msg = "{}-{} Stock.{} done!".format(d[:4],d[5:7],args.no)
                pushMsg(msg)
                day = subOneMonth(day)
        else:
            d = str(day.date())
            crawler.getStockByNo(d.replace('-',''),args.no)
            msg = "{}-{} Stock.{} done!".format(d[:4],d[5:7],args.no)
            pushMsg(msg)
    except ConnectionError:
        pushMsg("已超出最大請求數量!",1)
        return
    except Exception as e:
        pushMsg(e,1)

if __name__ == '__main__':
    main()


