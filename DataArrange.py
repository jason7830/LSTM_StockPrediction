import os
import argparse
import csv
from datetime import datetime

class Arranger():
    def __init__(self,file):
        self.file = file

    def minguo2Date(self, ming):
        y , m , d = ming.split('/')
        return datetime(int(y)+1911,int(m),int(d))

    def writeCSV(self,rows):
        with open(self.file,'w') as f:
            writer = csv.writer(f,lineterminator='\n')
            writer.writerows(rows)

    def Dupli_Sort(self):
        dict_rows = {}
        #remove Duplitcate
        with open(self.file,'r') as f:
            reader = csv.reader(f)
            #store the previous row temporary
            tmp = None
            for r in reader:
                #remove empty row
                if '--' in str(r):
                    continue
                #計算不比價的漲幅欄位
                if r[-2] == 'X0.00':
                    r[-2] = "{:.2f}".format(float(r[-3]) - float(tmp[-3]))
                dict_rows[r[0]] = r
                tmp = r
        #sort by date
        rows = [row for date, row in sorted(dict_rows.items(), key=lambda x: self.minguo2Date(x[0]))]
        self.writeCSV(rows)
        print("[{}] has been arranged!".format(self.file))



def main():    
    parser = argparse.ArgumentParser(description='Data Arraning')
    parser.add_argument('-f','--file', nargs='*',
        help='CSV file to be arranged.')
    parser.add_argument('-d','--dir', nargs='*',
        help='All csv files in dir to be arranged.')
    
    args = parser.parse_args()
    if args.dir != None:
        file_names = os.listdir(args.dir[0])
        for file_name in file_names:
            if not file_name.endswith('.csv'):
                continue
            arranger = Arranger("{}/{}".format(args.dir[0],file_name))
            arranger.Dupli_Sort()
    elif args.file != None:
        arranger = Arranger(args.file[0])
        arranger.Dupli_Sort()
    else:
        print('Enter atleast one arg, see help by "-h".')

if __name__ == '__main__':
    main()
