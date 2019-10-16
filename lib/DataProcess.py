import numpy as np
import pandas as pd

class Data():
    """
    sr: split rate
    file: path of a csv file or a dataframe or np.array
    type: 0 = read from path ; 1 = dataframe ; 2 = np.array
    """ 
    def __init__(self,sr,file,type=0,cols=None,drop_cols=None,drop_rows=None):
        if type < 2:
            if type == 0:
                df = pd.read_csv(filepath)
            else:
                df = file
            index = int(len(df)*sr)
            if drop_cols != None:
                df = df.drop(drop_cols,axis=1)
            if drop_rows != None:
                df = df.drop(drop_rows)
            cols = df.columns if cols == None else cols
            self.data_train = df.get(cols).values[:index]
            self.data_val = df.get(cols).values[index:]
        else:
            index = int(len(file)*sr)
            self.data_train = file[:index]
            self.data_val = file[index:]

    def data_steps(self,data,window_size):
        windows = []
        for i in range(len(data) - window_size):
            windows.append(data[i:i+window_size])
        return np.array(windows)

    def get_x_y(self,windows,target_index=0,if_single_col=False):
        if if_single_col:
            x = windows[:,:-1]
            y = windows[:,[-1]]
        else:
            x = windows[:,:-1]
            y = windows[:,-1,[target_index]]
        return x,y

    """
    target_index: the index of target y.
    is_normalised: if loaded data has been normalised, won't be normalised by the function below
    if_single_col: if its only one output
    """
    def get_train(self,window_size,target_index=0,indicator=True,is_normalised=False,if_single_col=False):
        windows = self.data_steps(self.data_train,window_size)
        if indicator and (not is_normalised):
            windows = self.normalise_indicators_window(np.array(windows).astype(float))
        windows = np.array(windows)
        return self.get_x_y(windows,target_index=target_index,if_single_col=if_single_col)

    def get_val(self,window_size,indicator=True,is_normalised=False,if_single_col=False):
        windows = []
        for i in range(len(self.data_val) - window_size):
            windows.append(self.data_val[i:i+window_size])
        if indicator and (not is_normalised):
            windows = self.normalise_indicators_window(np.array(windows).astype(float))
        windows = np.array(windows)
        if if_single_col:
            x = windows[:,:-1]
            y = windows[:,[-1]]
        else:
            x = windows[:,:-1]
            y = windows[:,-1,[0]]
        return x,y
    #normalise by cauculating the percentage of Variety from the first day.
    def normalise_indicators_window(self,windows,offset = 1):
        #offset to prevent zero-division
        normalised_data = []
        for w in windows:
            normalised_window = []
            for col_index in range(w.shape[1]):
                cols = [(((float(c)+offset) / (float(w[0,col_index])+offset)) - 1) for c in w[:,col_index]]
                normalised_window.append(cols)
            normalised_data.append(np.array(normalised_window).T)
        return np.array(normalised_data)

    def generate_train_batch(x_windows, y_windows, batch_size, is_multi_windows = False):
        x_windows = np.array(x_windows)
        y_windows = np.array(y_windows)
        i = 0
        windows_len = len(x_windows) if not is_multi_windows else x_windows.shape[1]
        while True:
            x_batch = None
            y_batch = None
            if is_multi_windows:
                #data must be a list as a multi IO
                x_batch = [c for c in x_windows[:,i:i+batch_size:,:,]]
                y_batch = [c for c in y_windows[:,i:i+batch_size,]]
            else:
                x_batch = x_windows[i:i+batch_size,]
                y_batch = y_windows[i:i+batch_size,]
            i += batch_size
            if i >= windows_len:
                i=0
            yield x_batch, y_batch

    def predict_trend(self,period):
        pass

#1. 分別讀入股票和籌碼後再丟來這，y只有收盤
#2. 多輸出