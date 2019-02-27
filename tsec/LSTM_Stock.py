from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val




df = pd.read_csv("data/2330_indicators.csv")
X_train = df[:-1].values
y_train = df.loc[1:,['closed']].values
X_train , Y_train , X_val , Y_val = splitData(X_train,y_train,0.2)


sc = MinMaxScaler()
X_train_sc = sc.fit_transform(X_train)
X_val_sc = sc.transform(X_val)

X_train_steps = []
Y_train_steps = []
for i in range(20, len(X_train_sc)):  # 1258 是訓練集總數
    X_train_steps.append(X_train_sc[i-20:i])
    Y_train_steps.append(Y_train[i])
X_train, Y_train = np.array(X_train_steps), np.array(Y_train_steps)

X_val_steps = []
Y_val_steps = []
for i in range(20,len(X_val_sc)):
	X_val_steps.append(X_val_sc[i-20:i])
	Y_val_steps.append(Y_val[i])
X_val , Y_val = np.array(X_val_steps) , np.array(Y_val_steps)


print(X_train.shape)

#X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
#X_val = np.reshape(X_val,(X_val.shape[0], 1,1))

model_lstm = Sequential()
model_lstm.add(LSTM(9,input_shape=X_train.shape[1:],activation='relu',kernel_initializer='lecun_uniform')) 
model_lstm.add(Dense(1))
model_lstm.summary()
model_lstm.compile(loss='mean_squared_error',optimizer='Adam')
history_model_lstm = model_lstm.fit(X_train, Y_train, epochs=200, batch_size=1, shuffle=False)

y_pred_test_lstm = model_lstm.predict(X_val)
y_train_pred_lstm = model_lstm.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(Y_train, y_train_pred_lstm)))
r2_train = r2_score(Y_train, y_train_pred_lstm)
"""
print("The Adjusted R2 score on the Train set is:\t{:0.3f}\n".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
r2_test = r2_score(Y_val, y_pred_test_lstm)
print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))
#train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=trainset.index)
#print(train_sc_df.tail())
"""