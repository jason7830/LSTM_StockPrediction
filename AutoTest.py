from sklearn.preprocessing import MinMaxScaler , StandardScaler , QuantileTransformer , normalize
from sklearn.metrics import r2_score
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras import metrics, regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class Tester():
    def __init__(self,inputs,outputs):
        self.model = None
        self.batch_size = 0
        self.units = 0
        self.epochs = 0
        self.l2rate = 0
        self.loss = 'mean_squared_error'
        self.optimizer = 'Adam'
        self.inputs = inputs
        self.outputs = outputs
        self.validation_data = []
        self.callbacks = []

    def newModel(self):
        self.model =  Model(inputs=self.inputs,outputs=self.outputs)
        self.model.compile(loss=self.loss,optimizer=self.optimizer)
        return self.model

    def visualize(self,preds,val,history,saveName,r2score,save=True):
        closed_val_loss = history['val_dense_9_loss']
        plt.figure(1)
        plt.subplot(211)
        plt.plot(preds[-1],'b',label='Predictions')
        plt.plot(val[1][-1],'r',label='Real')
        plt.title('LSTM_Stock-PredictionsV3',)
        plt.xlabel('Index')
        plt.ylabel('Price')
        score = "{:0.3f}".format(float(r2score))
        plt.text(0,plt.ylim()[1]*0.9,'R2 score: '+ score)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.subplot(212)
        plt.title("Trainning Lose")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(closed_val_loss,label='loss')
        plt.legend(loc='upper right')
        notate_xy = [np.argmin(closed_val_loss),np.min(closed_val_loss)]
        plt.annotate('epoch min: {:0.4f}'.format(notate_xy[1]), xy=notate_xy, xytext=(np.argmin(closed_val_loss)*0.5, np.min(closed_val_loss) + (np.max(closed_val_loss)-np.min(closed_val_loss))*0.5),arrowprops=dict(facecolor='black', shrink=0.05),)
        plt.grid(True)
        plt.subplots_adjust(hspace=0.7)
        if save :
            plt.savefig('pltfigs/'+saveName+'.png',dpi=200)
        return plt


    def fit(self,x,y,val,epochs,batch_size,save=True,showfig=True):
        self.epochs = epochs
        self.batch_size = batch_size
        history = self.model.fit(x,y,shuffle=False,epochs=epochs,batch_size=batch_size,validation_data=val,callbacks=self.callbacks)
        preds = self.model.predict(val[0])
        #closed price score
        histories = history.history
        closed_val_loss = histories[list(histories.keys())[-1]]
        self.finish_time = time.strftime("%Y-%m-%d %H_%M_%S",time.localtime())
        r2score = "{:0.3f}".format(float(r2_score(val[1][-1],preds[-1])))
        saveName = self.finish_time+' LSTM_Stock-Predictions'+'_r2score~'+str(r2score)+'_l2rate'+str(self.l2rate)+'_units'+str(self.units)+'_epochs'+str(self.epochs)+'_batch_size'+str(self.batch_size)
        fig = self.visualize(preds,val,history.history,saveName,r2score)
        if showfig :
            fig.show()
        if save:
            self.model.save(saveName+'.h5')
        return saveName + " Done!"




