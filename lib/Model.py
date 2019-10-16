import os
import numpy as np
from datetime import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
import keras
import matplotlib.pyplot as plt 

class Model():
    def __init__(self,if_sequential=False,load=None):
        self.save_path = 'save/{}'.format(dt.strftime(dt.today(), '%Y-%m-%d_%H.%M'))
        if load:
            self.model = load_model(load)
            return None
        #If model should be built as a sequential model.
        self.if_sequential = if_sequential
        if if_sequential:
            self.model = Sequential()
        self.input_layers = []
        self.hidden_layers = []
        self.output_layers = []

    def add_saveinfo(self,stock=None,units=None,batch_size=None,epochs=None,exinfo=''):
        stock = ' stock-'+stock if stock else ''
        units = ' units-'+units if units else ''
        batch_size = ' batch_size-'+batch_size if batch_size else ''
        epochs = ' epochs-'+epochs if epochs else ''
        self.save_path += (stock + units + batch_size + epochs + exinfo)

    def addLayer(self,layer):
        #add hidden layer.
        if self.if_sequential:
            self.model.add(layer)
        else:
            self.hidden_layers.append(layer)

    def addIOLayer(self,input_layer,output_layer):
        #Add Input/Output layers. Only for API model.
        self.input_layers.append(input_layer)
        self.output_layers.append(output_layer)

    def build_model(self,loss,optimizer,output_activation_layer=None):
        if self.if_sequential:
            self.model.compile(loss=loss,optimizer=optimizer)
            return self.model
        for i,ip in enumerate(self.input_layers):
            tmp = None
            for j,layer in enumerate(self.hidden_layers):
                if j == 0:
                    tmp = layer(ip)
                    continue
                tmp = layer(tmp)
            self.output_layers[i] = self.output_layers[i](tmp)
            if output_activation_layer:
                self.output_layers[i] = output_activation_layer(self.output_layers[i])
        self.model = keras.models.Model(inputs=self.input_layers,outputs=self.output_layers)
        self.model.compile(loss=loss,optimizer=optimizer)
        return self.model

    def train(self,train_x,train_y,val_x,val_y,epochs,batch_size,callbacks=[],save=True,shuffle=False):
        self.history = self.model.fit(train_x,train_y,shuffle=shuffle,epochs=epochs,batch_size=batch_size,callbacks=callbacks,validation_data=(val_x,val_y))
        print('Training Completed.')
        if save:
            self.model.save(self.save_path+'.h5')
            print('Model Saved at "{}"'.format(self.save_path))

    def train_generator(self,train_gen,train_steps_per_epoch,val_gen,val_steps_per_epoch,epochs,callbacks=[],workers=1,save=True,shuffle=False):
        self.history = self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch = train_steps_per_epoch,
            validation_data=val_gen,
            validation_steps=val_steps_per_epoch,
            epochs = epochs,
            shuffle=shuffle,
            workers=workers,
            callbacks=callbacks)
        print('Training Completed.')
        if save:
            self.model.save(self.save_path+'.h5')
            print('Model Saved at "{}"'.format(self.save_path))

    def predict_recurrent(self,x,prediction_len,multi_io=False):
        #Predict recursively.
        x=np.array(x)
        prediction_seqs = []
        loop = len(x) if not multi_io else x.shape[1] 
        loop = int(loop/prediction_len)
        for i in range(loop):
            print('Predicting recursively ... (window: {}/{}) '.format(i+1,loop),end='\r')
            curr_frame = x[i * prediction_len] if not multi_io else x[:,i*prediction_len][:,newaxis,:,:].tolist()
            preds=[]
            for j in range(prediction_len):
                if multi_io :
                    preds.append(np.array(self.model.predict(curr_frame))[:,0])
                    curr_frame = np.array(curr_frame)[:,:,1:]
                    tmp = []
                    for k,p in enumerate(preds[-1]):
                        tmp.append(np.insert(curr_frame[k][0],curr_frame.shape[2],p[0],axis=0))
                    curr_frame = np.array(tmp)[:,newaxis,:,:].tolist()
                else:
                    preds.append(self.model.predict(curr_frame[newaxis,:,:]))
                    curr_frame = curr_frame[1:]
                    curr_frame = np.insert(curr_frame,x.shape[1]-1,preds[-1],axis=0)
            prediction_seqs.append(preds)
        return prediction_seqs

    def plot_recurrent_results(self,predicted_data, true_data, prediction_len, target_index=0, multi_io=False, frames_to_plt = 0, plt_seperated=False, show_legend=True, save = False):
        frames_to_plt = len(predicted_data) if frames_to_plt == 0 else frames_to_plt
        true_data = true_data if not multi_io else true_data[target_index]
        predicted_data = np.array(predicted_data)
        for i, data in enumerate(predicted_data):
            if i == (frames_to_plt):
                break
            padding = [None for p in range(i * prediction_len)] if not plt_seperated else []
            data = data if not multi_io else data[:,target_index,0].tolist()
            plt.plot(padding + data)
            if show_legend:
                plt.legend()
            if plt_seperated:
                plt.plot(true_data[i*prediction_len:(i+1)*prediction_len], label='True Data')
                plt.show()
        if not plt_seperated:
            if show_legend:
                plt.legend()
            plt.plot(true_data, label='True Data')
            if save:
                plt.savefig(self.save_path[:5]+'figs/'+self.save_path[5:]+'.png',dpi=200)
                print('Figure saved [{}].'.format(self.save_path[5:]+'.png'))
            plt.show()

    def get_avg_price_error(self,predicted_data,true_data):
        self.avg_error = np.mean(np.abs(predicted_data-true_data)/true_data)
        print("Average Errors:",self.avg_error)

    def plot_results(self, predicted_data, true_data, plt_ammount=0, show_legend=True, save = False):
        plt_ammount = len(predicted_data) if plt_ammount == 0 else plt_ammount
        plt.plot(true_data[:plt_ammount],label=true_data)
        plt.plot(predicted_data[:plt_ammount])
        if show_legend:
            plt.legend()
        if save:
            plt.savefig(self.save_path[:5]+'figs/'+self.save_path[5:]+'.png',dpi=200)
        plt.show()





