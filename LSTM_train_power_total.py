# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:08:02 2019

@author: jbk48
"""
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras import regularizers
from keras.models import Sequential
from keras.models import Input
from keras.layers import LSTM, Dense, Dropout, GRU, TimeDistributed, Reshape
from keras.models import load_model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with open("./LSTM_train_data_power.pkl", 'rb') as file:
    X_train, Y_train = pickle.load(file)
    
with open("./LSTM_test_data_power.pkl", 'rb') as file:
    X_test, Y_test = pickle.load(file)    

window = 10
n_inputs = 6
n_lstm_nodes = 256

model = Sequential()
model.add(LSTM(n_lstm_nodes, input_shape=(window, n_inputs), return_sequences=True,
                 kernel_initializer="he_normal", recurrent_initializer="he_normal"))
model.add(LSTM(n_lstm_nodes, input_shape=(window, n_inputs), return_sequences=True,
                 kernel_initializer="he_normal", recurrent_initializer="he_normal"))
model.add(LSTM(n_lstm_nodes, input_shape=(window, n_inputs), return_sequences=True,
                 kernel_initializer="he_normal", recurrent_initializer="he_normal"))
model.add(LSTM(n_lstm_nodes, kernel_initializer="he_normal", recurrent_initializer="he_normal"))
## model.add(Dropout(0.9))
model.add(Dense(1))

optimizer = Adam(lr=0.001,  decay= 0.0001)
model.compile(optimizer=optimizer,loss='mse')

checkpoint = ModelCheckpoint('model-{epoch:03d}-{loss:.03f}-{val_loss:.03f}.h5', 
                             verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

history = model.fit(X_train,Y_train,
                    epochs=300,
                    batch_size=512,
                    validation_data=(X_test,Y_test),
                    shuffle=True,
                    callbacks=[checkpoint])

train_loss = history.history['loss']
test_loss = history.history['val_loss']


df = pd.DataFrame({'train_loss':train_loss ,'test_loss':test_loss})
                   
df.to_csv("LSTM_result.csv", sep="," , index=False)
