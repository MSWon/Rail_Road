# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:45:40 2018

@author: jbk48
"""

import os
import pandas as pd
import numpy as np
## os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

train_data = pd.read_csv("./data_1_north_train.csv", engine='python')
test_data = pd.read_csv("./data_1_north_test.csv", engine='python')

def scale_train(data):
    
    scaler = MinMaxScaler()
    speed_power = data.iloc[:, [2,7]]
    speed_power.columns = ['Speed2', 'Power2']
    data.iloc[:, [2,5,6,7,9,10]] = scaler.fit_transform(data.iloc[:, [2,5,6,7,9,10]])
    data = pd.concat([data, speed_power], axis = 1)
    
    return data, scaler

def scale_test(data, scaler):
    
    speed_power = data.iloc[:, [2,7]]
    speed_power.columns = ['Speed2', 'Power2']
    data.iloc[:, [2,5,6,7,9,10]] = scaler.transform(data.iloc[:, [2,5,6,7,9,10]])
    data = pd.concat([data, speed_power], axis = 1)
    
    return data
    

def make_batch(data):
    
    station_name = list(set(data['Next.Station.code']))
      
    total_batch = []
    
    for name in station_name:
        
        sub_data = data[data['Next.Station.code']==name]
        
        index_list = [0]
        
        for i in range(len(sub_data['No'])-1):            
            if(abs(sub_data['No'].values[i+1] - sub_data['No'].values[i]) > 3):
                index_list.append(i+1)
        
        index_list.append(len(sub_data['No']))
        
        split_list = []
        
        for j in range(len(index_list)-1):
            split_list.append(sub_data.iloc[index_list[j]:index_list[j+1], [2,5,6,7,9,10,11,12]].values)
        
        total_batch += split_list
    
    mask = np.random.permutation(len(total_batch))
    
    return np.array(total_batch)[mask]
 

def make_window(total_batch, window, mode = "speed"):
    
    batch_X = []
    batch_Y = []
    for batch in total_batch:
        for i in range(len(batch)-window):
            batch_X.append(batch[i:i+window,:6])
            if(mode == "speed"):
                batch_Y.append(batch[i+window][6])
            elif(mode == "power"):
                batch_Y.append(batch[i+window][7])
            
    batch_Y = np.reshape(batch_Y, (-1,1))
    batch_X = np.array(batch_X)
    return batch_X, batch_Y


window = 10
n_inputs = 6
n_lstm_nodes = 256


train_clean, scaler = scale_train(train_data)
test_clean = scale_test(test_data, scaler)

train_batch = make_batch(train_clean)
X_train, Y_train = make_window(train_batch, window, "power")

test_batch = make_batch(test_clean)
X_test, Y_test = make_window(test_batch, window, "power")


model = Sequential()
model.add(LSTM(n_lstm_nodes, input_shape=(window, n_inputs),
                 kernel_initializer="he_normal", recurrent_initializer="he_normal"))
## model.add(Dropout(0.9))
model.add(Dense(1))

optimizer = Adam(lr=0.001,  decay= 0.0001)
model.compile(optimizer=optimizer,loss='mse')

history = model.fit(X_train,Y_train,
                    epochs=300,
                    batch_size=512,
                    validation_data=(X_test,Y_test),shuffle=True)

model.save("north_power_model.keras")



