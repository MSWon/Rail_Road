# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 19:37:43 2018

@author: jbk48
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:34:56 2018

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
import re
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import matplotlib as mpl

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

## font for Korean
font_location = "C:/Windows/Fonts/NanumSquareR.ttf"
font_name = font_manager.FontProperties(fname=font_location)
rc('font', family="NanumSquare")
mpl.rcParams['axes.unicode_minus'] = False

train_data = pd.read_csv("./data_1_north_train.csv", engine='python')
test_data = pd.read_csv("./data_1_north_test.csv", engine='python')


def test_sample(data):

    code = data['Next.Station.code']
    
    index_list = []
    
    for i in range(len(data)-1): 
        if(re.findall("\d+", code[i])[0] == '132' and re.findall("\d+", code[i+1])[0] == '97'):
            index_list.append(i)
    
    
    return data.iloc[:index_list[0]+1,:]


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
 

def make_window(data, window, mode = "speed"):
    
    batch = data.iloc[:,[2,5,6,7,9,10,11,12]].values
    
    batch_X = []
    batch_Y = []
    
    for i in range(len(batch)-window):
            ## scale_batch = scaler.fit_transform(batch)
        batch_X.append(batch[i:i+window,:6])
        if(mode == "speed"):
            batch_Y.append(batch[i+window][6])
        elif(mode == "power"):
            batch_Y.append(batch[i+window][7])
            
    batch_Y = np.reshape(batch_Y, (-1,1))
    batch_X = np.array(batch_X)
    
    return batch_X, batch_Y

window = 10

test_data = test_sample(test_data)
train_clean, scaler = scale_train(train_data)

test_group = [g for _, g in test_data.groupby("Next.Station.code", sort =False)]
model = load_model("north_power_model.keras")


for split_df in test_group:
    
    filename = split_df.iloc[0,3]
    Y_real = list(split_df['Power'])
    test_clean = scale_test(split_df, scaler)
    X_test, Y_test = make_window(test_clean, window, "power")

   
    Y_pred = model.predict(X_test)
    
    plt.figure(figsize = (20,10))
    plt.plot(range(len(Y_real)), Y_real, label='real', color = "blue")
    plt.plot(range(10,len(Y_real)), Y_pred, label='pred', color = "red")
    plt.title("상행선 - {}".format(filename), size = 25)
    plt.ylabel('power', size = 15)
    plt.xlabel('time(seconds)', size = 15)
    plt.legend(prop = {'size' : 20})
    plt.savefig("./graph_power/{}.png".format(filename))
    plt.show()


## for MAPE


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


train_clean, scaler = scale_train(train_data)
test_clean = scale_test(test_data, scaler)

test_batch = make_batch(test_clean)
X_test, Y_test = make_window(test_batch, window, "power")

Y_real = Y_test
Y_pred = model.predict(X_test)

def MAPE(y, y_):
    
    return(np.sum(abs(y-y_)) / np.sum(abs(y)))


MAPE(Y_real, Y_pred) * 100

MAPE_list.sort(reverse = True)
np.mean(MAPE_list) * 100

per = int(len(MAPE_list) * 0.20)
np.mean(MAPE_list[per:]) * 100

