# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:56:34 2019

@author: jbk48
"""

import os
import pandas as pd
import numpy as np
import pickle
## os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from sklearn.preprocessing import MinMaxScaler


sub_dir = ["2018년 8월 14일","2018년 9월 4일"]
train_num = ["49편성","50편성","51편성"]

train_data = pd.DataFrame()


for date in sub_dir:
    for num in train_num:
        train_data_sub = pd.read_csv("./{}/{}/전처리3/data_1_north_train.csv".format(date, num), engine='python')
        train_data = train_data.append(train_data_sub, ignore_index=True)

def scale_train(data):
    
    scaler = MinMaxScaler()
    speed_power = data.iloc[:, [2,7]]
    speed_power.columns = ['Speed2', 'Power2']
    data.iloc[:, [2,5,6,7,9,10]] = scaler.fit_transform(data.iloc[:, [2,5,6,7,9,10]])
    data = pd.concat([data, speed_power], axis = 1)
    
    return data, scaler

    
def make_batch(data):
    
    start_index = [0]
    for i in range(1, len(data)):
        if(data['Next.Station.code'][i]=="낫개(97)" and data['Next.Station.code'][i-1]=="남산(132)"):
            start_index.append(i)
            
    end_index = []
    for i in range(0, len(data)-1):
        if(data['Next.Station.code'][i]=="남산(132)" and data['Next.Station.code'][i+1]=="낫개(97)"):
            end_index.append(i)
    
    end_index.append(len(data)-1)
    
    total_batch = []
    
    for j in range(len(start_index)):   
        total_batch.append(data.iloc[start_index[j]:end_index[j], [2,5,6,7,9,10,11,12]].values)
    
    return total_batch
 

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
    mask = np.random.permutation(len(batch_X))
    return batch_X[mask], batch_Y[mask]


window = 5
n_inputs = 6
n_lstm_nodes = 256

if(not os.path.exists("./window_{}".format(window))):
    os.mkdir("./window_{}".format(window))

train_clean, scaler = scale_train(train_data)

with open("./window_{}/LSTM_min_max_scaler.pkl".format(window), 'wb') as file:
    pickle.dump(scaler, file)

train_batch = make_batch(train_clean)
train_data = make_window(train_batch, window, "power")

with open("./window_{}/LSTM_train_data_power.pkl".format(window), 'wb') as file:
    pickle.dump(train_data, file)

