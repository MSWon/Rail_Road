# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:33:45 2019

@author: jbk48
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


def scale_test(data, scaler):
    speed_power = data.iloc[:, [2,7]]
    speed_power.columns = ['Speed2', 'Power2']
    data.iloc[:, [2,5,6,7,9,10]] = scaler.transform(data.iloc[:, [2,5,6,7,9,10]])
    data = pd.concat([data, speed_power], axis = 1)
    return data

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
    
    ## mask = np.random.permutation(len(total_batch))
    
    return np.array(total_batch)
 

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

sub_dir = ["2018년 9월 21일"]
train_num = ["49편성","50편성","51편성"]

test_data = pd.DataFrame()


for date in sub_dir:
    for num in train_num:
        test_data_sub = pd.read_csv("./{}/{}/전처리3/data_1_north_test.csv".format(date, num), engine='python')
        test_data = test_data.append(test_data_sub, ignore_index=True)

window = 5

if(not os.path.exists("./window_{}".format(window))):
    os.mkdir("./window_{}".format(window))

with open("./window_{}/LSTM_min_max_scaler.pkl".format(window), 'rb') as file:
    scaler = pickle.load(file)

test_clean = scale_test(test_data, scaler)
test_batch = make_batch(test_clean)
mask = np.random.permutation(len(test_batch))
test_batch = test_batch[mask]
test_batch = test_batch[:20]
test_data = make_window(test_batch, window, "power")

with open("./mask.pkl", 'wb') as file:
    pickle.dump(mask, file)

with open("./window_{}/LSTM_test_data_power.pkl".format(window), 'wb') as file:
    pickle.dump(test_data, file)

with open("./window_{}/LSTM_test_data_power.pkl".format(window), 'rb') as file:
    X_test2, Y_test2 = pickle.load(file)