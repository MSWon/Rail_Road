# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:26:10 2018

@author: jbk48
"""

import numpy as np
import pandas as pd
import time
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

train_data = pd.read_csv("./data_1_north_train.csv", engine='python')
test_data = pd.read_csv("./data_1_north_test.csv", engine='python')

    
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
            split_list.append(sub_data.iloc[index_list[j]:index_list[j+1], [2,5,6,7,9,10]].values)
        
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
                batch_Y.append(batch[i+window][0])
            elif(mode == "power"):
                batch_Y.append(batch[i+window][3])
            
    batch_X = np.reshape(batch_X, (-1,6*window))
    return batch_X, batch_Y

def MSE(a,b):
    return(np.mean(np.power(a-b,2)))


window = 10
path_model = "./north_power_model_ef.pkl"

train_batch = make_batch(train_data)
X_train, Y_train = make_window(train_batch, window, "power")

test_batch = make_batch(test_data)
X_test, Y_test = make_window(test_batch, window, "power")

start_time = time.time()

model = ExtraTreesRegressor(n_estimators = 100, max_depth = 20, max_features = 0.8)
model.fit(X_train, Y_train)

duration = time.time() - start_time
minute = int(duration / 60)
second = int(duration) % 60

print("%dminutes %dseconds" % (minute,second))

y_pred = model.predict(X_test)
y_real = Y_test

print("MSE : {}".format(MSE(y_real,y_pred)))

with open(path_model, 'wb') as file:
    pickle.dump(model, file)