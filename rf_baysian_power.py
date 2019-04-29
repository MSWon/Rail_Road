# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:02:47 2018

@author: jbk48
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args


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
    return np.array(batch_X), np.array(batch_Y)

def MSE(a,b):
    return(np.mean(np.power(a-b,2)))


#########################################################################
dim_n_estimators = Integer(low=50, high=200, name = 'n_estimators')
dim_max_features = Real(low=0.5, high=0.99, name = 'max_features')
dim_max_depth = Integer(low=10, high=50, name = 'max_depth')
dim_min_samples_split = Integer(low=2, high=30, name='min_samples_split')
dim_min_samples_leaf = Integer(low=1, high=10, name='min_samples_leaf')


dimensions=[dim_n_estimators,
            dim_max_features,
            dim_max_depth,
            dim_min_samples_split,
            dim_min_samples_leaf]


default_parameters = [100, 0.5, 30, 5, 5]
################################################################################ 
window = 10

train_data = pd.read_csv("./data_1_north_train.csv", engine='python')
test_data = pd.read_csv("./data_1_north_test.csv", engine='python')

train_batch = make_batch(train_data)
X, y = make_window(train_batch, window, "power")
test_batch = make_batch(test_data)
X_test, y_test = make_window(test_batch, window, "power")


kf = KFold(n_splits=4,random_state=1,shuffle=True)
kf.get_n_splits(X)

path_best_model = 'ef_best_model.pkl'
best_mse  = 100000000000000
lst = []
cnt = 0
@use_named_args(dimensions=dimensions)
def fitness(n_estimators, max_features, max_depth, min_samples_split,
                 min_samples_leaf):
    global cnt
    cnt = cnt + 1
    print("{}th".format(cnt))
    print("n_estimators : ", n_estimators)
    print("max_features : ", max_features)
    print("max_depth : ", max_depth)
    print("min_samples_split : ", min_samples_split)
    print("min_samples_leaf : ", min_samples_leaf)
    
    model = ExtraTreesRegressor(n_estimators=n_estimators,
                                   max_features=max_features,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   n_jobs=-1,
                                   random_state = 0)
    train = []
    valid = []
    test = []
    
    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        model.fit(X_train, y_train)
               
        train.append(MSE(model.predict(X_train),y_train))
        valid.append(MSE(model.predict(X_valid),y_valid))
    
    model.fit(X, y)
    test.append(MSE(model.predict(X_test),y_test))
    
    train = np.array(train)
    valid = np.array(valid)
    test = np.array(test)
    avg_train = train.sum()/4
    avg_valid = valid.sum()/4
    avg_mse = float((test + avg_valid)/2)
    
    print()
    print("Train : ", train)
    print("Train MSE : ", avg_train)
    print("Valid : ", valid)
    print("Valid MSE : ", avg_valid)
    print("Test MSE : ", test)
    print("Average MSE : ", avg_mse)
    print()
    
    global lst
    arr = [n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf,
           train,avg_train, valid, avg_valid, avg_mse, test]
    lst.append(arr)
    
    global best_mse
    
    if avg_mse < best_mse:
        with open(path_best_model, 'wb') as file:
            pickle.dump(model, file)
        best_mse = avg_mse
    
    del model
    return avg_mse

search_result  = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=50,
                            x0=default_parameters)

DATA = pd.DataFrame(lst)

print(search_result.x)
print(search_result.fun)

DATA.to_csv('./ef_history_power.csv', sep =',')