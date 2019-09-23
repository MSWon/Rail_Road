import pandas as pd
import os
import re
import numpy as np

def speed2accel(speed):
    accel = [0]
    for i in range(1, len(speed)):
        accel.append(speed[i] - speed[i-1])
    return accel

def abs_distance(distance, code_number):
    new_dist = [distance[0]]
    for i in range(1, len(distance)):
        if(code_number[i] != code_number[i-1]):
            new_dist.append(distance[i])
        elif(distance[i]==0 and distance[i-1] != 0):
            new_dist.append(distance[i])
        else:
            new_dist.append(distance[i] - distance[i-1])
    return np.cumsum(new_dist)

def preprocess(folder_name, file_name):
    
    data = pd.read_excel(folder_name + "/" + file_name)
    
    for col_name in data.columns:
        if(col_name != "Time" and col_name != "Next Station code"):
            if(data[col_name].dtype == 'O'):
                data[col_name] = data[col_name].map(lambda x: re.sub("[a-zA-Z]|/", "", x))
                data[col_name] = data[col_name].map(lambda x: float(re.sub("．", ".", x.rstrip())))
               
    data = data[data["Next Station code"] != "범어사(133)"]
    data = data[data["Next Station code"] != "노포(134)"]
    data = data[data["Next Station code"] != "다대포항(96)"]
    
    data = data.reset_index(drop=True)
    
    new_df = pd.DataFrame()
    new_df["Power"] = [0.0]*len(data)
    
    for idx in [5,9,11,13,15,17]:
        new_df["Power"] += ((data.iloc[:,idx]*data.iloc[:,idx+1])/1000)
          
    code_number = data["Next Station code"].map(lambda x: int(re.findall("\d+", x)[0]))
    
    new_df["Time"] = data["Time"]
    new_df["Speed"] = data["Speed"]
    new_df["Next Station code"] = data["Next Station code"]
    new_df["Powering"] = data["Car2_VVVF-POWERING"]
    new_df["Braking"] = data["Car2_VVVF-BRAKING"]
    new_df["Power"] *= data["Car2_VVVF-POWERING"] - data["Car2_VVVF-BRAKING"]
    new_df["abs_loc"] = abs_distance(data["Distance"], code_number)
    new_df["accel"] = speed2accel(data["Speed"]) 
    new_df = new_df[["Time","Speed","Next Station code","Powering","Braking","Power","abs_loc", "accel"]]
    return new_df

folder_dir = os.listdir()

north_df = pd.DataFrame()
south_df = pd.DataFrame()

for folder_name in folder_dir:
    file_name_list = os.listdir(folder_name)
    
    for file_name in file_name_list:
    
        if("상" in file_name):            
            new_df = preprocess(folder_name, file_name)
            north_df = north_df.append(new_df, ignore_index=True)            
        elif("하" in file_name):
            new_df = preprocess(folder_name, file_name)
            south_df = south_df.append(new_df, ignore_index=True)    

north_df.to_csv("./north_train.csv", index=False)
south_df.to_csv("./south_train.csv", index=False)