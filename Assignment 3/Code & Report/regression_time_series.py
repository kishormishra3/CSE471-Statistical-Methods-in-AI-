#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:55:07 2020

@author: kishor
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

import sys
import numpy as np
import pandas as pd
file=sys.argv[1]
data = pd.read_csv(file, sep=';')
data_=data["Global_active_power"].values
data=[]
for i in data_:
  if i=="?":
    i=np.nan
  data.append(i)
data=np.array(data)
rg=len(data)-61
X_train=[]
Y_train=[]
count=0
limit = 60000
while count < limit:
  i= np.random.randint(low=0, high=rg)
  temp=data[i:i+61]
  temp = np.float32(temp)
  if(np.isnan(temp).any()):
    continue
  X_train.append(temp[:-1])
  Y_train.append(temp[60])
  count+=1
X_train=np.array(X_train)
Y_train=np.array(Y_train)

model = Sequential()
model.add(Dense(100, activation ='relu', input_dim = X_train.shape[1]))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(1,activation = 'relu'))

model.compile(loss="mse", optimizer='adam')

model.fit(X_train,Y_train, epochs=100, batch_size=100,verbose=0)

data=np.float64(data)
temp=data[0:60]
mean=np.nanmean(temp)
value=[]
for i in range(60):
  if(np.isnan(data[i])):
    data[i]=mean
    value.append(mean)
for i in range(len(data)):
  if(np.isnan(data[i])):
    x=data[i-60:i]
    x=x.reshape([1,60])
    ynew = model.predict(x)
    data[i]=ynew[0][0]
    value.append(ynew[0][0])
for i in value:
    print(i)